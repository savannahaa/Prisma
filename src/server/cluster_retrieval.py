"""
候选检索与本地 rerank 工具集。

提供三类能力：
1) dense 分支：固定预算候选（支持簇内 fixed_k，优先 HNSW ANN）。
2) cluster payload 分支：按簇打包候选。
3) 统一后处理：用原始 query 在候选内做 exact top-k rerank。

这些函数被 `run_online_pipeline.py` 调用，是在线阶段的检索算子层。
"""

from __future__ import annotations

# Allow running this file directly: `python src/server/cluster_retrieval.py`
if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import pickle
import time
from typing import Optional, List, Tuple

import numpy as np

from shared.config import (
    WORKSET_CLUSTER_INFO_PATH,
    EVAL_K,
    FIXED_K,
    NUM_CLUSTERS,
    TARGET_CLUSTER_SIZE,
    EPS,
    TRACK1_DENSE_RETRIEVAL_BACKEND,
    TRACK1_HNSW_SPACE,
    TRACK1_HNSW_M,
    TRACK1_HNSW_EF_CONSTRUCTION,
    TRACK1_HNSW_EF_SEARCH_BASE,
)
from shared.cluster_info_contract import assert_cluster_info_contract
from shared.gpu_accel import (
    angular_distance_to_rows as gpu_angular_distance_to_rows,
    cosine_scores_1d as gpu_cosine_scores_1d,
)

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

# HNSW 参数（来自 shared.config，便于论文实验可复现调参）。
HNSW_SPACE = str(TRACK1_HNSW_SPACE)
HNSW_M = int(TRACK1_HNSW_M)
HNSW_EF_CONSTRUCTION = int(TRACK1_HNSW_EF_CONSTRUCTION)
HNSW_EF_SEARCH_BASE = int(TRACK1_HNSW_EF_SEARCH_BASE)
HNSW_ADD_BATCH_ROWS = 50000

_HNSW_CACHE: dict = {
    "docs_obj_id": None,
    "shape": None,
    "metric_mode": None,
    "index": None,
    "ef_search": None,
}

_CLUSTER_DOC_CACHE: dict = {
    "docs_obj_id": None,
    "shape": None,
    "clusters": {},  # cluster_id -> {"chunk_sig": tuple, "chunk": np.ndarray, "docs": np.ndarray}
}

_HNSW_CLUSTER_CACHE: dict = {}  # (docs_obj_id, shape, metric_mode) -> {"index": idx, "ef_search": int}


#读离线半径表
def load_cluster_info(path: str = WORKSET_CLUSTER_INFO_PATH):
    with open(path, "rb") as f:
        cluster_info = pickle.load(f)
    assert_cluster_info_contract(
        cluster_info,
        expected_eval_k=int(EVAL_K),
        expected_fixed_k=int(FIXED_K),
        expected_num_clusters=int(NUM_CLUSTERS),
        expected_target_cluster_size=int(TARGET_CLUSTER_SIZE),
    )
    return cluster_info


# 算余弦相似度 / 角距离（带稳定性保证的 tiebreak）
def cosine_scores(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    gpu_scores = gpu_cosine_scores_1d(
        query=np.asarray(query, dtype=np.float32).reshape(-1),
        docs=np.asarray(docs, dtype=np.float32),
        assume_unit_norm=True,
        prefer_float64=True,
        cache_docs_if_large=True,
        role="server",
    )
    if gpu_scores is not None:
        return np.asarray(gpu_scores, dtype=np.float64)

    # 使用 float64 做 exact rerank 打分，减少全量/子集矩阵乘法带来的 ulp 级翻转。
    query = np.asarray(query, dtype=np.float64)
    docs = np.asarray(docs, dtype=np.float64)

    q_norm = float(np.linalg.norm(query))
    if q_norm <= EPS:
        raise ValueError("query 向量范数过小。")

    # 逐行求和，保证同一 (query, doc) 在不同批量上下文中得到一致分数。
    numer = np.sum(docs * query[None, :], axis=1, dtype=np.float64)
    d_norms = np.sqrt(np.sum(docs * docs, axis=1, dtype=np.float64))
    denom = np.maximum(q_norm * d_norms, EPS)
    return (numer / denom).astype(np.float64)


def angular_distances(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    gpu_theta = gpu_angular_distance_to_rows(
        query=np.asarray(query, dtype=np.float32).reshape(-1),
        rows=np.asarray(docs, dtype=np.float32),
        assume_unit_norm=True,
        cache_rows_if_large=True,
        role="server",
    )
    if gpu_theta is not None:
        return np.asarray(gpu_theta, dtype=np.float64)

    sims = np.clip(cosine_scores(query, docs), -1.0, 1.0)
    return np.arccos(sims).astype(np.float64)


def topk_indices_with_tiebreak(scores: np.ndarray, top_k: int, global_indices: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    global_indices = np.asarray(global_indices, dtype=np.int32)
    if scores.ndim != 1 or global_indices.ndim != 1 or scores.shape[0] != global_indices.shape[0]:
        raise ValueError("scores/global_indices 形状不匹配，无法做稳定 top-k。")
    order = np.lexsort((global_indices, -scores))
    return order[:top_k].astype(np.int32)


def smallest_k_indices_with_tiebreak(
    distances: np.ndarray,
    top_k: int,
    global_indices: np.ndarray,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    global_indices = np.asarray(global_indices, dtype=np.int32)
    if distances.ndim != 1 or global_indices.ndim != 1 or distances.shape[0] != global_indices.shape[0]:
        raise ValueError("distances/global_indices 形状不匹配，无法做稳定 top-k。")
    order = np.lexsort((global_indices, distances))
    return order[:top_k].astype(np.int32)


def _resolve_faiss_metric_mode() -> tuple[Optional[int], Optional[str], Optional[str]]:
    """
    将 TRACK1_HNSW_SPACE 映射到 FAISS metric。
    论文主线要求检索按角距离排序；在单位球上，最小角距离与最大 cosine 等价，
    因此这里只允许 cosine / inner-product 等价实现。
    """
    space = str(HNSW_SPACE).strip().lower()
    if space in {"cosine", "ip", "inner_product"}:
        if faiss is None:
            return None, None, "faiss_not_installed"
        return int(faiss.METRIC_INNER_PRODUCT), "ip_cosine", None
    return None, None, f"unsupported_non_angular_hnsw_space_{HNSW_SPACE}"


def _normalize_rows_f32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return (x / norms).astype(np.float32)


def _cluster_chunk_signature(chunk: np.ndarray) -> tuple:
    chunk = np.asarray(chunk, dtype=np.int32).reshape(-1)
    if len(chunk) == 0:
        return (0, -1, -1, 0)
    return (
        int(len(chunk)),
        int(chunk[0]),
        int(chunk[-1]),
        int(np.sum(chunk, dtype=np.int64) % 1000000007),
    )


def _get_or_build_cluster_docs(
    *,
    cluster_id: int,
    chunks,
    docs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    docs_obj_id = int(id(docs))
    docs_shape = tuple(int(x) for x in getattr(docs, "shape", ()))

    if _CLUSTER_DOC_CACHE["docs_obj_id"] != docs_obj_id or _CLUSTER_DOC_CACHE["shape"] != docs_shape:
        _CLUSTER_DOC_CACHE["docs_obj_id"] = docs_obj_id
        _CLUSTER_DOC_CACHE["shape"] = docs_shape
        _CLUSTER_DOC_CACHE["clusters"] = {}

    chunk = np.asarray(chunks[int(cluster_id)], dtype=np.int32).reshape(-1)
    chunk_sig = _cluster_chunk_signature(chunk)
    cached = _CLUSTER_DOC_CACHE["clusters"].get(int(cluster_id))
    if cached is None or cached.get("chunk_sig") != chunk_sig:
        cluster_docs = np.asarray(docs[chunk], dtype=np.float32)
        cached = {
            "chunk_sig": chunk_sig,
            "chunk": chunk,
            "docs": cluster_docs,
        }
        _CLUSTER_DOC_CACHE["clusters"][int(cluster_id)] = cached

    return np.asarray(cached["docs"], dtype=np.float32), np.asarray(cached["chunk"], dtype=np.int32)


def _build_or_get_hnsw_index(docs: np.ndarray, fixed_k: int):
    """
    构建或复用 FAISS HNSW 索引，仅当 docs 指针/shape/metric 变化时重建。
    """
    backend = str(TRACK1_DENSE_RETRIEVAL_BACKEND).strip().lower()
    if backend != "faiss_hnsw_ann":
        return None, f"backend_disabled_{TRACK1_DENSE_RETRIEVAL_BACKEND}"
    if faiss is None:
        return None, "faiss_not_installed"

    metric_type, metric_mode, metric_reason = _resolve_faiss_metric_mode()
    if metric_reason is not None:
        return None, str(metric_reason)

    docs_shape = tuple(int(x) for x in getattr(docs, "shape", ()))
    if len(docs_shape) != 2:
        raise ValueError("docs 必须是二维矩阵。")

    docs_obj_id = int(id(docs))
    cache_hit = (
        _HNSW_CACHE["index"] is not None
        and _HNSW_CACHE["docs_obj_id"] == docs_obj_id
        and _HNSW_CACHE["shape"] == docs_shape
        and _HNSW_CACHE["metric_mode"] == metric_mode
    )

    if not cache_hit:
        num_docs, dim = docs_shape
        index = faiss.IndexHNSWFlat(int(dim), int(HNSW_M), int(metric_type))
        if hasattr(index, "hnsw"):
            index.hnsw.efConstruction = int(HNSW_EF_CONSTRUCTION)

        batch_rows = int(max(1, HNSW_ADD_BATCH_ROWS))
        for start in range(0, int(num_docs), int(batch_rows)):
            end = min(int(num_docs), int(start + batch_rows))
            docs_batch = np.asarray(docs[start:end], dtype=np.float32)
            if metric_mode == "ip_cosine":
                docs_batch = _normalize_rows_f32(docs_batch)
            index.add(np.ascontiguousarray(docs_batch, dtype=np.float32))

        _HNSW_CACHE["index"] = index
        _HNSW_CACHE["docs_obj_id"] = docs_obj_id
        _HNSW_CACHE["shape"] = docs_shape
        _HNSW_CACHE["metric_mode"] = metric_mode
        _HNSW_CACHE["ef_search"] = None

    index = _HNSW_CACHE["index"]
    wanted_ef = int(max(int(fixed_k), int(HNSW_EF_SEARCH_BASE)))
    if _HNSW_CACHE["ef_search"] != wanted_ef:
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = int(wanted_ef)
        _HNSW_CACHE["ef_search"] = int(wanted_ef)

    return index, None


def _build_or_get_cluster_hnsw_index(cluster_docs: np.ndarray, fixed_k: int):
    backend = str(TRACK1_DENSE_RETRIEVAL_BACKEND).strip().lower()
    if backend != "faiss_hnsw_ann":
        return None, f"backend_disabled_{TRACK1_DENSE_RETRIEVAL_BACKEND}"
    if faiss is None:
        return None, "faiss_not_installed"

    metric_type, metric_mode, metric_reason = _resolve_faiss_metric_mode()
    if metric_reason is not None:
        return None, str(metric_reason)

    cluster_docs = np.asarray(cluster_docs, dtype=np.float32)
    if cluster_docs.ndim != 2:
        raise ValueError("cluster_docs 必须是二维矩阵。")

    cache_key = (int(id(cluster_docs)), tuple(int(x) for x in cluster_docs.shape), str(metric_mode))
    cache_entry = _HNSW_CLUSTER_CACHE.get(cache_key)
    if cache_entry is None:
        num_docs, dim = cluster_docs.shape
        index = faiss.IndexHNSWFlat(int(dim), int(HNSW_M), int(metric_type))
        if hasattr(index, "hnsw"):
            index.hnsw.efConstruction = int(HNSW_EF_CONSTRUCTION)

        if metric_mode == "ip_cosine":
            docs_for_index = _normalize_rows_f32(cluster_docs)
        else:
            docs_for_index = np.asarray(cluster_docs, dtype=np.float32)
        index.add(np.ascontiguousarray(docs_for_index, dtype=np.float32))

        cache_entry = {"index": index, "ef_search": None}
        _HNSW_CLUSTER_CACHE[cache_key] = cache_entry

    index = cache_entry["index"]
    wanted_ef = int(max(int(fixed_k), int(HNSW_EF_SEARCH_BASE)))
    if cache_entry.get("ef_search") != wanted_ef:
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = int(wanted_ef)
        cache_entry["ef_search"] = int(wanted_ef)

    return index, None


def prewarm_dense_cluster_retrieval_runtime(
    *,
    chunks,
    docs: np.ndarray,
    cluster_local_fixed_k: int,
    multicluster_fixed_k_per_cluster: int,
) -> dict:
    """
    在正式 online 三阶段计时前预热 dense 路径的一次性运行时对象：
    1) 每簇 `chunk -> cluster_docs` 切片缓存；
    2) 每簇 FAISS HNSW 索引。

    这些对象依赖固定工作集 docs/chunks，属于 once-only setup，不应混入逐 query
    的 server_query 在线时延。
    """
    num_clusters = int(len(chunks))
    warm_fixed_k = int(
        max(
            1,
            int(cluster_local_fixed_k),
            int(multicluster_fixed_k_per_cluster),
        )
    )

    cluster_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    cluster_sizes: List[int] = []

    t0_docs = time.perf_counter()
    for cid in range(num_clusters):
        cluster_docs, cluster_global_idx = _get_or_build_cluster_docs(
            cluster_id=int(cid),
            chunks=chunks,
            docs=docs,
        )
        cluster_pairs.append((cluster_docs, cluster_global_idx))
        cluster_sizes.append(int(len(cluster_global_idx)))
    cluster_doc_cache_build_sec = float(time.perf_counter() - t0_docs)

    backend = str(TRACK1_DENSE_RETRIEVAL_BACKEND).strip().lower()
    hnsw_enabled = backend == "faiss_hnsw_ann"
    cluster_hnsw_build_sec = 0.0
    hnsw_warmed_cluster_count = 0
    hnsw_disabled_reason = None

    if hnsw_enabled:
        t0_hnsw = time.perf_counter()
        for cluster_docs, _cluster_global_idx in cluster_pairs:
            index, reason = _build_or_get_cluster_hnsw_index(
                cluster_docs=cluster_docs,
                fixed_k=int(warm_fixed_k),
            )
            if index is None:
                hnsw_disabled_reason = str(reason)
                raise RuntimeError(
                    "dense cluster-runtime prewarm failed while building cluster-local HNSW "
                    f"index: reason={hnsw_disabled_reason}"
                )
            hnsw_warmed_cluster_count += 1
        cluster_hnsw_build_sec = float(time.perf_counter() - t0_hnsw)
    else:
        hnsw_disabled_reason = f"backend_disabled_{TRACK1_DENSE_RETRIEVAL_BACKEND}"

    return {
        "retrieval_backend": str(TRACK1_DENSE_RETRIEVAL_BACKEND),
        "num_clusters": int(num_clusters),
        "cluster_sizes": [int(x) for x in cluster_sizes],
        "cluster_size_min": int(min(cluster_sizes)) if cluster_sizes else 0,
        "cluster_size_max": int(max(cluster_sizes)) if cluster_sizes else 0,
        "cluster_doc_cache_build_sec_once": float(cluster_doc_cache_build_sec),
        "cluster_hnsw_build_sec_once": float(cluster_hnsw_build_sec),
        "cluster_hnsw_warm_fixed_k": int(warm_fixed_k),
        "cluster_doc_cache_warmed_cluster_count": int(len(cluster_pairs)),
        "cluster_hnsw_warmed_cluster_count": int(hnsw_warmed_cluster_count),
        "cluster_hnsw_enabled": bool(hnsw_enabled),
        "cluster_hnsw_disabled_reason": hnsw_disabled_reason,
        "total_setup_sec_once": float(cluster_doc_cache_build_sec + cluster_hnsw_build_sec),
    }


def prewarm_dense_global_retrieval_runtime(
    *,
    docs: np.ndarray,
    fixed_k: int,
) -> dict:
    backend = str(TRACK1_DENSE_RETRIEVAL_BACKEND).strip().lower()
    hnsw_enabled = backend == "faiss_hnsw_ann"
    global_hnsw_build_sec = 0.0
    hnsw_disabled_reason = None
    if hnsw_enabled:
        t0_hnsw = time.perf_counter()
        index, reason = _build_or_get_hnsw_index(
            docs=docs,
            fixed_k=int(max(1, int(fixed_k))),
        )
        if index is None:
            hnsw_disabled_reason = str(reason)
            raise RuntimeError(
                "dense global-runtime prewarm failed while building global HNSW index: "
                f"reason={hnsw_disabled_reason}"
            )
        global_hnsw_build_sec = float(time.perf_counter() - t0_hnsw)
    else:
        hnsw_disabled_reason = f"backend_disabled_{TRACK1_DENSE_RETRIEVAL_BACKEND}"

    return {
        "retrieval_backend": str(TRACK1_DENSE_RETRIEVAL_BACKEND),
        "num_docs": int(len(docs)),
        "global_hnsw_build_sec_once": float(global_hnsw_build_sec),
        "global_hnsw_warm_fixed_k": int(max(1, int(fixed_k))),
        "global_hnsw_enabled": bool(hnsw_enabled),
        "global_hnsw_disabled_reason": hnsw_disabled_reason,
        "total_setup_sec_once": float(global_hnsw_build_sec),
    }


def _hnsw_ann_indices_and_scores(
    query_for_server: np.ndarray,
    docs: np.ndarray,
    fixed_k: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    返回：
    - idx: ANN 候选索引（长度 fixed_k）或 None
    - approx_angular_distances: ANN 近似角距离或 None
    - reason: ANN 不可用时的回退原因
    """
    index, reason = _build_or_get_hnsw_index(docs=docs, fixed_k=int(fixed_k))
    if index is None:
        return None, None, str(reason)

    _metric_type, metric_mode, _metric_reason = _resolve_faiss_metric_mode()
    if metric_mode is None:
        return None, None, "metric_mode_resolve_failed"

    q = np.asarray(query_for_server, dtype=np.float32).reshape(1, -1)
    if metric_mode == "ip_cosine":
        q = _normalize_rows_f32(q)
    q = np.ascontiguousarray(q, dtype=np.float32)
    probe_k = int(
        min(
            int(len(docs)),
            max(int(fixed_k), int(HNSW_EF_SEARCH_BASE)),
        )
    )
    distances, labels = index.search(q, probe_k)

    ann_idx = np.asarray(labels[0], dtype=np.int32)
    ann_dist = np.asarray(distances[0], dtype=np.float32)
    valid = ann_idx >= 0
    ann_idx = ann_idx[valid]
    ann_dist = ann_dist[valid]
    if len(ann_idx) == 0:
        return None, None, "faiss_hnsw_search_empty"

    if metric_mode != "ip_cosine":
        return None, None, f"unsupported_non_angular_metric_mode_{metric_mode}"
    approx_angular_distances = np.arccos(np.clip(ann_dist.astype(np.float64), -1.0, 1.0))

    order_local = smallest_k_indices_with_tiebreak(
        distances=approx_angular_distances,
        top_k=min(len(ann_idx), probe_k),
        global_indices=ann_idx,
    )
    ann_idx = ann_idx[order_local]
    approx_angular_distances = approx_angular_distances[order_local]

    if len(ann_idx) < int(fixed_k):
        theta = angular_distances(query_for_server, docs)
        order_local = smallest_k_indices_with_tiebreak(
            distances=theta,
            top_k=int(fixed_k),
            global_indices=np.arange(len(docs), dtype=np.int32),
        )
        return (
            order_local.astype(np.int32),
            np.asarray(theta[order_local], dtype=np.float32),
            None,
        )
    return (
        ann_idx[: int(fixed_k)].astype(np.int32),
        approx_angular_distances[: int(fixed_k)],
        None,
    )


def _hnsw_ann_ranked_indices_and_scores(
    query_for_server: np.ndarray,
    docs: np.ndarray,
    search_k: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    index, reason = _build_or_get_hnsw_index(docs=docs, fixed_k=int(search_k))
    if index is None:
        return None, None, str(reason)

    _metric_type, metric_mode, _metric_reason = _resolve_faiss_metric_mode()
    if metric_mode is None:
        return None, None, "metric_mode_resolve_failed"

    q = np.asarray(query_for_server, dtype=np.float32).reshape(1, -1)
    if metric_mode == "ip_cosine":
        q = _normalize_rows_f32(q)
    q = np.ascontiguousarray(q, dtype=np.float32)
    k = int(min(int(search_k), int(len(docs))))
    if k <= 0:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32), None

    distances, labels = index.search(q, k)
    ann_idx = np.asarray(labels[0], dtype=np.int32)
    ann_dist = np.asarray(distances[0], dtype=np.float32)
    valid = ann_idx >= 0
    ann_idx = ann_idx[valid]
    ann_dist = ann_dist[valid]
    if len(ann_idx) == 0:
        return None, None, "faiss_hnsw_search_empty"

    if metric_mode != "ip_cosine":
        return None, None, f"unsupported_non_angular_metric_mode_{metric_mode}"
    approx_angular_distances = np.arccos(np.clip(ann_dist.astype(np.float64), -1.0, 1.0))
    order_local = smallest_k_indices_with_tiebreak(
        distances=approx_angular_distances,
        top_k=min(len(ann_idx), k),
        global_indices=ann_idx,
    )
    ann_idx = ann_idx[order_local]
    approx_angular_distances = approx_angular_distances[order_local]
    return (
        ann_idx.astype(np.int32),
        np.asarray(approx_angular_distances, dtype=np.float32),
        None,
    )


def filtered_global_hnsw_topk_doc_indices_and_scores(
    *,
    query_for_server: np.ndarray,
    docs: np.ndarray,
    allowed_doc_indices,
    fixed_k: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    allowed = np.unique(np.asarray(allowed_doc_indices, dtype=np.int32).reshape(-1)).astype(np.int32)
    if allowed.size <= 0:
        return (
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.float32),
            {
                "requested_k": int(max(0, fixed_k)),
                "actual_k": 0,
                "support_size": 0,
                "probe_k_final": 0,
                "used_exact_refill": False,
            },
        )

    target_k = int(min(max(1, int(fixed_k)), int(allowed.size)))
    num_docs = int(len(docs))
    allowed_mask = np.zeros(num_docs, dtype=bool)
    allowed_mask[allowed] = True

    probe_k = int(min(num_docs, max(target_k, int(HNSW_EF_SEARCH_BASE))))
    filtered_idx = np.asarray([], dtype=np.int32)
    filtered_scores = np.asarray([], dtype=np.float32)
    used_exact_refill = False

    while probe_k > 0:
        idx, scores, _reason = _hnsw_ann_ranked_indices_and_scores(
            query_for_server=query_for_server,
            docs=docs,
            search_k=int(probe_k),
        )
        if idx is None or scores is None:
            filtered_idx = np.asarray([], dtype=np.int32)
            filtered_scores = np.asarray([], dtype=np.float32)
            break
        keep = allowed_mask[idx]
        filtered_idx = np.asarray(idx[keep], dtype=np.int32)
        filtered_scores = np.asarray(scores[keep], dtype=np.float32)
        if filtered_idx.size >= target_k or probe_k >= num_docs:
            break
        next_probe = int(min(num_docs, max(int(probe_k) * 2, target_k)))
        if next_probe == int(probe_k):
            break
        probe_k = int(next_probe)

    if filtered_idx.size < target_k:
        theta = angular_distances(query_for_server, docs[allowed])
        order = smallest_k_indices_with_tiebreak(
            distances=theta,
            top_k=int(target_k),
            global_indices=allowed,
        )
        filtered_idx = allowed[order].astype(np.int32)
        filtered_scores = np.asarray(theta[order], dtype=np.float32)
        used_exact_refill = True
    else:
        order = smallest_k_indices_with_tiebreak(
            distances=np.asarray(filtered_scores, dtype=np.float64),
            top_k=int(target_k),
            global_indices=np.asarray(filtered_idx, dtype=np.int32),
        )
        filtered_idx = np.asarray(filtered_idx[order], dtype=np.int32)[: int(target_k)]
        filtered_scores = np.asarray(filtered_scores[order], dtype=np.float32)[: int(target_k)]

    return (
        filtered_idx.astype(np.int32),
        filtered_scores.astype(np.float32),
        {
            "requested_k": int(max(0, fixed_k)),
            "actual_k": int(len(filtered_idx)),
            "support_size": int(len(allowed)),
            "probe_k_final": int(probe_k),
            "used_exact_refill": bool(used_exact_refill),
        },
    )


def _hnsw_ann_indices_and_scores_on_cluster(
    query_for_server: np.ndarray,
    cluster_docs: np.ndarray,
    fixed_k: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    index, reason = _build_or_get_cluster_hnsw_index(cluster_docs=cluster_docs, fixed_k=int(fixed_k))
    if index is None:
        return None, None, str(reason)

    _metric_type, metric_mode, _metric_reason = _resolve_faiss_metric_mode()
    if metric_mode is None:
        return None, None, "metric_mode_resolve_failed"

    q = np.asarray(query_for_server, dtype=np.float32).reshape(1, -1)
    if metric_mode == "ip_cosine":
        q = _normalize_rows_f32(q)
    q = np.ascontiguousarray(q, dtype=np.float32)
    k = int(min(int(fixed_k), int(len(cluster_docs))))
    distances, labels = index.search(q, k)

    ann_idx = np.asarray(labels[0], dtype=np.int32)
    ann_dist = np.asarray(distances[0], dtype=np.float32)
    valid = ann_idx >= 0
    ann_idx = ann_idx[valid]
    ann_dist = ann_dist[valid]
    if len(ann_idx) == 0:
        return None, None, "faiss_hnsw_search_empty"

    if metric_mode != "ip_cosine":
        return None, None, f"unsupported_non_angular_metric_mode_{metric_mode}"
    approx_angular_distances = np.arccos(np.clip(ann_dist.astype(np.float64), -1.0, 1.0))

    order_local = smallest_k_indices_with_tiebreak(
        distances=approx_angular_distances,
        top_k=min(len(ann_idx), k),
        global_indices=ann_idx,
    )
    ann_idx = ann_idx[order_local]
    approx_angular_distances = approx_angular_distances[order_local]

    if len(ann_idx) < int(fixed_k):
        # Keep the standard dense retrieval semantics even when HNSW undershoots
        # at the fixed-k boundary: refill by exact distances on the same cluster.
        theta = angular_distances(query_for_server, cluster_docs)
        order_local = smallest_k_indices_with_tiebreak(
            distances=theta,
            top_k=int(fixed_k),
            global_indices=np.arange(len(cluster_docs), dtype=np.int32),
        )
        return (
            order_local.astype(np.int32),
            np.asarray(theta[order_local], dtype=np.float32),
            None,
        )
    return (
        ann_idx[: int(fixed_k)].astype(np.int32),
        approx_angular_distances[: int(fixed_k)],
        None,
    )


#固定预算全局候选
def fixed_budget_global_knn_payload(
    query_for_server: np.ndarray,
    docs: np.ndarray,
    doc_ids,
    doc_texts,
    fixed_k: int,
):
    idx, ann_angular_distances, ann_reason = _hnsw_ann_indices_and_scores(
        query_for_server=query_for_server,
        docs=docs,
        fixed_k=int(fixed_k),
    )

    if idx is None:
        raise RuntimeError(
            "FAISS HNSW ANN failed and fallback is disabled by policy. "
            f"reason={ann_reason}"
        )

    payload_scores = np.asarray(ann_angular_distances, dtype=np.float32)

    return {
        "doc_indices": idx,
        "doc_ids": [str(doc_ids[i]) for i in idx.tolist()],
        "texts": [str(doc_texts[i]) for i in idx.tolist()],
        "embeddings": docs[idx].astype(np.float32),
        "scores": payload_scores,
        "retrieval_backend": "faiss_hnsw_ann_angular",
        "ann_fallback_reason": None,
    }


#本地精排
def rerank_candidate_payload_exact(
    original_query: np.ndarray,
    candidate_payload: dict,
    top_k: int,
):
    cand_emb = np.asarray(candidate_payload["embeddings"], dtype=np.float32)
    cand_idx = np.asarray(candidate_payload["doc_indices"], dtype=np.int32)
    cand_ids = list(candidate_payload["doc_ids"])
    cand_texts = list(candidate_payload["texts"])

    # 排名在 payload 局部空间完成，使用最小角距离做 exact rerank。
    theta = angular_distances(original_query, cand_emb)
    order = smallest_k_indices_with_tiebreak(theta, top_k=top_k, global_indices=cand_idx)

    return {
        "doc_indices": cand_idx[order],
        "doc_ids": [cand_ids[i] for i in order.tolist()],
        "texts": [cand_texts[i] for i in order.tolist()],
        "scores": theta[order].astype(np.float32),
    }


def build_payload_from_doc_indices(
    doc_indices,
    docs: np.ndarray,
    doc_ids,
    doc_texts,
):
    chunk = np.asarray(doc_indices, dtype=np.int32).reshape(-1)
    return {
        "doc_indices": chunk,
        "doc_ids": [str(doc_ids[i]) for i in chunk.tolist()],
        "texts": [str(doc_texts[i]) for i in chunk.tolist()],
        "embeddings": docs[chunk].astype(np.float32),
    }


#给稀疏路径用，把某个簇的全部文档打包出来
def build_cluster_payload(
    cluster_id: int,
    chunks,
    docs: np.ndarray,
    doc_ids,
    doc_texts,
):
    chunk = np.asarray(chunks[cluster_id], dtype=np.int32)
    return build_payload_from_doc_indices(
        doc_indices=chunk,
        docs=docs,
        doc_ids=doc_ids,
        doc_texts=doc_texts,
    )


def build_multi_cluster_payload(
    cluster_ids: List[int],
    chunks,
    docs: np.ndarray,
    doc_ids,
    doc_texts,
):
    if cluster_ids is None or len(cluster_ids) == 0:
        raise ValueError("cluster_ids is empty for build_multi_cluster_payload")
    ordered_unique = []
    seen = set()
    for cid in cluster_ids:
        cid = int(cid)
        if cid in seen:
            continue
        seen.add(cid)
        ordered_unique.append(cid)

    parts = [np.asarray(chunks[int(cid)], dtype=np.int32) for cid in ordered_unique]
    merged = np.concatenate(parts, axis=0).astype(np.int32) if parts else np.asarray([], dtype=np.int32)
    merged = np.unique(merged).astype(np.int32)
    payload = build_payload_from_doc_indices(
        doc_indices=merged,
        docs=docs,
        doc_ids=doc_ids,
        doc_texts=doc_texts,
    )
    payload["retrieval_backend"] = "cluster_payload_multicluster_union"
    payload["ann_fallback_reason"] = None
    payload["routed_cluster_ids"] = [int(x) for x in ordered_unique]
    return payload


if __name__ == "__main__":
    cluster_info = load_cluster_info()
    print("cluster_retrieval.py ready.")
    print("cluster_info keys:", list(cluster_info.keys()))
#all in all：生成候选集，并在客户端做最后那次选top-k的精排
