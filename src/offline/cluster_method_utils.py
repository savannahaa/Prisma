"""
cluster_method_utils 模块。
该模块用于当前检索与聚类流水线的对应阶段，
包含数据读取、核心计算（r_k/r_fixed/cluster-level r_max surrogate）与结果落盘逻辑。
"""

from __future__ import annotations

# Allow running this file directly: `python src/offline/cluster_method_utils.py`
if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import json
import os
import pickle
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shared.e5_dual_encoder import E5DualEncoder
from shared.gpu_accel import (
    cosine_similarity_matrix as gpu_cosine_similarity_matrix,
    squared_l2_distance_matrix as gpu_squared_l2_distance_matrix,
    topk_cosine_similarity_matrix as gpu_topk_cosine_similarity_matrix,
)
from shared.config import (
    WORKSET_DOCS_PATH,
    WORKSET_DOC_IDS_PATH,
    WORKSET_CORPUS_JSONL_PATH,
    WORKSET_META_PATH,
    WORKSET_CALIBRATION_QUERIES_PATH,
    WORKSET_CALIBRATION_QUERY_IDS_PATH,
    FULL_QUERIES_JSONL_PATH,
    NEW_MODEL_NAME,
    BATCH_SIZE,
    MAX_LENGTH,
    QUERY_CALIBRATION_RATIO,
    QUERY_CALIBRATION_MIN_COUNT,
    QUERY_EVALUATION_MIN_COUNT,
    QUERY_SPLIT_SEED,
    NUM_WORKSET_DOCS,
    NUM_CLUSTERS,
    TARGET_CLUSTER_SIZE,
    EVAL_K,
    FIXED_K,
    RMAX_CLUSTER_QUANTILE_GAMMA,
    RMAX_ANCHOR_POLICY,
    RMAX_TARGET_ANCHORS_PER_CLUSTER,
    RMAX_MIN_ANCHORS_PER_CLUSTER,
    RMAX_ANCHOR_MEMBERSHIP_MIN_RATIO,
    RMAX_ANCHOR_ENFORCE_MIN_PER_CLUSTER,
    RMAX_ANCHOR_NUM_DISTANCE_STRATA,
    RMAX_SUPPORT_AWARE_CONSERVATIVE_ENABLE,
    RMAX_SUPPORT_NORMAL_MIN,
    RMAX_SUPPORT_SOFT_MIN,
    RMAX_SUPPORT_SOFT_SCALE,
    RMAX_SUPPORT_HARD_RMAX_VALUE,
    RMAX_SHRINKAGE_ENABLE,
    RMAX_SHRINKAGE_TAU,
    RMAX_SHRINKAGE_MIN_BLEND,
    ROUTING_CLUSTER_SELECTION_POLICY,
    ROUTING_FIXED_TOP_C,
    WORKSET_QUERY_SPLIT_META_PATH,
    EPS,
)
from server.cluster_retrieval import filtered_global_hnsw_topk_doc_indices_and_scores


def _rmax_anchor_cache_paths() -> tuple[str, str, str]:
    emb_path = str(
        os.environ.get(
            "RMAX_ANCHOR_CACHE_EMB_PATH",
            str(WORKSET_CALIBRATION_QUERIES_PATH),
        )
    )
    ids_path = str(
        os.environ.get(
            "RMAX_ANCHOR_CACHE_IDS_PATH",
            str(WORKSET_CALIBRATION_QUERY_IDS_PATH),
        )
    )
    split_meta_path = str(
        os.environ.get(
            "RMAX_ANCHOR_CACHE_SPLIT_META_PATH",
            str(WORKSET_QUERY_SPLIT_META_PATH),
        )
    )
    return emb_path, ids_path, split_meta_path


def load_json(path: str):
    # Tolerate UTF-8 BOM in generated metadata so calibration cache checks
    # do not spuriously miss and fall back to full query re-encoding.
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return (x / norms).astype(np.float32)


def _env_flag_local(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def compute_topc_overlap_doc_indices(
    *,
    docs: np.ndarray,
    centers: np.ndarray,
    top_c: int,
) -> tuple[List[np.ndarray], np.ndarray]:
    docs = normalize_rows(np.asarray(docs, dtype=np.float32))
    centers = normalize_rows(np.asarray(centers, dtype=np.float32))
    num_clusters = int(centers.shape[0])
    top_c = int(max(1, min(int(top_c), int(num_clusters))))
    sims = np.clip(docs @ centers.T, -1.0, 1.0).astype(np.float32)
    topc_order = np.argsort(-sims, axis=1).astype(np.int32)[:, : int(top_c)]
    per_cluster = [[] for _ in range(num_clusters)]
    for didx, row in enumerate(topc_order.tolist()):
        for cid in row:
            per_cluster[int(cid)].append(int(didx))
    return (
        [np.asarray(vals, dtype=np.int32) for vals in per_cluster],
        np.asarray(topc_order, dtype=np.int32),
    )


def normalize_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = float(np.linalg.norm(x))
    if norm <= EPS:
        return x.astype(np.float32)
    return (x / norm).astype(np.float32)


def squared_l2_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    gpu_d2 = gpu_squared_l2_distance_matrix(
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        cache_x_if_large=True,
        cache_y_if_large=False,
    )
    if gpu_d2 is not None:
        return np.maximum(gpu_d2, 0.0).astype(np.float32, copy=False)

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x_sq = np.sum(x * x, axis=1, keepdims=True)
    y_sq = np.sum(y * y, axis=1, keepdims=True).T
    cross = x @ y.T
    d2 = x_sq + y_sq - 2.0 * cross
    return np.maximum(d2, 0.0).astype(np.float32)


def l2_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(squared_l2_distance_matrix(x, y)).astype(np.float32)


def cosine_similarity_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    gpu_sims = gpu_cosine_similarity_matrix(
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        cache_x_if_large=True,
        cache_y_if_large=False,
    )
    if gpu_sims is not None:
        return np.clip(gpu_sims, -1.0, 1.0).astype(np.float32, copy=False)

    sims = np.asarray(x, dtype=np.float32) @ np.asarray(y, dtype=np.float32).T
    return np.clip(sims, -1.0, 1.0).astype(np.float32)


def angular_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.arccos(cosine_similarity_matrix(x, y)).astype(np.float32)


def init_kmeans_pp(docs: np.ndarray, num_clusters: int, rng: np.random.Generator) -> np.ndarray:
    num_docs = docs.shape[0]
    centers = np.empty((num_clusters, docs.shape[1]), dtype=np.float32)

    first_idx = int(rng.integers(0, num_docs))
    centers[0] = docs[first_idx]
    min_d2 = squared_l2_distance_matrix(docs, centers[:1]).reshape(-1)

    for cid in range(1, num_clusters):
        total = float(np.sum(min_d2))
        if total <= EPS:
            next_idx = int(rng.integers(0, num_docs))
        else:
            probs = min_d2 / total
            next_idx = int(rng.choice(num_docs, p=probs))
        centers[cid] = docs[next_idx]
        cand_d2 = squared_l2_distance_matrix(docs, centers[cid:cid + 1]).reshape(-1)
        min_d2 = np.minimum(min_d2, cand_d2)

    return centers.astype(np.float32)


def assign_balanced_clusters(dists: np.ndarray, capacity: int) -> List[np.ndarray]:
    num_docs, num_clusters = dists.shape
    if num_docs != num_clusters * capacity:
        raise RuntimeError(
            "balanced assignment requires num_docs == num_clusters * capacity, "
            f"got {num_docs} vs {num_clusters} * {capacity}"
        )

    pref = np.argsort(dists, axis=1)
    if num_clusters > 1:
        best = dists[np.arange(num_docs), pref[:, 0]]
        second = dists[np.arange(num_docs), pref[:, 1]]
        margin = second - best
    else:
        margin = np.ones(num_docs, dtype=np.float32)

    order = np.lexsort((dists[np.arange(num_docs), pref[:, 0]], -margin))
    remaining = np.full(num_clusters, int(capacity), dtype=np.int32)
    assigned = np.full(num_docs, -1, dtype=np.int32)

    for doc_idx in order.tolist():
        for cid in pref[doc_idx].tolist():
            if remaining[cid] > 0:
                assigned[doc_idx] = int(cid)
                remaining[cid] -= 1
                break

    if np.any(remaining != 0):
        raise RuntimeError(f"balanced assignment failed, remaining capacities={remaining.tolist()}")

    chunks: List[np.ndarray] = []
    for cid in range(num_clusters):
        idx = np.where(assigned == cid)[0].astype(np.int32)
        if len(idx) != int(capacity):
            raise RuntimeError(
                f"cluster {cid} size mismatch, expected {capacity}, got {len(idx)}"
            )
        chunks.append(idx)
    return chunks


def assigned_chunks_from_labels(labels: np.ndarray, num_clusters: int) -> List[np.ndarray]:
    chunks: List[np.ndarray] = []
    for cid in range(int(num_clusters)):
        idx = np.where(labels == cid)[0].astype(np.int32)
        chunks.append(idx)
    return chunks


def compute_cluster_profile_with_center(
    cluster_docs: np.ndarray,
    center: np.ndarray,
    eval_k: int,
    fixed_k: int,
) -> dict:
    sims = np.clip(cluster_docs @ center.reshape(-1, 1), -1.0, 1.0).reshape(-1)
    dists = np.arccos(sims).astype(np.float32)
    sorted_d = np.sort(dists)
    r_k = float(sorted_d[int(eval_k) - 1])
    r_fixed_rank = int(max(1, min(int(fixed_k), int(len(sorted_d)))))
    r_fixed = float(sorted_d[int(r_fixed_rank) - 1])
    return {
        "center_norm": float(np.linalg.norm(center)),
        "r_k": r_k,
        "r_fixed": r_fixed,
        "r_fixed_rank_effective": int(r_fixed_rank),
        "gap_half": float((r_fixed - r_k) / 2.0),
        "min_doc_to_center": float(np.min(sorted_d)),
        "p50_doc_to_center": float(np.percentile(sorted_d, 50.0)),
        "avg_doc_to_center": float(np.mean(sorted_d)),
        "max_doc_to_center": float(np.max(sorted_d)),
    }


def load_workset_inputs() -> tuple[np.ndarray, np.ndarray, dict]:
    required = [
        WORKSET_DOCS_PATH,
        WORKSET_DOC_IDS_PATH,
        WORKSET_META_PATH,
    ]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"required file not found: {path}")

    docs = np.load(WORKSET_DOCS_PATH).astype(np.float32)
    doc_ids = np.load(WORKSET_DOC_IDS_PATH, allow_pickle=True)
    meta = load_json(WORKSET_META_PATH)

    if len(docs) != len(doc_ids):
        raise RuntimeError(f"docs/doc_ids mismatch: {len(docs)} vs {len(doc_ids)}")
    if len(docs) != int(NUM_WORKSET_DOCS):
        raise RuntimeError(f"expected {NUM_WORKSET_DOCS} docs, got {len(docs)}")

    docs = normalize_rows(docs)
    return docs, doc_ids, meta


def _clean_text_for_rmax(text: str) -> str:
    text = str(text).replace("\n", " ").replace("\t", " ").strip()
    return " ".join(text.split())


def _parse_query_row_for_rmax(row: dict) -> Tuple[str, str, Optional[str]]:
    raw_qid = None
    for key in ("query_id", "id", "_id"):
        if key in row:
            raw_qid = str(row[key]).strip()
            break

    qtext = None
    for key in ("text", "query", "contents", "content"):
        if key in row:
            qtext = _clean_text_for_rmax(row[key])
            break

    if not raw_qid or not qtext:
        raise ValueError("invalid query row for rmax calibration")
    source_query_id = row.get("source_query_id")
    if source_query_id is not None:
        source_query_id = str(source_query_id).strip()
        if len(source_query_id) == 0:
            source_query_id = None
    return str(raw_qid), str(qtext), source_query_id


def _canonical_query_id_for_rmax(raw_query_id: str, source_query_id: Optional[str]) -> str:
    if source_query_id is not None:
        s = str(source_query_id).strip()
        if s:
            return s
    return str(raw_query_id).strip()


def _stable_hash_to_unit_interval_for_rmax(s: str) -> float:
    h = hashlib.sha256(str(s).encode("utf-8")).hexdigest()
    n = int(h[:16], 16)
    return float(n / float(2**64))


def _split_query_rows_for_protocol_for_rmax(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    if len(rows) <= 1:
        raise RuntimeError("not enough queries for calibration/evaluation split.")

    scored = []
    for row in rows:
        qid = str(row["query_id"])
        score = _stable_hash_to_unit_interval_for_rmax(f"{int(QUERY_SPLIT_SEED)}::{qid}")
        scored.append((float(score), qid, row))
    scored.sort(key=lambda x: (float(x[0]), str(x[1])))

    n = int(len(scored))
    ratio = float(np.clip(float(QUERY_CALIBRATION_RATIO), 0.05, 0.95))
    min_cal = int(max(1, int(QUERY_CALIBRATION_MIN_COUNT)))
    min_eval = int(max(1, int(QUERY_EVALUATION_MIN_COUNT)))
    if min_cal + min_eval > n:
        min_cal = int(max(1, min(min_cal, n // 2)))
        min_eval = int(max(1, n - min_cal))
    desired_cal = int(round(ratio * n))
    desired_cal = int(max(min_cal, min(desired_cal, n - min_eval)))

    calibration_rows = [x[2] for x in scored[:desired_cal]]
    evaluation_rows = [x[2] for x in scored[desired_cal:]]
    if len(calibration_rows) == 0 or len(evaluation_rows) == 0:
        raise RuntimeError(
            "invalid calibration/evaluation split size: "
            f"calibration={len(calibration_rows)}, evaluation={len(evaluation_rows)}"
        )
    return calibration_rows, evaluation_rows


def _load_calibration_query_rows_for_rmax(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing FULL_QUERIES_JSONL_PATH for rmax anchors: {path}")

    raw_rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_rows.append(json.loads(line))
            except Exception:
                continue

    canonical_rows = []
    seen = set()
    for row in raw_rows:
        try:
            raw_qid, text, source_query_id = _parse_query_row_for_rmax(row)
        except Exception:
            continue
        qid = _canonical_query_id_for_rmax(raw_qid, source_query_id)
        if qid in seen:
            continue
        seen.add(qid)
        canonical_rows.append(
            {
                "query_id": str(qid),
                "raw_query_id": str(raw_qid),
                "source_query_id": (str(source_query_id) if source_query_id is not None else None),
                "text": str(text),
            }
        )
    if len(canonical_rows) == 0:
        raise RuntimeError(f"no valid query rows for rmax anchors from: {path}")

    calibration_rows, _ = _split_query_rows_for_protocol_for_rmax(canonical_rows)
    return calibration_rows


def _encode_query_rows_for_rmax(rows: List[dict]) -> Tuple[np.ndarray, List[str]]:
    texts = [str(r["text"]) for r in rows]
    qids = [str(r["query_id"]) for r in rows]
    if len(texts) == 0:
        return np.empty((0, 0), dtype=np.float32), []

    backend = E5DualEncoder(
        NEW_MODEL_NAME,
        log_prefix="rmax-anchor-encoder",
        device_role="client",
    )
    _raw, norm = backend.encode_queries(
        texts,
        batch_size=int(BATCH_SIZE),
        max_length=int(MAX_LENGTH),
        progress_name="encode-rmax-calibration-query",
    )
    return normalize_rows(np.asarray(norm, dtype=np.float32)), qids


def _load_cached_calibration_query_emb_for_rmax() -> Tuple[np.ndarray | None, List[str] | None, str]:
    emb_path, ids_path, split_meta_path = _rmax_anchor_cache_paths()
    if (not os.path.exists(emb_path)) or (not os.path.exists(ids_path)):
        return None, None, "missing_cache_file"
    try:
        q = normalize_rows(np.asarray(np.load(emb_path), dtype=np.float32))
        ids = np.asarray(np.load(ids_path, allow_pickle=True), dtype=object)
    except Exception as e:
        return None, None, f"cache_load_error:{type(e).__name__}"
    if q.ndim != 2 or len(q) <= 0:
        return None, None, "cache_emb_invalid_shape"
    if len(ids) != len(q):
        return None, None, "cache_emb_ids_length_mismatch"
    qids = [str(x) for x in ids.tolist()]

    # 严格协议检查：calibration/evaluation 必须完全不重叠。
    if os.path.exists(split_meta_path):
        try:
            split_meta = load_json(split_meta_path)
            overlap = int(split_meta.get("split_overlap_count", 0))
            if overlap != 0:
                return None, None, f"split_overlap_detected:{overlap}"
        except Exception:
            return None, None, "split_meta_load_error"
    return q.astype(np.float32), qids, "cache_hit"


def _allocate_even_quotas(total: int, num_groups: int) -> List[int]:
    total = int(max(0, total))
    num_groups = int(max(1, num_groups))
    q = [total // num_groups] * num_groups
    for i in range(total % num_groups):
        q[i] += 1
    return q


def _stratified_by_distance_pick(
    candidate_indices: np.ndarray,
    distances: np.ndarray,
    target_count: int,
    num_strata: int,
) -> np.ndarray:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int32)
    if int(target_count) <= 0 or len(candidate_indices) == 0:
        return np.asarray([], dtype=np.int32)
    if len(candidate_indices) <= int(target_count):
        order = np.argsort(distances[candidate_indices], kind="mergesort")
        return candidate_indices[order].astype(np.int32)

    num_strata = int(max(1, min(int(num_strata), len(candidate_indices))))
    order = np.argsort(distances[candidate_indices], kind="mergesort")
    sorted_idx = candidate_indices[order]
    bins = np.array_split(sorted_idx, num_strata)
    quotas = _allocate_even_quotas(int(target_count), num_strata)

    picked: List[int] = []
    for b, q in zip(bins, quotas):
        if int(q) <= 0 or len(b) == 0:
            continue
        if len(b) <= int(q):
            picked.extend(int(x) for x in b.tolist())
        else:
            pos = np.linspace(0, len(b) - 1, num=int(q), dtype=np.int32)
            picked.extend(int(b[int(p)]) for p in pos.tolist())

    if len(picked) < int(target_count):
        picked_set = set(picked)
        rest = [int(x) for x in sorted_idx.tolist() if int(x) not in picked_set]
        need = int(target_count - len(picked))
        picked.extend(rest[:need])
    elif len(picked) > int(target_count):
        picked = picked[: int(target_count)]

    picked = sorted(set(int(x) for x in picked), key=lambda x: x)
    if len(picked) < int(target_count):
        picked_set = set(picked)
        rest = [int(x) for x in sorted_idx.tolist() if int(x) not in picked_set]
        need = int(target_count - len(picked))
        picked.extend(rest[:need])
    return np.asarray(picked[: int(target_count)], dtype=np.int32)


def _select_clustered_calibration_query_anchors(
    *,
    query_emb: np.ndarray,
    query_ids: List[str],
    docs: np.ndarray,
    centers: np.ndarray,
    chunks: List[np.ndarray],
    fixed_k: int,
    target_per_cluster: int,
    min_per_cluster: int,
    membership_min_ratio: float,
    num_distance_strata: int,
    enforce_min_per_cluster: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    query_emb = normalize_rows(np.asarray(query_emb, dtype=np.float32))
    docs = normalize_rows(np.asarray(docs, dtype=np.float32))
    centers = normalize_rows(np.asarray(centers, dtype=np.float32))
    num_clusters = int(len(centers))

    if len(query_emb) == 0:
        raise RuntimeError("empty query embeddings for anchor selection.")
    if query_emb.shape[1] != docs.shape[1] or centers.shape[1] != docs.shape[1]:
        raise RuntimeError("dimension mismatch among query/docs/centers for anchor selection.")

    # 1) owner 定义：按最小角距离最近中心分簇（与在线单簇路由一致）
    dmat = angular_distance_matrix(query_emb, centers).astype(np.float32)
    owner_cluster = np.argmin(dmat, axis=1).astype(np.int32)
    d_to_owner_center = dmat[np.arange(len(query_emb), dtype=np.int32), owner_cluster]

    # 2) 代表性采样：不做 core-heavy / membership 过滤，直接在 owner 簇内分层均匀抽样。
    # 为兼容既有统计字段，owner_ratio 恒设为 1.0（表示“未启用过滤”）。
    owner_ratio = np.ones(len(query_emb), dtype=np.float32)
    valid_mask = np.ones(len(query_emb), dtype=bool)
    membership_min_ratio = float(membership_min_ratio)

    selected_indices: List[int] = []
    selected_owner_clusters: List[int] = []
    cluster_stats = []
    for cid in range(num_clusters):
        cid = int(cid)
        cid_all = np.where(owner_cluster == cid)[0].astype(np.int32)
        cid_valid = cid_all[valid_mask[cid_all]]

        picked = _stratified_by_distance_pick(
            candidate_indices=cid_valid,
            distances=d_to_owner_center,
            target_count=int(target_per_cluster),
            num_strata=int(num_distance_strata),
        )
        if len(picked) < int(min_per_cluster) and bool(enforce_min_per_cluster):
            raise RuntimeError(
                "r_max anchor selection failed min-per-cluster constraint: "
                f"cluster={cid}, valid={len(cid_valid)}, picked={len(picked)}, "
                f"required_min={int(min_per_cluster)}. "
                "Please increase calibration pool size or lower membership threshold."
            )
        selected_indices.extend(int(x) for x in picked.tolist())
        selected_owner_clusters.extend([cid] * int(len(picked)))
        cluster_stats.append(
            {
                "cluster_id": cid,
                "num_candidates_owner_cluster": int(len(cid_all)),
                "num_candidates_membership_valid": int(len(cid_valid)),
                "num_selected": int(len(picked)),
                "owner_ratio_p50": float(np.percentile(owner_ratio[cid_all], 50.0))
                if len(cid_all) > 0
                else None,
                "owner_ratio_p10": float(np.percentile(owner_ratio[cid_all], 10.0))
                if len(cid_all) > 0
                else None,
                "theta_to_owner_center_p50": float(np.percentile(d_to_owner_center[cid_all], 50.0))
                if len(cid_all) > 0
                else None,
            }
        )

    if len(selected_indices) == 0:
        raise RuntimeError("anchor selection produced empty set.")

    # 保持稳定顺序（按原始 query index 升序）
    pair = sorted(zip(selected_indices, selected_owner_clusters), key=lambda x: int(x[0]))
    selected_indices_arr = np.asarray([int(x[0]) for x in pair], dtype=np.int32)
    selected_owner_arr = np.asarray([int(x[1]) for x in pair], dtype=np.int32)

    selected = query_emb[selected_indices_arr]
    selected_qids = [str(query_ids[int(i)]) for i in selected_indices_arr.tolist()]
    clusters_below_min = [
        int(x["cluster_id"]) for x in cluster_stats if int(x["num_selected"]) < int(min_per_cluster)
    ]
    meta = {
        "selector": "nearest_cluster_representative_distance_stratified_sampling",
        "fixed_k_for_membership": None,
        "membership_scope": "not_applicable_representative_sampling",
        "owner_cluster_definition": "nearest_center_cluster_by_angular_distance",
        "owner_ratio_formula": "not_used_in_main_result_representative_sampling",
        "target_anchors_per_cluster": int(target_per_cluster),
        "min_anchors_per_cluster": int(min_per_cluster),
        "membership_min_ratio": float(membership_min_ratio),
        "num_distance_strata": int(num_distance_strata),
        "membership_filter_enabled": False,
        "enforce_min_per_cluster": bool(enforce_min_per_cluster),
        "num_candidates_total": int(len(query_emb)),
        "num_selected_total": int(len(selected)),
        "num_clusters_below_min": int(len(clusters_below_min)),
        "clusters_below_min": clusters_below_min,
        "selected_owner_cluster_ids_head20": [int(x) for x in selected_owner_arr[:20].tolist()],
        "selected_query_ids_head20": selected_qids[:20],
        "cluster_stats": cluster_stats,
    }
    return selected.astype(np.float32), selected_owner_arr.astype(np.int32), meta


def _select_clustered_docs_only_anchors(
    *,
    docs: np.ndarray,
    centers: np.ndarray,
    chunks: List[np.ndarray],
    target_per_cluster: int,
    num_distance_strata: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Lightweight docs-only anchor sampling.

    Instead of using every document as an r_max anchor, sample a small, balanced
    per-cluster subset by taking evenly spaced points along the center-similarity
    ranking. This preserves core/tail coverage while keeping the offline r_max
    phase tractable for large worksets.
    """
    docs_norm = normalize_rows(np.asarray(docs, dtype=np.float32))
    centers_norm = normalize_rows(np.asarray(centers, dtype=np.float32))

    selected_doc_indices: list[int] = []
    selected_owner_clusters: list[int] = []
    cluster_stats: list[dict] = []

    for cid, chunk in enumerate(chunks):
        chunk_idx = np.asarray(chunk, dtype=np.int32).reshape(-1)
        cluster_size = int(len(chunk_idx))
        if cluster_size <= 0:
            cluster_stats.append(
                {
                    "cluster_id": int(cid),
                    "cluster_size": 0,
                    "num_selected": 0,
                }
            )
            continue

        take = int(min(cluster_size, max(1, int(target_per_cluster))))
        chunk_docs = np.asarray(docs_norm[chunk_idx], dtype=np.float32)
        center = np.asarray(centers_norm[int(cid)], dtype=np.float32).reshape(-1)
        sims = np.clip(chunk_docs @ center, -1.0, 1.0).astype(np.float32)
        order = np.argsort(-sims, kind="stable").astype(np.int32)
        ranked_chunk_idx = np.asarray(chunk_idx[order], dtype=np.int32)

        if take >= cluster_size:
            picked = ranked_chunk_idx
        else:
            sample_pos = np.linspace(
                0,
                cluster_size - 1,
                num=int(take),
                endpoint=True,
                dtype=np.int64,
            )
            picked = np.asarray(ranked_chunk_idx[sample_pos], dtype=np.int32)
            picked = np.unique(picked).astype(np.int32)
            if len(picked) < int(take):
                missing = int(take) - int(len(picked))
                seen = set(int(x) for x in picked.tolist())
                tail_fill = [int(x) for x in ranked_chunk_idx.tolist() if int(x) not in seen][:missing]
                if tail_fill:
                    picked = np.asarray(
                        list(picked.tolist()) + [int(x) for x in tail_fill],
                        dtype=np.int32,
                    )

        selected_doc_indices.extend(int(x) for x in picked.tolist())
        selected_owner_clusters.extend([int(cid)] * int(len(picked)))
        cluster_stats.append(
            {
                "cluster_id": int(cid),
                "cluster_size": int(cluster_size),
                "num_selected": int(len(picked)),
                "target_anchors_per_cluster": int(target_per_cluster),
                "num_distance_strata": int(max(1, num_distance_strata)),
            }
        )

    if not selected_doc_indices:
        raise RuntimeError("docs_only anchor selection produced no anchors.")

    selected_idx_arr = np.asarray(selected_doc_indices, dtype=np.int32)
    selected_owner_arr = np.asarray(selected_owner_clusters, dtype=np.int32)
    meta = {
        "selector": "docs_only_cluster_balanced_equidistant",
        "target_anchors_per_cluster": int(target_per_cluster),
        "num_distance_strata": int(max(1, num_distance_strata)),
        "num_selected_total": int(len(selected_idx_arr)),
        "num_clusters_with_anchors": int(sum(1 for row in cluster_stats if int(row["num_selected"]) > 0)),
        "cluster_stats": cluster_stats,
    }
    return np.asarray(docs_norm[selected_idx_arr], dtype=np.float32), selected_owner_arr, meta


def load_rmax_calibration_anchors(
    docs: np.ndarray,
    centers: np.ndarray | None = None,
    chunks: List[np.ndarray] | None = None,
    calibration_queries: np.ndarray | None = None,
    anchor_policy: str = str(RMAX_ANCHOR_POLICY),
) -> tuple[np.ndarray, str, Dict[str, Any], np.ndarray | None]:
    """
    加载离线 r_max surrogate 的 anchor 集合。

    论文主结果默认：只使用独立 calibration query anchors（calibration_query_only）。
    可选补充实验：calibration_query_plus_docs / docs_only。
    """
    docs_norm = normalize_rows(np.asarray(docs, dtype=np.float32))
    policy = str(anchor_policy).strip().lower()
    valid = {"calibration_query_only", "calibration_query_plus_docs", "docs_only"}
    if policy not in valid:
        raise ValueError(
            f"invalid RMAX_ANCHOR_POLICY={anchor_policy}, expected one of {sorted(valid)}"
        )

    if policy == "docs_only":
        if centers is not None and chunks is not None:
            selected_docs, selected_owner_clusters, selector_meta = _select_clustered_docs_only_anchors(
                docs=docs_norm,
                centers=np.asarray(centers, dtype=np.float32),
                chunks=list(chunks),
                target_per_cluster=int(RMAX_TARGET_ANCHORS_PER_CLUSTER),
                num_distance_strata=int(RMAX_ANCHOR_NUM_DISTANCE_STRATA),
            )
            return (
                selected_docs,
                "docs_only_policy_cluster_sampled",
                selector_meta,
                selected_owner_clusters,
            )
        return (
            docs_norm,
            "docs_only_policy_all_docs_fallback",
            {"selector": "docs_only_all_docs_fallback", "num_selected_total": int(len(docs_norm))},
            None,
        )

    if centers is None or chunks is None:
        # 兼容旧行为：显式传入 query 时允许无分簇信息直接返回。
        if calibration_queries is not None:
            q = normalize_rows(np.asarray(calibration_queries, dtype=np.float32))
            if len(q) > 0:
                if policy == "calibration_query_plus_docs":
                    return (
                        np.concatenate([q, docs_norm], axis=0),
                        "explicit_calibration_queries_plus_docs",
                        {"selector": "explicit_calibration_queries", "num_selected_total": int(len(q))},
                        None,
                    )
                return (
                    q,
                    "explicit_calibration_queries_only",
                    {"selector": "explicit_calibration_queries", "num_selected_total": int(len(q))},
                    None,
                )
        raise RuntimeError(
            "centers/chunks are required for query anchor stratified selection. "
            "Please pass them into load_rmax_calibration_anchors."
        )

    q: np.ndarray | None = None
    qids: List[str] | None = None
    query_source = ""

    if calibration_queries is not None:
        q = normalize_rows(np.asarray(calibration_queries, dtype=np.float32))
        if len(q) > 0:
            qids = [f"explicit_calibration_query_{i}" for i in range(len(q))]
            query_source = "explicit_calibration_queries_input"

    if q is None or qids is None:
        q_cached, qids_cached, cache_status = _load_cached_calibration_query_emb_for_rmax()
        if q_cached is not None and qids_cached is not None:
            q = q_cached
            qids = qids_cached
            query_source = f"workset_calibration_cache:{cache_status}"
        else:
            calibration_rows = _load_calibration_query_rows_for_rmax(FULL_QUERIES_JSONL_PATH)
            q, qids = _encode_query_rows_for_rmax(calibration_rows)
            query_source = f"full_queries_reencode_fallback:{cache_status}"

    if q is None or qids is None or len(q) == 0:
        raise RuntimeError("empty calibration query anchors for representative sampling.")

    selected_q, selected_owner_clusters, selector_meta = _select_clustered_calibration_query_anchors(
        query_emb=q,
        query_ids=qids,
        docs=docs_norm,
        centers=np.asarray(centers, dtype=np.float32),
        chunks=list(chunks),
        fixed_k=int(FIXED_K),
        target_per_cluster=int(RMAX_TARGET_ANCHORS_PER_CLUSTER),
        min_per_cluster=int(RMAX_MIN_ANCHORS_PER_CLUSTER),
        membership_min_ratio=float(RMAX_ANCHOR_MEMBERSHIP_MIN_RATIO),
        num_distance_strata=int(RMAX_ANCHOR_NUM_DISTANCE_STRATA),
        enforce_min_per_cluster=bool(RMAX_ANCHOR_ENFORCE_MIN_PER_CLUSTER),
    )
    selector_meta["query_source"] = str(query_source)
    selector_meta["query_pool_size_before_selection"] = int(len(q))

    if policy == "calibration_query_plus_docs":
        return (
            np.concatenate([selected_q, docs_norm], axis=0),
            "calibration_query_plus_docs_policy",
            selector_meta,
            None,
        )
    return selected_q, "calibration_query_only_policy", selector_meta, selected_owner_clusters


def _compute_cluster_level_rmax_surrogate_fast_gpu(
    *,
    docs: np.ndarray,
    centers: np.ndarray,
    chunks: List[np.ndarray],
    overlap_doc_indices_by_cluster: List[np.ndarray] | None,
    eval_k: int,
    fixed_k: int,
    gamma: float,
    anchors: np.ndarray,
    anchor_source: str,
    anchor_cluster_ids: np.ndarray | None,
) -> dict | None:
    num_docs = int(docs.shape[0])
    num_clusters = int(centers.shape[0])
    overlap_top_c = int(max(1, min(int(num_clusters), int(ROUTING_FIXED_TOP_C))))
    use_soft_topc_overlap = bool(
        str(ROUTING_CLUSTER_SELECTION_POLICY).strip().lower() == "soft_topc_fixed"
        and int(overlap_top_c) > 1
    )

    anchor_to_center_sim = gpu_cosine_similarity_matrix(
        x=np.asarray(anchors, dtype=np.float32),
        y=np.asarray(centers, dtype=np.float32),
        cache_x_if_large=False,
        cache_y_if_large=False,
    )
    if anchor_to_center_sim is None:
        return None
    anchor_to_center_sim = np.clip(anchor_to_center_sim, -1.0, 1.0).astype(np.float32, copy=False)
    anchor_route_order = np.argsort(-anchor_to_center_sim, axis=1).astype(np.int32)

    if use_soft_topc_overlap:
        routed_cluster_matrix = np.asarray(
            anchor_route_order[:, : int(overlap_top_c)],
            dtype=np.int32,
        )
        nearest_cluster = np.asarray(routed_cluster_matrix[:, 0], dtype=np.int32)
        cluster_assign_source = f"topc_nearest_centroids_by_angular_distance:c={int(overlap_top_c)}"
    elif anchor_cluster_ids is not None:
        nearest_cluster = np.asarray(anchor_cluster_ids, dtype=np.int32)
        if nearest_cluster.shape != (len(anchors),):
            raise RuntimeError(
                "anchor_cluster_ids shape mismatch: "
                f"expected ({len(anchors)},), got {nearest_cluster.shape}"
            )
        if np.any(nearest_cluster < 0) or np.any(nearest_cluster >= int(num_clusters)):
            raise RuntimeError("anchor_cluster_ids has invalid cluster ids.")
        routed_cluster_matrix = nearest_cluster.reshape(-1, 1).astype(np.int32)
        cluster_assign_source = "provided_owner_cluster_ids"
    else:
        nearest_cluster = np.argmax(anchor_to_center_sim, axis=1).astype(np.int32)
        routed_cluster_matrix = nearest_cluster.reshape(-1, 1).astype(np.int32)
        cluster_assign_source = "nearest_center"

    # Large fixed-k values can trigger a sharp slowdown if the GPU prefilter
    # keeps too few global candidates, because many anchors then fall back to
    # per-route HNSW/exact refill just to recover enough support members.
    # Scale top-L more aggressively once fixed_k grows so the fast GPU path
    # continues to cover the routed support set with high probability.
    top_l_multiplier_env = str(os.environ.get("RMAX_GPU_GLOBAL_TOPL_MULTIPLIER", "")).strip()
    if top_l_multiplier_env:
        top_l_multiplier = int(max(1, int(float(top_l_multiplier_env))))
    elif int(fixed_k) >= 2000:
        top_l_multiplier = 48
    elif int(fixed_k) >= 1000:
        top_l_multiplier = 32
    elif int(fixed_k) >= 500:
        top_l_multiplier = 16
    else:
        top_l_multiplier = 8
    top_l_default = int(max(8192, int(fixed_k) * int(top_l_multiplier), int(eval_k) * 32))
    top_l = int(
        min(
            int(num_docs),
            max(
                int(eval_k),
                int(fixed_k),
                int(os.environ.get("RMAX_GPU_GLOBAL_TOPL", str(top_l_default))),
            ),
        )
    )
    global_topl_sims, global_topl_idx = gpu_topk_cosine_similarity_matrix(
        x=np.asarray(anchors, dtype=np.float32),
        y=np.asarray(docs, dtype=np.float32),
        top_k=int(top_l),
        cache_x_if_large=False,
        cache_y_if_large=True,
        y_chunk_rows=int(os.environ.get("RMAX_GPU_TOPL_CHUNK_ROWS", "8192")),
    )
    if global_topl_sims is None or global_topl_idx is None:
        return None

    global_topl_sims = np.clip(np.asarray(global_topl_sims, dtype=np.float32), -1.0, 1.0)
    global_topl_idx = np.asarray(global_topl_idx, dtype=np.int32)
    if int(global_topl_idx.shape[1]) < int(eval_k):
        raise RuntimeError(
            f"global top-L too small for eval_k: top_l={global_topl_idx.shape[1]} eval_k={eval_k}"
        )
    global_topl_theta = np.arccos(global_topl_sims).astype(np.float32)
    gt_topk_idx = np.asarray(global_topl_idx[:, : int(eval_k)], dtype=np.int32)
    gt_topk_worst_sim = np.min(global_topl_sims[:, : int(eval_k)], axis=1).astype(np.float64)

    cluster_support_indices = (
        [np.asarray(vals, dtype=np.int32).reshape(-1) for vals in overlap_doc_indices_by_cluster]
        if bool(use_soft_topc_overlap)
        else [np.asarray(chunk, dtype=np.int32).reshape(-1) for chunk in chunks]
    )
    support_mask_matrix = np.zeros((int(num_clusters), int(num_docs)), dtype=bool)
    for cid, vals in enumerate(cluster_support_indices):
        support_mask_matrix[int(cid), np.asarray(vals, dtype=np.int32)] = True

    anchor_ideal_rmax_by_cluster = np.zeros((len(anchors), num_clusters), dtype=np.float64)
    anchor_cover300_by_cluster = np.zeros((len(anchors), num_clusters), dtype=bool)
    anchor_cover500_by_cluster = np.zeros((len(anchors), num_clusters), dtype=bool)
    contain300_acc: list[float] = []
    contain500_acc: list[float] = []
    contain300_oracle_acc: list[float] = []
    contain500_oracle_acc: list[float] = []
    fallback_search_count = 0

    top_l_positions = np.arange(int(global_topl_idx.shape[1]), dtype=np.int32)
    for aidx in range(len(anchors)):
        row_idx = np.asarray(global_topl_idx[int(aidx)], dtype=np.int32)
        row_theta = np.asarray(global_topl_theta[int(aidx)], dtype=np.float32)
        gt_idx_row = np.asarray(gt_topk_idx[int(aidx)], dtype=np.int32)
        routed_ids = [int(x) for x in np.asarray(routed_cluster_matrix[int(aidx)], dtype=np.int32).tolist()]

        support_on_topl = np.asarray(support_mask_matrix[:, row_idx], dtype=bool)
        prefix_counts = np.cumsum(support_on_topl, axis=1, dtype=np.int32)
        support_counts = np.asarray(prefix_counts[:, -1], dtype=np.int32)
        enough_topl = np.asarray(support_counts >= int(fixed_k), dtype=bool)
        topfixed_mask = np.asarray(
            support_on_topl & (prefix_counts <= int(fixed_k)),
            dtype=bool,
        )
        gt_support_hit = np.sum(support_on_topl[:, : int(eval_k)], axis=1, dtype=np.int32)
        gt_topfixed_hit = np.sum(topfixed_mask[:, : int(eval_k)], axis=1, dtype=np.int32)
        boundary_pos = np.max(
            np.where(topfixed_mask, top_l_positions.reshape(1, -1), -1),
            axis=1,
        ).astype(np.int32)
        boundary_theta = np.zeros(int(num_clusters), dtype=np.float64)
        enough_idx = np.where(enough_topl)[0].astype(np.int32)
        if len(enough_idx) > 0:
            boundary_theta[enough_idx] = row_theta[boundary_pos[enough_idx]].astype(np.float64)

        # Only routed clusters affect the paperfaithful mainline r_max surrogate.
        # Falling back to exact filtered search for every prefilter-miss routed cluster turns the
        # supposed GPU fast path back into an almost full exhaustive pass.
        routed_ids_arr = np.asarray(routed_ids, dtype=np.int32)
        prefilter_miss_routed_idx = np.asarray(
            [int(oid) for oid in routed_ids_arr.tolist() if not bool(enough_topl[int(oid)])],
            dtype=np.int32,
        )
        for oid in prefilter_miss_routed_idx.tolist():
            topfixed_idx, topfixed_theta, _search_meta = filtered_global_hnsw_topk_doc_indices_and_scores(
                query_for_server=np.asarray(anchors[int(aidx)], dtype=np.float32),
                docs=docs,
                allowed_doc_indices=np.asarray(cluster_support_indices[int(oid)], dtype=np.int32),
                fixed_k=int(fixed_k),
            )
            fallback_search_count += 1
            topfixed_idx = np.asarray(topfixed_idx, dtype=np.int32)
            topfixed_theta = np.asarray(topfixed_theta, dtype=np.float32)
            exact_gt_hit = int(np.sum(np.isin(gt_idx_row, topfixed_idx, assume_unique=False)))
            gt_topfixed_hit[int(oid)] = int(exact_gt_hit)
            topfixed_mask[int(oid), : int(eval_k)] = np.isin(
                gt_idx_row,
                topfixed_idx,
                assume_unique=False,
            )
            if len(topfixed_theta) > 0:
                boundary_theta[int(oid)] = float(np.max(np.asarray(topfixed_theta, dtype=np.float64)))

        best500 = float(np.max(gt_support_hit) / max(1, int(eval_k)))
        # For non-routed clusters we keep the prefilter estimate instead of
        # issuing exact fallback searches. This keeps oracle-best-parent as a
        # diagnostic lower bound while preserving the routed-cluster protocol.
        best300 = float(np.max(gt_topfixed_hit) / max(1, int(eval_k)))
        contain300_oracle_acc.append(float(best300))
        contain500_oracle_acc.append(float(best500))

        routed_gt_support = np.asarray(gt_support_hit[routed_ids_arr], dtype=np.int32)
        routed_gt_topfixed = np.asarray(gt_topfixed_hit[routed_ids_arr], dtype=np.int32)
        contain300_acc.append(
            float(np.max(routed_gt_topfixed) / max(1, int(eval_k))) if len(routed_gt_topfixed) > 0 else 0.0
        )
        contain500_acc.append(
            float(np.max(routed_gt_support) / max(1, int(eval_k))) if len(routed_gt_support) > 0 else 0.0
        )

        if bool(use_soft_topc_overlap):
            cover300_union = bool(np.all(np.any(topfixed_mask[routed_ids_arr, : int(eval_k)], axis=0)))
            cover500_union = bool(np.all(np.any(support_on_topl[routed_ids_arr, : int(eval_k)], axis=0)))
            union_rmax = 0.0
            if bool(cover300_union) and len(routed_ids_arr) > 0:
                theta_gt_worst = float(np.arccos(gt_topk_worst_sim[int(aidx)]))
                theta_fixed = float(np.max(np.asarray(boundary_theta[routed_ids_arr], dtype=np.float64)))
                half_margin = max((theta_fixed - theta_gt_worst) / 2.0, 0.0)
                union_rmax = float(np.tan(half_margin))
            for oid in routed_ids:
                anchor_cover300_by_cluster[int(aidx), int(oid)] = bool(cover300_union)
                anchor_cover500_by_cluster[int(aidx), int(oid)] = bool(cover500_union)
                if bool(cover300_union):
                    anchor_ideal_rmax_by_cluster[int(aidx), int(oid)] = float(union_rmax)
        else:
            for oid in routed_ids:
                cover300 = bool(int(gt_topfixed_hit[int(oid)]) >= int(eval_k))
                cover500 = bool(int(gt_support_hit[int(oid)]) >= int(eval_k))
                anchor_cover300_by_cluster[int(aidx), int(oid)] = bool(cover300)
                anchor_cover500_by_cluster[int(aidx), int(oid)] = bool(cover500)
                if bool(cover300):
                    theta_gt_worst = float(np.arccos(gt_topk_worst_sim[int(aidx)]))
                    theta_fixed = float(boundary_theta[int(oid)])
                    half_margin = max((theta_fixed - theta_gt_worst) / 2.0, 0.0)
                    anchor_ideal_rmax_by_cluster[int(aidx), int(oid)] = float(np.tan(half_margin))

    rmax_vals_per_cluster: List[np.ndarray] = []
    rmax_vals_all: List[np.ndarray] = []
    covered_counts_per_cluster: List[int] = []
    covered500_counts_per_cluster: List[int] = []
    total_counts_per_cluster: List[int] = []
    zero_counts_per_cluster: List[int] = []
    cluster_r_max_raw = []
    cluster_anchor_counts = []
    for cid in range(num_clusters):
        if use_soft_topc_overlap:
            anchor_idx = np.where(np.any(routed_cluster_matrix == int(cid), axis=1))[0].astype(np.int32)
        else:
            anchor_idx = np.where(nearest_cluster == int(cid))[0].astype(np.int32)
        cluster_anchor_counts.append(int(len(anchor_idx)))
        total_counts_per_cluster.append(int(len(anchor_idx)))
        if len(anchor_idx) == 0:
            rmax_vals_per_cluster.append(np.asarray([], dtype=np.float64))
            covered_counts_per_cluster.append(0)
            covered500_counts_per_cluster.append(0)
            zero_counts_per_cluster.append(0)
            continue

        vals = np.asarray(anchor_ideal_rmax_by_cluster[anchor_idx, int(cid)], dtype=np.float64)
        covered = np.asarray(anchor_cover300_by_cluster[anchor_idx, int(cid)], dtype=bool)
        covered500 = np.asarray(anchor_cover500_by_cluster[anchor_idx, int(cid)], dtype=bool)
        covered_counts_per_cluster.append(int(np.sum(covered)))
        covered500_counts_per_cluster.append(int(np.sum(covered500)))
        zero_counts_per_cluster.append(int(np.sum(vals <= 1e-15)))
        rmax_vals_per_cluster.append(vals)
        rmax_vals_all.append(vals)

    if len(rmax_vals_all) > 0:
        all_vals = np.concatenate(rmax_vals_all, axis=0)
    else:
        all_vals = np.asarray([], dtype=np.float64)
    global_fallback = float(np.quantile(all_vals, gamma)) if len(all_vals) > 0 else 0.0

    for cid in range(num_clusters):
        vals = rmax_vals_per_cluster[cid]
        if len(vals) == 0:
            cluster_r_max_raw.append(float(global_fallback))
        else:
            cluster_r_max_raw.append(float(np.quantile(vals, gamma)))

    shrink_enabled = bool(RMAX_SHRINKAGE_ENABLE)
    shrink_tau = float(max(1e-9, float(RMAX_SHRINKAGE_TAU)))
    shrink_min_blend = float(np.clip(float(RMAX_SHRINKAGE_MIN_BLEND), 0.0, 1.0))
    cluster_r_max = []
    shrinkage_actions = []
    for cid in range(num_clusters):
        n_i = int(cluster_anchor_counts[cid])
        raw = float(cluster_r_max_raw[cid])
        if shrink_enabled:
            weight = float(max(shrink_min_blend, n_i / (n_i + shrink_tau)))
            shrunk = float(weight * raw + (1.0 - weight) * global_fallback)
        else:
            weight = 1.0
            shrunk = float(raw)
        cluster_r_max.append(float(max(0.0, shrunk)))
        shrinkage_actions.append(
            {
                "cluster_id": int(cid),
                "anchor_count": int(n_i),
                "raw_quantile": float(raw),
                "global_quantile": float(global_fallback),
                "blend_weight_raw": float(weight),
                "shrunk_quantile": float(shrunk),
            }
        )

    support_conservative_actions = []
    if bool(RMAX_SUPPORT_AWARE_CONSERVATIVE_ENABLE):
        normal_min = int(max(1, int(RMAX_SUPPORT_NORMAL_MIN)))
        soft_min = int(max(0, min(normal_min, int(RMAX_SUPPORT_SOFT_MIN))))
        soft_scale = float(np.clip(float(RMAX_SUPPORT_SOFT_SCALE), 0.0, 1.0))
        hard_value = float(max(0.0, float(RMAX_SUPPORT_HARD_RMAX_VALUE)))
        for cid in range(num_clusters):
            n_i = int(cluster_anchor_counts[cid])
            before = float(cluster_r_max[cid])
            after = float(before)
            action = "none"
            if n_i < int(soft_min):
                after = float(hard_value)
                action = "hard_floor_low_support"
            elif n_i < int(normal_min):
                after = float(max(0.0, before * soft_scale))
                action = "soft_scale_low_support"
            cluster_r_max[cid] = float(max(0.0, after))
            support_conservative_actions.append(
                {
                    "cluster_id": int(cid),
                    "anchor_count": int(n_i),
                    "normal_min": int(normal_min),
                    "soft_min": int(soft_min),
                    "soft_scale": float(soft_scale),
                    "hard_rmax_value": float(hard_value),
                    "value_before_support_adjust": float(before),
                    "value_after_support_adjust": float(cluster_r_max[cid]),
                    "action": str(action),
                }
            )
    else:
        for cid in range(num_clusters):
            support_conservative_actions.append(
                {
                    "cluster_id": int(cid),
                    "anchor_count": int(cluster_anchor_counts[cid]),
                    "action": "disabled",
                    "value_after_support_adjust": float(cluster_r_max[cid]),
                }
            )

    return {
        "cluster_r_max": np.asarray(cluster_r_max, dtype=np.float32),
        "cluster_r_max_raw": np.asarray(cluster_r_max_raw, dtype=np.float32),
        "cluster_r_max_shrunk": np.asarray(cluster_r_max, dtype=np.float32),
        "cluster_anchor_counts": [int(x) for x in cluster_anchor_counts],
        "cluster_track1_coverage_counts": [int(x) for x in covered_counts_per_cluster],
        "cluster_track1_total_counts": [int(x) for x in total_counts_per_cluster],
        "cluster_r_ideal_zero_counts": [int(x) for x in zero_counts_per_cluster],
        "anchor_source": str(anchor_source),
        "anchor_cluster_assign_source": str(cluster_assign_source),
        "rmax_scope": (
            "within_topc_overlap_route_union_docs"
            if bool(use_soft_topc_overlap)
            else "within_owner_cluster_docs"
        ),
        "routing_cluster_selection_policy": str(ROUTING_CLUSTER_SELECTION_POLICY),
        "routing_fixed_top_c": int(overlap_top_c),
        "gamma": float(gamma),
        "shrinkage_policy": {
            "enabled": bool(shrink_enabled),
            "tau": float(shrink_tau),
            "min_blend_weight_raw": float(shrink_min_blend),
            "actions": shrinkage_actions,
        },
        "support_aware_policy": {
            "enabled": bool(RMAX_SUPPORT_AWARE_CONSERVATIVE_ENABLE),
            "normal_min": int(RMAX_SUPPORT_NORMAL_MIN),
            "soft_min": int(RMAX_SUPPORT_SOFT_MIN),
            "soft_scale": float(np.clip(float(RMAX_SUPPORT_SOFT_SCALE), 0.0, 1.0)),
            "hard_rmax_value": float(max(0.0, float(RMAX_SUPPORT_HARD_RMAX_VALUE))),
            "actions": support_conservative_actions,
        },
        "paperfaithful_diagnostics": {
            "proxy_mode": "gpu_topl_prefilter_route_union",
            "nearest_route_contain_at_300_r0": float(np.mean(contain300_acc)) if contain300_acc else 0.0,
            "nearest_route_contain_at_500_r0": float(np.mean(contain500_acc)) if contain500_acc else 0.0,
            "oracle_best_parent_contain_at_300_r0": float(np.mean(contain300_oracle_acc))
            if contain300_oracle_acc
            else 0.0,
            "oracle_best_parent_contain_at_500_r0": float(np.mean(contain500_oracle_acc))
            if contain500_oracle_acc
            else 0.0,
            "gpu_global_topl": int(top_l),
            "gpu_prefilter_fallback_search_count": int(fallback_search_count),
            "gpu_prefilter_exact_fallback_scope": "routed_clusters_only",
            "oracle_best_parent_contain_at_300_r0_is_lower_bound": True,
        },
        "ideal_r_max_stats": {
            "min": float(np.min(all_vals)) if len(all_vals) > 0 else 0.0,
            "p50": float(np.percentile(all_vals, 50.0)) if len(all_vals) > 0 else 0.0,
            "mean": float(np.mean(all_vals)) if len(all_vals) > 0 else 0.0,
            "max": float(np.max(all_vals)) if len(all_vals) > 0 else 0.0,
            "global_gamma_quantile": float(global_fallback),
            "num_anchors": int(len(all_vals)),
            "num_zero_due_to_track1_noncoverage": int(np.sum(np.asarray(all_vals) <= 1e-15))
            if len(all_vals) > 0
            else 0,
        },
    }


def compute_cluster_level_rmax_surrogate(
    *,
    docs: np.ndarray,
    centers: np.ndarray,
    chunks: List[np.ndarray],
    overlap_doc_indices_by_cluster: List[np.ndarray] | None = None,
    eval_k: int,
    fixed_k: int,
    gamma: float,
    anchors: np.ndarray,
    anchor_source: str,
    anchor_cluster_ids: np.ndarray | None = None,
) -> dict:
    """
    论文主线 r_max surrogate（离线）：
    - 路由规则与在线一致：anchor -> Top-c routed clusters（或单 owner cluster）；
    - 先在全局 docs 上取 GT_topk（默认 k=5）；
    - 对 soft_topc_fixed：
      使用 routed Top-c 的 route-union 口径，让 c 真正参与 rescue。
      具体做法是：分别在每个 routed cluster 的 overlap 支持集 C_i 上做
      全局 HNSW 过滤检索取 top-fixed_k，再把这些 top-fixed_k 做并集；
    - 若 GT_topk 不被 route-union top-fixed_k 完整覆盖，则该 anchor 的
      r_max^ideal_track1 记为 0；
    - 否则：
      r_max^ideal_track1 = tan((theta_fixed_boundary - theta_gt_topk_worst)/2)
      其中 theta_fixed_boundary 取 route-union 中最宽松的 per-cluster fixed-k 边界；
    - 最后把该 anchor 的 union-level r_max^ideal 赋给其 routed Top-c 内所有簇，
      再按簇取 quantile_gamma，得到 cluster_r_max[i]。
    """
    docs = normalize_rows(np.asarray(docs, dtype=np.float32))
    centers = normalize_rows(np.asarray(centers, dtype=np.float32))
    anchors = normalize_rows(np.asarray(anchors, dtype=np.float32))

    num_docs = int(docs.shape[0])
    num_clusters = int(centers.shape[0])
    if num_docs <= 0 or num_clusters <= 0:
        raise RuntimeError("compute_cluster_level_rmax_surrogate expects non-empty docs/centers.")
    if len(chunks) != int(num_clusters):
        raise RuntimeError(
            f"chunks/centers size mismatch: len(chunks)={len(chunks)} vs num_clusters={num_clusters}"
        )
    if int(eval_k) < 1 or int(fixed_k) < 1:
        raise RuntimeError("eval_k/fixed_k must be positive.")
    if int(eval_k) > int(fixed_k):
        raise RuntimeError(f"eval_k={eval_k} > fixed_k={fixed_k}, invalid rank setup.")
    if anchors.shape[1] != docs.shape[1]:
        raise RuntimeError(
            f"anchor dim mismatch: anchors dim={anchors.shape[1]} vs docs dim={docs.shape[1]}"
        )

    gamma = float(np.clip(float(gamma), 0.0, 1.0))

    overlap_top_c = int(max(1, min(int(num_clusters), int(ROUTING_FIXED_TOP_C))))
    use_soft_topc_overlap = bool(
        str(ROUTING_CLUSTER_SELECTION_POLICY).strip().lower() == "soft_topc_fixed"
        and int(overlap_top_c) > 1
    )
    if bool(use_soft_topc_overlap):
        if overlap_doc_indices_by_cluster is None:
            overlap_doc_indices_by_cluster, _doc_topc_order = compute_topc_overlap_doc_indices(
                docs=docs,
                centers=centers,
                top_c=int(overlap_top_c),
            )
        if len(overlap_doc_indices_by_cluster) != int(num_clusters):
            raise RuntimeError(
                "overlap_doc_indices_by_cluster size mismatch: "
                f"expected {int(num_clusters)}, got {len(overlap_doc_indices_by_cluster)}"
            )

    fast_gpu_profile = _compute_cluster_level_rmax_surrogate_fast_gpu(
        docs=docs,
        centers=centers,
        chunks=chunks,
        overlap_doc_indices_by_cluster=overlap_doc_indices_by_cluster,
        eval_k=int(eval_k),
        fixed_k=int(fixed_k),
        gamma=float(gamma),
        anchors=anchors,
        anchor_source=anchor_source,
        anchor_cluster_ids=anchor_cluster_ids,
    )
    if fast_gpu_profile is not None:
        return fast_gpu_profile

    anchor_to_center_sim = np.clip(anchors @ centers.T, -1.0, 1.0).astype(np.float64)
    anchor_route_order = np.argsort(-anchor_to_center_sim, axis=1).astype(np.int32)

    if use_soft_topc_overlap:
        routed_cluster_matrix = np.asarray(
            anchor_route_order[:, : int(overlap_top_c)],
            dtype=np.int32,
        )
        nearest_cluster = np.asarray(routed_cluster_matrix[:, 0], dtype=np.int32)
        cluster_assign_source = f"topc_nearest_centroids_by_angular_distance:c={int(overlap_top_c)}"
    elif anchor_cluster_ids is not None:
        nearest_cluster = np.asarray(anchor_cluster_ids, dtype=np.int32)
        if nearest_cluster.shape != (len(anchors),):
            raise RuntimeError(
                "anchor_cluster_ids shape mismatch: "
                f"expected ({len(anchors)},), got {nearest_cluster.shape}"
            )
        if np.any(nearest_cluster < 0) or np.any(nearest_cluster >= int(num_clusters)):
            raise RuntimeError("anchor_cluster_ids has invalid cluster ids.")
        routed_cluster_matrix = nearest_cluster.reshape(-1, 1).astype(np.int32)
        cluster_assign_source = "provided_owner_cluster_ids"
    else:
        # 默认回退：anchor -> nearest public centroid（余弦空间等价于最大点积）
        nearest_cluster = np.argmax(anchor_to_center_sim, axis=1).astype(np.int32)
        routed_cluster_matrix = nearest_cluster.reshape(-1, 1).astype(np.int32)
        cluster_assign_source = "nearest_center"

    # 先在全局 docs 上计算每个 anchor 的 GT_topk 及“最差 GT 邻居”相似度。
    global_sims = np.clip(anchors @ docs.T, -1.0, 1.0).astype(np.float64)
    idx_k_global = int(num_docs - int(eval_k))
    topk_global_idx = np.argpartition(global_sims, kth=idx_k_global, axis=1)[:, idx_k_global:]
    topk_global_sims = np.take_along_axis(global_sims, topk_global_idx, axis=1)
    gt_topk_worst_sim = np.clip(np.min(topk_global_sims, axis=1), -1.0, 1.0)
    gt_topk_sets = [set(int(x) for x in row.tolist()) for row in topk_global_idx]

    anchor_ideal_rmax_by_cluster = np.zeros((len(anchors), num_clusters), dtype=np.float64)
    anchor_cover300_by_cluster = np.zeros((len(anchors), num_clusters), dtype=bool)
    anchor_cover500_by_cluster = np.zeros((len(anchors), num_clusters), dtype=bool)
    contain300_acc = []
    contain500_acc = []
    contain300_oracle_acc = []
    contain500_oracle_acc = []

    cluster_support_indices = (
        [np.asarray(vals, dtype=np.int32).reshape(-1) for vals in overlap_doc_indices_by_cluster]
        if bool(use_soft_topc_overlap)
        else [np.asarray(chunk, dtype=np.int32).reshape(-1) for chunk in chunks]
    )
    support_sets = [set(int(x) for x in vals.tolist()) for vals in cluster_support_indices]
    for aidx in range(len(anchors)):
        routed_ids = [int(x) for x in np.asarray(routed_cluster_matrix[aidx], dtype=np.int32).tolist()]
        gt_set = gt_topk_sets[int(aidx)]
        per_routed_cover300 = []
        per_routed_cover500 = []
        union_topfixed = set()
        union_support = set()
        union_boundary_theta = []
        for oid in routed_ids:
            support_idx = np.asarray(cluster_support_indices[int(oid)], dtype=np.int32).reshape(-1)
            topfixed_idx, topfixed_theta, _search_meta = filtered_global_hnsw_topk_doc_indices_and_scores(
                query_for_server=np.asarray(anchors[int(aidx)], dtype=np.float32),
                docs=docs,
                allowed_doc_indices=support_idx,
                fixed_k=int(fixed_k),
            )
            topfixed_set = set(int(x) for x in np.asarray(topfixed_idx, dtype=np.int32).tolist())
            support_set = support_sets[int(oid)]
            per_routed_cover300.append(float(len(gt_set & topfixed_set) / max(1, int(eval_k))))
            per_routed_cover500.append(float(len(gt_set & support_set) / max(1, int(eval_k))))
            union_topfixed.update(topfixed_set)
            union_support.update(support_set)
            if len(topfixed_theta) > 0:
                union_boundary_theta.append(float(np.max(np.asarray(topfixed_theta, dtype=np.float64))))

        if bool(use_soft_topc_overlap):
            cover300_union = bool(gt_set.issubset(union_topfixed))
            cover500_union = bool(gt_set.issubset(union_support))
            contain300_acc.append(float(len(gt_set & union_topfixed) / max(1, int(eval_k))))
            contain500_acc.append(float(len(gt_set & union_support) / max(1, int(eval_k))))
            union_rmax = 0.0
            if bool(cover300_union) and union_boundary_theta:
                theta_gt_worst = float(np.arccos(gt_topk_worst_sim[int(aidx)]))
                theta_fixed = float(np.max(np.asarray(union_boundary_theta, dtype=np.float64)))
                half_margin = max((theta_fixed - theta_gt_worst) / 2.0, 0.0)
                union_rmax = float(np.tan(half_margin))
            for oid in routed_ids:
                anchor_cover300_by_cluster[int(aidx), int(oid)] = bool(cover300_union)
                anchor_cover500_by_cluster[int(aidx), int(oid)] = bool(cover500_union)
                if bool(cover300_union):
                    anchor_ideal_rmax_by_cluster[int(aidx), int(oid)] = float(union_rmax)
        else:
            for oid in routed_ids:
                support_idx = np.asarray(cluster_support_indices[int(oid)], dtype=np.int32).reshape(-1)
                topfixed_idx, topfixed_theta, _search_meta = filtered_global_hnsw_topk_doc_indices_and_scores(
                    query_for_server=np.asarray(anchors[int(aidx)], dtype=np.float32),
                    docs=docs,
                    allowed_doc_indices=support_idx,
                    fixed_k=int(fixed_k),
                )
                topfixed_set = set(int(x) for x in np.asarray(topfixed_idx, dtype=np.int32).tolist())
                support_set = support_sets[int(oid)]
                cover300 = bool(gt_set.issubset(topfixed_set))
                cover500 = bool(gt_set.issubset(support_set))
                anchor_cover300_by_cluster[int(aidx), int(oid)] = bool(cover300)
                anchor_cover500_by_cluster[int(aidx), int(oid)] = bool(cover500)
                if bool(cover300) and len(topfixed_theta) > 0:
                    theta_gt_worst = float(np.arccos(gt_topk_worst_sim[int(aidx)]))
                    theta_fixed = float(np.max(np.asarray(topfixed_theta, dtype=np.float64)))
                    half_margin = max((theta_fixed - theta_gt_worst) / 2.0, 0.0)
                    anchor_ideal_rmax_by_cluster[int(aidx), int(oid)] = float(np.tan(half_margin))
            contain300_acc.append(float(max(per_routed_cover300)) if per_routed_cover300 else 0.0)
            contain500_acc.append(float(max(per_routed_cover500)) if per_routed_cover500 else 0.0)

        # oracle-best-parent 仅诊断：在所有父簇里选择 coverage 最好的那个。
        best300 = 0.0
        best500 = 0.0
        for cand_set in support_sets:
            best500 = max(best500, float(len(gt_set & cand_set) / max(1, int(eval_k))))
        for oid, support_idx in enumerate(cluster_support_indices):
            topfixed_idx, _topfixed_theta, _search_meta = filtered_global_hnsw_topk_doc_indices_and_scores(
                query_for_server=np.asarray(anchors[int(aidx)], dtype=np.float32),
                docs=docs,
                allowed_doc_indices=support_idx,
                fixed_k=int(fixed_k),
            )
            otop300_set = set(int(x) for x in np.asarray(topfixed_idx, dtype=np.int32).tolist())
            best300 = max(best300, float(len(gt_set & otop300_set) / max(1, int(eval_k))))
        contain300_oracle_acc.append(float(best300))
        contain500_oracle_acc.append(float(best500))

    rmax_vals_per_cluster: List[np.ndarray] = []
    rmax_vals_all: List[np.ndarray] = []
    covered_counts_per_cluster: List[int] = []
    covered500_counts_per_cluster: List[int] = []
    total_counts_per_cluster: List[int] = []
    zero_counts_per_cluster: List[int] = []
    cluster_r_max_raw = []
    cluster_anchor_counts = []
    for cid in range(num_clusters):
        if use_soft_topc_overlap:
            anchor_idx = np.where(np.any(routed_cluster_matrix == int(cid), axis=1))[0].astype(np.int32)
        else:
            anchor_idx = np.where(nearest_cluster == int(cid))[0].astype(np.int32)
        cluster_anchor_counts.append(int(len(anchor_idx)))
        total_counts_per_cluster.append(int(len(anchor_idx)))
        if len(anchor_idx) == 0:
            rmax_vals_per_cluster.append(np.asarray([], dtype=np.float64))
            covered_counts_per_cluster.append(0)
            covered500_counts_per_cluster.append(0)
            zero_counts_per_cluster.append(0)
            continue

        vals = np.asarray(anchor_ideal_rmax_by_cluster[anchor_idx, int(cid)], dtype=np.float64)
        covered = np.asarray(anchor_cover300_by_cluster[anchor_idx, int(cid)], dtype=bool)
        covered500 = np.asarray(anchor_cover500_by_cluster[anchor_idx, int(cid)], dtype=bool)
        covered_counts_per_cluster.append(int(np.sum(covered)))
        covered500_counts_per_cluster.append(int(np.sum(covered500)))
        zero_counts_per_cluster.append(int(np.sum(vals <= 1e-15)))
        rmax_vals_per_cluster.append(vals)
        rmax_vals_all.append(vals)

    if len(rmax_vals_all) > 0:
        all_vals = np.concatenate(rmax_vals_all, axis=0)
    else:
        all_vals = np.asarray([], dtype=np.float64)
    global_fallback = float(np.quantile(all_vals, gamma)) if len(all_vals) > 0 else 0.0

    for cid in range(num_clusters):
        vals = rmax_vals_per_cluster[cid]
        if len(vals) == 0:
            cluster_r_max_raw.append(float(global_fallback))
        else:
            cluster_r_max_raw.append(float(np.quantile(vals, gamma)))

    # 低样本簇保守收缩：向全局 routed-query quantile 做 shrinkage（离线常量）。
    shrink_enabled = bool(RMAX_SHRINKAGE_ENABLE)
    shrink_tau = float(max(1e-9, float(RMAX_SHRINKAGE_TAU)))
    shrink_min_blend = float(np.clip(float(RMAX_SHRINKAGE_MIN_BLEND), 0.0, 1.0))
    cluster_r_max = []
    shrinkage_actions = []
    for cid in range(num_clusters):
        n_i = int(cluster_anchor_counts[cid])
        raw = float(cluster_r_max_raw[cid])
        if shrink_enabled:
            weight = float(max(shrink_min_blend, n_i / (n_i + shrink_tau)))
            shrunk = float(weight * raw + (1.0 - weight) * global_fallback)
        else:
            weight = 1.0
            shrunk = float(raw)
        cluster_r_max.append(float(max(0.0, shrunk)))
        shrinkage_actions.append(
            {
                "cluster_id": int(cid),
                "anchor_count": int(n_i),
                "raw_quantile": float(raw),
                "global_quantile": float(global_fallback),
                "blend_weight_raw": float(weight),
                "shrunk_quantile": float(shrunk),
            }
        )

    support_conservative_actions = []
    if bool(RMAX_SUPPORT_AWARE_CONSERVATIVE_ENABLE):
        normal_min = int(max(1, int(RMAX_SUPPORT_NORMAL_MIN)))
        soft_min = int(max(0, min(normal_min, int(RMAX_SUPPORT_SOFT_MIN))))
        soft_scale = float(np.clip(float(RMAX_SUPPORT_SOFT_SCALE), 0.0, 1.0))
        hard_value = float(max(0.0, float(RMAX_SUPPORT_HARD_RMAX_VALUE)))
        for cid in range(num_clusters):
            n_i = int(cluster_anchor_counts[cid])
            before = float(cluster_r_max[cid])
            after = float(before)
            action = "none"
            if n_i < int(soft_min):
                after = float(hard_value)
                action = "hard_floor_low_support"
            elif n_i < int(normal_min):
                after = float(max(0.0, before * soft_scale))
                action = "soft_scale_low_support"
            cluster_r_max[cid] = float(max(0.0, after))
            support_conservative_actions.append(
                {
                    "cluster_id": int(cid),
                    "anchor_count": int(n_i),
                    "normal_min": int(normal_min),
                    "soft_min": int(soft_min),
                    "soft_scale": float(soft_scale),
                    "hard_rmax_value": float(hard_value),
                    "value_before_support_adjust": float(before),
                    "value_after_support_adjust": float(cluster_r_max[cid]),
                    "action": str(action),
                }
            )
    else:
        for cid in range(num_clusters):
            support_conservative_actions.append(
                {
                    "cluster_id": int(cid),
                    "anchor_count": int(cluster_anchor_counts[cid]),
                    "action": "disabled",
                    "value_after_support_adjust": float(cluster_r_max[cid]),
                }
            )

    return {
        "cluster_r_max": np.asarray(cluster_r_max, dtype=np.float32),
        "cluster_r_max_raw": np.asarray(cluster_r_max_raw, dtype=np.float32),
        "cluster_r_max_shrunk": np.asarray(cluster_r_max, dtype=np.float32),
        "cluster_anchor_counts": [int(x) for x in cluster_anchor_counts],
        "cluster_track1_coverage_counts": [int(x) for x in covered_counts_per_cluster],
        "cluster_track1_total_counts": [int(x) for x in total_counts_per_cluster],
        "cluster_r_ideal_zero_counts": [int(x) for x in zero_counts_per_cluster],
        "anchor_source": str(anchor_source),
        "anchor_cluster_assign_source": str(cluster_assign_source),
        "rmax_scope": (
            "within_topc_overlap_route_union_docs"
            if bool(use_soft_topc_overlap)
            else "within_owner_cluster_docs"
        ),
        "routing_cluster_selection_policy": str(ROUTING_CLUSTER_SELECTION_POLICY),
        "routing_fixed_top_c": int(overlap_top_c),
        "gamma": float(gamma),
        "shrinkage_policy": {
            "enabled": bool(shrink_enabled),
            "tau": float(shrink_tau),
            "min_blend_weight_raw": float(shrink_min_blend),
            "actions": shrinkage_actions,
        },
        "support_aware_policy": {
            "enabled": bool(RMAX_SUPPORT_AWARE_CONSERVATIVE_ENABLE),
            "normal_min": int(max(1, int(RMAX_SUPPORT_NORMAL_MIN))),
            "soft_min": int(max(0, min(int(max(1, int(RMAX_SUPPORT_NORMAL_MIN))), int(RMAX_SUPPORT_SOFT_MIN)))),
            "soft_scale": float(np.clip(float(RMAX_SUPPORT_SOFT_SCALE), 0.0, 1.0)),
            "hard_rmax_value": float(max(0.0, float(RMAX_SUPPORT_HARD_RMAX_VALUE))),
            "actions": support_conservative_actions,
        },
        "paperfaithful_diagnostics": {
            "nearest_route_contain_at_300_r0": float(np.mean(contain300_acc)) if contain300_acc else 0.0,
            "nearest_route_contain_at_500_r0": float(np.mean(contain500_acc)) if contain500_acc else 0.0,
            "oracle_best_parent_contain_at_300_r0": float(np.mean(contain300_oracle_acc))
            if contain300_oracle_acc
            else 0.0,
            "oracle_best_parent_contain_at_500_r0": float(np.mean(contain500_oracle_acc))
            if contain500_oracle_acc
            else 0.0,
        },
        "ideal_r_max_stats": {
            "min": float(np.min(all_vals)) if len(all_vals) > 0 else 0.0,
            "p50": float(np.percentile(all_vals, 50.0)) if len(all_vals) > 0 else 0.0,
            "mean": float(np.mean(all_vals)) if len(all_vals) > 0 else 0.0,
            "max": float(np.max(all_vals)) if len(all_vals) > 0 else 0.0,
            "global_gamma_quantile": float(global_fallback),
            "num_anchors": int(len(all_vals)),
            "num_zero_due_to_track1_noncoverage": int(np.sum(np.asarray(all_vals) <= 1e-15))
            if len(all_vals) > 0
            else 0,
        },
    }


def save_cluster_artifacts(
    pipeline_name: str,
    chunks: List[np.ndarray],
    centers: np.ndarray,
    docs: np.ndarray | None,
    doc_ids: np.ndarray,
    meta: dict,
    out_pkl_path: str,
    out_json_path: str,
    method_info: dict,
    calibration_queries: np.ndarray | None = None,
    rmax_quantile_gamma: float = float(RMAX_CLUSTER_QUANTILE_GAMMA),
    num_clusters: int | None = None,
    target_cluster_size: int | None = None,
    eval_k: int | None = None,
    fixed_k: int | None = None,
):
    centers = np.asarray(centers, dtype=np.float32)
    num_clusters_runtime = int(NUM_CLUSTERS if num_clusters is None else num_clusters)
    target_cluster_size_runtime = int(
        TARGET_CLUSTER_SIZE if target_cluster_size is None else target_cluster_size
    )
    eval_k_runtime = int(EVAL_K if eval_k is None else eval_k)
    fixed_k_runtime = int(FIXED_K if fixed_k is None else fixed_k)
    if len(chunks) != int(num_clusters_runtime):
        raise RuntimeError(f"expected {num_clusters_runtime} clusters, got {len(chunks)}")

    cluster_r_k = []
    cluster_r_fixed = []
    cluster_summaries = []
    docid_to_parent_cluster: Dict[str, int] = {}
    for cid, chunk in enumerate(chunks):
        if len(chunk) != int(target_cluster_size_runtime):
            raise RuntimeError(
                f"cluster {cid} size mismatch, expected {target_cluster_size_runtime}, got {len(chunk)}"
            )

    if docs is None:
        docs = normalize_rows(np.load(WORKSET_DOCS_PATH).astype(np.float32))
    else:
        docs = normalize_rows(np.asarray(docs, dtype=np.float32))

    use_docs_only_fast_surrogate = bool(
        str(RMAX_ANCHOR_POLICY).strip().lower() == "docs_only"
        and _env_flag_local("RMAX_DOCS_ONLY_FAST_SURROGATE", False)
    )
    overlap_top_c = int(max(1, min(int(num_clusters_runtime), int(ROUTING_FIXED_TOP_C))))
    use_soft_topc_overlap = bool(
        str(ROUTING_CLUSTER_SELECTION_POLICY).strip().lower() == "soft_topc_fixed"
        and int(overlap_top_c) > 1
    )
    if bool(use_docs_only_fast_surrogate):
        overlap_doc_indices_by_cluster = [
            np.asarray(chunk, dtype=np.int32).reshape(-1) for chunk in chunks
        ]
        doc_topc_order = np.full((int(len(doc_ids)), 1), -1, dtype=np.int32)
    elif bool(use_soft_topc_overlap):
        overlap_doc_indices_by_cluster, doc_topc_order = compute_topc_overlap_doc_indices(
            docs=docs,
            centers=centers,
            top_c=int(overlap_top_c),
        )
    else:
        overlap_doc_indices_by_cluster = [
            np.asarray(chunk, dtype=np.int32).reshape(-1) for chunk in chunks
        ]
        doc_topc_order = np.argmax(np.clip(docs @ centers.T, -1.0, 1.0), axis=1).astype(np.int32).reshape(-1, 1)

    cluster_profiles: list[dict] = []
    for cid, chunk in enumerate(chunks):
        cluster_docs = docs[np.asarray(chunk, dtype=np.int32)]
        prof = compute_cluster_profile_with_center(
            cluster_docs=cluster_docs,
            center=centers[cid],
            eval_k=int(eval_k_runtime),
            fixed_k=int(fixed_k_runtime),
        )
        cluster_profiles.append(dict(prof))
        cluster_r_k.append(float(prof["r_k"]))
        cluster_r_fixed.append(float(prof["r_fixed"]))
        for idx in np.asarray(chunk, dtype=np.int32).tolist():
            docid_to_parent_cluster[str(doc_ids[int(idx)])] = int(cid)
            if bool(use_docs_only_fast_surrogate):
                doc_topc_order[int(idx), 0] = int(cid)

    if bool(use_docs_only_fast_surrogate):
        raw_vals = np.asarray(
            [
                float(np.tan(max((float(prof["r_fixed"]) - float(prof["r_k"])) / 2.0, 0.0)))
                for prof in cluster_profiles
            ],
            dtype=np.float32,
        )
        global_fallback = float(
            np.quantile(raw_vals, float(np.clip(float(rmax_quantile_gamma), 0.0, 1.0)))
        ) if len(raw_vals) > 0 else 0.0
        cluster_r_max = np.asarray(raw_vals, dtype=np.float32)
        cluster_r_max_raw = np.asarray(raw_vals, dtype=np.float32)
        cluster_anchor_counts = [1 for _ in range(int(num_clusters_runtime))]
        cluster_track1_coverage_counts = [1 for _ in range(int(num_clusters_runtime))]
        cluster_track1_total_counts = [1 for _ in range(int(num_clusters_runtime))]
        cluster_zero_counts = [int(float(v) <= 1e-15) for v in raw_vals.tolist()]
        anchor_selector_meta = {
            "selector": "docs_only_fast_cluster_gap_proxy",
            "target_anchors_per_cluster": 1,
            "num_selected_total": int(num_clusters_runtime),
            "membership_scope": "not_applicable_representative_sampling",
        }
        rmax_profile = {
            "cluster_r_max": np.asarray(cluster_r_max, dtype=np.float32),
            "cluster_r_max_raw": np.asarray(cluster_r_max_raw, dtype=np.float32),
            "cluster_r_max_shrunk": np.asarray(cluster_r_max, dtype=np.float32),
            "cluster_anchor_counts": [int(x) for x in cluster_anchor_counts],
            "cluster_track1_coverage_counts": [int(x) for x in cluster_track1_coverage_counts],
            "cluster_track1_total_counts": [int(x) for x in cluster_track1_total_counts],
            "cluster_r_ideal_zero_counts": [int(x) for x in cluster_zero_counts],
            "anchor_source": "docs_only_fast_cluster_gap_proxy",
            "anchor_cluster_assign_source": "parent_cluster_membership",
            "rmax_scope": (
                "within_topc_overlap_route_union_docs"
                if bool(use_soft_topc_overlap)
                else "within_owner_cluster_docs"
            ),
            "routing_cluster_selection_policy": str(ROUTING_CLUSTER_SELECTION_POLICY),
            "routing_fixed_top_c": int(overlap_top_c),
            "gamma": float(rmax_quantile_gamma),
            "shrinkage_policy": {
                "enabled": False,
                "tau": 0.0,
                "min_blend_weight_raw": 1.0,
                "actions": [],
            },
            "support_aware_policy": {
                "enabled": False,
                "normal_min": 1,
                "soft_min": 1,
                "soft_scale": 1.0,
                "hard_rmax_value": 0.0,
                "actions": [],
            },
            "paperfaithful_diagnostics": {
                "proxy_mode": "docs_only_fast_cluster_gap_proxy",
                "nearest_route_contain_at_300_r0": 1.0,
                "nearest_route_contain_at_500_r0": 1.0,
                "oracle_best_parent_contain_at_300_r0": 1.0,
                "oracle_best_parent_contain_at_500_r0": 1.0,
            },
            "ideal_r_max_stats": {
                "min": float(np.min(raw_vals)) if len(raw_vals) > 0 else 0.0,
                "p50": float(np.percentile(raw_vals, 50.0)) if len(raw_vals) > 0 else 0.0,
                "mean": float(np.mean(raw_vals)) if len(raw_vals) > 0 else 0.0,
                "max": float(np.max(raw_vals)) if len(raw_vals) > 0 else 0.0,
                "global_gamma_quantile": float(global_fallback),
                "num_anchors": int(len(raw_vals)),
                "num_zero_due_to_track1_noncoverage": int(np.sum(raw_vals <= 1e-15)) if len(raw_vals) > 0 else 0,
            },
        }
    else:
        anchors, anchor_source, anchor_selector_meta, anchor_cluster_ids = load_rmax_calibration_anchors(
            docs=docs,
            centers=centers,
            chunks=chunks,
            calibration_queries=calibration_queries,
        )
        rmax_profile = compute_cluster_level_rmax_surrogate(
            docs=docs,
            centers=centers,
            chunks=chunks,
            overlap_doc_indices_by_cluster=overlap_doc_indices_by_cluster,
            eval_k=int(eval_k_runtime),
            fixed_k=int(fixed_k_runtime),
            gamma=float(rmax_quantile_gamma),
            anchors=anchors,
            anchor_source=anchor_source,
            anchor_cluster_ids=anchor_cluster_ids,
        )
        cluster_r_max = np.asarray(rmax_profile["cluster_r_max"], dtype=np.float32)
        cluster_r_max_raw = np.asarray(
            rmax_profile.get("cluster_r_max_raw", cluster_r_max),
            dtype=np.float32,
        )
        cluster_anchor_counts = list(rmax_profile["cluster_anchor_counts"])
        cluster_track1_coverage_counts = list(
            rmax_profile.get("cluster_track1_coverage_counts", [0] * len(cluster_anchor_counts))
        )
        cluster_track1_total_counts = list(
            rmax_profile.get("cluster_track1_total_counts", cluster_anchor_counts)
        )
        cluster_zero_counts = list(
            rmax_profile.get("cluster_r_ideal_zero_counts", [0] * len(cluster_anchor_counts))
        )

    for cid, chunk in enumerate(chunks):
        prof = dict(cluster_profiles[int(cid)])
        cluster_summaries.append({
            "cluster_id": int(cid),
            "cluster_size": int(len(chunk)),
            "topc_overlap_support_size": int(len(np.asarray(overlap_doc_indices_by_cluster[int(cid)], dtype=np.int32))),
            "center_norm": float(prof["center_norm"]),
            "r_k": float(prof["r_k"]),
            "r_fixed": float(prof["r_fixed"]),
            "r_fixed_rank_effective": int(prof.get("r_fixed_rank_effective", fixed_k_runtime)),
            "gap_half": float(prof["gap_half"]),
            "r_max_cluster": float(cluster_r_max[cid]),
            "r_max_cluster_raw": float(cluster_r_max_raw[cid]),
            "r_max_cluster_anchor_count": int(cluster_anchor_counts[cid]),
            "r_max_track1_covered_anchor_count": int(cluster_track1_coverage_counts[cid]),
            "r_ideal_zero_count": int(cluster_zero_counts[cid]),
            "r_max_track1_coverage_ratio": float(
                float(cluster_track1_coverage_counts[cid]) / max(1, int(cluster_track1_total_counts[cid]))
            ),
            "min_doc_to_center": float(prof["min_doc_to_center"]),
            "p50_doc_to_center": float(prof["p50_doc_to_center"]),
            "avg_doc_to_center": float(prof["avg_doc_to_center"]),
            "max_doc_to_center": float(prof["max_doc_to_center"]),
            "doc_indices_head10": [int(x) for x in np.asarray(chunk, dtype=np.int32)[:10].tolist()],
        })

    cluster_info = {
        "pipeline": str(pipeline_name),
        "source_workset_pipeline": meta.get("pipeline"),
        "docs_path": WORKSET_DOCS_PATH,
        "doc_ids_path": WORKSET_DOC_IDS_PATH,
        "corpus_path": WORKSET_CORPUS_JSONL_PATH,
        "num_clusters": int(num_clusters_runtime),
        "chunks": [np.asarray(chunk, dtype=np.int32) for chunk in chunks],
        "cluster_topc_overlap_doc_indices": [
            np.asarray(vals, dtype=np.int32) for vals in overlap_doc_indices_by_cluster
        ],
        "centers": centers.astype(np.float32),
        "cluster_r_k": np.asarray(cluster_r_k, dtype=np.float32),
        "cluster_r_fixed": np.asarray(cluster_r_fixed, dtype=np.float32),
        "cluster_r_max": cluster_r_max.astype(np.float32),
        "rmax_surrogate": {
            "formula": (
                "topc_overlap_route_union: "
                "r_max_ideal_track1=0_if_global_topk_not_subset_of_route_union_top_fixedk_per_cluster_else_"
                "tan((theta_fixed-theta_k)/2), "
                "cluster_r_max=quantile_gamma(r_max_ideal_track1)"
                if str(rmax_profile.get("rmax_scope", "")) == "within_topc_overlap_route_union_docs"
                else "within_cluster: "
                "r_max_ideal_track1=0_if_global_topk_not_subset_of_routed_cluster_top_fixedk_else_"
                "tan((theta_fixed-theta_k)/2), "
                "cluster_r_max=quantile_gamma(r_max_ideal_track1)"
            ),
            "gamma": float(rmax_profile["gamma"]),
            "rmax_scope": str(rmax_profile.get("rmax_scope", "unknown")),
            "anchor_policy": str(RMAX_ANCHOR_POLICY),
            "anchor_source": str(rmax_profile["anchor_source"]),
            "anchor_cluster_assign_source": str(rmax_profile["anchor_cluster_assign_source"]),
            "routing_cluster_selection_policy": str(
                rmax_profile.get("routing_cluster_selection_policy", "")
            ),
            "routing_fixed_top_c": int(rmax_profile.get("routing_fixed_top_c", 1)),
            "cluster_topc_overlap_doc_counts": [
                int(len(np.asarray(vals, dtype=np.int32))) for vals in overlap_doc_indices_by_cluster
            ],
            "anchor_selector_meta": anchor_selector_meta,
            "cluster_anchor_counts": [int(x) for x in cluster_anchor_counts],
            "cluster_track1_coverage_counts": [int(x) for x in cluster_track1_coverage_counts],
            "cluster_track1_total_counts": [int(x) for x in cluster_track1_total_counts],
            "cluster_r_ideal_zero_counts": [int(x) for x in cluster_zero_counts],
            "cluster_r_max_raw": [float(x) for x in cluster_r_max_raw.tolist()],
            "cluster_r_max_shrunk": [float(x) for x in cluster_r_max.tolist()],
            "shrinkage_policy": dict(rmax_profile.get("shrinkage_policy", {})),
            "support_aware_policy": dict(rmax_profile.get("support_aware_policy", {})),
            "paperfaithful_diagnostics": dict(rmax_profile.get("paperfaithful_diagnostics", {})),
            "ideal_r_max_stats": dict(rmax_profile["ideal_r_max_stats"]),
        },
        "doc_topc_nearest_centroids": np.asarray(doc_topc_order, dtype=np.int32),
        "docid_to_parent_cluster": docid_to_parent_cluster,
        "eval_k": int(eval_k_runtime),
        "fixed_k": int(fixed_k_runtime),
        "target_cluster_size": int(target_cluster_size_runtime),
        "clustering_method": method_info,
    }

    os.makedirs(os.path.dirname(out_pkl_path), exist_ok=True)
    with open(out_pkl_path, "wb") as f:
        pickle.dump(cluster_info, f)

    save_json(
        out_json_path,
        {
            "pipeline": cluster_info["pipeline"],
            "num_clusters": int(num_clusters_runtime),
            "eval_k": int(eval_k_runtime),
            "fixed_k": int(fixed_k_runtime),
            "clusters": cluster_summaries,
            "rmax_surrogate": cluster_info["rmax_surrogate"],
            "clustering_method": method_info,
        },
    )
