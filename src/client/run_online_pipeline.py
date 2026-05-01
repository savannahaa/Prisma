"""
读取文档工作集、query 工作集、离线 cluster_info；
先基于缓存 / 上传版本地 fallback 生成 RRDP 区间画像；
读取 epsilon_min / epsilon_max / epsilon_used；
对每个 query 调 gate_one_query(...)；
当前 upload-ready 主线只保留 dense_rdp：走 polar_rdp.py 加扰，再仅把扰动后的 query 向量
发给服务端做全局 HNSW fixed-k 检索；
最后统一用 cluster_retrieval.py 做 exact rerank；
算 strict / relaxed / exact 三套指标；
把每个 query 的结果写入在线结果文件；
再汇总成 summary。

out:
·results/online_results_e5_workset_2000.jsonl
逐 query 结果，一行一个 query。

·results/online_summary_e5_workset_2000.json
汇总指标。
"""

from __future__ import annotations

# Allow running this file directly: `python src/client/run_online_pipeline.py`
if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import json
import os
import pickle
import csv
import time
from typing import List

import numpy as np

MAINLINE_RECALL_SCOPE = os.getenv("MAINLINE_RECALL_SCOPE", "overall").strip().lower()

from shared.config import (
    WORKSET_QRELS_PATH,
    WORKSET_RELAXED_QRELS_PATH,
    WORKSET_CALIBRATION_QUERIES_JSONL_PATH,
    WORKSET_CALIBRATION_QUERIES_PATH,
    WORKSET_CALIBRATION_QUERY_IDS_PATH,
    WORKSET_QUERY_SPLIT_META_PATH,
    WORKSET_CLUSTER_INFO_PATH,
    ONLINE_RESULTS_JSONL,
    ONLINE_SUMMARY_JSON,
    RRDP_PROFILE_JSON,
    PAPERFAITHFUL_MAINLINE_AUDIT_JSON,
    PAPERFAITHFUL_MAINLINE_AUDIT_CSV,
    PAPERFAITHFUL_MAINLINE_AUDIT_PKL,
    RRDP_ETA,
    RRDP_BETA,
    RRDP_K_SAFE,
    RRDP_ENFORCE_EPSILON_INTERVAL,
    RRDP_GLOBAL_INTERVAL_POLICY,
    RRDP_PROFILE_QUERY_LIMIT,
    NEW_MODEL_NAME,
    BATCH_SIZE,
    MAX_LENGTH,
    ALPHA,
    EPSILON,
    EVAL_K,
    FIXED_K,
    PAPERFAITHFUL_MAINLINE_GATE_MODE,
    PAPERFAITHFUL_MAINLINE_TRACK1_ONLY,
    TRACK1_FORCE_PERTURB_R,
    TRACK1_FORCE_PERTURB_R_CLIP_TO_RMAX,
    TRACK1_HNSW_EF_SEARCH_BASE,
    NUM_CLUSTERS,
    TARGET_CLUSTER_SIZE,
    ROUTING_CLUSTER_SELECTION_POLICY,
    ROUTING_FIXED_TOP_C,
    ROUTING_ENABLE_BOUNDARY_MULTICLUSTER,
    ROUTING_BOUNDARY_TOP_M,
    ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD,
    ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_2,
    ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_3,
    ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_4,
    ROUTING_MULTICLUSTER_RMAX_FILTER_ENABLE,
    RNG_SEED,
    EPS,
)
from shared.gpu_accel import (
    angular_distance_to_rows as gpu_angular_distance_to_rows,
    resolve_device_spec,
)
from shared.offline_state import load_online_offline_states
from client.privacy_gate import gate_one_query
from client.polar_rdp import perturb_query_track1
from server.cluster_retrieval import (
    fixed_budget_global_knn_payload,
    prewarm_dense_global_retrieval_runtime,
    rerank_candidate_payload_exact,
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return int(default)


# <=0 表示使用 query workset 全量，不做截断。
ONLINE_QUERY_LIMIT = _env_int("ONLINE_QUERY_LIMIT", 0)
PAPERFAITHFUL_MAINLINE_SKIP_GATE = _env_flag("PAPERFAITHFUL_MAINLINE_SKIP_GATE", False)
ONLINE_ALLOW_MISSING_REFERENCE = _env_flag("ONLINE_ALLOW_MISSING_REFERENCE", False)
ONLINE_ALLOW_SYNTHETIC_DOCID_TEXT_FALLBACK = _env_flag(
    "ONLINE_ALLOW_SYNTHETIC_DOCID_TEXT_FALLBACK",
    False,
)
BYTES_PER_FLOAT32 = 4
BYTES_PER_INT32 = 4
ONLINE_PATH_WARMUP_ROUNDS = 3


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _float_or_none(value):
    if value is None:
        return None
    return float(value)


def _float_or_default(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def build_upload_ready_rrdp_profile_fallback(
    *,
    profile_query_emb: np.ndarray | None,
    profile_query_ids: np.ndarray | None,
    profile_query_emb_source: str,
    epsilon_input: float,
    out_json_path: str,
    skip_reason: str,
    interval_mode: str,
) -> dict:
    num_profile_queries = 0
    if profile_query_emb is not None:
        num_profile_queries = int(len(np.asarray(profile_query_emb)))
    elif profile_query_ids is not None:
        num_profile_queries = int(len(np.asarray(profile_query_ids, dtype=object)))

    profile = {
        "profile_skipped": True,
        "profile_skip_reason": str(skip_reason),
        "profile_query_source": (
            "precomputed_emb_upload_ready_fallback"
            if num_profile_queries > 0
            else "disabled_missing_profile_inputs"
        ),
        "profile_query_emb_source": str(profile_query_emb_source),
        "num_profile_queries": int(num_profile_queries),
        "r_1_minus_eta_global": None,
        "density_proxy": {
            "rho_min": None,
        },
        "epsilon_interval": {
            "epsilon_min": None,
            "epsilon_max": None,
            "epsilon_min_strict_utility": None,
            "epsilon_max_strict_privacy": None,
            "interval_mode": str(interval_mode),
            "strict_interval_feasible": True,
            "interval_feasible": True,
            "enforce_interval_active": False,
            "epsilon_used": float(epsilon_input),
        },
    }
    save_json(out_json_path, profile)
    return profile


def _warmup_reference_query(queries: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if len(queries) > 0:
        return np.asarray(queries[0], dtype=np.float32).reshape(-1)
    if len(centers) > 0:
        return np.asarray(centers[0], dtype=np.float32).reshape(-1)
    raise ValueError("cannot warm up online path without queries or centers")


def _prewarm_dense_online_three_stage_path(
    *,
    queries: np.ndarray,
    centers: np.ndarray,
    chunks,
    docs: np.ndarray,
    doc_ids,
    doc_texts,
    top_k: int,
    fixed_k: int,
) -> dict:
    warm_query = _warmup_reference_query(queries=queries, centers=centers)
    per_round_totals = []
    last_round = None
    for _ in range(int(ONLINE_PATH_WARMUP_ROUNDS)):
        stage_generate_t0 = time.perf_counter()
        query_for_server = np.asarray(warm_query, dtype=np.float32)
        stage_generate_sec = float(time.perf_counter() - stage_generate_t0)

        stage_server_t0 = time.perf_counter()
        candidate_payload = fixed_budget_global_knn_payload(
            query_for_server=query_for_server,
            docs=docs,
            doc_ids=doc_ids,
            doc_texts=doc_texts,
            fixed_k=int(fixed_k),
        )
        stage_server_sec = float(time.perf_counter() - stage_server_t0)

        stage_recover_t0 = time.perf_counter()
        rerank_candidate_payload_exact(
            original_query=warm_query,
            candidate_payload=candidate_payload,
            top_k=int(top_k),
        )
        stage_recover_sec = float(time.perf_counter() - stage_recover_t0)
        round_total = float(stage_generate_sec + stage_server_sec + stage_recover_sec)
        per_round_totals.append(round_total)
        last_round = {
            "candidate_count": int(len(candidate_payload["doc_indices"])),
            "client_generate_query_sec_once": float(stage_generate_sec),
            "server_query_sec_once": float(stage_server_sec),
            "client_recover_docs_sec_once": float(stage_recover_sec),
            "time_three_stage_total_sec_once": float(round_total),
        }

    if last_round is None:
        raise RuntimeError("dense warmup did not execute")

    return {
        "warm_query_source": "evaluation_query_0" if len(queries) > 0 else "center_0",
        "warm_cluster_ids": [],
        "warmup_rounds": int(ONLINE_PATH_WARMUP_ROUNDS),
        "per_round_time_three_stage_total_sec_once": [float(x) for x in per_round_totals],
        "total_time_three_stage_total_sec_once": float(np.sum(per_round_totals)),
        "candidate_count": int(last_round["candidate_count"]),
        "client_generate_query_sec_once": float(last_round["client_generate_query_sec_once"]),
        "server_query_sec_once": float(last_round["server_query_sec_once"]),
        "client_recover_docs_sec_once": float(last_round["client_recover_docs_sec_once"]),
        "time_three_stage_total_sec_once": float(last_round["time_three_stage_total_sec_once"]),
        "counted_in_online_latency": False,
    }


def save_paperfaithful_mainline_audit(
    *,
    cluster_info: dict,
    online_summary: dict,
):
    rmax = dict(cluster_info.get("rmax_surrogate", {}))
    diag = dict(rmax.get("paperfaithful_diagnostics", {}))
    anchor_counts = [int(x) for x in rmax.get("cluster_anchor_counts", [])]
    cov300_counts = [int(x) for x in rmax.get("cluster_track1_coverage_counts", [])]
    total_counts = [int(x) for x in rmax.get("cluster_track1_total_counts", anchor_counts)]
    rmax_raw = [float(x) for x in rmax.get("cluster_r_max_raw", [])]
    rmax_shrunk = [float(x) for x in rmax.get("cluster_r_max_shrunk", [])]

    n = int(max(
        len(anchor_counts),
        len(cov300_counts),
        len(total_counts),
        len(rmax_raw),
        len(rmax_shrunk),
    ))
    per_cluster = []
    for cid in range(n):
        a = int(anchor_counts[cid]) if cid < len(anchor_counts) else 0
        c300 = int(cov300_counts[cid]) if cid < len(cov300_counts) else 0
        t = int(total_counts[cid]) if cid < len(total_counts) else 0
        rraw = float(rmax_raw[cid]) if cid < len(rmax_raw) else 0.0
        rshr = float(rmax_shrunk[cid]) if cid < len(rmax_shrunk) else 0.0
        per_cluster.append(
            {
                "cluster_id": int(cid),
                "anchor_count": int(a),
                "gt_top5_full_cover_count_top300": int(c300),
                "gt_top5_full_cover_ratio_top300": float(c300 / max(1, t)),
                "r_ideal_zero_count": int(
                    rmax.get("cluster_r_ideal_zero_counts", [0] * n)[cid]
                    if cid < len(rmax.get("cluster_r_ideal_zero_counts", []))
                    else 0
                ),
                "r_max_raw": float(rraw),
                "r_max_shrunk": float(rshr),
            }
        )

    audit = {
        "nearest_route_contain_at_300_r0": _float_or_default(
            diag.get("nearest_route_contain_at_300_r0", 0.0)
        ),
        "nearest_route_contain_at_500_r0": _float_or_default(
            diag.get("nearest_route_contain_at_500_r0", 0.0)
        ),
        "oracle_best_parent_contain_at_300_r0": _float_or_default(
            diag.get("oracle_best_parent_contain_at_300_r0", 0.0)
        ),
        "oracle_best_parent_contain_at_500_r0": _float_or_default(
            diag.get("oracle_best_parent_contain_at_500_r0", 0.0)
        ),
        "per_cluster": per_cluster,
        "online_avg_exact_recall_at_k": _float_or_none(online_summary.get("avg_exact_recall_at_k", 0.0)),
        "online_original_track_counter": dict(online_summary.get("original_track_counter", {})),
        "online_executed_track_counter": dict(online_summary.get("executed_track_counter", {})),
        "single_cluster_only": bool(
            (not bool(online_summary.get("routing_protocol", {}).get("boundary_multicluster_enabled", False)))
            and float(online_summary.get("routing_multicluster_applied_ratio", 0.0)) == 0.0
        ),
    }

    os.makedirs(os.path.dirname(PAPERFAITHFUL_MAINLINE_AUDIT_JSON), exist_ok=True)
    with open(PAPERFAITHFUL_MAINLINE_AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    with open(PAPERFAITHFUL_MAINLINE_AUDIT_PKL, "wb") as f:
        pickle.dump(audit, f)

    fieldnames = [
        "cluster_id",
        "anchor_count",
        "gt_top5_full_cover_count_top300",
        "gt_top5_full_cover_ratio_top300",
        "gt_top5_full_cover_count_top500",
        "gt_top5_full_cover_ratio_top500",
        "r_ideal_zero_count",
        "r_max_raw",
        "r_max_shrunk",
        "nearest_route_contain_at_300_r0",
        "nearest_route_contain_at_500_r0",
        "oracle_best_parent_contain_at_300_r0",
        "oracle_best_parent_contain_at_500_r0",
        "online_avg_exact_recall_at_k",
        "single_cluster_only",
    ]
    with open(PAPERFAITHFUL_MAINLINE_AUDIT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_cluster:
            writer.writerow(
                {
                    **row,
                    "nearest_route_contain_at_300_r0": float(audit["nearest_route_contain_at_300_r0"]),
                    "nearest_route_contain_at_500_r0": float(audit["nearest_route_contain_at_500_r0"]),
                    "oracle_best_parent_contain_at_300_r0": float(audit["oracle_best_parent_contain_at_300_r0"]),
                    "oracle_best_parent_contain_at_500_r0": float(audit["oracle_best_parent_contain_at_500_r0"]),
                    "online_avg_exact_recall_at_k": _float_or_none(audit["online_avg_exact_recall_at_k"]),
                    "single_cluster_only": int(bool(audit["single_cluster_only"])),
                }
            )


def finite_or_none(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def evaluate_with_qrels(pred_doc_ids, positive_doc_ids):
    pred_set = set(pred_doc_ids)
    hit_docs = pred_set & positive_doc_ids

    num_hits = len(hit_docs)
    num_positives = len(positive_doc_ids)
    hit_at_k = 1.0 if num_hits > 0 else 0.0
    recall_on_qrels = float(num_hits / num_positives) if num_positives > 0 else 0.0

    return {
        "hit_at_k": hit_at_k,
        "recall_on_qrels": recall_on_qrels,
        "num_hits": num_hits,
        "num_positives": num_positives,
        "hit_doc_ids": sorted(list(hit_docs)),
    }


def evaluate_exact_recall_at_k(pred_doc_ids, gt_indices: np.ndarray, all_doc_ids):
    pred_set = {str(x) for x in pred_doc_ids}
    gt_set = {str(all_doc_ids[int(idx)]) for idx in gt_indices.tolist()}
    if len(gt_set) == 0:
        return 0.0
    return len(pred_set & gt_set) / len(gt_set)


def evaluate_candidate_inclusion_recall_at_k(candidate_doc_ids, gt_indices: np.ndarray, all_doc_ids):
    candidate_set = {str(x) for x in candidate_doc_ids}
    gt_set = {str(all_doc_ids[int(idx)]) for idx in gt_indices.tolist()}
    if len(gt_set) == 0:
        return 0.0
    return len(candidate_set & gt_set) / len(gt_set)


def bytes_to_mb(x_bytes: int) -> float:
    return float(x_bytes) / (1024.0 * 1024.0)


def estimate_client_request_bytes(
    *,
    routing_cluster_ids: List[int],
    query_for_server: np.ndarray | None,
) -> int:
    # 当前协议下 dense 路径只发送 query 向量本身；routing_cluster_ids 仅保留
    # 作为显式协议上下文，便于后续需要时扩展。
    total = 0
    if query_for_server is not None:
        q = np.asarray(query_for_server, dtype=np.float32).reshape(-1)
        total += int(q.size) * int(BYTES_PER_FLOAT32)
    return int(total)


def estimate_server_response_payload_bytes(candidate_payload: dict) -> int:
    # 响应消息：按当前实现中实际用于客户端恢复的 payload 字段估算：
    # doc_indices / scores / embeddings（数值数组） + doc_ids/texts（UTF-8 字节）。
    total = 0
    if "doc_indices" in candidate_payload:
        arr = np.asarray(candidate_payload["doc_indices"], dtype=np.int32).reshape(-1)
        total += int(arr.nbytes)
    if "scores" in candidate_payload and candidate_payload.get("scores") is not None:
        arr = np.asarray(candidate_payload["scores"], dtype=np.float32).reshape(-1)
        total += int(arr.nbytes)
    if "embeddings" in candidate_payload and candidate_payload.get("embeddings") is not None:
        arr = np.asarray(candidate_payload["embeddings"], dtype=np.float32)
        total += int(arr.nbytes)
    for key in ("doc_ids", "texts"):
        vals = candidate_payload.get(key, [])
        if vals is None:
            continue
        total += sum(len(str(v).encode("utf-8")) for v in vals)
    return int(total)


def estimate_server_query_docs_touched(
    *,
    retrieval_backend: str,
    candidate_payload: dict,
    routing_cluster_ids: List[int],
    primary_cluster_id: int,
    chunks,
    docs: np.ndarray,
    fixed_k: int,
) -> int:
    docs_shape = tuple(int(x) for x in getattr(docs, "shape", ()))
    num_docs = int(docs_shape[0]) if len(docs_shape) == 2 else 0

    def _cluster_size(cid: int) -> int:
        return int(len(np.asarray(chunks[int(cid)], dtype=np.int32).reshape(-1)))

    def _global_filtered_touch_count(search_meta: dict | None, support_size: int) -> int:
        meta = dict(search_meta or {})
        if bool(meta.get("used_exact_refill", False)):
            return int(max(0, support_size))
        probe_k = int(meta.get("probe_k_final", 0))
        return int(min(int(num_docs), max(int(probe_k), int(fixed_k), int(HNSW_EF_SEARCH_BASE))))

    backend = str(retrieval_backend)
    if backend in {"cluster_payload", "cluster_payload_multicluster_union"}:
        routed_ids = [
            int(x) for x in candidate_payload.get("routed_cluster_ids", routing_cluster_ids)
        ]
        if not routed_ids:
            routed_ids = [int(primary_cluster_id)]
        return int(sum(_cluster_size(int(cid)) for cid in routed_ids))
    if backend == "faiss_hnsw_ann_angular":
        return int(min(int(num_docs), max(int(fixed_k), int(TRACK1_HNSW_EF_SEARCH_BASE))))
    return int(len(np.asarray(candidate_payload.get("doc_indices", []), dtype=np.int32).reshape(-1)))


def estimate_server_query_io_bytes(
    *,
    retrieval_backend: str,
    candidate_payload: dict,
    routing_cluster_ids: List[int],
    primary_cluster_id: int,
    chunks,
    docs: np.ndarray,
    fixed_k: int,
) -> int:
    docs_shape = tuple(int(x) for x in getattr(docs, "shape", ()))
    embedding_dim = int(docs_shape[1]) if len(docs_shape) == 2 else 0
    docs_touched = int(
        estimate_server_query_docs_touched(
            retrieval_backend=str(retrieval_backend),
            candidate_payload=candidate_payload,
            routing_cluster_ids=[int(x) for x in routing_cluster_ids],
            primary_cluster_id=int(primary_cluster_id),
            chunks=chunks,
            docs=docs,
            fixed_k=int(fixed_k),
        )
    )
    # 这里只估计服务器检索阶段在云端内部触达的向量/索引读取量。
    # 该量仅作 system / I/O 诊断，不计入跨 client-server 边界的 communication。
    embedding_read_bytes = int(max(0, docs_touched) * max(0, embedding_dim) * int(BYTES_PER_FLOAT32))
    return int(embedding_read_bytes)


def load_cached_rrdp_profile_queries() -> tuple[np.ndarray | None, np.ndarray | None, str]:
    emb_path = str(WORKSET_CALIBRATION_QUERIES_PATH)
    ids_path = str(WORKSET_CALIBRATION_QUERY_IDS_PATH)
    if (not os.path.exists(emb_path)) or (not os.path.exists(ids_path)):
        return None, None, "missing_cache_file"
    try:
        emb = np.asarray(np.load(emb_path), dtype=np.float32)
        ids = np.load(ids_path, allow_pickle=True)
    except Exception as e:
        return None, None, f"cache_load_error:{type(e).__name__}"
    if emb.ndim != 2 or len(emb) <= 0:
        return None, None, "cache_emb_invalid_shape"
    if len(ids) != len(emb):
        return None, None, "cache_emb_ids_length_mismatch"
    return emb.astype(np.float32), np.asarray(ids, dtype=object), "cache_hit"


def angular_distance_to_centers(query: np.ndarray, centers: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    centers = np.asarray(centers, dtype=np.float32)
    gpu_theta = gpu_angular_distance_to_rows(
        query=query,
        rows=centers,
        assume_unit_norm=True,
        cache_rows_if_large=False,
        role="client",
    )
    if gpu_theta is not None:
        return np.asarray(gpu_theta, dtype=np.float32)
    sims = np.clip(centers @ query, -1.0, 1.0).astype(np.float32)
    return np.arccos(sims).astype(np.float32)


def build_skip_gate_result(
    *,
    query: np.ndarray,
    centers: np.ndarray,
    cluster_r_max: np.ndarray,
    epsilon_input: float,
    epsilon_used: float,
    epsilon_min,
    epsilon_max,
) -> dict:
    theta = angular_distance_to_centers(query=query, centers=centers)
    primary_cluster_id = int(np.argmin(theta)) if len(theta) > 0 else 0
    primary_r_max = float(cluster_r_max[int(primary_cluster_id)]) if len(cluster_r_max) > 0 else 0.0
    interval_defined = epsilon_min is not None or epsilon_max is not None
    interval_feasible = True
    if epsilon_min is not None and float(epsilon_used) < float(epsilon_min):
        interval_feasible = False
    if epsilon_max is not None and float(epsilon_used) > float(epsilon_max):
        interval_feasible = False
    return {
        "cluster_id": int(primary_cluster_id),
        "r_max": float(primary_r_max),
        "r_max_cluster": float(primary_r_max),
        "r_max_mode": "skip_gate_nearest_centroid",
        "sigma": 0.0,
        "r_rdp_bar": 0.0,
        "track": "dense_rdp",
        "track_reason": "paperfaithful_mainline_skip_gate_nearest_centroid_dense",
        "epsilon_input": float(epsilon_input),
        "epsilon_used": float(epsilon_used),
        "epsilon_min": epsilon_min,
        "epsilon_max": epsilon_max,
        "epsilon_interval_defined": bool(interval_defined),
        "epsilon_interval_feasible": bool(interval_feasible),
        "epsilon_clipped": False,
        "interval_infeasible_warning": False,
    }


def select_routing_clusters_for_query(
    *,
    query: np.ndarray,
    centers: np.ndarray,
    primary_cluster_id: int,
) -> dict:
    centers = np.asarray(centers, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    theta = angular_distance_to_centers(query=query, centers=centers)
    order = np.argsort(theta).astype(np.int32)

    primary_cluster_id = int(primary_cluster_id)
    if primary_cluster_id < 0 or primary_cluster_id >= len(centers):
        raise ValueError(f"invalid primary_cluster_id={primary_cluster_id}")

    others = [int(cid) for cid in order.tolist() if int(cid) != int(primary_cluster_id)]
    route_pref = [int(primary_cluster_id)] + others
    second_cluster = int(route_pref[1]) if len(route_pref) > 1 else int(primary_cluster_id)
    primary_dist = float(theta[int(primary_cluster_id)])
    second_dist = float(theta[int(second_cluster)])
    ordered_dists = [float(theta[int(cid)]) for cid in route_pref]
    gap_to_second = float(ordered_dists[1] - primary_dist) if len(ordered_dists) > 1 else float("inf")
    gap_to_third = float(ordered_dists[2] - primary_dist) if len(ordered_dists) > 2 else float("inf")
    gap_to_fourth = float(ordered_dists[3] - primary_dist) if len(ordered_dists) > 3 else float("inf")
    top_m = int(max(1, min(int(ROUTING_BOUNDARY_TOP_M), len(route_pref))))
    top_c = int(max(1, min(int(ROUTING_FIXED_TOP_C), len(route_pref))))

    if str(ROUTING_CLUSTER_SELECTION_POLICY) == "soft_topc_fixed":
        routed_cluster_ids = [int(x) for x in route_pref[:top_c]]
        apply_multicluster = bool(len(routed_cluster_ids) > 1)
        routing_rule_version = "soft_topc_fixed_v1"
        return {
            "routed_cluster_ids": routed_cluster_ids,
            "routing_num_clusters": int(len(routed_cluster_ids)),
            "routing_rule_version": str(routing_rule_version),
            "routing_boundary_gap_to_second": float(gap_to_second),
            "routing_boundary_gap_to_third": float(gap_to_third),
            "routing_boundary_gap_to_fourth": float(gap_to_fourth),
            "routing_boundary_threshold": float(ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD),
            "routing_boundary_threshold_to_second": float(
                ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_2
            ),
            "routing_boundary_threshold_to_third": (
                float(ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_3)
                if ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_3 is not None
                else None
            ),
            "routing_boundary_threshold_to_fourth": (
                float(ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_4)
                if ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_4 is not None
                else None
            ),
            "routing_boundary_multicluster_applied": bool(apply_multicluster),
            "routing_primary_center_theta": float(primary_dist),
            "routing_second_center_theta": float(second_dist),
            "routing_ordered_cluster_ids": [int(x) for x in route_pref],
            "routing_ordered_center_theta": [float(x) for x in ordered_dists],
            "routing_fixed_top_c": int(top_c),
        }

    use_gap_ladder = bool(
        bool(ROUTING_ENABLE_BOUNDARY_MULTICLUSTER)
        and top_m > 1
        and (
            ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_3 is not None
            or ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_4 is not None
        )
    )
    threshold2 = float(ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_2)
    threshold3 = (
        float(ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_3)
        if ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_3 is not None
        else None
    )
    threshold4 = (
        float(ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_4)
        if ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_4 is not None
        else None
    )

    if use_gap_ladder:
        routed_cluster_ids = [int(primary_cluster_id)]
        if np.isfinite(gap_to_second) and gap_to_second <= threshold2:
            routed_cluster_ids.append(int(route_pref[1]))
            threshold3_eff = float(threshold3) if threshold3 is not None else float(threshold2)
            if (
                top_m >= 3
                and len(route_pref) >= 3
                and np.isfinite(gap_to_third)
                and gap_to_third <= threshold3_eff
            ):
                routed_cluster_ids.append(int(route_pref[2]))
                threshold4_eff = (
                    float(threshold4)
                    if threshold4 is not None
                    else float(threshold3_eff)
                )
                if (
                    top_m >= 4
                    and len(route_pref) >= 4
                    and np.isfinite(gap_to_fourth)
                    and gap_to_fourth <= threshold4_eff
                ):
                    routed_cluster_ids.append(int(route_pref[3]))
        apply_multicluster = bool(len(routed_cluster_ids) > 1)
        routing_rule_version = "boundary_gap_ladder_v1"
    else:
        apply_multicluster = bool(
            bool(ROUTING_ENABLE_BOUNDARY_MULTICLUSTER)
            and top_m > 1
            and np.isfinite(gap_to_second)
            and float(gap_to_second) <= float(ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD)
        )
        if apply_multicluster:
            routed_cluster_ids = [int(x) for x in route_pref[:top_m]]
        else:
            routed_cluster_ids = [int(primary_cluster_id)]
        routing_rule_version = "boundary_top_m_legacy"

    return {
        "routed_cluster_ids": routed_cluster_ids,
        "routing_num_clusters": int(len(routed_cluster_ids)),
        "routing_rule_version": str(routing_rule_version),
        "routing_boundary_gap_to_second": float(gap_to_second),
        "routing_boundary_gap_to_third": float(gap_to_third),
        "routing_boundary_gap_to_fourth": float(gap_to_fourth),
        "routing_boundary_threshold": float(ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD),
        "routing_boundary_threshold_to_second": float(threshold2),
        "routing_boundary_threshold_to_third": (
            float(threshold3) if threshold3 is not None else None
        ),
        "routing_boundary_threshold_to_fourth": (
            float(threshold4) if threshold4 is not None else None
        ),
        "routing_boundary_multicluster_applied": bool(apply_multicluster),
        "routing_primary_center_theta": float(primary_dist),
        "routing_second_center_theta": float(second_dist),
        "routing_ordered_cluster_ids": [int(x) for x in route_pref],
        "routing_ordered_center_theta": [float(x) for x in ordered_dists],
        "routing_fixed_top_c": int(top_c),
    }


def main():
    client_state, server_state = load_online_offline_states(
        query_limit=int(ONLINE_QUERY_LIMIT),
        allow_missing_reference=bool(ONLINE_ALLOW_MISSING_REFERENCE),
        allow_synthetic_docid_text_fallback=bool(ONLINE_ALLOW_SYNTHETIC_DOCID_TEXT_FALLBACK),
    )
    shared_cluster_state = client_state.cluster

    queries = client_state.queries
    query_ids = client_state.query_ids
    gt_topk = client_state.gt_topk
    strict_qrels_map = client_state.strict_qrels_map
    relaxed_qrels_map = client_state.relaxed_qrels_map
    reference_metrics_available = bool(client_state.reference_metrics_available)
    docs = server_state.docs
    doc_ids = server_state.doc_ids
    doc_texts = server_state.doc_texts
    cluster_info = shared_cluster_state.cluster_info
    cluster_info_contract = shared_cluster_state.cluster_info_contract
    centers = shared_cluster_state.centers
    cluster_r_k = shared_cluster_state.cluster_r_k
    cluster_r_fixed = shared_cluster_state.cluster_r_fixed
    cluster_r_max = shared_cluster_state.cluster_r_max
    chunks = shared_cluster_state.chunks
    overlap_doc_indices_by_cluster = shared_cluster_state.overlap_doc_indices_by_cluster

    if not reference_metrics_available:
        print(
            "[warn] reference metrics unavailable; running latency/communication-only mode "
            f"(allow_missing_reference={ONLINE_ALLOW_MISSING_REFERENCE})"
        )

    top_k = int(shared_cluster_state.top_k)
    fixed_k = int(shared_cluster_state.fixed_k)
    track1_dense_runtime_stats = prewarm_dense_global_retrieval_runtime(
        docs=docs,
        fixed_k=int(fixed_k),
    )
    dense_online_warmup_stats = _prewarm_dense_online_three_stage_path(
        queries=queries,
        centers=centers,
        chunks=chunks,
        docs=docs,
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        top_k=int(top_k),
        fixed_k=int(fixed_k),
    )
    rng = np.random.default_rng(RNG_SEED)
    cached_profile_q_emb, cached_profile_q_ids, cached_profile_q_status = load_cached_rrdp_profile_queries()
    if cached_profile_q_emb is None or cached_profile_q_ids is None:
        rrdp_profile = build_upload_ready_rrdp_profile_fallback(
            profile_query_emb=None,
            profile_query_ids=None,
            profile_query_emb_source=str(cached_profile_q_status),
            epsilon_input=float(EPSILON),
            out_json_path=RRDP_PROFILE_JSON,
            skip_reason=f"missing_calibration_cache:{cached_profile_q_status}",
            interval_mode="disabled_missing_calibration",
        )
    else:
        rrdp_profile = build_upload_ready_rrdp_profile_fallback(
            profile_query_emb=cached_profile_q_emb,
            profile_query_ids=cached_profile_q_ids,
            profile_query_emb_source=str(cached_profile_q_status),
            epsilon_input=float(EPSILON),
            out_json_path=RRDP_PROFILE_JSON,
            skip_reason="upload_ready_external_rrdp_profile_removed",
            interval_mode="disabled_upload_ready_rrdp_profile_removed",
        )
    eps_interval = dict(rrdp_profile.get("epsilon_interval", {}))
    epsilon_min = eps_interval.get("epsilon_min")
    epsilon_max = eps_interval.get("epsilon_max")
    epsilon_min_strict = eps_interval.get("epsilon_min_strict_utility")
    epsilon_max_strict = eps_interval.get("epsilon_max_strict_privacy")
    interval_mode = str(eps_interval.get("interval_mode", "strict"))
    profile_interval_feasible_strict = bool(
        eps_interval.get("strict_interval_feasible", eps_interval.get("interval_feasible", False))
    )
    enforce_interval_active_profile = bool(
        eps_interval.get(
            "enforce_interval_active",
            bool(RRDP_ENFORCE_EPSILON_INTERVAL) and profile_interval_feasible_strict,
        )
    )
    epsilon_used_profile = float(eps_interval.get("epsilon_used", EPSILON))
    epsilon_for_gate = float(epsilon_used_profile)
    profile_interval_feasible = bool(eps_interval.get("interval_feasible", False))
    epsilon_min_for_gate = epsilon_min
    epsilon_max_for_gate = epsilon_max
    global_interval_policy = str(RRDP_GLOBAL_INTERVAL_POLICY).strip().lower()
    if global_interval_policy not in {"warn_only"}:
        print(
            "[warn] invalid RRDP_GLOBAL_INTERVAL_POLICY="
            f"{RRDP_GLOBAL_INTERVAL_POLICY}, fallback to warn_only"
        )
        global_interval_policy = "warn_only"
    global_force_dense_only_warning = bool(not profile_interval_feasible_strict)
    if bool(PAPERFAITHFUL_MAINLINE_SKIP_GATE):
        global_force_dense_only_warning = False
    if global_force_dense_only_warning:
        raise RuntimeError(
            "upload-ready release keeps only the dense/track1 mainline path, "
            "but the current epsilon-interval profile is infeasible for the selected queries. "
            "Use a feasible epsilon/workset/query split under the warn_only policy."
        )

    print("=" * 80)
    print("online evaluation (paper-aligned strict mode)")
    print("=" * 80)
    print(f"client gpu role : {resolve_device_spec('client')}")
    print(f"server gpu role : {resolve_device_spec('server')}")
    print(f"docs shape      : {docs.shape}")
    print(f"queries shape   : {queries.shape}")
    print(f"query_limit     : {query_limit}")
    print(f"calib queries   : {WORKSET_CALIBRATION_QUERIES_JSONL_PATH}")
    print(f"calib emb cache : {WORKSET_CALIBRATION_QUERIES_PATH} ({cached_profile_q_status})")
    print(f"num_clusters    : {len(chunks)}")
    print(f"top_k/fixed_k   : {top_k}/{fixed_k}")
    print(f"rrdp_k_safe     : {int(RRDP_K_SAFE)}")
    print(f"alpha/epsilon   : {ALPHA}/{EPSILON}")
    if str(ROUTING_CLUSTER_SELECTION_POLICY) == "soft_topc_fixed":
        print(
            "routing rule    : fixed Top-c nearest centroids -> "
            "pick argmax cluster_r_max over Top-c -> "
            "global HNSW fixed-k over the full workset using only the perturbed query "
            f"(top_c={int(ROUTING_FIXED_TOP_C)})"
        )
    elif bool(ROUTING_ENABLE_BOUNDARY_MULTICLUSTER):
        if (
            ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_3 is not None
            or ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_4 is not None
        ):
            print(
                "routing rule    : nearest_center_theta primary -> boundary gap ladder "
                f"(th2={ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_2}, "
                f"th3={ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_3}, "
                f"th4={ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_4}) -> "
                "per-cluster r_max filter -> dense ANN over the full workset"
            )
        else:
            print(
                "routing rule    : nearest_center_theta primary -> "
                "if boundary gap small then top-m clusters -> cluster_r_max lookup by primary"
            )
    else:
        print(
            "routing rule    : nearest_center_theta -> single cluster -> "
            "cluster_r_max lookup -> global HNSW retrieval over the full workset"
        )
    print(f"cluster_contract: {cluster_info_contract['signature']}")
    print(
        "rrdp interval   : "
        f"epsilon_min={finite_or_none(epsilon_min)}, "
        f"epsilon_max={finite_or_none(epsilon_max)}, "
        f"epsilon_used={epsilon_used_profile}, "
        f"feasible={profile_interval_feasible}"
    )
    print(
        "rrdp strict     : "
        f"epsilon_min_strict={finite_or_none(epsilon_min_strict)}, "
        f"epsilon_max_strict={finite_or_none(epsilon_max_strict)}, "
        f"strict_feasible={profile_interval_feasible_strict}, "
        f"mode={interval_mode}, "
        f"enforce_active={enforce_interval_active_profile}"
    )
    print(f"global_policy   : {global_interval_policy}")
    print(f"skip_gate       : {bool(PAPERFAITHFUL_MAINLINE_SKIP_GATE)}")
    if not profile_interval_feasible_strict:
        print(
            "[warn] strict epsilon interval infeasible; "
            f"profile keeps {interval_mode} bounds (enforce_active={enforce_interval_active_profile}), "
            "per-query gate still compares cluster_r_max vs r_rdp_bar."
        )
    print("=" * 80)

    original_track_counter = {}
    executed_track_counter = {}
    strict_hit_list = []
    strict_recall_qrels_list = []
    strict_positive_query_count = 0
    strict_hit_positive_list = []
    strict_recall_positive_list = []
    relaxed_hit_list = []
    relaxed_recall_qrels_list = []
    relaxed_positive_query_count = 0
    relaxed_hit_positive_list = []
    relaxed_recall_positive_list = []
    exact_recall_list = []
    exact_recall_track1_only_list = []
    candidate_inclusion_recall_list = []
    candidate_inclusion_recall_track1_only_list = []
    r_max_list = []
    dense_margin_list = []
    routing_selected_max_rmax_list = []
    routing_selected_max_margin_list = []
    routing_dense_exec_rmax_list = []
    routing_dense_exec_margin_list = []
    epsilon_used_list = []
    sigma_gate_list = []
    perturb_radius_all_queries_list = []
    perturb_radius_dense_queries_list = []
    perturb_radius_requested_dense_queries_list = []
    delta_alpha_deg_dense_queries_list = []
    time_client_generate_sec_list = []
    time_server_query_sec_list = []
    time_client_recover_sec_list = []
    time_client_generate_sec_track1_only_list = []
    time_server_query_sec_track1_only_list = []
    time_client_recover_sec_track1_only_list = []
    server_query_est_docs_touched_list = []
    server_query_est_docs_touched_track1_only_list = []
    comm_client_generate_mb_list = []
    comm_server_query_mb_list = []
    comm_client_generate_mb_track1_only_list = []
    comm_server_query_mb_track1_only_list = []
    io_server_query_est_mb_list = []
    io_server_query_est_mb_track1_only_list = []
    system_total_est_mb_list = []
    system_total_est_mb_track1_only_list = []
    epsilon_clipped_count = 0
    epsilon_interval_infeasible_count = 0
    global_policy_applied_count = 0
    global_policy_override_count = 0
    perturb_radius_clipped_count = 0
    r_max_mode_counter = {}
    perturb_radius_mode_counter = {}
    routing_multicluster_applied_count = 0
    routing_num_clusters_counter = {}
    routing_selected_num_clusters_counter = {}
    routing_dense_eligible_num_clusters_counter = {}
    routing_rmax_filtered_query_count = 0
    track1_candidate_budget_list = []
    returned_candidate_count_list = []
    primary_cluster_query_counts = {str(int(cid)): 0 for cid in range(len(chunks))}
    primary_cluster_local_track_counts = {
        str(int(cid)): {"dense_rdp": 0} for cid in range(len(chunks))
    }
    primary_cluster_final_track_counts = {
        str(int(cid)): {"dense_rdp": 0} for cid in range(len(chunks))
    }
    pipeline_three_stage_start = time.perf_counter()

    with open(ONLINE_RESULTS_JSONL, "w", encoding="utf-8") as writer:
        for i in range(len(queries)):
            q = queries[i]
            qid = str(query_ids[i])

            stage_client_generate_t0 = time.perf_counter()
            if bool(PAPERFAITHFUL_MAINLINE_SKIP_GATE):
                gate_result = build_skip_gate_result(
                    query=q,
                    centers=centers,
                    cluster_r_max=cluster_r_max,
                    epsilon_input=float(EPSILON),
                    epsilon_used=float(epsilon_for_gate),
                    epsilon_min=epsilon_min_for_gate,
                    epsilon_max=epsilon_max_for_gate,
                )
            else:
                gate_result = gate_one_query(
                    query=q,
                    centers=centers,
                    cluster_r_max=cluster_r_max,
                    cluster_r_k=cluster_r_k,
                    cluster_r_fixed=cluster_r_fixed,
                    alpha=ALPHA,
                    epsilon=epsilon_for_gate,
                    epsilon_min=epsilon_min_for_gate,
                    epsilon_max=epsilon_max_for_gate,
                    enforce_epsilon_interval=bool(enforce_interval_active_profile),
                )
            r_max_mode = str(gate_result.get("r_max_mode", "unknown"))

            cluster_id = int(gate_result["cluster_id"])
            cluster_key = str(int(cluster_id))
            routing_info = select_routing_clusters_for_query(
                query=q,
                centers=centers,
                primary_cluster_id=int(cluster_id),
            )
            routing_selected_cluster_ids = [int(x) for x in routing_info["routed_cluster_ids"]]
            routing_selected_cluster_rmax = [
                float(cluster_r_max[int(cid)]) for cid in routing_selected_cluster_ids
            ]
            selected_max_rmax = (
                float(max(routing_selected_cluster_rmax))
                if routing_selected_cluster_rmax
                else float(gate_result["r_max"])
            )
            selected_max_rmax_cluster_id = (
                int(routing_selected_cluster_ids[int(np.argmax(np.asarray(routing_selected_cluster_rmax)))])
                if routing_selected_cluster_rmax
                else int(cluster_id)
            )
            track1_return_embeddings = int(fixed_k)
            track1_selected_cluster_id = int(selected_max_rmax_cluster_id)
            if str(ROUTING_CLUSTER_SELECTION_POLICY) == "soft_topc_fixed":
                if str(PAPERFAITHFUL_MAINLINE_GATE_MODE) == "force_dense":
                    local_track = "dense_rdp"
                    local_track_reason = "paperfaithful_mainline_gate_mode_force_dense"
                else:
                    if float(selected_max_rmax) < float(gate_result["r_rdp_bar"]):
                        raise RuntimeError(
                            "upload-ready release keeps only track1/dense queries, "
                            "but the current evaluation query falls outside that scope: "
                            f"query_id={qid}, cluster_id={cluster_id}, "
                            f"selected_max_rmax={float(selected_max_rmax)}, "
                            f"r_rdp_bar={float(gate_result['r_rdp_bar'])}"
                        )
                    local_track = "dense_rdp"
                    local_track_reason = "soft_topc_max_cluster_r_max_vs_r_rdp_bar"
            else:
                local_track = "dense_rdp"
                local_track_reason = str(gate_result.get("track_reason", ""))
            if str(gate_result.get("track")) != "dense_rdp":
                raise RuntimeError(
                    "upload-ready release keeps only track1/dense queries, "
                    "but gate_one_query marked this evaluation query out of scope: "
                    f"query_id={qid}, gate_track={gate_result.get('track')}, "
                    f"cluster_id={cluster_id}, r_max={float(gate_result['r_max'])}, "
                    f"r_rdp_bar={float(gate_result['r_rdp_bar'])}"
                )
            original_track_counter[local_track] = original_track_counter.get(local_track, 0) + 1
            r_max_mode_counter[r_max_mode] = r_max_mode_counter.get(r_max_mode, 0) + 1
            primary_cluster_query_counts[cluster_key] = int(primary_cluster_query_counts.get(cluster_key, 0)) + 1
            if local_track not in primary_cluster_local_track_counts[cluster_key]:
                primary_cluster_local_track_counts[cluster_key][local_track] = 0
            primary_cluster_local_track_counts[cluster_key][local_track] = int(
                primary_cluster_local_track_counts[cluster_key].get(local_track, 0)
            ) + 1
            routing_cluster_ids = list(routing_selected_cluster_ids)
            cluster_size = int(len(chunks[cluster_id]))
            routing_selected_total_size = int(
                np.sum([len(np.asarray(chunks[int(cid)], dtype=np.int32)) for cid in routing_selected_cluster_ids])
            )
            routing_selected_overlap_support_total_size = int(
                np.sum(
                    [
                        len(np.asarray(overlap_doc_indices_by_cluster[int(cid)], dtype=np.int32))
                        for cid in routing_selected_cluster_ids
                    ]
                )
            )
            routed_cluster_total_size = int(routing_selected_total_size)
            routed_overlap_support_total_size = int(routing_selected_overlap_support_total_size)
            routing_selected_num_clusters_counter[str(int(len(routing_selected_cluster_ids)))] = (
                int(
                    routing_selected_num_clusters_counter.get(
                        str(int(len(routing_selected_cluster_ids))),
                        0,
                    )
                )
                + 1
            )
            if bool(routing_info.get("routing_boundary_multicluster_applied", False)):
                routing_multicluster_applied_count += 1
            final_track = str(local_track)
            final_track_reason = str(local_track_reason)
            global_policy_applied = False
            global_policy_overrode_track = False
            global_policy_override_reason = ""

            dense_eligible_cluster_ids = []
            if str(ROUTING_CLUSTER_SELECTION_POLICY) == "soft_topc_fixed":
                dense_eligible_cluster_ids = [int(track1_selected_cluster_id)]
                routing_dense_eligible_num_clusters_counter[str(int(len(dense_eligible_cluster_ids)))] = (
                    int(
                        routing_dense_eligible_num_clusters_counter.get(
                            str(int(len(dense_eligible_cluster_ids))),
                            0,
                        )
                    )
                    + 1
                )
                if final_track == "dense_rdp":
                    routing_cluster_ids = [int(cid) for cid in routing_selected_cluster_ids]
            elif final_track == "dense_rdp":
                dense_eligible_cluster_ids = [
                    int(cid)
                    for cid in routing_selected_cluster_ids
                    if float(cluster_r_max[int(cid)]) >= float(gate_result["r_rdp_bar"])
                ]
                routing_dense_eligible_num_clusters_counter[str(int(len(dense_eligible_cluster_ids)))] = (
                    int(
                        routing_dense_eligible_num_clusters_counter.get(
                            str(int(len(dense_eligible_cluster_ids))),
                            0,
                        )
                    )
                    + 1
                )
                if bool(ROUTING_MULTICLUSTER_RMAX_FILTER_ENABLE):
                    if len(dense_eligible_cluster_ids) <= 0:
                        raise RuntimeError(
                            "upload-ready release keeps only track1/dense queries, "
                            "but boundary multicluster routing found no dense-eligible cluster: "
                            f"query_id={qid}, primary_cluster_id={cluster_id}"
                        )
                    else:
                        routing_cluster_ids = [int(cid) for cid in dense_eligible_cluster_ids]
                        if len(routing_cluster_ids) != len(routing_selected_cluster_ids):
                            routing_rmax_filtered_query_count += 1
                else:
                    routing_cluster_ids = [int(cid) for cid in routing_selected_cluster_ids]

            routing_num_clusters_counter[str(int(len(routing_cluster_ids)))] = (
                int(routing_num_clusters_counter.get(str(int(len(routing_cluster_ids))), 0)) + 1
            )
            routed_cluster_total_size = int(
                np.sum([len(np.asarray(chunks[int(cid)], dtype=np.int32)) for cid in routing_cluster_ids])
            )

            dense_exec_r_max = None
            if final_track == "dense_rdp":
                if str(ROUTING_CLUSTER_SELECTION_POLICY) == "soft_topc_fixed":
                    dense_exec_r_max = float(selected_max_rmax)
                else:
                    dense_exec_r_max = float(
                        np.min([float(cluster_r_max[int(cid)]) for cid in routing_cluster_ids])
                    )
                    track1_selected_cluster_id = int(routing_cluster_ids[0])
            track1_selected_cluster_size = int(
                len(np.asarray(chunks[int(track1_selected_cluster_id)], dtype=np.int32))
            )
            track1_selected_overlap_support_size = int(
                len(
                    np.asarray(
                        overlap_doc_indices_by_cluster[int(track1_selected_cluster_id)],
                        dtype=np.int32,
                    )
                )
            )

            if global_policy_applied:
                global_policy_applied_count += 1
            if global_policy_overrode_track:
                global_policy_override_count += 1
            if final_track not in primary_cluster_final_track_counts[cluster_key]:
                primary_cluster_final_track_counts[cluster_key][final_track] = 0
            primary_cluster_final_track_counts[cluster_key][final_track] = int(
                primary_cluster_final_track_counts[cluster_key].get(final_track, 0)
            ) + 1

            query_for_server = None
            pert = None
            use_clean_query_without_perturb = bool(
                PAPERFAITHFUL_MAINLINE_SKIP_GATE
                and TRACK1_FORCE_PERTURB_R is not None
                and abs(float(TRACK1_FORCE_PERTURB_R)) <= float(EPS)
            )
            if use_clean_query_without_perturb:
                pert = {
                    "perturbed_query_for_server": np.asarray(q, dtype=np.float32),
                    "radius_mode": "forced_zero_clean_query",
                    "requested_r": 0.0,
                    "sampled_r": 0.0,
                    "sampled_r_raw": 0.0,
                    "sampled_r_clipped_to_r_max": False,
                    "delta_alpha_rad": 0.0,
                    "delta_alpha_deg": 0.0,
                }
            else:
                pert = perturb_query_track1(
                    query=q,
                    alpha=ALPHA,
                    epsilon=float(gate_result["epsilon_used"]),
                    r_max=float(dense_exec_r_max),
                    rng=rng,
                    forced_radius=TRACK1_FORCE_PERTURB_R,
                    force_clip_to_r_max=bool(TRACK1_FORCE_PERTURB_R_CLIP_TO_RMAX),
                )
            query_for_server = np.asarray(pert["perturbed_query_for_server"], dtype=np.float32)
            stage_client_generate_sec = float(time.perf_counter() - stage_client_generate_t0)
            comm_client_generate_mb = float(
                bytes_to_mb(
                    estimate_client_request_bytes(
                        routing_cluster_ids=routing_cluster_ids,
                        query_for_server=query_for_server,
                    )
                )
            )

            stage_server_query_t0 = time.perf_counter()
            candidate_payload = fixed_budget_global_knn_payload(
                query_for_server=query_for_server,
                docs=docs,
                doc_ids=doc_ids,
                doc_texts=doc_texts,
                fixed_k=int(fixed_k),
            )
            executed_track = "dense_rdp"
            stage_server_query_sec = float(time.perf_counter() - stage_server_query_t0)
            comm_server_query_mb = float(
                bytes_to_mb(estimate_server_response_payload_bytes(candidate_payload))
            )
            stage_client_recover_t0 = time.perf_counter()
            reranked = rerank_candidate_payload_exact(
                original_query=q,
                candidate_payload=candidate_payload,
                top_k=top_k,
            )
            pred_doc_ids = [str(x) for x in reranked["doc_ids"]]
            stage_client_recover_sec = float(time.perf_counter() - stage_client_recover_t0)

            retrieval_backend = str(candidate_payload.get("retrieval_backend", "cluster_payload"))
            ann_fallback_reason = candidate_payload.get("ann_fallback_reason", None)
            if final_track == "dense_rdp":
                if retrieval_backend != "faiss_hnsw_ann_angular":
                    raise RuntimeError(
                        "dense branch retrieval backend mismatch: "
                        "expected faiss_hnsw_ann_angular, "
                        f"got {retrieval_backend}"
                    )
                if ann_fallback_reason is not None:
                    raise RuntimeError(
                        "dense branch ANN fallback is disabled by policy, "
                        f"but got ann_fallback_reason={ann_fallback_reason}"
                    )
            # 约束：dense 候选必须来自被选中的 Track1 簇支持集。
            candidate_indices = np.asarray(candidate_payload.get("doc_indices", []), dtype=np.int32)
            if candidate_indices.size <= 0:
                raise RuntimeError("candidate payload is empty; expected non-empty candidates")

            executed_track_counter[executed_track] = executed_track_counter.get(executed_track, 0) + 1

            server_query_est_docs_touched = int(
                estimate_server_query_docs_touched(
                    retrieval_backend=str(retrieval_backend),
                    candidate_payload=candidate_payload,
                    routing_cluster_ids=[int(x) for x in routing_cluster_ids],
                    primary_cluster_id=int(cluster_id),
                    chunks=chunks,
                    docs=docs,
                    fixed_k=int(fixed_k),
                )
            )
            io_server_query_est_mb = float(
                bytes_to_mb(
                    estimate_server_query_io_bytes(
                        retrieval_backend=str(retrieval_backend),
                        candidate_payload=candidate_payload,
                        routing_cluster_ids=[int(x) for x in routing_cluster_ids],
                        primary_cluster_id=int(cluster_id),
                        chunks=chunks,
                        docs=docs,
                        fixed_k=int(fixed_k),
                    )
                )
            )
            stage_total_sec = float(
                stage_client_generate_sec + stage_server_query_sec + stage_client_recover_sec
            )
            stage_total_comm_mb = float(comm_client_generate_mb + comm_server_query_mb)
            stage_total_system_est_mb = float(
                comm_client_generate_mb + io_server_query_est_mb + comm_server_query_mb
            )

            if reference_metrics_available and gt_topk is not None:
                candidate_inclusion_recall = evaluate_candidate_inclusion_recall_at_k(
                    candidate_doc_ids=[str(x) for x in candidate_payload.get("doc_ids", [])],
                    gt_indices=gt_topk[i],
                    all_doc_ids=doc_ids,
                )
                relaxed_qrels_eval = evaluate_with_qrels(
                    pred_doc_ids=pred_doc_ids,
                    positive_doc_ids=relaxed_qrels_map.get(qid, set()),
                )
                strict_qrels_eval = evaluate_with_qrels(
                    pred_doc_ids=pred_doc_ids,
                    positive_doc_ids=strict_qrels_map.get(qid, set()),
                )
                exact_recall = evaluate_exact_recall_at_k(
                    pred_doc_ids=pred_doc_ids,
                    gt_indices=gt_topk[i],
                    all_doc_ids=doc_ids,
                )
            else:
                relaxed_qrels_eval = {
                    "hit_at_k": None,
                    "recall_on_qrels": None,
                    "num_hits": 0,
                    "num_positives": 0,
                    "hit_doc_ids": [],
                }
                strict_qrels_eval = {
                    "hit_at_k": None,
                    "recall_on_qrels": None,
                    "num_hits": 0,
                    "num_positives": 0,
                    "hit_doc_ids": [],
                }
                candidate_inclusion_recall = None
                exact_recall = None

            result = {
                "query_index": int(i),
                "query_id": qid,
                "cluster_id": cluster_id,
                "primary_cluster_id": int(cluster_id),
                "routing_rule_version": str(routing_info.get("routing_rule_version", "unknown")),
                "routing_cluster_selection_policy": str(ROUTING_CLUSTER_SELECTION_POLICY),
                "routing_selected_cluster_ids_before_rmax_filter": [
                    int(x) for x in routing_selected_cluster_ids
                ],
                "routed_cluster_ids": [int(x) for x in routing_cluster_ids],
                "cluster_size": cluster_size,
                "routing_selected_total_size_before_rmax_filter": int(routing_selected_total_size),
                "routed_cluster_total_size": int(routed_cluster_total_size),
                "routing_selected_overlap_support_total_size_before_rmax_filter": int(
                    routing_selected_overlap_support_total_size
                ),
                "routed_overlap_support_total_size": int(routed_overlap_support_total_size),
                "track1_selected_cluster_id": int(track1_selected_cluster_id),
                "track1_selected_cluster_size": int(track1_selected_cluster_size),
                "track1_selected_overlap_support_size": int(track1_selected_overlap_support_size),
                "r_max": float(gate_result["r_max"]),
                "r_max_cluster": float(gate_result.get("r_max_cluster", gate_result["r_max"])),
                "r_max_mode": str(gate_result.get("r_max_mode", "unknown")),
                "sigma_gate": float(gate_result["sigma"]),
                "r_rdp_bar_gate": float(gate_result["r_rdp_bar"]),
                "perturb_radius_mode": (
                    str(pert.get("radius_mode", "truncated_sigma_chi"))
                    if pert is not None
                    else "not_applied"
                ),
                "perturb_radius_requested": (
                    float(pert["requested_r"])
                    if pert is not None and pert.get("requested_r") is not None
                    else None
                ),
                "perturb_radius_used": float(pert["sampled_r"]) if pert is not None else None,
                "perturb_radius_raw": float(pert["sampled_r_raw"]) if pert is not None else None,
                "perturb_radius_clipped_to_r_max": (
                    bool(pert.get("sampled_r_clipped_to_r_max", False))
                    if pert is not None
                    else False
                ),
                "delta_alpha_rad": float(pert["delta_alpha_rad"]) if pert is not None else None,
                "delta_alpha_deg": float(pert["delta_alpha_deg"]) if pert is not None else None,
                "dense_margin_to_track1": float(gate_result["r_max"] - gate_result["r_rdp_bar"]),
                "routing_dense_exec_margin_to_track1": (
                    float(dense_exec_r_max - float(gate_result["r_rdp_bar"]))
                    if dense_exec_r_max is not None
                    else None
                ),
                "routing_selected_cluster_rmax": [
                    float(cluster_r_max[int(cid)]) for cid in routing_selected_cluster_ids
                ],
                "routing_selected_max_rmax": float(selected_max_rmax),
                "routing_selected_max_rmax_cluster_id": int(selected_max_rmax_cluster_id),
                "routing_dense_eligible_cluster_ids": [
                    int(x) for x in dense_eligible_cluster_ids
                ],
                "routing_dense_eligible_cluster_rmax": [
                    float(cluster_r_max[int(cid)]) for cid in dense_eligible_cluster_ids
                ],
                "routing_dense_exec_rmax": (
                    float(dense_exec_r_max) if dense_exec_r_max is not None else None
                ),
                "track_reason": str(final_track_reason),
                "routing_cluster_source": "nearest_center_angular_boundary_multicluster_dynamic",
                "routing_num_clusters": int(len(routing_cluster_ids)),
                "routing_selected_num_clusters_before_rmax_filter": int(len(routing_selected_cluster_ids)),
                "routing_dense_eligible_num_clusters": int(len(dense_eligible_cluster_ids)),
                "routing_boundary_gap_to_second": float(routing_info["routing_boundary_gap_to_second"]),
                "routing_boundary_gap_to_third": float(routing_info["routing_boundary_gap_to_third"]),
                "routing_boundary_gap_to_fourth": float(routing_info["routing_boundary_gap_to_fourth"]),
                "routing_boundary_threshold": float(routing_info["routing_boundary_threshold"]),
                "routing_boundary_threshold_to_second": float(
                    routing_info["routing_boundary_threshold_to_second"]
                ),
                "routing_boundary_threshold_to_third": (
                    float(routing_info["routing_boundary_threshold_to_third"])
                    if routing_info["routing_boundary_threshold_to_third"] is not None
                    else None
                ),
                "routing_boundary_threshold_to_fourth": (
                    float(routing_info["routing_boundary_threshold_to_fourth"])
                    if routing_info["routing_boundary_threshold_to_fourth"] is not None
                    else None
                ),
                "routing_boundary_multicluster_applied": bool(
                    routing_info["routing_boundary_multicluster_applied"]
                ),
                "paperfaithful_mainline_gate_mode": str(PAPERFAITHFUL_MAINLINE_GATE_MODE),
                "routing_primary_center_theta": float(routing_info["routing_primary_center_theta"]),
                "routing_second_center_theta": float(routing_info["routing_second_center_theta"]),
                "routing_ordered_cluster_ids": [
                    int(x) for x in routing_info.get("routing_ordered_cluster_ids", [])
                ],
                "routing_ordered_center_theta": [
                    float(x) for x in routing_info.get("routing_ordered_center_theta", [])
                ],
                "local_track_reason": str(local_track_reason),
                "final_track_reason": str(final_track_reason),
                "epsilon_input": float(gate_result.get("epsilon_input", EPSILON)),
                "epsilon_used": float(gate_result.get("epsilon_used", EPSILON)),
                "epsilon_min": finite_or_none(gate_result.get("epsilon_min", np.nan))
                if gate_result.get("epsilon_min", None) is not None
                else None,
                "epsilon_max": finite_or_none(gate_result.get("epsilon_max", np.nan))
                if gate_result.get("epsilon_max", None) is not None
                else None,
                "epsilon_interval_defined": bool(gate_result.get("epsilon_interval_defined", False)),
                "epsilon_interval_feasible": bool(gate_result.get("epsilon_interval_feasible", False)),
                "epsilon_clipped": bool(gate_result.get("epsilon_clipped", False)),
                "interval_infeasible_warning": bool(gate_result.get("interval_infeasible_warning", False)),
                "global_interval_policy": str(global_interval_policy),
                "global_policy_applied": bool(global_policy_applied),
                "global_policy_overrode_track": bool(global_policy_overrode_track),
                "global_policy_override": bool(global_policy_overrode_track),
                "global_policy_override_reason": str(global_policy_override_reason),
                "local_track": str(local_track),
                "final_track": str(final_track),
                "original_track": str(local_track),
                "executed_track": executed_track,
                "track1_return_embeddings": int(track1_return_embeddings),
                "returned_candidate_count": int(len(candidate_indices)),
                "server_query_est_docs_touched": int(server_query_est_docs_touched),
                "retrieval_backend": retrieval_backend,
                "ann_fallback_reason": ann_fallback_reason,
                "time_client_generate_query_sec": stage_client_generate_sec,
                "time_server_query_sec": stage_server_query_sec,
                "time_client_recover_docs_sec": stage_client_recover_sec,
                "time_three_stage_total_sec": stage_total_sec,
                "comm_client_generate_query_mb": comm_client_generate_mb,
                "comm_server_query_mb": float(comm_server_query_mb),
                "comm_two_stage_total_mb": stage_total_comm_mb,
                "io_server_query_est_mb": float(io_server_query_est_mb),
                "system_total_est_mb": float(stage_total_system_est_mb),
                "pred_doc_ids": pred_doc_ids,
                "strict_hit_at_k": (
                    float(strict_qrels_eval["hit_at_k"])
                    if strict_qrels_eval["hit_at_k"] is not None
                    else None
                ),
                "strict_recall_on_qrels": (
                    float(strict_qrels_eval["recall_on_qrels"])
                    if strict_qrels_eval["recall_on_qrels"] is not None
                    else None
                ),
                "strict_num_hits": int(strict_qrels_eval["num_hits"]),
                "strict_num_positives": int(strict_qrels_eval["num_positives"]),
                "strict_hit_doc_ids": [str(x) for x in strict_qrels_eval["hit_doc_ids"]],
                "relaxed_hit_at_k": (
                    float(relaxed_qrels_eval["hit_at_k"])
                    if relaxed_qrels_eval["hit_at_k"] is not None
                    else None
                ),
                "relaxed_recall_on_qrels": (
                    float(relaxed_qrels_eval["recall_on_qrels"])
                    if relaxed_qrels_eval["recall_on_qrels"] is not None
                    else None
                ),
                "relaxed_num_hits": int(relaxed_qrels_eval["num_hits"]),
                "relaxed_num_positives": int(relaxed_qrels_eval["num_positives"]),
                "relaxed_hit_doc_ids": [str(x) for x in relaxed_qrels_eval["hit_doc_ids"]],
                "candidate_inclusion_recall_at_k": (
                    float(candidate_inclusion_recall) if candidate_inclusion_recall is not None else None
                ),
                "exact_recall_at_k": (float(exact_recall) if exact_recall is not None else None),
            }
            writer.write(json.dumps(result, ensure_ascii=False) + "\n")

            if strict_qrels_eval["hit_at_k"] is not None:
                strict_hit_list.append(float(strict_qrels_eval["hit_at_k"]))
            if strict_qrels_eval["recall_on_qrels"] is not None:
                strict_recall_qrels_list.append(float(strict_qrels_eval["recall_on_qrels"]))
            if (
                strict_qrels_eval["hit_at_k"] is not None
                and strict_qrels_eval["recall_on_qrels"] is not None
                and int(strict_qrels_eval["num_positives"]) > 0
            ):
                strict_positive_query_count += 1
                strict_hit_positive_list.append(float(strict_qrels_eval["hit_at_k"]))
                strict_recall_positive_list.append(float(strict_qrels_eval["recall_on_qrels"]))
            if relaxed_qrels_eval["hit_at_k"] is not None:
                relaxed_hit_list.append(float(relaxed_qrels_eval["hit_at_k"]))
            if relaxed_qrels_eval["recall_on_qrels"] is not None:
                relaxed_recall_qrels_list.append(float(relaxed_qrels_eval["recall_on_qrels"]))
            if (
                relaxed_qrels_eval["hit_at_k"] is not None
                and relaxed_qrels_eval["recall_on_qrels"] is not None
                and int(relaxed_qrels_eval["num_positives"]) > 0
            ):
                relaxed_positive_query_count += 1
                relaxed_hit_positive_list.append(float(relaxed_qrels_eval["hit_at_k"]))
                relaxed_recall_positive_list.append(float(relaxed_qrels_eval["recall_on_qrels"]))
            if exact_recall is not None:
                exact_recall_value = float(exact_recall)
                exact_recall_list.append(exact_recall_value)
                if str(result.get("final_track")) == "dense_rdp":
                    exact_recall_track1_only_list.append(exact_recall_value)
            if candidate_inclusion_recall is not None:
                candidate_inclusion_recall_value = float(candidate_inclusion_recall)
                candidate_inclusion_recall_list.append(candidate_inclusion_recall_value)
                if str(result.get("final_track")) == "dense_rdp":
                    candidate_inclusion_recall_track1_only_list.append(
                        candidate_inclusion_recall_value
                    )
            r_max_list.append(float(result["r_max"]))
            dense_margin_list.append(float(result["r_max"] - result["r_rdp_bar_gate"]))
            routing_selected_max_rmax_list.append(float(result["routing_selected_max_rmax"]))
            routing_selected_max_margin_list.append(
                float(result["routing_selected_max_rmax"] - result["r_rdp_bar_gate"])
            )
            if result.get("routing_dense_exec_rmax") is not None:
                routing_dense_exec_rmax_list.append(float(result["routing_dense_exec_rmax"]))
            if result.get("routing_dense_exec_margin_to_track1") is not None:
                routing_dense_exec_margin_list.append(float(result["routing_dense_exec_margin_to_track1"]))
            epsilon_used_list.append(float(result["epsilon_used"]))
            sigma_gate_list.append(float(result["sigma_gate"]))
            perturb_radius_mode = str(result.get("perturb_radius_mode", "unknown"))
            perturb_radius_mode_counter[perturb_radius_mode] = (
                int(perturb_radius_mode_counter.get(perturb_radius_mode, 0)) + 1
            )
            perturb_radius_used = result.get("perturb_radius_used")
            perturb_radius_all_queries_list.append(
                float(perturb_radius_used) if perturb_radius_used is not None else 0.0
            )
            if perturb_radius_used is not None:
                perturb_radius_dense_queries_list.append(float(perturb_radius_used))
                requested_r = result.get("perturb_radius_requested")
                perturb_radius_requested_dense_queries_list.append(
                    float(requested_r) if requested_r is not None else float(perturb_radius_used)
                )
                delta_alpha_deg = result.get("delta_alpha_deg")
                if delta_alpha_deg is not None:
                    delta_alpha_deg_dense_queries_list.append(float(delta_alpha_deg))
            if bool(result.get("perturb_radius_clipped_to_r_max", False)):
                perturb_radius_clipped_count += 1
            time_client_generate_sec_list.append(float(stage_client_generate_sec))
            time_server_query_sec_list.append(float(stage_server_query_sec))
            time_client_recover_sec_list.append(float(stage_client_recover_sec))
            track1_candidate_budget_list.append(int(track1_return_embeddings))
            returned_candidate_count_list.append(int(len(candidate_indices)))
            server_query_est_docs_touched_list.append(int(server_query_est_docs_touched))
            comm_client_generate_mb_list.append(float(comm_client_generate_mb))
            comm_server_query_mb_list.append(float(comm_server_query_mb))
            io_server_query_est_mb_list.append(float(io_server_query_est_mb))
            system_total_est_mb_list.append(float(stage_total_system_est_mb))
            if str(executed_track) == "dense_rdp":
                time_client_generate_sec_track1_only_list.append(float(stage_client_generate_sec))
                time_server_query_sec_track1_only_list.append(float(stage_server_query_sec))
                time_client_recover_sec_track1_only_list.append(float(stage_client_recover_sec))
                server_query_est_docs_touched_track1_only_list.append(
                    int(server_query_est_docs_touched)
                )
                comm_client_generate_mb_track1_only_list.append(float(comm_client_generate_mb))
                comm_server_query_mb_track1_only_list.append(float(comm_server_query_mb))
                io_server_query_est_mb_track1_only_list.append(float(io_server_query_est_mb))
                system_total_est_mb_track1_only_list.append(float(stage_total_system_est_mb))
            if bool(result["epsilon_clipped"]):
                epsilon_clipped_count += 1
            if bool(result["epsilon_interval_defined"]) and (not bool(result["epsilon_interval_feasible"])):
                epsilon_interval_infeasible_count += 1

    pipeline_three_stage_wall_sec = float(time.perf_counter() - pipeline_three_stage_start)
    query_split_meta = {}
    if os.path.exists(WORKSET_QUERY_SPLIT_META_PATH):
        try:
            with open(WORKSET_QUERY_SPLIT_META_PATH, "r", encoding="utf-8") as f:
                query_split_meta = json.load(f)
        except Exception:
            query_split_meta = {}

    cluster_rmax_raw = [float(x) for x in cluster_info.get("rmax_surrogate", {}).get("cluster_r_max_raw", [])]
    cluster_rmax_shrunk = [float(x) for x in cluster_info.get("rmax_surrogate", {}).get("cluster_r_max_shrunk", [])]
    primary_cluster_track1_counts = {
        str(cid): int(primary_cluster_final_track_counts.get(str(cid), {}).get("dense_rdp", 0))
        for cid in range(len(chunks))
    }
    primary_cluster_track1_ratios = {
        str(cid): float(
            float(primary_cluster_track1_counts.get(str(cid), 0))
            / max(1, int(primary_cluster_query_counts.get(str(cid), 0)))
        )
        for cid in range(len(chunks))
    }
    overall_local_track1_ratio = float(
        float(original_track_counter.get("dense_rdp", 0)) / max(1, int(len(queries)))
    )
    overall_final_track1_ratio = float(
        float(sum(primary_cluster_track1_counts.values())) / max(1, int(len(queries)))
    )
    per_cluster_summary = []
    for cid in range(len(chunks)):
        cid_key = str(int(cid))
        per_cluster_summary.append(
            {
                "cluster_id": int(cid),
                "query_count": int(primary_cluster_query_counts.get(cid_key, 0)),
                "local_track_counts": dict(primary_cluster_local_track_counts.get(cid_key, {})),
                "final_track_counts": dict(primary_cluster_final_track_counts.get(cid_key, {})),
                "track1_count": int(primary_cluster_track1_counts.get(cid_key, 0)),
                "track1_ratio": float(primary_cluster_track1_ratios.get(cid_key, 0.0)),
                "r_max": float(cluster_r_max[int(cid)]),
                "r_max_raw": (
                    float(cluster_rmax_raw[int(cid)]) if int(cid) < len(cluster_rmax_raw) else float(cluster_r_max[int(cid)])
                ),
                "r_max_shrunk": (
                    float(cluster_rmax_shrunk[int(cid)])
                    if int(cid) < len(cluster_rmax_shrunk)
                    else float(cluster_r_max[int(cid)])
                ),
            }
        )

    summary = {
        "num_queries": int(len(queries)),
        "num_docs": int(len(docs)),
        "embedding_dim_n": int(docs.shape[1]),
        "top_k": int(top_k),
        "online_query_limit_requested": int(ONLINE_QUERY_LIMIT),
        "online_query_limit_effective": int(len(queries)),
        "fixed_k": int(fixed_k),
        "track1_return_embeddings": int(fixed_k),
        "rrdp_k_safe": int(RRDP_K_SAFE),
        "alpha": float(ALPHA),
        "epsilon": float(EPSILON),
        "rrdp_profile_json": RRDP_PROFILE_JSON,
        "rrdp_profile_query_pool_path": WORKSET_CALIBRATION_QUERIES_JSONL_PATH,
        "rrdp_profile_query_emb_cache_path": WORKSET_CALIBRATION_QUERIES_PATH,
        "rrdp_profile_query_ids_cache_path": WORKSET_CALIBRATION_QUERY_IDS_PATH,
        "rrdp_profile_query_source": str(rrdp_profile.get("profile_query_source", "unknown")),
        "rrdp_profile_query_emb_source": str(rrdp_profile.get("profile_query_emb_source", "")),
        "query_split_meta_path": WORKSET_QUERY_SPLIT_META_PATH,
        "query_pool_disjoint_protocol": (
            str(query_split_meta.get("protocol_version"))
            if isinstance(query_split_meta, dict) and query_split_meta.get("protocol_version") is not None
            else "calibration_eval_disjoint_v1"
        ),
        "routing_protocol": {
            "cluster_selection": (
                "nearest_center_angular_fixed_topc"
                if str(ROUTING_CLUSTER_SELECTION_POLICY) == "soft_topc_fixed"
                else "nearest_center_angular_primary_with_boundary_gap_ladder"
                if bool(ROUTING_ENABLE_BOUNDARY_MULTICLUSTER)
                and (
                    ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_3 is not None
                    or ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_4 is not None
                )
                else "nearest_center_angular_primary_with_optional_boundary_multicluster"
                if bool(ROUTING_ENABLE_BOUNDARY_MULTICLUSTER)
                else "nearest_center_angular_single_cluster"
            ),
            "rmax_lookup": (
                "max(cluster_r_max[top_c_nearest_centroids]) selects the single Track1 cluster"
                if str(ROUTING_CLUSTER_SELECTION_POLICY) == "soft_topc_fixed"
                else "cluster_r_max[primary_cluster] for gate; "
                "cluster_r_max[selected_cluster] for dense multicluster filter"
            ),
            "candidate_scope": "global_fixed_k_over_full_workset_on_perturbed_query_only",
            "cluster_selection_policy": str(ROUTING_CLUSTER_SELECTION_POLICY),
            "fixed_top_c": int(ROUTING_FIXED_TOP_C),
            "boundary_multicluster_enabled": bool(ROUTING_ENABLE_BOUNDARY_MULTICLUSTER),
            "boundary_top_m": int(ROUTING_BOUNDARY_TOP_M),
            "boundary_distance_gap_threshold": float(ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD),
            "boundary_distance_gap_threshold_2": float(ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_2),
            "boundary_distance_gap_threshold_3": (
                float(ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_3)
                if ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_3 is not None
                else None
            ),
            "boundary_distance_gap_threshold_4": (
                float(ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_4)
                if ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_4 is not None
                else None
            ),
            "multicluster_rmax_filter_enabled": bool(ROUTING_MULTICLUSTER_RMAX_FILTER_ENABLE),
            "track1_selected_cluster_policy": "argmax_cluster_r_max_over_routed_topc_for_local_gate_only",
            "track1_selected_cluster_return_embeddings": int(fixed_k),
        },
        "track1_dense_runtime": {
            "retrieval_backend": str(
                track1_dense_runtime_stats.get("retrieval_backend", "unknown")
            ),
            "num_docs": int(track1_dense_runtime_stats.get("num_docs", len(docs))),
            "num_clusters": int(track1_dense_runtime_stats.get("num_clusters", len(chunks))),
            "cluster_size_min": int(track1_dense_runtime_stats.get("cluster_size_min", 0)),
            "cluster_size_max": int(track1_dense_runtime_stats.get("cluster_size_max", 0)),
            "cluster_doc_cache_build_sec_once": float(
                track1_dense_runtime_stats.get("cluster_doc_cache_build_sec_once", 0.0)
            ),
            "cluster_hnsw_build_sec_once": float(
                track1_dense_runtime_stats.get("cluster_hnsw_build_sec_once", 0.0)
            ),
            "global_hnsw_build_sec_once": float(
                track1_dense_runtime_stats.get("global_hnsw_build_sec_once", 0.0)
            ),
            "cluster_hnsw_warm_fixed_k": int(
                track1_dense_runtime_stats.get("cluster_hnsw_warm_fixed_k", 0)
            ),
            "global_hnsw_warm_fixed_k": int(
                track1_dense_runtime_stats.get("global_hnsw_warm_fixed_k", 0)
            ),
            "cluster_doc_cache_warmed_cluster_count": int(
                track1_dense_runtime_stats.get("cluster_doc_cache_warmed_cluster_count", 0)
            ),
            "cluster_hnsw_warmed_cluster_count": int(
                track1_dense_runtime_stats.get("cluster_hnsw_warmed_cluster_count", 0)
            ),
            "cluster_hnsw_enabled": bool(
                track1_dense_runtime_stats.get("cluster_hnsw_enabled", False)
            ),
            "global_hnsw_enabled": bool(
                track1_dense_runtime_stats.get("global_hnsw_enabled", False)
            ),
            "cluster_hnsw_disabled_reason": track1_dense_runtime_stats.get(
                "cluster_hnsw_disabled_reason"
            ),
            "global_hnsw_disabled_reason": track1_dense_runtime_stats.get(
                "global_hnsw_disabled_reason"
            ),
            "total_setup_sec_once": float(
                track1_dense_runtime_stats.get("total_setup_sec_once", 0.0)
            ),
        },
        "online_path_prewarm": {
            "dense_three_stage": {
                "warm_query_source": dense_online_warmup_stats.get("warm_query_source"),
                "warm_cluster_ids": dense_online_warmup_stats.get("warm_cluster_ids", []),
                "warmup_rounds": int(dense_online_warmup_stats.get("warmup_rounds", 0)),
                "per_round_time_three_stage_total_sec_once": [
                    float(x)
                    for x in dense_online_warmup_stats.get(
                        "per_round_time_three_stage_total_sec_once", []
                    )
                ],
                "total_time_three_stage_total_sec_once": float(
                    dense_online_warmup_stats.get("total_time_three_stage_total_sec_once", 0.0)
                ),
                "candidate_count": int(dense_online_warmup_stats.get("candidate_count", 0)),
                "client_generate_query_sec_once": float(
                    dense_online_warmup_stats.get("client_generate_query_sec_once", 0.0)
                ),
                "server_query_sec_once": float(
                    dense_online_warmup_stats.get("server_query_sec_once", 0.0)
                ),
                "client_recover_docs_sec_once": float(
                    dense_online_warmup_stats.get("client_recover_docs_sec_once", 0.0)
                ),
                "time_three_stage_total_sec_once": float(
                    dense_online_warmup_stats.get("time_three_stage_total_sec_once", 0.0)
                ),
                "counted_in_online_latency": False,
            },
        },
        "single_cluster_only": bool(
            (str(ROUTING_CLUSTER_SELECTION_POLICY) != "soft_topc_fixed" or int(ROUTING_FIXED_TOP_C) <= 1)
            and (not bool(ROUTING_ENABLE_BOUNDARY_MULTICLUSTER))
            and int(routing_multicluster_applied_count) == 0
        ),
        "query_split_overlap_count": int(query_split_meta.get("split_overlap_count", -1))
        if isinstance(query_split_meta, dict)
        else -1,
        "query_split_num_calibration": int(query_split_meta.get("num_queries_calibration", 0))
        if isinstance(query_split_meta, dict)
        else 0,
        "query_split_num_evaluation": int(query_split_meta.get("num_queries_evaluation", 0))
        if isinstance(query_split_meta, dict)
        else 0,
        "rrdp_eta": float(RRDP_ETA),
        "rrdp_beta": float(RRDP_BETA),
        "paperfaithful_mainline_gate_mode": str(PAPERFAITHFUL_MAINLINE_GATE_MODE),
        "paperfaithful_mainline_skip_gate": bool(PAPERFAITHFUL_MAINLINE_SKIP_GATE),
        "global_interval_policy": str(global_interval_policy),
        "global_dense_only_guard_active": bool(global_force_dense_only_warning),
        "num_queries_global_policy_applied": int(global_policy_applied_count),
        "num_queries_global_policy_overrode_track": int(global_policy_override_count),
        "rrdp_profile_num_queries": int(rrdp_profile.get("num_profile_queries", 0)),
        "rrdp_r_1_minus_eta_global": finite_or_none(rrdp_profile.get("r_1_minus_eta_global", np.nan)),
        "rrdp_rho_min": finite_or_none(rrdp_profile.get("density_proxy", {}).get("rho_min", np.nan)),
        "epsilon_interval_mode": str(interval_mode),
        "epsilon_interval_enforce_active_profile": bool(enforce_interval_active_profile),
        "epsilon_min_strict_profile": (
            finite_or_none(epsilon_min_strict) if epsilon_min_strict is not None else None
        ),
        "epsilon_max_strict_profile": (
            finite_or_none(epsilon_max_strict) if epsilon_max_strict is not None else None
        ),
        "epsilon_interval_feasible_strict_profile": bool(profile_interval_feasible_strict),
        "epsilon_min_used_for_gate": (
            finite_or_none(epsilon_min_for_gate) if epsilon_min_for_gate is not None else None
        ),
        "epsilon_max_used_for_gate": (
            finite_or_none(epsilon_max_for_gate) if epsilon_max_for_gate is not None else None
        ),
        "epsilon_min_profile": finite_or_none(epsilon_min) if epsilon_min is not None else None,
        "epsilon_max_profile": finite_or_none(epsilon_max) if epsilon_max is not None else None,
        "epsilon_used_profile": float(epsilon_used_profile),
        "epsilon_interval_feasible_profile": bool(profile_interval_feasible),
        "avg_epsilon_used_gate": float(np.mean(epsilon_used_list)) if epsilon_used_list else float(EPSILON),
        "avg_sigma_gate": float(np.mean(sigma_gate_list)) if sigma_gate_list else 0.0,
        "num_epsilon_clipped_queries": int(epsilon_clipped_count),
        "num_queries_with_infeasible_epsilon_interval": int(epsilon_interval_infeasible_count),
        "track1_force_perturb_r": (
            finite_or_none(TRACK1_FORCE_PERTURB_R) if TRACK1_FORCE_PERTURB_R is not None else None
        ),
        "track1_force_perturb_r_clip_to_rmax": bool(TRACK1_FORCE_PERTURB_R_CLIP_TO_RMAX),
        "perturb_radius_mode_counter": perturb_radius_mode_counter,
        "num_queries_with_applied_perturb_radius": int(len(perturb_radius_dense_queries_list)),
        "num_perturb_radius_clipped_queries": int(perturb_radius_clipped_count),
        "perturb_radius_clipped_ratio_dense_queries": float(
            perturb_radius_clipped_count / max(1, len(perturb_radius_dense_queries_list))
        ),
        "avg_perturb_radius_all_queries": float(np.mean(perturb_radius_all_queries_list))
        if perturb_radius_all_queries_list
        else 0.0,
        "avg_perturb_radius_dense_queries": float(np.mean(perturb_radius_dense_queries_list))
        if perturb_radius_dense_queries_list
        else 0.0,
        "avg_perturb_radius_requested_dense_queries": float(
            np.mean(perturb_radius_requested_dense_queries_list)
        )
        if perturb_radius_requested_dense_queries_list
        else 0.0,
        "avg_delta_alpha_deg_dense_queries": float(np.mean(delta_alpha_deg_dense_queries_list))
        if delta_alpha_deg_dense_queries_list
        else 0.0,
        "min_perturb_radius_dense_queries": float(np.min(perturb_radius_dense_queries_list))
        if perturb_radius_dense_queries_list
        else 0.0,
        "max_perturb_radius_dense_queries": float(np.max(perturb_radius_dense_queries_list))
        if perturb_radius_dense_queries_list
        else 0.0,
        "original_track_counter": original_track_counter,
        "executed_track_counter": executed_track_counter,
        "paperfaithful_mainline_track1_only": bool(PAPERFAITHFUL_MAINLINE_TRACK1_ONLY),
        "overall_track1_ratio": float(overall_final_track1_ratio),
        "overall_local_track1_ratio": float(overall_local_track1_ratio),
        "overall_final_track1_ratio": float(overall_final_track1_ratio),
        "routing_num_clusters_counter": routing_num_clusters_counter,
        "routing_selected_num_clusters_counter": routing_selected_num_clusters_counter,
        "routing_dense_eligible_num_clusters_counter": routing_dense_eligible_num_clusters_counter,
        "routing_multicluster_applied_count": int(routing_multicluster_applied_count),
        "routing_multicluster_applied_ratio": float(routing_multicluster_applied_count / max(1, len(queries))),
        "routing_rmax_filtered_query_count": int(routing_rmax_filtered_query_count),
        "routing_rmax_filtered_query_ratio": float(
            routing_rmax_filtered_query_count / max(1, len(queries))
        ),
        "primary_cluster_query_counts": primary_cluster_query_counts,
        "primary_cluster_local_track_counts": primary_cluster_local_track_counts,
        "primary_cluster_final_track_counts": primary_cluster_final_track_counts,
        "primary_cluster_track1_counts": primary_cluster_track1_counts,
        "primary_cluster_track1_ratios": primary_cluster_track1_ratios,
        "cluster_rmax": [float(x) for x in cluster_r_max.tolist()],
        "cluster_rmax_raw": cluster_rmax_raw,
        "cluster_rmax_shrunk": cluster_rmax_shrunk,
        "per_cluster_summary": per_cluster_summary,
        "r_max_mode_counter": r_max_mode_counter,
        "rmax_surrogate": cluster_info.get("rmax_surrogate", {}),
        "cluster_info_contract": cluster_info_contract,
        "reference_metrics_available": bool(reference_metrics_available),
        "strict_qrels_path": WORKSET_QRELS_PATH,
        "num_queries_with_strict_positives": int(strict_positive_query_count),
        "avg_strict_hit_at_k_overall": float(np.mean(strict_hit_list)) if strict_hit_list else None,
        "avg_strict_recall_on_qrels_overall": (
            float(np.mean(strict_recall_qrels_list)) if strict_recall_qrels_list else None
        ),
        "avg_strict_hit_at_k_on_positive_queries": (
            float(np.mean(strict_hit_positive_list)) if strict_hit_positive_list else None
        ),
        "avg_strict_recall_on_qrels_on_positive_queries": (
            float(np.mean(strict_recall_positive_list)) if strict_recall_positive_list else None
        ),
        "relaxed_qrels_path": WORKSET_RELAXED_QRELS_PATH,
        "num_queries_with_relaxed_positives": int(relaxed_positive_query_count),
        "avg_relaxed_hit_at_k_overall": float(np.mean(relaxed_hit_list)) if relaxed_hit_list else None,
        "avg_relaxed_recall_on_qrels_overall": (
            float(np.mean(relaxed_recall_qrels_list)) if relaxed_recall_qrels_list else None
        ),
        "avg_relaxed_hit_at_k_on_positive_queries": (
            float(np.mean(relaxed_hit_positive_list)) if relaxed_hit_positive_list else None
        ),
        "avg_relaxed_recall_on_qrels_on_positive_queries": (
            float(np.mean(relaxed_recall_positive_list)) if relaxed_recall_positive_list else None
        ),
        "avg_exact_recall_at_k": float(np.mean(exact_recall_list)) if exact_recall_list else None,
        "avg_candidate_inclusion_recall_at_k": (
            float(np.mean(candidate_inclusion_recall_list))
            if candidate_inclusion_recall_list
            else None
        ),
        "num_track1_queries_for_candidate_inclusion_recall": int(
            len(candidate_inclusion_recall_track1_only_list)
        ),
        "avg_candidate_inclusion_recall_at_k_on_track1_queries": (
            float(np.mean(candidate_inclusion_recall_track1_only_list))
            if candidate_inclusion_recall_track1_only_list
            else None
        ),
        "num_track1_queries_for_exact_recall": int(len(exact_recall_track1_only_list)),
        "avg_exact_recall_at_k_on_track1_queries": (
            float(np.mean(exact_recall_track1_only_list))
            if exact_recall_track1_only_list
            else None
        ),
        "avg_track1_return_embeddings_budget": (
            float(np.mean(track1_candidate_budget_list)) if track1_candidate_budget_list else 0.0
        ),
        "min_track1_return_embeddings_budget": (
            int(np.min(track1_candidate_budget_list)) if track1_candidate_budget_list else 0
        ),
        "max_track1_return_embeddings_budget": (
            int(np.max(track1_candidate_budget_list)) if track1_candidate_budget_list else 0
        ),
        "avg_returned_candidate_count": (
            float(np.mean(returned_candidate_count_list)) if returned_candidate_count_list else 0.0
        ),
        "min_returned_candidate_count": (
            int(np.min(returned_candidate_count_list)) if returned_candidate_count_list else 0
        ),
        "max_returned_candidate_count": (
            int(np.max(returned_candidate_count_list)) if returned_candidate_count_list else 0
        ),
        "avg_r_max": float(np.mean(r_max_list)) if r_max_list else 0.0,
        "avg_dense_margin_to_rbar": float(np.mean(dense_margin_list)) if dense_margin_list else 0.0,
        "avg_routing_selected_max_rmax": (
            float(np.mean(routing_selected_max_rmax_list)) if routing_selected_max_rmax_list else 0.0
        ),
        "avg_routing_selected_max_margin_to_rbar": (
            float(np.mean(routing_selected_max_margin_list)) if routing_selected_max_margin_list else 0.0
        ),
        "avg_routing_dense_exec_rmax": (
            float(np.mean(routing_dense_exec_rmax_list)) if routing_dense_exec_rmax_list else 0.0
        ),
        "avg_routing_dense_exec_margin_to_rbar": (
            float(np.mean(routing_dense_exec_margin_list)) if routing_dense_exec_margin_list else 0.0
        ),
        "min_r_max": float(np.min(r_max_list)) if r_max_list else 0.0,
        "max_r_max": float(np.max(r_max_list)) if r_max_list else 0.0,
        "cost_stage_definition": {
            "client_generate_query": "local gate decision + perturbed dense query construction; excludes server retrieval, response transfer, and rerank",
            "server_query": "server-side dense retrieval + response construction; excludes client-side rerank and evaluation",
            "client_recover_docs": "client-side exact rerank with the original query only",
            "comm_client_generate_query_mb": "client->server per-query request bytes: perturbed dense query vector only",
            "comm_server_query_mb": "server->client per-query response bytes: dense candidate payload bytes",
            "server_query_est_docs_touched": "diagnostic only: estimated number of docs whose embeddings are touched during server retrieval; not counted as communication",
            "io_server_query_est_mb": "diagnostic only: estimated server-internal retrieval bytes touched during retrieval itself; not counted as communication",
            "comm_two_stage_total_mb": "true per-query cross-boundary communication total = client->server request bytes + server->client response bytes",
            "system_total_est_mb": "diagnostic system-byte estimate = request bytes + server-internal retrieval I/O estimate + response payload bytes",
            "track1_dense_runtime.cluster_doc_cache_build_sec_once": "Dense path only: once-only cluster_docs cache materialization before pipeline_three_stage_start, excluded from per-query online latency",
            "track1_dense_runtime.cluster_hnsw_build_sec_once": "Dense path only: once-only cluster-local HNSW build before pipeline_three_stage_start, excluded from per-query online latency",
            "track1_dense_runtime.global_hnsw_build_sec_once": "Dense path only: once-only global HNSW build before pipeline_three_stage_start, excluded from per-query online latency",
            "online_path_prewarm.dense_three_stage.time_three_stage_total_sec_once": "Dense path only: final round of multiple untimed dummy queries through retrieval + rerank before pipeline_three_stage_start, excluded from online latency",
        },
        "time_client_generate_query_sec_sum": float(np.sum(time_client_generate_sec_list))
        if time_client_generate_sec_list
        else 0.0,
        "time_server_query_sec_sum": float(np.sum(time_server_query_sec_list))
        if time_server_query_sec_list
        else 0.0,
        "time_client_recover_docs_sec_sum": float(np.sum(time_client_recover_sec_list))
        if time_client_recover_sec_list
        else 0.0,
        "time_client_generate_query_sec_sum_on_track1_queries": float(
            np.sum(time_client_generate_sec_track1_only_list)
        )
        if time_client_generate_sec_track1_only_list
        else 0.0,
        "time_server_query_sec_sum_on_track1_queries": float(
            np.sum(time_server_query_sec_track1_only_list)
        )
        if time_server_query_sec_track1_only_list
        else 0.0,
        "time_client_recover_docs_sec_sum_on_track1_queries": float(
            np.sum(time_client_recover_sec_track1_only_list)
        )
        if time_client_recover_sec_track1_only_list
        else 0.0,
        "time_three_stage_total_sec_sum": float(
            (np.sum(time_client_generate_sec_list) if time_client_generate_sec_list else 0.0)
            + (np.sum(time_server_query_sec_list) if time_server_query_sec_list else 0.0)
            + (np.sum(time_client_recover_sec_list) if time_client_recover_sec_list else 0.0)
        ),
        "time_three_stage_total_sec_sum_on_track1_queries": float(
            (np.sum(time_client_generate_sec_track1_only_list) if time_client_generate_sec_track1_only_list else 0.0)
            + (np.sum(time_server_query_sec_track1_only_list) if time_server_query_sec_track1_only_list else 0.0)
            + (np.sum(time_client_recover_sec_track1_only_list) if time_client_recover_sec_track1_only_list else 0.0)
        ),
        "time_client_generate_query_sec_avg": float(np.mean(time_client_generate_sec_list))
        if time_client_generate_sec_list
        else 0.0,
        "time_server_query_sec_avg": float(np.mean(time_server_query_sec_list))
        if time_server_query_sec_list
        else 0.0,
        "time_client_recover_docs_sec_avg": float(np.mean(time_client_recover_sec_list))
        if time_client_recover_sec_list
        else 0.0,
        "time_client_generate_query_sec_avg_on_track1_queries": float(
            np.mean(time_client_generate_sec_track1_only_list)
        )
        if time_client_generate_sec_track1_only_list
        else 0.0,
        "time_server_query_sec_avg_on_track1_queries": float(
            np.mean(time_server_query_sec_track1_only_list)
        )
        if time_server_query_sec_track1_only_list
        else 0.0,
        "time_client_recover_docs_sec_avg_on_track1_queries": float(
            np.mean(time_client_recover_sec_track1_only_list)
        )
        if time_client_recover_sec_track1_only_list
        else 0.0,
        "time_three_stage_wall_clock_sec": float(pipeline_three_stage_wall_sec),
        "comm_client_generate_query_mb_sum": float(np.sum(comm_client_generate_mb_list))
        if comm_client_generate_mb_list
        else 0.0,
        "comm_server_query_mb_sum": float(np.sum(comm_server_query_mb_list))
        if comm_server_query_mb_list
        else 0.0,
        "comm_client_generate_query_mb_sum_on_track1_queries": float(
            np.sum(comm_client_generate_mb_track1_only_list)
        )
        if comm_client_generate_mb_track1_only_list
        else 0.0,
        "comm_server_query_mb_sum_on_track1_queries": float(
            np.sum(comm_server_query_mb_track1_only_list)
        )
        if comm_server_query_mb_track1_only_list
        else 0.0,
        "server_query_est_docs_touched_sum": int(np.sum(server_query_est_docs_touched_list))
        if server_query_est_docs_touched_list
        else 0,
        "server_query_est_docs_touched_sum_on_track1_queries": int(
            np.sum(server_query_est_docs_touched_track1_only_list)
        )
        if server_query_est_docs_touched_track1_only_list
        else 0,
        "io_server_query_est_mb_sum": float(np.sum(io_server_query_est_mb_list))
        if io_server_query_est_mb_list
        else 0.0,
        "io_server_query_est_mb_sum_on_track1_queries": float(
            np.sum(io_server_query_est_mb_track1_only_list)
        )
        if io_server_query_est_mb_track1_only_list
        else 0.0,
        "comm_two_stage_total_mb_sum": float(
            (np.sum(comm_client_generate_mb_list) if comm_client_generate_mb_list else 0.0)
            + (np.sum(comm_server_query_mb_list) if comm_server_query_mb_list else 0.0)
        ),
        "comm_two_stage_total_mb_sum_on_track1_queries": float(
            (np.sum(comm_client_generate_mb_track1_only_list) if comm_client_generate_mb_track1_only_list else 0.0)
            + (np.sum(comm_server_query_mb_track1_only_list) if comm_server_query_mb_track1_only_list else 0.0)
        ),
        "system_total_est_mb_sum": float(np.sum(system_total_est_mb_list))
        if system_total_est_mb_list
        else 0.0,
        "system_total_est_mb_sum_on_track1_queries": float(
            np.sum(system_total_est_mb_track1_only_list)
        )
        if system_total_est_mb_track1_only_list
        else 0.0,
        "comm_client_generate_query_mb_avg": float(np.mean(comm_client_generate_mb_list))
        if comm_client_generate_mb_list
        else 0.0,
        "comm_server_query_mb_avg": float(np.mean(comm_server_query_mb_list))
        if comm_server_query_mb_list
        else 0.0,
        "comm_client_generate_query_mb_avg_on_track1_queries": float(
            np.mean(comm_client_generate_mb_track1_only_list)
        )
        if comm_client_generate_mb_track1_only_list
        else 0.0,
        "comm_server_query_mb_avg_on_track1_queries": float(
            np.mean(comm_server_query_mb_track1_only_list)
        )
        if comm_server_query_mb_track1_only_list
        else 0.0,
        "server_query_est_docs_touched_avg": float(np.mean(server_query_est_docs_touched_list))
        if server_query_est_docs_touched_list
        else 0.0,
        "server_query_est_docs_touched_avg_on_track1_queries": float(
            np.mean(server_query_est_docs_touched_track1_only_list)
        )
        if server_query_est_docs_touched_track1_only_list
        else 0.0,
        "io_server_query_est_mb_avg": float(np.mean(io_server_query_est_mb_list))
        if io_server_query_est_mb_list
        else 0.0,
        "io_server_query_est_mb_avg_on_track1_queries": float(
            np.mean(io_server_query_est_mb_track1_only_list)
        )
        if io_server_query_est_mb_track1_only_list
        else 0.0,
        "comm_two_stage_total_mb_avg": float(
            (np.mean(comm_client_generate_mb_list) if comm_client_generate_mb_list else 0.0)
            + (np.mean(comm_server_query_mb_list) if comm_server_query_mb_list else 0.0)
        ),
        "comm_two_stage_total_mb_avg_on_track1_queries": float(
            (np.mean(comm_client_generate_mb_track1_only_list) if comm_client_generate_mb_track1_only_list else 0.0)
            + (np.mean(comm_server_query_mb_track1_only_list) if comm_server_query_mb_track1_only_list else 0.0)
        ),
        "system_total_est_mb_avg": float(np.mean(system_total_est_mb_list))
        if system_total_est_mb_list
        else 0.0,
        "system_total_est_mb_avg_on_track1_queries": float(
            np.mean(system_total_est_mb_track1_only_list)
        )
        if system_total_est_mb_track1_only_list
        else 0.0,
    }
    save_json(ONLINE_SUMMARY_JSON, summary)
    save_paperfaithful_mainline_audit(
        cluster_info=cluster_info,
        online_summary=summary,
    )

    print("=" * 80)
    print("online stage finished")
    print(f"WORKSET_CLUSTER_INFO : {WORKSET_CLUSTER_INFO_PATH}")
    print(f"ONLINE_RESULTS_JSONL : {ONLINE_RESULTS_JSONL}")
    print(f"ONLINE_SUMMARY_JSON  : {ONLINE_SUMMARY_JSON}")
    print(f"AUDIT_JSON           : {PAPERFAITHFUL_MAINLINE_AUDIT_JSON}")
    print(f"AUDIT_CSV            : {PAPERFAITHFUL_MAINLINE_AUDIT_CSV}")
    print(f"AUDIT_PKL            : {PAPERFAITHFUL_MAINLINE_AUDIT_PKL}")
    print(f"RRDP_CALIB_QUERIES   : {WORKSET_CALIBRATION_QUERIES_JSONL_PATH}")
    if isinstance(query_split_meta, dict) and query_split_meta:
        print(
            "QUERY_SPLIT_META     : "
            f"overlap={query_split_meta.get('split_overlap_count')}, "
            f"calib={query_split_meta.get('num_queries_calibration')}, "
            f"eval={query_split_meta.get('num_queries_evaluation')}"
        )
    if int(summary["num_queries_with_strict_positives"]) == 0:
        print(
            "[warn] strict qrels has zero positives for all selected queries; "
            "strict metrics should not be used as primary evidence."
        )
    use_track1_only_summary = str(MAINLINE_RECALL_SCOPE) == "track1_only"
    time_avg_key = (
        "time_three_stage_total_sec_avg_on_track1_queries"
        if use_track1_only_summary
        else "time_three_stage_total_sec_avg"
    )
    comm_avg_key = (
        "comm_two_stage_total_mb_avg_on_track1_queries"
        if use_track1_only_summary
        else "comm_two_stage_total_mb_avg"
    )
    system_avg_key = (
        "system_total_est_mb_avg_on_track1_queries"
        if use_track1_only_summary
        else "system_total_est_mb_avg"
    )
    server_io_avg_key = (
        "io_server_query_est_mb_avg_on_track1_queries"
        if use_track1_only_summary
        else "io_server_query_est_mb_avg"
    )
    print(
        "metrics: "
        f"exact={summary['avg_exact_recall_at_k']:.6f}, "
        f"cand_incl={summary['avg_candidate_inclusion_recall_at_k']}, "
        f"exact_track1={summary['avg_exact_recall_at_k_on_track1_queries']}, "
        f"cand_incl_track1={summary['avg_candidate_inclusion_recall_at_k_on_track1_queries']}, "
        f"strict_pos={summary['avg_strict_recall_on_qrels_on_positive_queries']}, "
        f"relaxed_pos={summary['avg_relaxed_recall_on_qrels_on_positive_queries']}, "
        f"dense_ratio={summary['overall_track1_ratio']:.3f}, "
        f"route_multi_ratio={summary['routing_multicluster_applied_ratio']:.3f}, "
        f"time_avg={summary.get(time_avg_key, 0.0):.6f}s, "
        f"comm_avg={summary.get(comm_avg_key, 0.0):.6f}MB, "
        f"server_io_avg={summary.get(server_io_avg_key, 0.0):.6f}MB, "
        f"system_avg={summary.get(system_avg_key, 0.0):.6f}MB"
    )
    print(f"primary_cluster_query_counts : {summary['primary_cluster_query_counts']}")
    print(f"primary_cluster_track1_ratios: {summary['primary_cluster_track1_ratios']}")
    print(f"routing_selected_clusters    : {summary['routing_selected_num_clusters_counter']}")
    print(f"routing_final_clusters       : {summary['routing_num_clusters_counter']}")
    print(f"routing_dense_eligible       : {summary['routing_dense_eligible_num_clusters_counter']}")
    print(
        "routing_rmax_filtered       : "
        f"{summary['routing_rmax_filtered_query_count']} "
        f"({summary['routing_rmax_filtered_query_ratio']:.3f})"
    )
    print(f"cluster_rmax_shrunk          : {summary['cluster_rmax_shrunk']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
