"""
cluster_offline_method4_balanced_spherical 模块。

paper-faithful 主结果线（docs-only）：
1) 纯 docs-only Step1: 在归一化文档 embedding 上做 spherical k-means 原型训练；
2) 一次固定容量修正: 以 Step1 分配为起点，修正到 4x500；
3) docs-only seed 扫描 + J_doc 选优（不使用 query/qrels/containment）；
4) 轻量 docs-only 边缘交换 refinement（仅当 J_doc 提升才接受）；
5) 一次最终几何冻结: 公开中心严格使用 full-500 centroid。
"""

# Allow running this file directly: `python src/offline/cluster_offline_method4_balanced_spherical.py`
if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import os
import csv
import json
import heapq
import hashlib
from collections import Counter

import numpy as np

from shared.config import (
    RESULTS_DIR,
    PIPELINE_OUTPUT_SUFFIX,
    WORKSET_CLUSTER_INFO_METHOD4_PATH,
    WORKSET_CLUSTER_SUMMARY_METHOD4_JSON,
    WORKSET_CORPUS_JSONL_PATH,
    NUM_CLUSTERS,
    TARGET_CLUSTER_SIZE,
    FIXED_K,
    RANDOM_STATE,
    METHOD4_RNG_SEED_OFFSET,
    METHOD4_SEED_SWEEP_ENABLED,
    METHOD4_SEED_SWEEP_START,
    METHOD4_SEED_SWEEP_END,
    METHOD4_DOC_REFINE_ENABLED,
    METHOD4_DOC_REFINE_MAX_SWAPS,
    RMAX_CLUSTER_QUANTILE_GAMMA,
    RMAX_ANCHOR_POLICY,
    RMAX_TARGET_ANCHORS_PER_CLUSTER,
    RMAX_MIN_ANCHORS_PER_CLUSTER,
    RMAX_ANCHOR_MEMBERSHIP_MIN_RATIO,
    RMAX_ANCHOR_ENFORCE_MIN_PER_CLUSTER,
    RMAX_ANCHOR_NUM_DISTANCE_STRATA,
)
from offline.cluster_method_utils import (
    load_workset_inputs,
    init_kmeans_pp,
    l2_distance_matrix,
    assign_balanced_clusters,
    assigned_chunks_from_labels,
    normalize_vec,
    save_cluster_artifacts,
)

OUT_PKL = WORKSET_CLUSTER_INFO_METHOD4_PATH
OUT_JSON = WORKSET_CLUSTER_SUMMARY_METHOD4_JSON
CLUSTER_ARTIFACT_SCHEMA_VERSION = "paperfaithful_method4_schema_v2_canonicalized_seedbound"

# Step1 训练只做单次，不做多重启。
STEP1_MAX_ITER = int(max(1, int(os.environ.get("METHOD4_STEP1_MAX_ITER", "80"))))
# 论文主结果线：公开中心必须是 full-500 centroid。
PUBLIC_CENTER_CORE_SIZE = int(TARGET_CLUSTER_SIZE)
RNG_SEED_OFFSET = int(
    os.environ.get("METHOD4_RNG_SEED_OFFSET", str(METHOD4_RNG_SEED_OFFSET))
)
SEED_SCAN_OUT_JSON = os.path.join(
    RESULTS_DIR, f"docs_only_seed_scoring{PIPELINE_OUTPUT_SUFFIX}.json"
)
SEED_SCAN_OUT_CSV = os.path.join(
    RESULTS_DIR, f"docs_only_seed_scoring{PIPELINE_OUTPUT_SUFFIX}.csv"
)


def _build_spherical_centers_from_labels(
    docs: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
    fallback_order: np.ndarray,
) -> np.ndarray:
    """
    用标签重算球面中心。

    当某簇临时为空时，用 fallback_order 里还没被取过的样本向量做重置，
    保证 Step1 训练不会因空簇中断。
    """
    centers = []
    used_fallback = set()
    for cid in range(int(num_clusters)):
        member_idx = np.where(labels == int(cid))[0].astype(np.int32)
        if len(member_idx) == 0:
            pick = None
            for idx in fallback_order.tolist():
                if int(idx) not in used_fallback:
                    pick = int(idx)
                    used_fallback.add(pick)
                    break
            if pick is None:
                pick = int(fallback_order[0])
            centers.append(normalize_vec(docs[pick]))
        else:
            mean_center = np.mean(docs[member_idx], axis=0).astype(np.float32)
            centers.append(normalize_vec(mean_center))
    return np.asarray(centers, dtype=np.float32)


def train_step1_spherical_prototypes(
    docs: np.ndarray,
    num_clusters: int,
    rng_seed: int,
    max_iter: int,
) -> dict:
    """
    纯 Step1 训练：标准 nearest-prototype spherical k-means（无容量约束）。
    """
    rng = np.random.default_rng(int(rng_seed))
    prototypes = init_kmeans_pp(docs=docs, num_clusters=int(num_clusters), rng=rng)
    prototypes = np.asarray([normalize_vec(c) for c in prototypes], dtype=np.float32)

    prev_labels = None
    used_iter = 0
    for it in range(int(max_iter)):
        # Step1 训练阶段按 normalized Euclidean（与 cosine 排序等价）做最近原型分配。
        costs = l2_distance_matrix(docs, prototypes)
        labels = np.argmin(costs, axis=1).astype(np.int32)
        nearest_cost = costs[np.arange(len(docs), dtype=np.int32), labels]
        fallback_order = np.argsort(-nearest_cost).astype(np.int32)
        prototypes = _build_spherical_centers_from_labels(
            docs=docs,
            labels=labels,
            num_clusters=int(num_clusters),
            fallback_order=fallback_order,
        )
        used_iter = it + 1

        if prev_labels is not None and np.array_equal(labels, prev_labels):
            break
        prev_labels = labels

    final_costs = l2_distance_matrix(docs, prototypes)
    final_labels = np.argmin(final_costs, axis=1).astype(np.int32)
    objective_cost_sum = float(
        np.sum(final_costs[np.arange(len(docs), dtype=np.int32), final_labels], dtype=np.float64)
    )
    objective_cost_mean = float(objective_cost_sum / max(1, len(docs)))
    return {
        "prototypes": prototypes.astype(np.float32),
        "labels": final_labels.astype(np.int32),
        "iters": int(used_iter),
        "objective_cost_sum": float(objective_cost_sum),
        "objective_cost_mean": float(objective_cost_mean),
    }


def repair_capacity_by_min_delta_moves(
    cost: np.ndarray,
    init_labels: np.ndarray,
    capacity: int,
) -> tuple[np.ndarray, str, str | None]:
    """
    一次固定容量修正（轻量）：
    - 先用 Step1 最近原型的自然分配；
    - 只从超额簇向缺额簇搬运文档；
    - 每次选择增量代价 delta 最小的搬运候选。

    复杂度（建堆 + 弹堆）约 O(N*K*log(N*K))，适合当前 2000x4 规模。
    """
    cost = np.asarray(cost, dtype=np.float64)
    labels = np.asarray(init_labels, dtype=np.int32).copy()
    if cost.ndim != 2:
        raise RuntimeError("repair_capacity_by_min_delta_moves expects a 2D cost matrix.")
    num_docs, num_clusters = cost.shape
    if labels.shape != (num_docs,):
        raise RuntimeError(
            f"init_labels shape mismatch, expected ({num_docs},), got {labels.shape}"
        )
    if np.any(labels < 0) or np.any(labels >= int(num_clusters)):
        raise RuntimeError("init_labels contains invalid cluster ids.")
    if int(num_docs) != int(num_clusters) * int(capacity):
        raise RuntimeError(
            "balanced assignment requires num_docs == num_clusters * capacity, "
            f"got {num_docs} vs {num_clusters} * {capacity}"
        )

    target = np.full(int(num_clusters), int(capacity), dtype=np.int32)
    counts = np.bincount(labels, minlength=int(num_clusters)).astype(np.int32)
    if np.array_equal(counts, target):
        return labels, "step1_natural_assignment_already_balanced", None

    # 全量候选 move 的最小堆：key = 增量代价 delta = cost(i, dst) - cost(i, src_step1)。
    heap = []
    for doc_idx in range(int(num_docs)):
        src = int(labels[doc_idx])
        src_cost = float(cost[doc_idx, src])
        for dst in range(int(num_clusters)):
            if dst == src:
                continue
            delta = float(cost[doc_idx, dst] - src_cost)
            heap.append((delta, int(doc_idx), int(dst)))
    heapq.heapify(heap)

    moved_once = np.zeros(int(num_docs), dtype=bool)
    while np.any(counts > target):
        if not heap:
            break
        _delta, doc_idx, dst = heapq.heappop(heap)
        if moved_once[int(doc_idx)]:
            continue

        src = int(labels[int(doc_idx)])
        if src == int(dst):
            continue
        if counts[src] <= target[src]:
            continue
        if counts[int(dst)] >= target[int(dst)]:
            continue

        labels[int(doc_idx)] = int(dst)
        counts[src] -= 1
        counts[int(dst)] += 1
        moved_once[int(doc_idx)] = True

    if np.array_equal(counts, target):
        return labels, "step1_min_delta_boundary_repair_heap", None

    # 极少发生：兜底回到稳定容量贪心，保证一定产出固定 500 配置。
    chunks = assign_balanced_clusters(dists=cost.astype(np.float32), capacity=int(capacity))
    fallback_labels = np.full(int(num_docs), -1, dtype=np.int32)
    for cid, chunk in enumerate(chunks):
        fallback_labels[np.asarray(chunk, dtype=np.int32)] = int(cid)
    if np.any(fallback_labels < 0):
        raise RuntimeError("fallback balanced assignment produced unassigned documents.")
    fallback_counts = np.bincount(fallback_labels, minlength=int(num_clusters)).astype(np.int32)
    if not np.array_equal(fallback_counts, target):
        raise RuntimeError(
            "fallback balanced assignment failed to satisfy fixed capacities: "
            f"counts={fallback_counts.tolist()}, target={target.tolist()}"
        )
    return fallback_labels, "stable_margin_greedy_fallback", "heap_boundary_repair_incomplete"


def global_balanced_assignment_once(
    docs: np.ndarray,
    prototypes: np.ndarray,
    capacity: int,
    step1_labels: np.ndarray | None = None,
) -> dict:
    """
    一次固定容量修正：
    - 先按 Step1 最近原型做自然分配；
    - 再用最小增量代价搬运，把超额簇修正到固定容量。
    """
    base_cost = l2_distance_matrix(docs, prototypes).astype(np.float64)
    num_docs, num_clusters = base_cost.shape
    if int(num_docs) != int(num_clusters) * int(capacity):
        raise RuntimeError(
            "global balanced assignment requires num_docs == num_clusters * capacity, "
            f"got {num_docs} vs {num_clusters} * {capacity}"
        )

    if step1_labels is None:
        natural_labels = np.argmin(base_cost, axis=1).astype(np.int32)
    else:
        natural_labels = np.asarray(step1_labels, dtype=np.int32)
        if natural_labels.shape != (num_docs,):
            raise RuntimeError(
                f"step1_labels shape mismatch, expected ({num_docs},), got {natural_labels.shape}"
            )
    labels, solver_used, fallback_reason = repair_capacity_by_min_delta_moves(
        cost=base_cost,
        init_labels=natural_labels,
        capacity=int(capacity),
    )
    chunks = assigned_chunks_from_labels(labels=labels, num_clusters=int(num_clusters))

    objective_cost_sum = float(
        np.sum(base_cost[np.arange(num_docs, dtype=np.int32), labels], dtype=np.float64)
    )
    objective_cost_mean = float(objective_cost_sum / max(1, num_docs))
    natural_counts = np.bincount(natural_labels, minlength=int(num_clusters)).astype(np.int32)
    return {
        "labels": labels.astype(np.int32),
        "chunks": [np.asarray(chunk, dtype=np.int32) for chunk in chunks],
        "cost_matrix": base_cost,
        "natural_cluster_sizes": [int(x) for x in natural_counts.tolist()],
        "objective_cost_sum": float(objective_cost_sum),
        "objective_cost_mean": float(objective_cost_mean),
        "solver_used": str(solver_used),
        "fallback_reason": fallback_reason,
    }


def freeze_final_geometry_with_core_proxy_center(
    docs: np.ndarray,
    chunks: list[np.ndarray],
) -> dict:
    """
    一次最终几何冻结：
    - 父簇成员保持 500 不变；
    - 公开中心（online 路由）使用 full-500 centroid；
    - 额外计算 core-300 centroid 作为诊断字段（不参与在线主结果路由）。
    """
    proxy_centers = []
    full_centers = []
    core300_centers = []
    core_sizes = []
    core_share = []
    full_vs_core_cos = []
    for cid, chunk in enumerate(chunks):
        member_idx = np.asarray(chunk, dtype=np.int32)
        if len(member_idx) == 0:
            raise RuntimeError(f"cluster {cid} is empty after balanced assignment.")
        cluster_docs = docs[member_idx]
        full_center = normalize_vec(np.mean(cluster_docs, axis=0).astype(np.float32))

        # core-300 仅用于诊断。
        core_k = int(max(1, min(300, int(len(member_idx)))))
        sims = (cluster_docs @ full_center).astype(np.float32)
        if core_k >= int(len(member_idx)):
            core_local = np.arange(len(member_idx), dtype=np.int32)
        else:
            keep = int(len(member_idx) - core_k)
            core_local = np.argpartition(sims, kth=keep)[keep:].astype(np.int32)
        core_docs = cluster_docs[core_local]
        core_center = normalize_vec(np.mean(core_docs, axis=0).astype(np.float32))

        full_centers.append(full_center)
        # 主结果公开中心严格使用 full centroid。
        proxy_centers.append(full_center)
        core300_centers.append(core_center)
        core_sizes.append(int(core_k))
        core_share.append(float(core_k / max(1, len(member_idx))))
        full_vs_core_cos.append(float(np.clip(float(full_center @ core_center), -1.0, 1.0)))

    return {
        "proxy_centers": np.asarray(proxy_centers, dtype=np.float32),
        "full_centers": np.asarray(full_centers, dtype=np.float32),
        "core300_centers": np.asarray(core300_centers, dtype=np.float32),
        "core_sizes": [int(x) for x in core_sizes],
        "core_share": [float(x) for x in core_share],
        "full_vs_core_center_cosine": [float(x) for x in full_vs_core_cos],
    }


def freeze_final_geometry(
    docs: np.ndarray,
    chunks: list[np.ndarray],
) -> np.ndarray:
    """
    兼容旧调用方：返回公开 proxy centers。
    """
    frozen = freeze_final_geometry_with_core_proxy_center(docs=docs, chunks=chunks)
    return np.asarray(frozen["proxy_centers"], dtype=np.float32)


def _compute_proxy_center_for_chunk(docs: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    member_idx = np.asarray(chunk, dtype=np.int32)
    if len(member_idx) == 0:
        raise RuntimeError("cannot compute proxy center for an empty cluster")
    cluster_docs = np.asarray(docs[member_idx], dtype=np.float32)
    return normalize_vec(np.mean(cluster_docs, axis=0).astype(np.float32))


def _compute_proxy_centers_only(
    docs: np.ndarray,
    chunks: list[np.ndarray],
) -> np.ndarray:
    return np.asarray(
        [_compute_proxy_center_for_chunk(docs=docs, chunk=np.asarray(chunk, dtype=np.int32)) for chunk in chunks],
        dtype=np.float32,
    )


def summarize_step1_alignment(
    step1_labels: np.ndarray,
    final_labels: np.ndarray,
    num_clusters: int,
) -> dict:
    """
    统计容量修正前后的一致性：
    - Step1 各簇大小；
    - 最终各簇大小（应固定为 500）；
    - 从 Step1 被迁移的文档比例。
    """
    step1_labels = np.asarray(step1_labels, dtype=np.int32)
    final_labels = np.asarray(final_labels, dtype=np.int32)
    if step1_labels.shape != final_labels.shape:
        raise RuntimeError(
            f"step1/final labels shape mismatch: {step1_labels.shape} vs {final_labels.shape}"
        )

    step1_sizes = np.bincount(step1_labels, minlength=int(num_clusters)).astype(np.int32)
    final_sizes = np.bincount(final_labels, minlength=int(num_clusters)).astype(np.int32)
    unchanged = int(np.sum(step1_labels == final_labels))
    moved = int(len(step1_labels) - unchanged)

    moved_out = []
    moved_in = []
    for cid in range(int(num_clusters)):
        out_cnt = int(np.sum((step1_labels == cid) & (final_labels != cid)))
        in_cnt = int(np.sum((step1_labels != cid) & (final_labels == cid)))
        moved_out.append(out_cnt)
        moved_in.append(in_cnt)

    return {
        "step1_cluster_sizes": [int(x) for x in step1_sizes.tolist()],
        "final_cluster_sizes": [int(x) for x in final_sizes.tolist()],
        "moved_docs": int(moved),
        "moved_ratio": float(moved / max(1, len(step1_labels))),
        "moved_out_per_cluster": moved_out,
        "moved_in_per_cluster": moved_in,
    }


def _load_workset_corpus_lookup() -> dict:
    if not os.path.exists(WORKSET_CORPUS_JSONL_PATH):
        return {}
    lookup = {}
    with open(WORKSET_CORPUS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            did = str(row.get("doc_id", "")).strip()
            if not did:
                continue
            lookup[did] = {
                "source_doc_id": str(row.get("source_doc_id", did)),
                "text": str(row.get("text", "")),
            }
    return lookup


def _simhash64(text: str) -> int:
    toks = [t for t in str(text).lower().split() if t]
    if not toks:
        return 0
    vec = [0] * 64
    for tok in toks:
        h = int.from_bytes(
            __import__("hashlib").sha1(tok.encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=False,
        )
        for b in range(64):
            if (h >> b) & 1:
                vec[b] += 1
            else:
                vec[b] -= 1
    out = 0
    for b, v in enumerate(vec):
        if v >= 0:
            out |= (1 << b)
    return int(out)


def _hamming64(a: int, b: int) -> int:
    return int((int(a) ^ int(b)).bit_count())


def _build_doc_metadata_arrays(
    *,
    doc_ids: np.ndarray,
    corpus_lookup: dict,
) -> tuple[np.ndarray, np.ndarray]:
    doc_id_list = [str(x) for x in np.asarray(doc_ids, dtype=object).tolist()]
    source_ids = []
    simhashes = []
    for did in doc_id_list:
        row = corpus_lookup.get(did, {})
        sid = str(row.get("source_doc_id", did.split("::")[0]))
        txt = str(row.get("text", ""))
        source_ids.append(sid)
        simhashes.append(int(_simhash64(txt)))
    return np.asarray(source_ids, dtype=object), np.asarray(simhashes, dtype=np.uint64)


def _cluster_membership_signature(
    *,
    chunk: np.ndarray,
    doc_ids: np.ndarray,
) -> tuple[str, str]:
    member_doc_ids = sorted(str(doc_ids[int(i)]) for i in np.asarray(chunk, dtype=np.int32).tolist())
    if not member_doc_ids:
        return "", ""
    digest = hashlib.sha1("\n".join(member_doc_ids).encode("utf-8")).hexdigest()
    return str(member_doc_ids[0]), str(digest)


def _canonicalize_final_chunks_by_membership(
    *,
    chunks: list[np.ndarray],
    doc_ids: np.ndarray,
) -> tuple[list[np.ndarray], list[dict]]:
    keyed = []
    for old_cid, chunk in enumerate(chunks):
        chunk_arr = np.asarray(chunk, dtype=np.int32)
        min_doc_id, membership_sha1 = _cluster_membership_signature(
            chunk=chunk_arr,
            doc_ids=doc_ids,
        )
        keyed.append(
            (
                str(min_doc_id),
                str(membership_sha1),
                int(old_cid),
                chunk_arr,
            )
        )
    keyed.sort(key=lambda x: (x[0], x[1], x[2]))
    reordered_chunks = [np.asarray(item[3], dtype=np.int32) for item in keyed]
    canonicalization_meta = [
        {
            "new_cluster_id": int(new_cid),
            "old_cluster_id": int(item[2]),
            "membership_min_doc_id": str(item[0]),
            "membership_sha1": str(item[1]),
        }
        for new_cid, item in enumerate(keyed)
    ]
    return reordered_chunks, canonicalization_meta


def _compose_j_doc_components(metrics: dict) -> dict:
    sep = float(max(0.0, metrics.get("center_separation_mean", 0.0)))
    core = float(max(0.0, metrics.get("core300_radius_mean", 0.0)))
    tail = float(max(0.0, metrics.get("tail_inflation_mean", 0.0)))
    dom = float(np.clip(metrics.get("source_dominance_mean", 0.0), 0.0, 1.0))
    dup = float(np.clip(metrics.get("duplicate_density_mean", 0.0), 0.0, 1.0))
    return {
        "center_separation": float(sep / (1.0 + sep)),
        "core300_radius": float(1.0 / (1.0 + core)),
        "tail_inflation": float(1.0 / (1.0 + tail)),
        "source_dominance": float(1.0 - dom),
        "duplicate_density": float(1.0 - dup),
    }


def _compose_j_doc_score(components: dict) -> float:
    vals = [float(components[k]) for k in (
        "center_separation",
        "core300_radius",
        "tail_inflation",
        "source_dominance",
        "duplicate_density",
    )]
    return float(np.mean(vals))


def _compute_docs_only_cluster_metrics(
    *,
    docs: np.ndarray,
    chunks: list[np.ndarray],
    centers: np.ndarray,
    source_ids: np.ndarray,
    simhashes: np.ndarray,
) -> dict:
    docs = np.asarray(docs, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    source_ids = np.asarray(source_ids, dtype=object)
    simhashes = np.asarray(simhashes, dtype=np.uint64)

    per_cluster = [
        _compute_single_cluster_docs_only_metrics(
            docs=docs,
            chunk=np.asarray(chunk, dtype=np.int32),
            center=centers[int(cid)],
            source_ids=source_ids,
            simhashes=simhashes,
            cluster_id=int(cid),
        )
        for cid, chunk in enumerate(chunks)
    ]
    return _aggregate_docs_only_cluster_metrics(
        centers=centers,
        per_cluster=per_cluster,
    )


def _compute_single_cluster_docs_only_metrics(
    *,
    docs: np.ndarray,
    chunk: np.ndarray,
    center: np.ndarray,
    source_ids: np.ndarray,
    simhashes: np.ndarray,
    cluster_id: int,
) -> dict:
    idx = np.asarray(chunk, dtype=np.int32)
    cdocs = np.asarray(docs[idx], dtype=np.float32)
    center = np.asarray(center, dtype=np.float32).reshape(-1)
    dists = np.linalg.norm(cdocs - center[None, :], axis=1).astype(np.float64)
    order = np.argsort(dists, kind="mergesort")
    core_k = int(min(300, len(order)))
    core_d = dists[order[:core_k]]
    tail_d = dists[order]
    r_core = float(np.percentile(core_d, 95.0)) if len(core_d) > 0 else 0.0
    r_tail = float(np.percentile(tail_d, 95.0)) if len(tail_d) > 0 else 0.0
    tail_inflation = float(max(0.0, r_tail - r_core))

    sids = [str(x) for x in source_ids[idx].tolist()]
    cnt = Counter(sids)
    source_dom = float(max(cnt.values()) / max(1, len(sids))) if cnt else 0.0

    local_hash = np.asarray(simhashes[idx], dtype=np.uint64)
    prefixes = [int(int(h) >> 48) for h in local_hash.tolist()]
    pcount = Counter(prefixes)
    dup_cnt = int(sum(int(c) * (int(c) - 1) // 2 for c in pcount.values()))
    total_pairs = int(len(local_hash) * (len(local_hash) - 1) // 2)
    dup_density = float(dup_cnt / max(1, total_pairs))

    return {
        "cluster_id": int(cluster_id),
        "core300_radius_p95": float(r_core),
        "tail500_radius_p95": float(r_tail),
        "tail_inflation_500_minus_300": float(tail_inflation),
        "source_dominance": float(source_dom),
        "duplicate_density": float(dup_density),
    }


def _center_separation_sum(centers: np.ndarray) -> float:
    centers = np.asarray(centers, dtype=np.float32)
    sep_sum = 0.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            sep_sum += float(np.linalg.norm(centers[i] - centers[j]))
    return float(sep_sum)


def _aggregate_docs_only_cluster_metrics(
    *,
    centers: np.ndarray,
    per_cluster: list[dict],
    precomputed_center_separation_sum: float | None = None,
) -> dict:
    centers = np.asarray(centers, dtype=np.float32)
    pair_count = int(len(centers) * (len(centers) - 1) // 2)
    if precomputed_center_separation_sum is None:
        center_separation_sum = _center_separation_sum(centers)
    else:
        center_separation_sum = float(precomputed_center_separation_sum)
    center_separation_mean = float(center_separation_sum / max(1, pair_count))

    core300_radius_list = [float(x.get("core300_radius_p95", 0.0)) for x in per_cluster]
    tail_inflation_list = [float(x.get("tail_inflation_500_minus_300", 0.0)) for x in per_cluster]
    source_dominance_list = [float(x.get("source_dominance", 0.0)) for x in per_cluster]
    duplicate_density_list = [float(x.get("duplicate_density", 0.0)) for x in per_cluster]

    return {
        "center_separation_mean": float(center_separation_mean),
        "core300_radius_mean": float(np.mean(core300_radius_list)) if core300_radius_list else 0.0,
        "tail_inflation_mean": float(np.mean(tail_inflation_list)) if tail_inflation_list else 0.0,
        "source_dominance_mean": float(np.mean(source_dominance_list)) if source_dominance_list else 0.0,
        "duplicate_density_mean": float(np.mean(duplicate_density_list)) if duplicate_density_list else 0.0,
        "per_cluster": [dict(x) for x in per_cluster],
    }


def _select_best_seed_docs_only(
    *,
    docs: np.ndarray,
    seed_start: int,
    seed_end: int,
    source_ids: np.ndarray,
    simhashes: np.ndarray,
) -> tuple[dict, list[dict]]:
    rows = []
    total = max(0, int(seed_end) - int(seed_start))
    best_row: dict | None = None
    for seed in range(int(seed_start), int(seed_end)):
        idx = int(seed - int(seed_start) + 1)
        print(
            f"[method4 sweep] seed {idx}/{total} (rng_seed={int(seed)}) started",
            flush=True,
        )
        step1 = train_step1_spherical_prototypes(
            docs=docs,
            num_clusters=int(NUM_CLUSTERS),
            rng_seed=int(seed),
            max_iter=int(STEP1_MAX_ITER),
        )
        balanced = global_balanced_assignment_once(
            docs=docs,
            prototypes=step1["prototypes"],
            capacity=int(TARGET_CLUSTER_SIZE),
            step1_labels=step1["labels"],
        )
        centers = _compute_proxy_centers_only(docs=docs, chunks=balanced["chunks"])
        metrics = _compute_docs_only_cluster_metrics(
            docs=docs,
            chunks=balanced["chunks"],
            centers=centers,
            source_ids=source_ids,
            simhashes=simhashes,
        )
        j_doc_components = _compose_j_doc_components(metrics)
        j_doc = _compose_j_doc_score(j_doc_components)
        rows.append(
            {
                "seed": int(seed),
                "method4_rng_seed_offset": int(seed - int(RANDOM_STATE)),
                "step1_objective_cost_sum": float(step1["objective_cost_sum"]),
                "balanced_objective_cost_sum": float(balanced["objective_cost_sum"]),
                **metrics,
                "j_doc_components": j_doc_components,
                "j_doc": float(j_doc),
            }
        )
        current = rows[-1]
        if best_row is None or float(current["j_doc"]) > float(best_row["j_doc"]):
            best_row = current
            print(
                "[method4 sweep] "
                f"seed {idx}/{total} completed; "
                f"j_doc={float(current['j_doc']):.6f}; "
                f"new_best_seed={int(current['seed'])}",
                flush=True,
            )
        else:
            print(
                "[method4 sweep] "
                f"seed {idx}/{total} completed; "
                f"j_doc={float(current['j_doc']):.6f}; "
                f"best_so_far_seed={int(best_row['seed'])}; "
                f"best_so_far_j_doc={float(best_row['j_doc']):.6f}",
                flush=True,
            )

    best = max(rows, key=lambda x: float(x["j_doc"]))
    return best, rows


def _refine_by_boundary_swaps_docs_only(
    *,
    docs: np.ndarray,
    chunks: list[np.ndarray],
    source_ids: np.ndarray,
    simhashes: np.ndarray,
    max_swaps: int,
) -> tuple[list[np.ndarray], dict]:
    if int(max_swaps) <= 0:
        return [np.asarray(c, dtype=np.int32) for c in chunks], {
            "accepted_swaps": 0,
            "j_doc_before": None,
            "j_doc_after": None,
            "components_before": None,
            "components_after": None,
        }

    cur_chunks = [np.asarray(c, dtype=np.int32).copy() for c in chunks]
    cur_centers = _compute_proxy_centers_only(docs=docs, chunks=cur_chunks)
    cur_metrics = _compute_docs_only_cluster_metrics(
        docs=docs,
        chunks=cur_chunks,
        centers=cur_centers,
        source_ids=source_ids,
        simhashes=simhashes,
    )
    init_metrics = dict(cur_metrics)
    cur_comp = _compose_j_doc_components(cur_metrics)
    cur_score = _compose_j_doc_score(cur_comp)
    init_score = float(cur_score)
    init_comp = dict(cur_comp)
    accepted = 0
    boundary_k = 10
    cur_per_cluster = [dict(x) for x in cur_metrics.get("per_cluster", [])]
    cur_sep_sum = _center_separation_sum(cur_centers)
    pair_count = int(len(cur_centers) * (len(cur_centers) - 1) // 2)

    for _ in range(int(max_swaps)):
        # 取每簇边缘文档候选
        boundary = []
        for cid, c in enumerate(cur_chunks):
            cidx = np.asarray(c, dtype=np.int32)
            d = np.linalg.norm(docs[cidx] - cur_centers[int(cid)][None, :], axis=1)
            order = np.argsort(-d)[: int(min(boundary_k, len(cidx)))]
            boundary.append(cidx[order].astype(np.int32))

        improved = False
        for a in range(int(NUM_CLUSTERS)):
            for b in range(a + 1, int(NUM_CLUSTERS)):
                for da in boundary[a].tolist():
                    for db in boundary[b].tolist():
                        na = cur_chunks[a].copy()
                        nb = cur_chunks[b].copy()
                        ia = int(np.where(na == int(da))[0][0])
                        ib = int(np.where(nb == int(db))[0][0])
                        na[ia], nb[ib] = int(db), int(da)

                        new_center_a = _compute_proxy_center_for_chunk(docs=docs, chunk=na)
                        new_center_b = _compute_proxy_center_for_chunk(docs=docs, chunk=nb)
                        new_metric_a = _compute_single_cluster_docs_only_metrics(
                            docs=docs,
                            chunk=na,
                            center=new_center_a,
                            source_ids=source_ids,
                            simhashes=simhashes,
                            cluster_id=int(a),
                        )
                        new_metric_b = _compute_single_cluster_docs_only_metrics(
                            docs=docs,
                            chunk=nb,
                            center=new_center_b,
                            source_ids=source_ids,
                            simhashes=simhashes,
                            cluster_id=int(b),
                        )
                        test_centers = np.asarray(cur_centers, dtype=np.float32).copy()
                        test_centers[a] = new_center_a
                        test_centers[b] = new_center_b

                        sep_delta = 0.0
                        for cid in range(int(NUM_CLUSTERS)):
                            if cid in {int(a), int(b)}:
                                continue
                            sep_delta -= float(np.linalg.norm(cur_centers[a] - cur_centers[cid]))
                            sep_delta -= float(np.linalg.norm(cur_centers[b] - cur_centers[cid]))
                            sep_delta += float(np.linalg.norm(new_center_a - cur_centers[cid]))
                            sep_delta += float(np.linalg.norm(new_center_b - cur_centers[cid]))
                        if pair_count > 0:
                            sep_delta -= float(np.linalg.norm(cur_centers[a] - cur_centers[b]))
                            sep_delta += float(np.linalg.norm(new_center_a - new_center_b))
                        test_sep_sum = float(cur_sep_sum + sep_delta)

                        test_per_cluster = [dict(x) for x in cur_per_cluster]
                        test_per_cluster[a] = new_metric_a
                        test_per_cluster[b] = new_metric_b
                        m = _aggregate_docs_only_cluster_metrics(
                            centers=test_centers,
                            per_cluster=test_per_cluster,
                            precomputed_center_separation_sum=test_sep_sum,
                        )
                        comp = _compose_j_doc_components(m)
                        s = _compose_j_doc_score(comp)
                        if s > cur_score + 1e-9:
                            cur_chunks[a] = np.asarray(na, dtype=np.int32)
                            cur_chunks[b] = np.asarray(nb, dtype=np.int32)
                            cur_centers = np.asarray(test_centers, dtype=np.float32)
                            cur_per_cluster = [dict(x) for x in test_per_cluster]
                            cur_sep_sum = float(test_sep_sum)
                            cur_metrics = m
                            cur_comp = comp
                            cur_score = float(s)
                            accepted += 1
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return cur_chunks, {
        "accepted_swaps": int(accepted),
        "j_doc_before": float(init_score),
        "j_doc_after": float(cur_score),
        "components_before": init_comp,
        "components_after": dict(cur_comp),
        "metrics_before": init_metrics,
        "metrics_after": cur_metrics,
    }


def _save_seed_scoring(rows: list[dict], best_seed: int):
    if not rows:
        return
    os.makedirs(os.path.dirname(SEED_SCAN_OUT_JSON), exist_ok=True)
    payload = {
        "scoring_name": "docs_only_j_doc",
        "best_seed": int(best_seed),
        "num_seeds": int(len(rows)),
        "rows": rows,
    }
    with open(SEED_SCAN_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "seed",
        "method4_rng_seed_offset",
        "j_doc",
        "j_center_separation",
        "j_core300_radius",
        "j_tail_inflation",
        "j_source_dominance",
        "j_duplicate_density",
        "center_separation_mean",
        "core300_radius_mean",
        "tail_inflation_mean",
        "source_dominance_mean",
        "duplicate_density_mean",
        "step1_objective_cost_sum",
        "balanced_objective_cost_sum",
        "is_best_seed",
    ]
    with open(SEED_SCAN_OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            comp = dict(r.get("j_doc_components", {}))
            writer.writerow(
                {
                    "seed": int(r.get("seed", -1)),
                    "method4_rng_seed_offset": int(r.get("method4_rng_seed_offset", -1)),
                    "j_doc": float(r.get("j_doc", 0.0)),
                    "j_center_separation": float(comp.get("center_separation", 0.0)),
                    "j_core300_radius": float(comp.get("core300_radius", 0.0)),
                    "j_tail_inflation": float(comp.get("tail_inflation", 0.0)),
                    "j_source_dominance": float(comp.get("source_dominance", 0.0)),
                    "j_duplicate_density": float(comp.get("duplicate_density", 0.0)),
                    "center_separation_mean": float(r.get("center_separation_mean", 0.0)),
                    "core300_radius_mean": float(r.get("core300_radius_mean", 0.0)),
                    "tail_inflation_mean": float(r.get("tail_inflation_mean", 0.0)),
                    "source_dominance_mean": float(r.get("source_dominance_mean", 0.0)),
                    "duplicate_density_mean": float(r.get("duplicate_density_mean", 0.0)),
                    "step1_objective_cost_sum": float(r.get("step1_objective_cost_sum", 0.0)),
                    "balanced_objective_cost_sum": float(r.get("balanced_objective_cost_sum", 0.0)),
                    "is_best_seed": int(int(r.get("seed", -1)) == int(best_seed)),
                }
            )


def main():
    docs, doc_ids, meta = load_workset_inputs()
    corpus_lookup = _load_workset_corpus_lookup()
    source_ids, simhashes = _build_doc_metadata_arrays(
        doc_ids=doc_ids,
        corpus_lookup=corpus_lookup,
    )

    seed_rows: list[dict] = []
    seed_scan_meta: dict = {
        "enabled": bool(METHOD4_SEED_SWEEP_ENABLED),
        "seed_scan_start": None,
        "seed_scan_end_exclusive": None,
        "num_seed_candidates": 0,
        "best_seed": None,
        "best_seed_j_doc": None,
        "seed_scoring_json": None,
        "seed_scoring_csv": None,
    }
    if bool(METHOD4_SEED_SWEEP_ENABLED):
        seed_start = int(RANDOM_STATE) + int(METHOD4_SEED_SWEEP_START)
        seed_end = int(RANDOM_STATE) + int(METHOD4_SEED_SWEEP_END)
        if seed_end <= seed_start:
            raise RuntimeError(
                "invalid METHOD4_SEED_SWEEP range: "
                f"start={seed_start}, end={seed_end}"
            )
        best_seed_row, seed_rows = _select_best_seed_docs_only(
            docs=docs,
            seed_start=seed_start,
            seed_end=seed_end,
            source_ids=source_ids,
            simhashes=simhashes,
        )
        selected_seed = int(best_seed_row["seed"])
        _save_seed_scoring(seed_rows, selected_seed)
        seed_scan_meta.update(
            {
                "seed_scan_start": int(seed_start),
                "seed_scan_end_exclusive": int(seed_end),
                "num_seed_candidates": int(len(seed_rows)),
                "best_seed": int(selected_seed),
                "best_seed_j_doc": float(best_seed_row["j_doc"]),
                "seed_scoring_json": str(SEED_SCAN_OUT_JSON),
                "seed_scoring_csv": str(SEED_SCAN_OUT_CSV),
            }
        )
    else:
        selected_seed = int(RANDOM_STATE) + int(RNG_SEED_OFFSET)
        seed_scan_meta.update(
            {
                "best_seed": int(selected_seed),
                "seed_scan_start": int(selected_seed),
                "seed_scan_end_exclusive": int(selected_seed + 1),
                "num_seed_candidates": 1,
            }
        )

    step1 = train_step1_spherical_prototypes(
        docs=docs,
        num_clusters=int(NUM_CLUSTERS),
        rng_seed=int(selected_seed),
        max_iter=int(STEP1_MAX_ITER),
    )
    balanced = global_balanced_assignment_once(
        docs=docs,
        prototypes=step1["prototypes"],
        capacity=int(TARGET_CLUSTER_SIZE),
        step1_labels=step1["labels"],
    )

    pre_refine_chunks = [np.asarray(c, dtype=np.int32) for c in balanced["chunks"]]
    pre_refine_frozen = freeze_final_geometry_with_core_proxy_center(
        docs=docs,
        chunks=pre_refine_chunks,
    )
    pre_refine_centers = np.asarray(pre_refine_frozen["proxy_centers"], dtype=np.float32)
    pre_refine_metrics = _compute_docs_only_cluster_metrics(
        docs=docs,
        chunks=pre_refine_chunks,
        centers=pre_refine_centers,
        source_ids=source_ids,
        simhashes=simhashes,
    )
    pre_refine_components = _compose_j_doc_components(pre_refine_metrics)
    pre_refine_j_doc = _compose_j_doc_score(pre_refine_components)

    final_chunks = pre_refine_chunks
    refinement_info = {
        "enabled": bool(METHOD4_DOC_REFINE_ENABLED),
        "max_swaps": int(METHOD4_DOC_REFINE_MAX_SWAPS),
        "accepted_swaps": 0,
        "j_doc_before": float(pre_refine_j_doc),
        "j_doc_after": float(pre_refine_j_doc),
        "components_before": pre_refine_components,
        "components_after": pre_refine_components,
        "metrics_before": pre_refine_metrics,
        "metrics_after": pre_refine_metrics,
    }
    if bool(METHOD4_DOC_REFINE_ENABLED):
        refined_chunks, refine_meta = _refine_by_boundary_swaps_docs_only(
            docs=docs,
            chunks=pre_refine_chunks,
            source_ids=source_ids,
            simhashes=simhashes,
            max_swaps=int(METHOD4_DOC_REFINE_MAX_SWAPS),
        )
        final_chunks = [np.asarray(c, dtype=np.int32) for c in refined_chunks]
        refinement_info.update(dict(refine_meta))

    final_chunks, cluster_id_canonicalization = _canonicalize_final_chunks_by_membership(
        chunks=final_chunks,
        doc_ids=doc_ids,
    )

    frozen_geometry = freeze_final_geometry_with_core_proxy_center(
        docs=docs,
        chunks=final_chunks,
    )
    final_centers = np.asarray(frozen_geometry["proxy_centers"], dtype=np.float32)

    final_metrics = _compute_docs_only_cluster_metrics(
        docs=docs,
        chunks=final_chunks,
        centers=final_centers,
        source_ids=source_ids,
        simhashes=simhashes,
    )
    final_components = _compose_j_doc_components(final_metrics)
    final_j_doc = _compose_j_doc_score(final_components)

    final_labels = np.full(int(len(docs)), -1, dtype=np.int32)
    for cid, chunk in enumerate(final_chunks):
        final_labels[np.asarray(chunk, dtype=np.int32)] = int(cid)
    if np.any(final_labels < 0):
        raise RuntimeError("final chunks produced unassigned docs")

    step1_alignment = summarize_step1_alignment(
        step1_labels=step1["labels"],
        final_labels=final_labels,
        num_clusters=int(NUM_CLUSTERS),
    )

    base_cost = l2_distance_matrix(docs, step1["prototypes"]).astype(np.float64)
    final_cost_sum = float(
        np.sum(base_cost[np.arange(len(docs), dtype=np.int32), final_labels], dtype=np.float64)
    )
    final_cost_mean = float(final_cost_sum / max(1, int(len(docs))))

    greedy_chunks = assign_balanced_clusters(
        dists=base_cost.astype(np.float32),
        capacity=int(TARGET_CLUSTER_SIZE),
    )
    greedy_cost = float(
        sum(
            np.sum(
                base_cost[np.asarray(chunk, dtype=np.int32), cid],
                dtype=np.float64,
            )
            for cid, chunk in enumerate(greedy_chunks)
        )
    )
    repair_vs_greedy_gap = float(final_cost_sum - greedy_cost)

    print("=" * 90)
    print("Method-4 docs-only paper-faithful mainline finished")
    print(f"seed_selected                     : {int(selected_seed)}")
    if bool(METHOD4_SEED_SWEEP_ENABLED):
        print(
            "seed_scan                         : "
            f"[{seed_scan_meta['seed_scan_start']}, {seed_scan_meta['seed_scan_end_exclusive']}) "
            f"num={seed_scan_meta['num_seed_candidates']}, "
            f"best_j_doc={seed_scan_meta['best_seed_j_doc']:.6f}"
        )
    print(f"step1 iters                       : {step1['iters']}")
    print(f"step1 objective_cost_sum          : {step1['objective_cost_sum']:.6f}")
    print(f"step1 objective_cost_mean         : {step1['objective_cost_mean']:.6f}")
    print(f"step1 cluster sizes               : {step1_alignment['step1_cluster_sizes']}")
    print(f"natural cluster sizes             : {balanced['natural_cluster_sizes']}")
    print(f"final assignment objective_cost_sum: {final_cost_sum:.6f}")
    print(f"final assignment objective_cost_mean: {final_cost_mean:.6f}")
    print(f"final cluster sizes               : {step1_alignment['final_cluster_sizes']}")
    print(
        "docs moved from step1->fixed500  : "
        f"{step1_alignment['moved_docs']} ({step1_alignment['moved_ratio']:.2%})"
    )
    print(f"greedy baseline objective_cost_sum: {greedy_cost:.6f}")
    print(f"final minus greedy (cost gap)     : {repair_vs_greedy_gap:.6f}")
    print(f"solver used                       : {balanced['solver_used']}")
    if balanced["fallback_reason"] is not None:
        print(f"fallback reason                   : {balanced['fallback_reason']}")
    print(
        "docs-only J_doc                  : "
        f"before={float(refinement_info.get('j_doc_before', pre_refine_j_doc)):.6f}, "
        f"after={float(refinement_info.get('j_doc_after', final_j_doc)):.6f}, "
        f"accepted_swaps={int(refinement_info.get('accepted_swaps', 0))}"
    )
    print("public center definition          : full-500 centroid")
    print(
        "full-vs-core center cosine       : "
        f"{[round(float(x), 6) for x in frozen_geometry['full_vs_core_center_cosine']]}"
    )
    print("=" * 90)

    method_info = {
        "name": "paperfaithful_docs_only_step1_balanced_spherical_mainline",
        "artifact_schema_version": CLUSTER_ARTIFACT_SCHEMA_VERSION,
        "workflow": [
            "docs_only_seed_selection_by_j_doc",
            "step1_spherical_prototype_training",
            "one_shot_fixed_capacity_boundary_repair",
            "docs_only_boundary_swap_refinement",
            "single_final_geometry_freeze",
        ],
        "seed_selection": {
            "enabled": bool(METHOD4_SEED_SWEEP_ENABLED),
            "criterion": "docs_only_j_doc",
            "j_doc_components": [
                "center_separation",
                "core300_radius",
                "tail_inflation",
                "source_dominance",
                "duplicate_density",
            ],
            "seed_scan_start": seed_scan_meta["seed_scan_start"],
            "seed_scan_end_exclusive": seed_scan_meta["seed_scan_end_exclusive"],
            "num_seed_candidates": int(seed_scan_meta["num_seed_candidates"]),
            "selected_seed": int(seed_scan_meta["best_seed"]),
            "selected_seed_j_doc": (
                float(seed_scan_meta["best_seed_j_doc"])
                if seed_scan_meta.get("best_seed_j_doc") is not None
                else None
            ),
            "seed_scoring_json": seed_scan_meta.get("seed_scoring_json"),
            "seed_scoring_csv": seed_scan_meta.get("seed_scoring_csv"),
        },
        "prototype_training": {
            "algorithm": "spherical_kmeans",
            "max_iter": int(STEP1_MAX_ITER),
            "capacity_constraint": "none",
            "assignment_cost": "normalized_euclidean_l2",
            "ranking_equivalence_note": "normalized_l2_and_cosine_are_monotonic_equivalent",
            "objective_cost_sum": float(step1["objective_cost_sum"]),
            "objective_cost_mean": float(step1["objective_cost_mean"]),
            "iters_used": int(step1["iters"]),
            "cluster_sizes_before_balance": [
                int(x) for x in step1_alignment["step1_cluster_sizes"]
            ],
        },
        "global_balanced_assignment": {
            "solver": str(balanced["solver_used"]),
            "solver_fallback_reason": balanced["fallback_reason"],
            "objective": "min_incremental_cost_boundary_repair_from_step1",
            "cost": "normalized_euclidean_l2",
            "capacity_per_cluster": int(TARGET_CLUSTER_SIZE),
            "complexity": "O(N*K*log(N*K))",
            "natural_cluster_sizes_before_repair": [
                int(x) for x in balanced["natural_cluster_sizes"]
            ],
            "objective_cost_sum": float(final_cost_sum),
            "objective_cost_mean": float(final_cost_mean),
            "moved_docs_from_step1": int(step1_alignment["moved_docs"]),
            "moved_ratio_from_step1": float(step1_alignment["moved_ratio"]),
            "moved_out_per_cluster": [int(x) for x in step1_alignment["moved_out_per_cluster"]],
            "moved_in_per_cluster": [int(x) for x in step1_alignment["moved_in_per_cluster"]],
            "greedy_capacity_baseline_cost_sum": float(greedy_cost),
            "final_minus_greedy_cost_gap": float(repair_vs_greedy_gap),
        },
        "docs_only_refinement": {
            "enabled": bool(METHOD4_DOC_REFINE_ENABLED),
            "max_swaps": int(METHOD4_DOC_REFINE_MAX_SWAPS),
            "accepted_swaps": int(refinement_info.get("accepted_swaps", 0)),
            "j_doc_before": float(refinement_info.get("j_doc_before", pre_refine_j_doc)),
            "j_doc_after": float(refinement_info.get("j_doc_after", final_j_doc)),
            "components_before": dict(refinement_info.get("components_before", pre_refine_components)),
            "components_after": dict(refinement_info.get("components_after", final_components)),
            "metrics_before": dict(refinement_info.get("metrics_before", pre_refine_metrics)),
            "metrics_after": dict(refinement_info.get("metrics_after", final_metrics)),
        },
        "docs_only_metrics_final": {
            **final_metrics,
            "j_doc_components": final_components,
            "j_doc": float(final_j_doc),
        },
        "final_geometry": {
            "center_source": "full_500_centroid_then_normalize",
            "public_center_member_count": int(PUBLIC_CENTER_CORE_SIZE),
            "public_center_definition": "full_500_centroid",
            "cluster_core_sizes_diagnostic": [int(x) for x in frozen_geometry["core_sizes"]],
            "cluster_core_share_diagnostic": [float(x) for x in frozen_geometry["core_share"]],
            "cluster_full_vs_core_center_cosine_diagnostic": [
                float(x) for x in frozen_geometry["full_vs_core_center_cosine"]
            ],
            "radius_source": "final_real_members",
            "distance_metric": "angular_distance_on_unit_sphere",
            "online_routing_metric": "angular_distance",
            "single_freeze_pass": True,
        },
        "cluster_id_canonicalization": {
            "enabled": True,
            "rule": "sort final clusters by (min_member_doc_id, sha1(sorted_member_doc_ids))",
            "mapping": [dict(x) for x in cluster_id_canonicalization],
        },
        "rmax_surrogate": {
            "mode": "cluster_level_offline_quantile",
            "scope": "within_topc_selected_cluster_global_hnsw",
            "ideal_formula": (
                "r_max_ideal_track1(q)=0 if global_topk(q) is not covered by "
                "topc_selected_cluster_global_hnsw(q); "
                "else tan((theta_fixed_boundary-theta_gt_topk_worst)/2)"
            ),
            "cluster_capacity_formula": (
                "R_max^(i)=quantile_gamma(r_max_ideal_track1 over anchors whose Top-c nearest "
                "centroids include i)"
            ),
            "gamma": float(RMAX_CLUSTER_QUANTILE_GAMMA),
            "anchor_policy": str(RMAX_ANCHOR_POLICY),
            "anchor_selection_constraints": {
                "target_anchors_per_cluster": int(RMAX_TARGET_ANCHORS_PER_CLUSTER),
                "min_anchors_per_cluster": int(RMAX_MIN_ANCHORS_PER_CLUSTER),
                "membership_min_ratio": float(RMAX_ANCHOR_MEMBERSHIP_MIN_RATIO),
                "num_distance_strata": int(RMAX_ANCHOR_NUM_DISTANCE_STRATA),
                "membership_filter_enabled": False,
                "enforce_min_per_cluster": bool(RMAX_ANCHOR_ENFORCE_MIN_PER_CLUSTER),
            },
            "online_dependency": "none_query_independent_constant_time_lookup",
        },
        "random_state": int(selected_seed),
        "query_participation": (
            "workset_clustering_seed_center=docs_only;"
            "calibration_queries_used_only_after_cluster_freeze_for_rmax"
        ),
    }
    save_cluster_artifacts(
        pipeline_name="method4_docs_only_paperfaithful_mainline",
        chunks=final_chunks,
        centers=final_centers,
        docs=docs,
        doc_ids=doc_ids,
        meta=meta,
        out_pkl_path=OUT_PKL,
        out_json_path=OUT_JSON,
        method_info=method_info,
        rmax_quantile_gamma=float(RMAX_CLUSTER_QUANTILE_GAMMA),
    )

    print(f"OUT_PKL        : {OUT_PKL}")
    print(f"OUT_JSON       : {OUT_JSON}")
    if bool(METHOD4_SEED_SWEEP_ENABLED):
        print(f"SEED_SCAN_JSON : {SEED_SCAN_OUT_JSON}")
        print(f"SEED_SCAN_CSV  : {SEED_SCAN_OUT_CSV}")


if __name__ == "__main__":
    main()
