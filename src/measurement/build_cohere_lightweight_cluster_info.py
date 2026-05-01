from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from shared.config import (
    EVAL_K,
    FIXED_K,
    NUM_CLUSTERS,
    ROUTING_CLUSTER_SELECTION_POLICY,
    ROUTING_FIXED_TOP_C,
    TARGET_CLUSTER_SIZE,
    WORKSET_CLUSTER_INFO_METHOD4_PATH,
    WORKSET_CLUSTER_SUMMARY_METHOD4_JSON,
    WORKSET_DOC_IDS_PATH,
    WORKSET_DOCS_PATH,
    WORKSET_META_PATH,
    WORKSET_NAME,
)


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (arr / norms).astype(np.float32)


def _normalize_vec(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr.astype(np.float32)
    return (arr / norm).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a lightweight sequential cluster_info for Cohere latency-only experiments."
    )
    parser.add_argument("--out-pkl", default=WORKSET_CLUSTER_INFO_METHOD4_PATH)
    parser.add_argument("--out-json", default=WORKSET_CLUSTER_SUMMARY_METHOD4_JSON)
    parser.add_argument("--rmax-value", type=float, default=10.0)
    parser.add_argument("--pipeline-name", default="cohere_latency_lightweight_cluster_info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    docs = np.load(WORKSET_DOCS_PATH, mmap_mode="r")
    doc_ids = np.load(WORKSET_DOC_IDS_PATH, allow_pickle=True)
    meta = _load_json(WORKSET_META_PATH)

    num_docs = int(docs.shape[0])
    dim = int(docs.shape[1])
    num_clusters = int(NUM_CLUSTERS)
    target_cluster_size = int(TARGET_CLUSTER_SIZE)
    eval_k = int(EVAL_K)
    fixed_k = int(FIXED_K)
    if num_docs != int(num_clusters * target_cluster_size):
        raise RuntimeError(
            "lightweight cluster builder requires exact balanced partition: "
            f"num_docs={num_docs} vs num_clusters*target_cluster_size={num_clusters * target_cluster_size}"
        )

    centers = np.empty((num_clusters, dim), dtype=np.float32)
    cluster_r_k = np.empty((num_clusters,), dtype=np.float32)
    cluster_r_fixed = np.empty((num_clusters,), dtype=np.float32)
    cluster_r_max = np.full((num_clusters,), float(args.rmax_value), dtype=np.float32)
    chunks: list[np.ndarray] = []
    overlap_chunks: list[np.ndarray] = []
    cluster_sizes: list[int] = []

    for cid in range(num_clusters):
        start = int(cid * target_cluster_size)
        end = int(start + target_cluster_size)
        idx = np.arange(start, end, dtype=np.int32)
        chunk_docs = _normalize_rows(np.asarray(docs[start:end], dtype=np.float32))
        center = _normalize_vec(np.mean(chunk_docs, axis=0))
        sims = np.clip(chunk_docs @ center.reshape(-1, 1), -1.0, 1.0).reshape(-1)
        dists = np.sort(np.arccos(sims).astype(np.float32))
        centers[cid] = center
        cluster_r_k[cid] = float(dists[max(0, min(len(dists) - 1, eval_k - 1))])
        cluster_r_fixed[cid] = float(dists[max(0, min(len(dists) - 1, fixed_k - 1))])
        chunks.append(idx)
        overlap_chunks.append(idx)
        cluster_sizes.append(int(len(idx)))

    cluster_info = {
        "pipeline": str(args.pipeline_name),
        "source_workset_pipeline": str(meta.get("pipeline", "")),
        "docs_path": str(WORKSET_DOCS_PATH),
        "doc_ids_path": str(WORKSET_DOC_IDS_PATH),
        "corpus_path": "",
        "workset_name": str(WORKSET_NAME),
        "num_clusters": int(num_clusters),
        "target_cluster_size": int(target_cluster_size),
        "eval_k": int(eval_k),
        "fixed_k": int(fixed_k),
        "chunks": chunks,
        "cluster_topc_overlap_doc_indices": overlap_chunks,
        "centers": centers.astype(np.float32),
        "cluster_r_k": cluster_r_k.astype(np.float32),
        "cluster_r_fixed": cluster_r_fixed.astype(np.float32),
        "cluster_r_max": cluster_r_max.astype(np.float32),
        "doc_topc_nearest_centroids": np.repeat(
            np.arange(num_clusters, dtype=np.int32).reshape(-1, 1), target_cluster_size, axis=0
        ).reshape(num_docs, 1),
        "rmax_surrogate": {
            "formula": "topc_overlap_route_union: theta_fixed-theta_k lightweight_constant_rmax",
            "rmax_scope": "within_topc_overlap_route_union_docs",
            "anchor_policy": "docs_only",
            "anchor_source": "docs_only_no_calibration",
            "anchor_cluster_assign_source": "docs_only_sequential_balanced_partition",
            "routing_cluster_selection_policy": str(ROUTING_CLUSTER_SELECTION_POLICY),
            "routing_fixed_top_c": int(ROUTING_FIXED_TOP_C),
            "anchor_selector_meta": {
                "selector": "docs_only_lightweight_no_membership_filter",
                "membership_scope": "",
            },
            "cluster_anchor_counts": [0 for _ in range(num_clusters)],
            "cluster_track1_coverage_counts": [0 for _ in range(num_clusters)],
            "cluster_track1_total_counts": [0 for _ in range(num_clusters)],
            "cluster_r_ideal_zero_counts": [0 for _ in range(num_clusters)],
            "cluster_r_max_raw": [float(x) for x in cluster_r_max.tolist()],
            "cluster_r_max_shrunk": [float(x) for x in cluster_r_max.tolist()],
            "cluster_topc_overlap_doc_counts": [int(x) for x in cluster_sizes],
        },
        "clustering_method": {
            "name": "cohere_latency_lightweight_cluster_info",
            "description": "Sequential balanced chunking for latency-only paperfaithful Cohere experiments.",
            "final_geometry": {
                "center_source": "full_500_centroid_then_normalize",
                "public_center_member_count": int(target_cluster_size),
            },
        },
    }

    out_pkl = Path(args.out_pkl).resolve()
    out_json = Path(args.out_json).resolve()
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(cluster_info, f)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pipeline": str(cluster_info["pipeline"]),
                "workset_name": str(cluster_info["workset_name"]),
                "num_clusters": int(num_clusters),
                "target_cluster_size": int(target_cluster_size),
                "eval_k": int(eval_k),
                "fixed_k": int(fixed_k),
                "rmax_value": float(args.rmax_value),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(
        json.dumps(
            {
                "status": "completed",
                "out_pkl": str(out_pkl),
                "out_json": str(out_json),
                "num_clusters": int(num_clusters),
                "target_cluster_size": int(target_cluster_size),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
