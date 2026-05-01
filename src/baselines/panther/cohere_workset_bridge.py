from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from baselines.common import ComparisonContractRow, write_contract_rows_csv, write_contract_rows_jsonl
from baselines.panther.config import PantherConfig
from baselines.panther.common import external_repo_root, load_cluster_info, save_json, write_jsonl
from baselines.panther.ms_author_bridge import (
    _choose_poly_modulus_degree,
    _compute_openpanther_bridge_config,
    _panther_partition_layout,
    _quantize_panther_uint,
    _safe_max_points_per_cluster,
    _selected_cluster_topk,
    _bridge_header_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MB = 1024.0 * 1024.0


def _parse_sizes(text: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("sizes list is empty")
    return sorted(values)


def _size_paths(project_root: Path, num_docs: int) -> dict[str, Path]:
    size = int(num_docs)
    data_root = project_root / "data"
    results_root = project_root / "results"
    raw_root = data_root / "raw"
    suffix = f"cohere_workset_{size}_paperfaithful_mainline_cohere_track1_{size}"
    return {
        "docs": data_root / f"docs_{suffix}.npy",
        "doc_ids": data_root / f"doc_ids_{suffix}.npy",
        "queries": data_root / f"queries_{suffix}.npy",
        "query_ids": data_root / f"query_ids_{suffix}.npy",
        "gt_topk": data_root / f"gt_topk_{suffix}.npy",
        "queries_jsonl": raw_root / f"queries_{suffix}.jsonl",
        "cluster_info_pkl": results_root / f"cluster_info_cohere_workset_{size}_balanced_spherical_paperfaithful_mainline_cohere_track1_{size}.pkl",
        "cluster_info_json": results_root / f"cluster_info_cohere_workset_{size}_balanced_spherical_paperfaithful_mainline_cohere_track1_{size}.json",
    }


def _ensure_exists(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("missing required Panther Cohere assets:\n" + "\n".join(missing))


def _stage_root(cfg: PantherConfig) -> Path:
    return external_repo_root(cfg) / "experimental" / "panther" / "dataset"


def _result_root(project_root: Path) -> Path:
    return project_root / "results" / "repro_workflows" / "panther"


def _dataset_slug(
    num_docs: int,
    selected_clusters: int,
    query_limit: int,
    top_k: int,
    poly_modulus_degree: int,
) -> str:
    return (
        f"cohere_track1_n{int(num_docs)}_sc{int(selected_clusters)}_q{int(query_limit)}"
        f"_k{int(top_k)}_N{int(poly_modulus_degree)}"
    )


def _u32bin_name(dataset_slug: str, kind: str) -> str:
    return f"{dataset_slug}_{kind}.u32bin"


def _write_u32_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(np.asarray(matrix, dtype=np.uint32))
    with open(path, "wb") as f:
        array.tofile(f)


def _write_quantized_docs_binary(path: Path, docs_path: Path, *, chunk_rows: int = 8192) -> None:
    docs = np.load(docs_path, mmap_mode="r")
    total = int(docs.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for start in range(0, total, int(chunk_rows)):
            end = min(total, start + int(chunk_rows))
            quantized = _quantize_panther_uint(np.asarray(docs[start:end], dtype=np.float32))
            np.ascontiguousarray(quantized, dtype=np.uint32).tofile(f)


def _last_match(text: str, pattern: str) -> re.Match[str] | None:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        return None
    return matches[-1]


def _last_float(text: str, pattern: str) -> float | None:
    match = _last_match(text, pattern)
    if match is None:
        return None
    return float(match.group(1))


def _parse_topk_ids(text: str) -> tuple[int | None, list[int]]:
    header = _last_match(text, r"(\d+)-NNs IDs:")
    if header is None:
        return None, []
    top_k = int(header.group(1))
    tail = text[header.end() :]
    ids_match = re.search(r"\(([0-9 ]+)\)", tail, flags=re.DOTALL)
    if ids_match is None:
        return top_k, []
    ids = [int(part) for part in ids_match.group(1).strip().split() if part.strip()]
    return top_k, ids[:top_k]


def _parse_query_metrics(*, client_log: Path, server_log: Path) -> dict:
    client_text = client_log.read_text(encoding="utf-8", errors="replace")
    server_text = server_log.read_text(encoding="utf-8", errors="replace")
    top_k, topk_ids = _parse_topk_ids(client_text)
    return {
        "latency_total_sec": (_last_float(client_text, r"Total time:\s*([0-9.]+)\s*ms") or 0.0) / 1000.0,
        "latency_server_sec": (_last_float(server_text, r"Total time:\s*([0-9.]+)\s*ms") or 0.0) / 1000.0,
        "comm_request_bytes": ((_last_float(client_text, r"Total comm:\s*([0-9.]+)\s*MB") or 0.0) * _MB),
        "comm_response_bytes": ((_last_float(server_text, r"Total comm:\s*([0-9.]+)\s*MB") or 0.0) * _MB),
        "distance_time_sec": (_last_float(client_text, r"Distance time:\s*([0-9.]+)\s*ms") or 0.0) / 1000.0,
        "topk_time_sec": (_last_float(client_text, r"Topk time:\s*([0-9.]+)\s*ms") or 0.0) / 1000.0,
        "pir_time_sec": (_last_float(client_text, r"Pir time:\s*([0-9.]+)\s*ms") or 0.0) / 1000.0,
        "top_k": int(top_k) if top_k is not None else None,
        "ranked_doc_indices": [int(x) for x in topk_ids],
        "accuracy_ratio_logged": _last_float(client_text, r"Accuracy:\s*\d+\s*/\s*\d+\s*=\s*([0-9.]+)"),
    }


def _prepare_bridge_assets(
    *,
    cfg: PantherConfig,
    num_docs: int,
    query_limit: int,
    top_k: int,
    requested_selected_clusters: int,
    requested_max_points_per_cluster: int,
    requested_poly_modulus_degree: int,
    reuse_existing: bool,
) -> dict:
    asset_paths = _size_paths(cfg.project_root, int(num_docs))
    _ensure_exists(asset_paths)

    docs_mmap = np.load(asset_paths["docs"], mmap_mode="r")
    queries = np.asarray(np.load(asset_paths["queries"]), dtype=np.float32)
    query_ids = [str(x) for x in np.load(asset_paths["query_ids"], allow_pickle=True).tolist()]
    gt_topk = np.asarray(np.load(asset_paths["gt_topk"]), dtype=np.int32)
    doc_ids = [str(x) for x in np.load(asset_paths["doc_ids"], allow_pickle=True).tolist()]
    if int(query_limit) > 0:
        limit = min(int(query_limit), int(len(queries)))
        queries = queries[:limit]
        query_ids = query_ids[:limit]
        gt_topk = gt_topk[:limit]
    if int(len(queries)) <= 0:
        raise RuntimeError("no queries available for Panther Cohere bridge")

    cluster_info = load_cluster_info(asset_paths["cluster_info_pkl"])
    base_chunks = [np.asarray(chunk, dtype=np.int32).reshape(-1) for chunk in cluster_info["chunks"]]
    poly_modulus_degree = (
        int(requested_poly_modulus_degree)
        if int(requested_poly_modulus_degree) > 0
        else _choose_poly_modulus_degree(
            dims=int(docs_mmap.shape[1]),
            num_docs=int(docs_mmap.shape[0]),
            requested_max_points_per_cluster=int(requested_max_points_per_cluster),
            default_degree=int(cfg.openpanther_default_poly_modulus_degree),
        )
    )
    safe_max_points_per_cluster = _safe_max_points_per_cluster(
        dims=int(docs_mmap.shape[1]),
        num_docs=int(docs_mmap.shape[0]),
        poly_modulus_degree=int(poly_modulus_degree),
    )
    max_points_per_cluster = min(int(requested_max_points_per_cluster), int(safe_max_points_per_cluster))
    layout = _panther_partition_layout(
        docs=docs_mmap,
        base_chunks=base_chunks,
        max_points_per_cluster=int(max_points_per_cluster),
    )
    selected_clusters_fallback = int(max(1, requested_selected_clusters))
    selected_clusters = _selected_cluster_topk(
        int(requested_selected_clusters),
        fallback=int(selected_clusters_fallback),
        cluster_num=int(layout["cluster_num"]),
    )
    dataset_slug = _dataset_slug(
        num_docs=int(num_docs),
        selected_clusters=int(selected_clusters),
        query_limit=int(len(queries)),
        top_k=int(top_k),
        poly_modulus_degree=int(poly_modulus_degree),
    )

    stage_root = _stage_root(cfg)
    stage_root.mkdir(parents=True, exist_ok=True)
    result_root = _result_root(cfg.project_root)
    result_root.mkdir(parents=True, exist_ok=True)
    bridge_run_root = result_root / dataset_slug
    bridge_run_root.mkdir(parents=True, exist_ok=True)

    dataset_bin = stage_root / _u32bin_name(dataset_slug, "dataset")
    test_bin = stage_root / _u32bin_name(dataset_slug, "test")
    neighbors_bin = stage_root / _u32bin_name(dataset_slug, "neighbors")
    centroids_bin = stage_root / _u32bin_name(dataset_slug, "centroids")
    ptoc_bin = stage_root / _u32bin_name(dataset_slug, "ptoc")
    stash_bin = stage_root / _u32bin_name(dataset_slug, "stash")

    quant_queries = _quantize_panther_uint(queries)
    centroids = np.asarray(layout["centroids"], dtype=np.float32)
    quant_centroids = _quantize_panther_uint(centroids)
    ptoc = np.asarray(layout["ptoc"], dtype=np.uint32)
    stash = np.asarray(layout["stash_indices"], dtype=np.uint32).reshape(-1, 1)

    if (not reuse_existing) or (not dataset_bin.exists()):
        _write_quantized_docs_binary(dataset_bin, asset_paths["docs"])
    if (not reuse_existing) or (not test_bin.exists()):
        _write_u32_matrix(test_bin, quant_queries)
    if (not reuse_existing) or (not neighbors_bin.exists()):
        _write_u32_matrix(neighbors_bin, np.asarray(gt_topk, dtype=np.uint32))
    if (not reuse_existing) or (not centroids_bin.exists()):
        _write_u32_matrix(centroids_bin, quant_centroids)
    if (not reuse_existing) or (not ptoc_bin.exists()):
        _write_u32_matrix(ptoc_bin, ptoc)
    if (not reuse_existing) or (not stash_bin.exists()):
        if int(stash.size) > 0:
            _write_u32_matrix(stash_bin, stash)
        else:
            stash_bin.write_bytes(b"")

    openpanther_config = _compute_openpanther_bridge_config(
        cfg=cfg,
        dataset_slug=str(dataset_slug),
        docs=docs_mmap,
        queries=queries,
        layout=layout,
        top_k=int(top_k),
        selected_clusters=int(selected_clusters),
        max_points_per_cluster=int(max_points_per_cluster),
        poly_modulus_degree=int(poly_modulus_degree),
    )
    openpanther_config["cluster_data_path"] = f"dataset/{centroids_bin.name}"
    openpanther_config["dataset_path"] = f"dataset/{dataset_bin.name}"
    openpanther_config["ptoc_path"] = f"dataset/{ptoc_bin.name}"
    openpanther_config["stash_path"] = f"dataset/{stash_bin.name}"
    openpanther_config["test_path"] = f"dataset/{test_bin.name}"
    openpanther_config["neighbors_path"] = f"dataset/{neighbors_bin.name}"

    config_header = external_repo_root(cfg) / "experimental" / "panther" / "demo" / "panther_bridge_config.h"
    config_header.write_text(_bridge_header_text(openpanther_config), encoding="utf-8")

    manifest = {
        "baseline_slug": "panther",
        "mode": "cohere_workset_openpanther_bridge",
        "dataset_slug": str(dataset_slug),
        "num_docs": int(num_docs),
        "num_queries": int(len(queries)),
        "top_k": int(top_k),
        "selected_clusters": int(selected_clusters),
        "requested_max_points_per_cluster": int(requested_max_points_per_cluster),
        "bridge_max_points_per_cluster": int(max_points_per_cluster),
        "poly_modulus_degree": int(poly_modulus_degree),
        "bridge_partition_cluster_num": int(layout["cluster_num"]),
        "bridge_partition_stash_size": int(layout["stash_size"]),
        "paths": {
            "docs": str(asset_paths["docs"]),
            "doc_ids": str(asset_paths["doc_ids"]),
            "queries": str(asset_paths["queries"]),
            "query_ids": str(asset_paths["query_ids"]),
            "gt_topk": str(asset_paths["gt_topk"]),
            "cluster_info_pkl": str(asset_paths["cluster_info_pkl"]),
            "dataset_bin": str(dataset_bin),
            "test_bin": str(test_bin),
            "neighbors_bin": str(neighbors_bin),
            "centroids_bin": str(centroids_bin),
            "ptoc_bin": str(ptoc_bin),
            "stash_bin": str(stash_bin),
            "config_header": str(config_header),
        },
        "openpanther_bridge_config": dict(openpanther_config),
    }
    manifest_path = bridge_run_root / "manifest.json"
    save_json(manifest_path, manifest)
    return {
        "asset_paths": asset_paths,
        "queries": queries,
        "query_ids": query_ids,
        "gt_topk": gt_topk,
        "doc_ids": doc_ids,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "bridge_run_root": bridge_run_root,
        "dataset_slug": dataset_slug,
    }


def _build_openpanther_bridge(cfg: PantherConfig) -> tuple[Path, Path]:
    openpanther_root = external_repo_root(cfg)
    cache_root = cfg.project_root / ".cache"
    bazel_distdir = cache_root / "panther_bazel_distdir"
    bazel_user_root = cache_root / "bazel_user_root"
    bazel_distdir.mkdir(parents=True, exist_ok=True)
    bazel_user_root.mkdir(parents=True, exist_ok=True)
    build_cmd = [
        "bazel",
        "--batch",
        f"--output_user_root={str(bazel_user_root)}",
        "build",
        f"--distdir={str(bazel_distdir)}",
        "-c",
        "opt",
        "//experimental/panther:panther_server_bridge",
        "//experimental/panther:panther_client_bridge",
    ]
    subprocess.run(build_cmd, cwd=str(openpanther_root), check=True)
    server_bin = openpanther_root / "bazel-bin" / "experimental" / "panther" / "panther_server_bridge"
    client_bin = openpanther_root / "bazel-bin" / "experimental" / "panther" / "panther_client_bridge"
    if not server_bin.exists() or not client_bin.exists():
        raise FileNotFoundError("OpenPanther bridge binaries missing after build")
    return server_bin, client_bin


def _run_query_pair(
    *,
    cfg: PantherConfig,
    server_bin: Path,
    client_bin: Path,
    query_index: int,
    run_root: Path,
    server_wait_sec: float,
) -> tuple[Path, Path]:
    openpanther_root = external_repo_root(cfg)
    log_root = run_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    client_log = log_root / f"query_{int(query_index):03d}_client.log"
    server_log = log_root / f"query_{int(query_index):03d}_server.log"
    with open(server_log, "w", encoding="utf-8") as sf:
        server_proc = subprocess.Popen(
            [str(server_bin)],
            cwd=str(openpanther_root),
            stdout=sf,
            stderr=subprocess.STDOUT,
        )
    try:
        time.sleep(float(server_wait_sec))
        with open(client_log, "w", encoding="utf-8") as cf:
            subprocess.run(
                [str(client_bin), f"--query_index={int(query_index)}"],
                cwd=str(openpanther_root),
                stdout=cf,
                stderr=subprocess.STDOUT,
                check=True,
            )
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=10)
    return client_log, server_log


def _aggregate_size_run(
    *,
    cfg: PantherConfig,
    num_docs: int,
    query_limit: int,
    top_k: int,
    selected_clusters: int,
    max_points_per_cluster: int,
    poly_modulus_degree: int,
    reuse_existing: bool,
    server_wait_sec: float,
) -> tuple[ComparisonContractRow, Path, Path]:
    prepared = _prepare_bridge_assets(
        cfg=cfg,
        num_docs=int(num_docs),
        query_limit=int(query_limit),
        top_k=int(top_k),
        requested_selected_clusters=int(selected_clusters),
        requested_max_points_per_cluster=int(max_points_per_cluster),
        requested_poly_modulus_degree=int(poly_modulus_degree),
        reuse_existing=bool(reuse_existing),
    )
    bridge_run_root = Path(prepared["bridge_run_root"])
    server_bin, client_bin = _build_openpanther_bridge(cfg)

    doc_ids = list(prepared["doc_ids"])
    query_ids = list(prepared["query_ids"])
    gt_topk = np.asarray(prepared["gt_topk"], dtype=np.int32)

    per_query_rows: list[dict] = []
    exact_recall_list: list[float] = []
    order_match_list: list[float] = []
    top1_hit_list: list[float] = []
    latency_total_list: list[float] = []
    latency_server_list: list[float] = []
    comm_request_list: list[float] = []
    comm_response_list: list[float] = []

    for query_index in range(int(len(query_ids))):
        client_log, server_log = _run_query_pair(
            cfg=cfg,
            server_bin=server_bin,
            client_bin=client_bin,
            query_index=int(query_index),
            run_root=bridge_run_root,
            server_wait_sec=float(server_wait_sec),
        )
        metrics = _parse_query_metrics(client_log=client_log, server_log=server_log)
        ranked_indices = [int(x) for x in metrics["ranked_doc_indices"][: int(top_k)] if 0 <= int(x) < int(num_docs)]
        ranked_doc_ids = [str(doc_ids[int(x)]) for x in ranked_indices]
        truth_indices = [int(x) for x in np.asarray(gt_topk[query_index], dtype=np.int32).tolist()[: int(top_k)]]
        truth_set = set(int(x) for x in truth_indices)
        overlap_count = float(len(set(ranked_indices) & truth_set))
        exact_recall = float(overlap_count / max(1, int(top_k)))
        order_match = 1.0 if list(ranked_indices[: int(top_k)]) == list(truth_indices[: int(top_k)]) else 0.0
        top1_hit = 1.0 if ranked_indices and truth_indices and int(ranked_indices[0]) == int(truth_indices[0]) else 0.0

        exact_recall_list.append(float(exact_recall))
        order_match_list.append(float(order_match))
        top1_hit_list.append(float(top1_hit))
        latency_total_list.append(float(metrics["latency_total_sec"]))
        latency_server_list.append(float(metrics["latency_server_sec"]))
        comm_request_list.append(float(metrics["comm_request_bytes"]))
        comm_response_list.append(float(metrics["comm_response_bytes"]))
        per_query_rows.append(
            {
                "query_index": int(query_index),
                "query_id": str(query_ids[query_index]),
                "top_k": int(top_k),
                "ranked_doc_indices": [int(x) for x in ranked_indices],
                "ranked_doc_ids": [str(x) for x in ranked_doc_ids],
                "exact_topk_doc_indices": [int(x) for x in truth_indices],
                "exact_topk_doc_ids": [str(doc_ids[int(x)]) for x in truth_indices],
                "exact_topk_overlap_count": float(overlap_count),
                "exact_recall_at_k": float(exact_recall),
                "exact_topk_full_order_match": bool(order_match > 0.0),
                "top1_hit": bool(top1_hit > 0.0),
                "latency_total_sec": float(metrics["latency_total_sec"]),
                "latency_server_sec": float(metrics["latency_server_sec"]),
                "comm_request_bytes": float(metrics["comm_request_bytes"]),
                "comm_response_bytes": float(metrics["comm_response_bytes"]),
                "source_client_log": str(client_log),
                "source_server_log": str(server_log),
            }
        )

    summary_path = bridge_run_root / "summary.json"
    rows_path = bridge_run_root / "rankings.jsonl"
    write_jsonl(rows_path, per_query_rows)
    summary = {
        "baseline_slug": "panther",
        "baseline_display_name": "Panther",
        "paper_url": str(cfg.paper_url),
        "code_url": str(cfg.code_url),
        "mode": "cohere_workset_openpanther_bridge",
        "comparison_axis": "database_size",
        "run_label": f"panther_cohere_track1_n{int(num_docs)}",
        "dataset": "cohere_track1",
        "num_docs": int(num_docs),
        "num_clusters": int(prepared["manifest"]["bridge_partition_cluster_num"]),
        "num_queries": int(len(per_query_rows)),
        "top_k": int(top_k),
        "latency_total_sec_avg": float(statistics.fmean(latency_total_list)),
        "latency_client_generate_sec_avg": None,
        "latency_server_query_sec_avg": float(statistics.fmean(latency_server_list)),
        "latency_client_recover_sec_avg": None,
        "comm_request_bytes_avg": float(statistics.fmean(comm_request_list)),
        "comm_response_bytes_avg": float(statistics.fmean(comm_response_list)),
        "comm_downstream_bytes_avg": None,
        "avg_exact_recall_at_k": float(statistics.fmean(exact_recall_list)),
        "exact_topk_overlap_mean": float(statistics.fmean(exact_recall_list)),
        "exact_topk_order_match_rate": float(statistics.fmean(order_match_list)),
        "top1_hit_rate": float(statistics.fmean(top1_hit_list)),
        "rows_jsonl": str(rows_path),
        "notes": [
            "Measured from the real OpenPanther bridge binaries on Cohere track1 worksets.",
            f"selected_clusters={int(prepared['manifest']['selected_clusters'])}",
            f"max_points_per_cluster={int(prepared['manifest']['bridge_max_points_per_cluster'])}",
            f"poly_modulus_degree={int(prepared['manifest']['poly_modulus_degree'])}",
            "latency_total_sec_avg uses the end-to-end client-observed total time from panther_client_bridge logs.",
            "exact_topk_overlap_mean stores normalized exact Recall@k against the Cohere track1 gt_topk file.",
        ],
    }
    save_json(summary_path, summary)

    row = ComparisonContractRow(
        baseline_slug="panther",
        baseline_display_name="Panther",
        paper_url=str(cfg.paper_url),
        contract_version="v1",
        comparison_axis="database_size",
        run_label=f"panther_n{int(num_docs)}",
        num_docs=int(num_docs),
        num_clusters=int(prepared["manifest"]["bridge_partition_cluster_num"]),
        num_queries=int(len(per_query_rows)),
        top_k=int(top_k),
        latency_total_sec_avg=float(summary["latency_total_sec_avg"]),
        latency_client_generate_sec_avg=None,
        latency_server_query_sec_avg=float(summary["latency_server_query_sec_avg"]),
        latency_client_recover_sec_avg=None,
        comm_request_bytes_avg=float(summary["comm_request_bytes_avg"]),
        comm_response_bytes_avg=float(summary["comm_response_bytes_avg"]),
        comm_downstream_bytes_avg=None,
        mean_first_relevant_rank=None,
        top1_hit_rate=float(summary["top1_hit_rate"]),
        exact_topk_overlap_mean=float(summary["exact_topk_overlap_mean"]),
        exact_topk_order_match_rate=float(summary["exact_topk_order_match_rate"]),
        candidate_cover_rate=float(summary["avg_exact_recall_at_k"]),
        direct_retrieve_rate=None,
        ot_retrieve_rate=None,
        real_cluster_hit_rate=None,
        source_summary_json=str(summary_path),
        source_rows_jsonl=str(rows_path),
        notes=tuple(str(x) for x in summary["notes"]),
    )
    return row, summary_path, rows_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real OpenPanther bridge on Cohere track1 worksets and export comparison-contract rows."
    )
    parser.add_argument("--sizes", type=str, default="10000", help="Comma-separated sizes, e.g. 10000,100000,1000000")
    parser.add_argument("--query-limit", type=int, default=64, help="How many staged queries to measure per size.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k retrieved results.")
    parser.add_argument("--selected-clusters", type=int, default=68, help="Fixed OpenPanther first-stage cluster count.")
    parser.add_argument("--max-points-per-cluster", type=int, default=20, help="Bridge partition size.")
    parser.add_argument("--poly-modulus-degree", type=int, default=0, help="Optional OpenPanther PIR poly_modulus_degree override. 0 auto-selects the smallest supported value that can fit the requested max-points-per-cluster for the current embedding dimension.")
    parser.add_argument("--server-wait-sec", type=float, default=2.0, help="Seconds to wait after starting the bridge server.")
    parser.set_defaults(reuse_existing=True)
    parser.add_argument("--reuse-existing", dest="reuse_existing", action="store_true")
    parser.add_argument("--no-reuse-existing", dest="reuse_existing", action="store_false")
    parser.add_argument("--output-prefix", type=str, default="cohere_panther_bridge_sc68_q64", help="Output stem under results/repro_workflows/panther/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PantherConfig(project_root=PROJECT_ROOT)
    result_root = _result_root(PROJECT_ROOT)
    result_root.mkdir(parents=True, exist_ok=True)

    rows: list[ComparisonContractRow] = []
    run_index: list[dict] = []
    for num_docs in _parse_sizes(str(args.sizes)):
        row, summary_path, rows_path = _aggregate_size_run(
            cfg=cfg,
            num_docs=int(num_docs),
            query_limit=int(args.query_limit),
            top_k=int(args.top_k),
            selected_clusters=int(args.selected_clusters),
            max_points_per_cluster=int(args.max_points_per_cluster),
            poly_modulus_degree=int(args.poly_modulus_degree),
            reuse_existing=bool(args.reuse_existing),
            server_wait_sec=float(args.server_wait_sec),
        )
        rows.append(row)
        run_index.append(
            {
                "num_docs": int(num_docs),
                "run_label": str(row.run_label),
                "summary_json": str(summary_path),
                "rows_jsonl": str(rows_path),
            }
        )

    jsonl_path = result_root / f"{str(args.output_prefix).strip()}.jsonl"
    csv_path = result_root / f"{str(args.output_prefix).strip()}.csv"
    summary_path = result_root / f"{str(args.output_prefix).strip()}.json"
    write_contract_rows_jsonl(jsonl_path, rows)
    write_contract_rows_csv(csv_path, rows)
    save_json(
        summary_path,
        {
            "baseline_slug": "panther",
            "baseline_display_name": "Panther",
            "sizes": [int(row.num_docs) for row in rows if row.num_docs is not None],
            "query_limit": int(args.query_limit),
            "top_k": int(args.top_k),
            "selected_clusters": int(args.selected_clusters),
            "max_points_per_cluster": int(args.max_points_per_cluster),
            "poly_modulus_degree": int(args.poly_modulus_degree),
            "artifacts": {
                "summary_json": str(summary_path),
                "contract_jsonl": str(jsonl_path),
                "contract_csv": str(csv_path),
            },
            "rows": run_index,
        },
    )
    print(f"[saved] {summary_path}")
    print(f"[saved] {jsonl_path}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
