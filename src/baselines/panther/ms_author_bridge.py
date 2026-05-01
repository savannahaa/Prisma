from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import csv
import math
import shutil
from pathlib import Path

import numpy as np

from baselines.panther.common import (
    bridge_root,
    copy_corpus_jsonl,
    copy_json,
    external_repo_root,
    exact_topk_indices,
    first_relevant_rank,
    load_bundle,
    load_json,
    load_jsonl,
    normalize_rows,
    save_json,
    write_jsonl,
    write_lines,
    write_matrix_txt,
)
from baselines.panther.config import PantherConfig


DOC_ID_FIELDS = (
    "ranked_doc_ids",
    "pred_doc_ids_top100",
    "pred_doc_ids",
    "doc_ids",
    "topk_doc_ids",
    "retrieved_doc_ids",
)
DOC_INDEX_FIELDS = (
    "ranked_doc_indices",
    "pred_doc_indices",
    "doc_indices",
    "topk_doc_indices",
    "retrieved_doc_indices",
)
_MB = 1024.0 * 1024.0


def _coerce_sequence(values) -> list:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return list(values)
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, str):
        text = values.strip()
        if not text:
            return []
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [part for part in text.split() if part]
    return list(values)


def _dedupe_keep_order(values: list[str], *, allowed: set[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw)
        if value not in allowed or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _selected_dataset_slug(cfg: PantherConfig, raw: str) -> str:
    value = str(raw).strip()
    return value if value else str(cfg.default_author_input_slug)


def _selected_routing_c(cfg: PantherConfig, raw: int) -> int:
    return int(raw) if int(raw) > 0 else int(cfg.default_cluster_info_selector_c)


def _selected_top_k(cfg: PantherConfig, raw: int) -> int:
    return int(raw) if int(raw) > 0 else int(cfg.default_top_k)


def _selected_query_limit(cfg: PantherConfig, raw: int) -> int | None:
    return int(raw) if int(raw) > 0 else int(cfg.default_query_limit)


def _selected_summary_stem(cfg: PantherConfig, raw: str) -> str:
    value = str(raw).strip()
    return value if value else str(cfg.default_summary_stem)


def _selected_max_points_per_cluster(cfg: PantherConfig, raw: int) -> int:
    return int(raw) if int(raw) > 0 else int(cfg.openpanther_default_max_points_per_cluster)


def _selected_cluster_topk(raw: int, *, fallback: int, cluster_num: int) -> int:
    requested = int(raw) if int(raw) > 0 else int(fallback)
    return max(1, min(int(cluster_num), requested))


def _maybe_float(value: float | None) -> float | None:
    if value is None:
        return None
    raw = float(value)
    if not math.isfinite(raw):
        return None
    return raw


def _int_log2_ceil(value: int) -> int:
    value = max(1, int(value))
    return int(math.ceil(math.log2(float(value))))


def _int_log256_ceil(value: int) -> int:
    value = max(1, int(value))
    return int(math.ceil(math.log(float(value), 256.0)))


def _bridge_message_size(*, dims: int, num_docs: int) -> int:
    max_sq_norm = int(dims) * 255 * 255
    return max(1, _int_log256_ceil(max(max_sq_norm, int(num_docs) - 1) + 1))


def _safe_max_points_per_cluster(*, dims: int, num_docs: int, poly_modulus_degree: int) -> int:
    message_size = _bridge_message_size(dims=int(dims), num_docs=int(num_docs))
    coeffs_per_point = int(dims) + 2 * int(message_size)
    return max(1, int(poly_modulus_degree) // max(1, coeffs_per_point))


def _choose_poly_modulus_degree(
    *,
    dims: int,
    num_docs: int,
    requested_max_points_per_cluster: int,
    default_degree: int,
) -> int:
    supported = [4096, 8192, 16384, 32768]
    requested = max(1, int(requested_max_points_per_cluster))
    default_degree = int(default_degree)
    if default_degree not in supported:
        supported.append(int(default_degree))
        supported = sorted(set(int(x) for x in supported))
    for degree in supported:
        if _safe_max_points_per_cluster(
            dims=int(dims),
            num_docs=int(num_docs),
            poly_modulus_degree=int(degree),
        ) >= int(requested):
            return int(degree)
    return int(max(supported))


def _quantize_panther_uint(matrix: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(matrix, dtype=np.float32), -1.0, 1.0)
    return np.clip(np.rint((clipped + 1.0) * 127.5), 0.0, 255.0).astype(np.uint32)


def _bridge_text_paths(root: Path, dataset_slug: str) -> dict[str, Path]:
    return {
        "dataset_txt": root / f"{dataset_slug}_dataset.txt",
        "test_txt": root / f"{dataset_slug}_test.txt",
        "neighbors_txt": root / f"{dataset_slug}_neighbors.txt",
        "centroids_txt": root / f"{dataset_slug}_centroids.txt",
        "ptoc_txt": root / f"{dataset_slug}_ptoc.txt",
        "stash_txt": root / f"{dataset_slug}_stash.txt",
    }


def _panther_partition_layout(
    *,
    docs: np.ndarray,
    base_chunks: list[np.ndarray],
    max_points_per_cluster: int,
) -> dict:
    cluster_rows: list[np.ndarray] = []
    stash_indices: list[int] = []
    for chunk in base_chunks:
        indices = np.asarray(chunk, dtype=np.int32).reshape(-1)
        if int(indices.size) <= 0:
            continue
        for start in range(0, int(indices.size), int(max_points_per_cluster)):
            sub = np.asarray(indices[start : start + int(max_points_per_cluster)], dtype=np.int32)
            if int(sub.size) == int(max_points_per_cluster):
                cluster_rows.append(sub)
            else:
                stash_indices.extend(int(x) for x in sub.tolist())
    if not cluster_rows:
        raise RuntimeError("failed to derive any Panther-compatible clusters from the MS bundle")

    cluster_num = int(len(cluster_rows))
    stash = np.asarray(stash_indices, dtype=np.int32).reshape(-1)
    ptoc = np.full((cluster_num, int(max_points_per_cluster)), 111111112, dtype=np.uint32)
    point_to_cluster = np.full(int(len(docs)), -1, dtype=np.int32)
    centroid_rows: list[np.ndarray] = []
    for cluster_id, row in enumerate(cluster_rows):
        ptoc[cluster_id, : int(len(row))] = np.asarray(row, dtype=np.uint32)
        point_to_cluster[row] = int(cluster_id)
        centroid_rows.append(np.mean(np.asarray(docs[row], dtype=np.float32), axis=0))
    if int(stash.size) > 0:
        centroid_rows.extend(np.asarray(docs[stash], dtype=np.float32))
    centroids = normalize_rows(np.asarray(centroid_rows, dtype=np.float32))
    return {
        "cluster_rows": [np.asarray(row, dtype=np.int32) for row in cluster_rows],
        "cluster_num": int(cluster_num),
        "stash_indices": stash.astype(np.int32),
        "stash_size": int(stash.size),
        "ptoc": ptoc.astype(np.uint32),
        "point_to_cluster": point_to_cluster.astype(np.int32),
        "centroids": centroids.astype(np.float32),
    }


def _compute_openpanther_bridge_config(
    *,
    cfg: PantherConfig,
    dataset_slug: str,
    docs: np.ndarray,
    queries: np.ndarray,
    layout: dict,
    top_k: int,
    selected_clusters: int,
    max_points_per_cluster: int,
) -> dict:
    dims = int(docs.shape[1])
    total_points_num = int(docs.shape[0])
    cluster_num = int(layout["cluster_num"])
    stash_size = int(layout["stash_size"])
    max_sq_norm = int(dims) * 255 * 255
    logt = min(31, max(24, _int_log2_ceil(2 * max_sq_norm + 1) + 2))
    message_size = _bridge_message_size(dims=int(dims), num_docs=int(total_points_num))
    group_bin_number_last = int(stash_size) if int(stash_size) > 0 else 1
    k_c = [int(cluster_num), int(stash_size)]
    group_bin_number = [1, int(group_bin_number_last)]
    group_k_number = [int(selected_clusters), int(top_k)]
    text_paths = _bridge_text_paths(Path("."), dataset_slug)
    return {
        "dataset_slug": str(dataset_slug),
        "dims": int(dims),
        "num_docs": int(total_points_num),
        "num_queries": int(queries.shape[0]),
        "topk_k": int(top_k),
        "selected_clusters": int(selected_clusters),
        "max_cluster_points": int(max_points_per_cluster),
        "cluster_num": int(cluster_num),
        "stash_size": int(stash_size),
        "sum_k_c": int(cluster_num + stash_size),
        "total_cluster_size": int(cluster_num),
        "pir_logt": int(cfg.openpanther_default_pir_logt),
        "pir_fixt": int(cfg.openpanther_default_pir_fixt),
        "logt": int(logt),
        "poly_modulus_degree": int(cfg.openpanther_default_poly_modulus_degree),
        "distance_poly_degree": int(cfg.openpanther_default_distance_poly_degree),
        "compare_radix": int(cfg.openpanther_default_compare_radix),
        "pointer_dc_bits": int(cfg.openpanther_default_pointer_dc_bits),
        "cluster_dc_bits": int(cfg.openpanther_default_cluster_dc_bits),
        "message_size": int(message_size),
        "ele_size": int((dims + 2 * message_size) * int(max_points_per_cluster)),
        "k_c": [int(x) for x in k_c],
        "group_bin_number": [int(x) for x in group_bin_number],
        "group_k_number": [int(x) for x in group_k_number],
        "cluster_data_path": f"dataset/{dataset_slug}_centroids.txt",
        "dataset_path": f"dataset/{dataset_slug}_dataset.txt",
        "ptoc_path": f"dataset/{dataset_slug}_ptoc.txt",
        "stash_path": f"dataset/{dataset_slug}_stash.txt",
        "test_path": f"dataset/{dataset_slug}_test.txt",
        "neighbors_path": f"dataset/{dataset_slug}_neighbors.txt",
        "text_paths": {key: str(path) for key, path in text_paths.items()},
    }


def _bridge_header_text(config: dict) -> str:
    k_c = ", ".join(str(int(x)) for x in config["k_c"])
    group_bin = ", ".join(str(int(x)) for x in config["group_bin_number"])
    group_k = ", ".join(str(int(x)) for x in config["group_k_number"])
    return f"""#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace panther_bridge {{

inline constexpr size_t pir_logt = {int(config["pir_logt"])};
inline constexpr size_t pir_fixt = {int(config["pir_fixt"])};
inline constexpr size_t logt = {int(config["logt"])};
inline constexpr size_t N = {int(config["poly_modulus_degree"])};
inline constexpr size_t dis_N = {int(config["distance_poly_degree"])};
inline constexpr size_t compare_radix = {int(config["compare_radix"])};
inline constexpr size_t max_cluster_points = {int(config["max_cluster_points"])};
inline constexpr size_t total_points_num = {int(config["num_docs"])};
inline constexpr size_t num_queries = {int(config["num_queries"])};
inline constexpr uint32_t dims = {int(config["dims"])};
inline constexpr size_t topk_k = {int(config["topk_k"])};
inline constexpr size_t selected_clusters = {int(config["selected_clusters"])};
inline constexpr size_t cluster_num = {int(config["cluster_num"])};
inline constexpr size_t stash_size = {int(config["stash_size"])};
inline constexpr size_t pointer_dc_bits = {int(config["pointer_dc_bits"])};
inline constexpr size_t cluster_dc_bits = {int(config["cluster_dc_bits"])};
inline constexpr size_t message_size = {int(config["message_size"])};
inline constexpr size_t ele_size = {int(config["ele_size"])};
inline constexpr uint32_t MASK = (1u << logt) - 1u;
inline constexpr uint32_t sum_k_c = {int(config["sum_k_c"])};
inline constexpr uint32_t total_cluster_size = {int(config["total_cluster_size"])};
inline const std::vector<int64_t> k_c = {{{k_c}}};
inline const std::vector<int64_t> group_bin_number = {{{group_bin}}};
inline const std::vector<int64_t> group_k_number = {{{group_k}}};
inline constexpr const char* cluster_data_path = "{config["cluster_data_path"]}";
inline constexpr const char* stash_path = "{config["stash_path"]}";
inline constexpr const char* dataset_path = "{config["dataset_path"]}";
inline constexpr const char* ptoc_path = "{config["ptoc_path"]}";
inline constexpr const char* test_path = "{config["test_path"]}";
inline constexpr const char* neighbors_path = "{config["neighbors_path"]}";
inline constexpr const char* dataset_slug = "{config["dataset_slug"]}";

}}  // namespace panther_bridge
"""


def _stage_into_openpanther(
    *,
    cfg: PantherConfig,
    dataset_slug: str,
    text_paths: dict[str, Path],
    openpanther_config: dict,
) -> dict | None:
    openpanther_root = external_repo_root(cfg)
    if not openpanther_root.exists():
        return None
    dataset_root = openpanther_root / "experimental" / "panther" / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    staged_paths: dict[str, str] = {}
    for key, src in text_paths.items():
        dst = dataset_root / src.name
        shutil.copy2(src, dst)
        staged_paths[key] = str(dst)
    config_header = openpanther_root / "experimental" / "panther" / "demo" / "panther_bridge_config.h"
    config_header.write_text(_bridge_header_text(openpanther_config), encoding="utf-8")
    staged_paths["bridge_config_header"] = str(config_header)
    stage_manifest = dataset_root / f"{dataset_slug}_bridge_manifest.json"
    save_json(
        stage_manifest,
        {
            "dataset_slug": str(dataset_slug),
            "dataset_root": str(dataset_root),
            "staged_paths": dict(staged_paths),
            "openpanther_config": dict(openpanther_config),
        },
    )
    staged_paths["stage_manifest"] = str(stage_manifest)
    return staged_paths


def _extract_ranked_doc_ids(row: dict, runtime: dict) -> list[str]:
    doc_ids = runtime["doc_ids"]
    for key in DOC_ID_FIELDS:
        values = row.get(key)
        if values is None:
            continue
        return [str(x) for x in _coerce_sequence(values)]
    for key in DOC_INDEX_FIELDS:
        values = row.get(key)
        if values is None:
            continue
        return [str(doc_ids[int(x)]) for x in _coerce_sequence(values)]
    raise ValueError(f"could not find ranked doc ids in row keys={sorted(row.keys())}")


def _standardize_payload_rows(payload_rows: list, runtime: dict) -> list[dict]:
    query_ids = runtime["query_ids"]
    qid_to_index = {str(qid): int(i) for i, qid in enumerate(query_ids)}
    standardized: list[dict] = []
    for pos, row in enumerate(payload_rows):
        if isinstance(row, list):
            query_index = int(pos)
            if query_index >= len(query_ids):
                raise IndexError(f"query_index={query_index} out of range for bundle size={len(query_ids)}")
            query_id = str(query_ids[query_index])
            ranked_doc_ids = [str(x) for x in row]
        elif isinstance(row, dict):
            raw_qid = row.get("query_id")
            raw_qindex = row.get("query_index")
            if raw_qid is None and raw_qindex is None:
                raw_qindex = int(pos)
            if raw_qid is not None:
                query_id = str(raw_qid)
                if query_id not in qid_to_index:
                    raise KeyError(f"query_id={query_id!r} not found in the MS bundle")
                query_index = int(qid_to_index[query_id])
            else:
                query_index = int(raw_qindex)
                if query_index < 0 or query_index >= len(query_ids):
                    raise IndexError(f"query_index={query_index} out of range for bundle size={len(query_ids)}")
                query_id = str(query_ids[query_index])
            ranked_doc_ids = _extract_ranked_doc_ids(row, runtime)
        else:
            raise TypeError(f"unsupported ranking row type: {type(row)!r}")
        standardized.append(
            {
                "query_index": int(query_index),
                "query_id": str(query_id),
                "ranked_doc_ids": list(ranked_doc_ids),
            }
        )
    standardized.sort(key=lambda row: (int(row["query_index"]), str(row["query_id"])))
    return standardized


def _load_rankings_jsonl(path: Path, runtime: dict) -> list[dict]:
    return _standardize_payload_rows(load_jsonl(path), runtime)


def _load_rankings_json(path: Path, runtime: dict) -> list[dict]:
    payload = load_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        list_value = None
        for key in ("rows", "rankings", "results", "queries"):
            if isinstance(payload.get(key), list):
                list_value = payload.get(key)
                break
        if list_value is not None:
            rows = list_value
        else:
            rows = [{"query_id": str(qid), "ranked_doc_ids": _coerce_sequence(docs)} for qid, docs in payload.items()]
    else:
        raise TypeError(f"unsupported json ranking payload type: {type(payload)!r}")
    return _standardize_payload_rows(list(rows), runtime)


def _load_rankings_tsv(path: Path, runtime: dict) -> list[dict]:
    grouped: dict[tuple[int, str], list[tuple[int, str]]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"missing header in TSV ranking file: {path}")
        for line_no, row in enumerate(reader, start=2):
            raw_qid = row.get("query_id")
            raw_qindex = row.get("query_index")
            if raw_qid is None and raw_qindex is None:
                raise ValueError(f"TSV row {line_no} is missing query_id/query_index")
            raw_rank = row.get("rank")
            rank = int(raw_rank) if raw_rank not in (None, "") else int(line_no)
            if row.get("doc_id") not in (None, ""):
                doc_id = str(row["doc_id"])
            elif row.get("doc_index") not in (None, ""):
                doc_id = str(runtime["doc_ids"][int(row["doc_index"])])
            else:
                raise ValueError(f"TSV row {line_no} is missing doc_id/doc_index")
            if raw_qid not in (None, ""):
                query_id = str(raw_qid)
                query_index = int({str(q): i for i, q in enumerate(runtime["query_ids"])}[query_id])
            else:
                query_index = int(raw_qindex)
                query_id = str(runtime["query_ids"][query_index])
            grouped.setdefault((query_index, query_id), []).append((int(rank), str(doc_id)))
    rows = []
    for (query_index, query_id), ranked_items in sorted(grouped.items()):
        ranked_items.sort(key=lambda item: int(item[0]))
        rows.append(
            {
                "query_index": int(query_index),
                "query_id": str(query_id),
                "ranked_doc_ids": [str(doc_id) for _, doc_id in ranked_items],
            }
        )
    return rows


def _load_rankings_npy(path: Path, runtime: dict) -> list[dict]:
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 1:
        arr = np.asarray(arr).reshape(1, -1)
    rows: list[dict] = []
    numeric = np.issubdtype(arr.dtype, np.number)
    for query_index in range(int(arr.shape[0])):
        values = arr[query_index].tolist()
        if numeric:
            ranked_doc_ids = [str(runtime["doc_ids"][int(x)]) for x in values]
        else:
            ranked_doc_ids = [str(x) for x in values]
        rows.append(
            {
                "query_index": int(query_index),
                "query_id": str(runtime["query_ids"][query_index]),
                "ranked_doc_ids": ranked_doc_ids,
            }
        )
    return rows


def _load_rankings(args: argparse.Namespace, runtime: dict) -> tuple[list[dict], Path]:
    candidates = [
        ("jsonl", str(args.rankings_jsonl).strip()),
        ("json", str(args.rankings_json).strip()),
        ("tsv", str(args.rankings_tsv).strip()),
        ("npy", str(args.rankings_npy).strip()),
    ]
    selected = [(kind, raw) for kind, raw in candidates if raw]
    if len(selected) != 1:
        raise ValueError("provide exactly one of --rankings-jsonl / --rankings-json / --rankings-tsv / --rankings-npy")
    kind, raw_path = selected[0]
    path = Path(raw_path).resolve()
    if kind == "jsonl":
        return _load_rankings_jsonl(path, runtime), path
    if kind == "json":
        return _load_rankings_json(path, runtime), path
    if kind == "tsv":
        return _load_rankings_tsv(path, runtime), path
    if kind == "npy":
        return _load_rankings_npy(path, runtime), path
    raise RuntimeError(f"unsupported ranking input kind: {kind}")


def _write_topk_tsv(path: Path, rows: list[dict]) -> None:
    headers = [
        "query_index",
        "query_id",
        "rank",
        "doc_id",
        "is_qrels_relevant",
        "first_relevant_rank",
        "exact_topk_overlap_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def export_ms_assets(cfg: PantherConfig, args: argparse.Namespace) -> dict:
    routing_c = _selected_routing_c(cfg, int(args.routing_c))
    query_limit = _selected_query_limit(cfg, int(args.query_limit))
    top_k = _selected_top_k(cfg, int(args.top_k))
    dataset_slug = _selected_dataset_slug(cfg, str(args.dataset_slug))
    requested_max_points_per_cluster = _selected_max_points_per_cluster(cfg, int(args.max_points_per_cluster))
    runtime = load_bundle(
        cfg,
        query_limit=query_limit,
        cluster_info_selector_c=routing_c,
    )
    cluster_info = runtime["cluster_info"]
    if cluster_info is None:
        raise RuntimeError("missing cluster_info payload for Panther MS asset export")

    out_root = bridge_root(cfg) / str(dataset_slug)
    out_root.mkdir(parents=True, exist_ok=True)

    docs = normalize_rows(np.asarray(runtime["docs"], dtype=np.float32))
    queries = normalize_rows(np.asarray(runtime["queries"], dtype=np.float32))
    safe_max_points_per_cluster = _safe_max_points_per_cluster(
        dims=int(docs.shape[1]),
        num_docs=int(docs.shape[0]),
        poly_modulus_degree=int(cfg.openpanther_default_poly_modulus_degree),
    )
    max_points_per_cluster = min(int(requested_max_points_per_cluster), int(safe_max_points_per_cluster))
    doc_ids = [str(x) for x in runtime["doc_ids"]]
    query_ids = [str(x) for x in runtime["query_ids"]]
    base_chunks = [np.asarray(chunk, dtype=np.int32).reshape(-1) for chunk in cluster_info["chunks"]]
    layout = _panther_partition_layout(
        docs=docs,
        base_chunks=base_chunks,
        max_points_per_cluster=int(max_points_per_cluster),
    )
    point_to_cluster = np.asarray(layout["point_to_cluster"], dtype=np.int32)
    stash_indices = np.asarray(layout["stash_indices"], dtype=np.int32)
    centers = normalize_rows(np.asarray(layout["centroids"], dtype=np.float32))

    candidate_budget = int(cluster_info.get("fixed_k", top_k))
    effective_candidate_budget = max(1, min(int(candidate_budget), int(top_k)))
    selected_clusters_fallback = int(math.ceil(float(effective_candidate_budget) / float(max_points_per_cluster)))
    selected_clusters = _selected_cluster_topk(
        int(args.selected_clusters),
        fallback=selected_clusters_fallback,
        cluster_num=int(layout["cluster_num"]),
    )

    quant_docs = _quantize_panther_uint(docs)
    quant_queries = _quantize_panther_uint(queries)
    quant_centers = _quantize_panther_uint(centers)
    quant_docs_float = np.asarray(quant_docs, dtype=np.float32)
    quant_queries_float = np.asarray(quant_queries, dtype=np.float32)

    exact_neighbors = np.stack(
        [exact_topk_indices(docs=quant_docs_float, query=quant_queries_float[i], top_k=int(top_k)) for i in range(int(len(queries)))],
        axis=0,
    ).astype(np.int32)

    np.save(out_root / "docs.npy", docs.astype(np.float32))
    np.save(out_root / "doc_ids.npy", np.asarray(doc_ids, dtype=object))
    np.save(out_root / "queries.npy", queries.astype(np.float32))
    np.save(out_root / "query_ids.npy", np.asarray(query_ids, dtype=object))
    np.save(out_root / "exact_neighbors.npy", exact_neighbors.astype(np.int32))
    np.save(out_root / "centroids.npy", centers.astype(np.float32))
    np.save(out_root / "point_to_cluster.npy", point_to_cluster.astype(np.int32))
    np.save(out_root / "stash_indices.npy", stash_indices.astype(np.int32))
    np.save(out_root / "panther_quantized_docs.npy", quant_docs.astype(np.uint32))
    np.save(out_root / "panther_quantized_queries.npy", quant_queries.astype(np.uint32))
    np.save(out_root / "panther_quantized_centroids.npy", quant_centers.astype(np.uint32))
    np.save(out_root / "panther_ptoc.npy", np.asarray(layout["ptoc"], dtype=np.uint32))

    copy_json(out_root / "qrels.json", {qid: sorted(runtime["qrels"].get(qid, set())) for qid in query_ids})
    copy_corpus_jsonl(out_root / "corpus.jsonl", runtime["corpus_rows"])
    write_lines(out_root / "doc_ids.txt", doc_ids)
    write_lines(out_root / "query_ids.txt", query_ids)

    text_path_map = _bridge_text_paths(out_root, dataset_slug)
    if bool(args.write_openpanther_text) or bool(args.stage_into_openpanther):
        write_matrix_txt(text_path_map["dataset_txt"], quant_docs, fmt="%d")
        write_matrix_txt(text_path_map["test_txt"], quant_queries, fmt="%d")
        write_matrix_txt(text_path_map["neighbors_txt"], exact_neighbors, fmt="%d")
        write_matrix_txt(text_path_map["centroids_txt"], quant_centers, fmt="%d")
        write_matrix_txt(text_path_map["ptoc_txt"], np.asarray(layout["ptoc"], dtype=np.uint32), fmt="%d")
        stash_path = text_path_map["stash_txt"]
        if int(len(stash_indices)) > 0:
            write_matrix_txt(stash_path, stash_indices.reshape(-1, 1), fmt="%d")
        else:
            stash_path.parent.mkdir(parents=True, exist_ok=True)
            stash_path.write_text("", encoding="utf-8")

    openpanther_config = _compute_openpanther_bridge_config(
        cfg=cfg,
        dataset_slug=str(dataset_slug),
        docs=docs,
        queries=queries,
        layout=layout,
        top_k=int(top_k),
        selected_clusters=int(selected_clusters),
        max_points_per_cluster=int(max_points_per_cluster),
    )
    staged_into_openpanther = None
    if bool(args.stage_into_openpanther):
        staged_into_openpanther = _stage_into_openpanther(
            cfg=cfg,
            dataset_slug=str(dataset_slug),
            text_paths=text_path_map,
            openpanther_config=openpanther_config,
        )

    manifest = {
        "baseline_slug": "panther",
        "mode": "ms_author_bridge_assets",
        "paper_url": str(cfg.paper_url),
        "code_url": str(cfg.code_url),
        "dataset": "ms",
        "dataset_slug": str(dataset_slug),
        "source_bundle_root": str(runtime["paths"]["root"]),
        "bundle_meta_path": str(runtime["paths"]["bundle_meta"]),
        "cluster_info_selector_c": int(routing_c),
        "bridge_partition_cluster_num": int(layout["cluster_num"]),
        "bridge_partition_stash_size": int(layout["stash_size"]),
        "bridge_max_points_per_cluster": int(max_points_per_cluster),
        "bridge_requested_max_points_per_cluster": int(requested_max_points_per_cluster),
        "bridge_safe_max_points_per_cluster": int(safe_max_points_per_cluster),
        "bridge_selected_clusters": int(selected_clusters),
        "bridge_effective_candidate_budget": int(effective_candidate_budget),
        "actual_num_clusters": int(cluster_info.get("num_clusters", len(base_chunks))),
        "num_docs": int(docs.shape[0]),
        "num_queries": int(queries.shape[0]),
        "embedding_dim": int(docs.shape[1]),
        "top_k_exact_neighbors": int(top_k),
        "openpanther_style_text_emitted": bool(args.write_openpanther_text) or bool(args.stage_into_openpanther),
        "openpanther_staged_into_repo": bool(staged_into_openpanther is not None),
        "notes": [
            "This bridge package keeps the repo's MS qrels bundle intact while exporting a Panther-compatible integer dataset staging directory.",
            "The OpenPanther txt files are quantized to [0, 255] via round((x + 1) * 127.5), matching the author code's integer input convention.",
            "ptoc.txt is now a 2-D posting-list matrix (cluster -> point ids), and stash.txt is a stash point-id list, matching convert_model_to_input.py/common.cc semantics.",
            f"max_points_per_cluster is capped to {int(max_points_per_cluster)} so ele_size fits under poly_modulus_degree={int(cfg.openpanther_default_poly_modulus_degree)} for {int(docs.shape[1])}-d embeddings.",
            f"selected_clusters fallback uses effective_candidate_budget=min(fixed_k, top_k)={int(effective_candidate_budget)} to keep the PIR batch size protocol-safe.",
        ],
        "paths": {
            "bridge_root": str(out_root),
            "docs_npy": str(out_root / "docs.npy"),
            "doc_ids_npy": str(out_root / "doc_ids.npy"),
            "queries_npy": str(out_root / "queries.npy"),
            "query_ids_npy": str(out_root / "query_ids.npy"),
            "qrels_json": str(out_root / "qrels.json"),
            "corpus_jsonl": str(out_root / "corpus.jsonl"),
            "exact_neighbors_npy": str(out_root / "exact_neighbors.npy"),
            "centroids_npy": str(out_root / "centroids.npy"),
            "point_to_cluster_npy": str(out_root / "point_to_cluster.npy"),
            "stash_indices_npy": str(out_root / "stash_indices.npy"),
            "panther_quantized_docs_npy": str(out_root / "panther_quantized_docs.npy"),
            "panther_quantized_queries_npy": str(out_root / "panther_quantized_queries.npy"),
            "panther_quantized_centroids_npy": str(out_root / "panther_quantized_centroids.npy"),
            "panther_ptoc_npy": str(out_root / "panther_ptoc.npy"),
            **{key: str(path) for key, path in text_path_map.items()},
        },
        "openpanther_bridge_config": dict(openpanther_config),
    }
    if staged_into_openpanther is not None:
        manifest["staged_into_openpanther"] = dict(staged_into_openpanther)
    manifest_path = out_root / "manifest.json"
    save_json(manifest_path, manifest)
    print(f"[saved] {manifest_path}")
    return manifest


def build_summary_from_rankings(cfg: PantherConfig, args: argparse.Namespace) -> dict:
    routing_c = _selected_routing_c(cfg, int(args.routing_c))
    query_limit = _selected_query_limit(cfg, int(args.query_limit))
    top_k = _selected_top_k(cfg, int(args.top_k))
    summary_stem = _selected_summary_stem(cfg, str(args.summary_stem))
    runtime = load_bundle(
        cfg,
        query_limit=query_limit,
        cluster_info_selector_c=routing_c,
    )
    cluster_info = runtime["cluster_info"]
    ranking_rows, ranking_input_path = _load_rankings(args, runtime)
    result_root = cfg.project_root / "results" / "repro_workflows" / "panther"
    result_root.mkdir(parents=True, exist_ok=True)
    rows_out_path = result_root / f"{summary_stem}_rankings.jsonl"
    topk_tsv_path = result_root / f"{summary_stem}_top{int(top_k)}.tsv"
    summary_path = result_root / f"{summary_stem}.json"

    doc_id_set = {str(x) for x in runtime["doc_ids"]}
    docid_to_index = {str(doc_id): int(i) for i, doc_id in enumerate(runtime["doc_ids"])}
    docs = np.asarray(runtime["docs"], dtype=np.float32)
    queries = np.asarray(runtime["queries"], dtype=np.float32)

    per_query_rows: list[dict] = []
    topk_rows: list[dict] = []
    first_ranks: list[int] = []
    top1_hits: list[float] = []
    candidate_cover_flags: list[float] = []
    exact_overlap_counts: list[float] = []
    exact_recall_at_k_list: list[float] = []
    exact_order_match_flags: list[float] = []

    for row in ranking_rows:
        query_index = int(row["query_index"])
        query_id = str(row["query_id"])
        pred_doc_ids = _dedupe_keep_order(list(row["ranked_doc_ids"]), allowed=doc_id_set)[: int(top_k)]
        pred_doc_indices = [int(docid_to_index[doc_id]) for doc_id in pred_doc_ids]
        positive_doc_ids = set(runtime["qrels"].get(query_id, set()))
        exact_indices = exact_topk_indices(docs=docs, query=queries[query_index], top_k=int(top_k))
        exact_doc_ids = [str(runtime["doc_ids"][int(idx)]) for idx in exact_indices.tolist()]

        first_rank = first_relevant_rank(pred_doc_ids, positive_doc_ids, cutoff=int(top_k))
        top1_hit = 1.0 if pred_doc_ids and pred_doc_ids[0] in positive_doc_ids else 0.0
        candidate_contains_relevant = 1.0 if set(pred_doc_ids) & positive_doc_ids else 0.0
        exact_overlap_count = float(len(set(pred_doc_ids) & set(exact_doc_ids)))
        exact_recall_at_k = float(exact_overlap_count / max(1, int(top_k)))
        exact_order_match = 1.0 if list(pred_doc_ids[: int(top_k)]) == list(exact_doc_ids[: int(top_k)]) else 0.0

        if first_rank is not None:
            first_ranks.append(int(first_rank))
        top1_hits.append(float(top1_hit))
        candidate_cover_flags.append(float(candidate_contains_relevant))
        exact_overlap_counts.append(float(exact_overlap_count))
        exact_recall_at_k_list.append(float(exact_recall_at_k))
        exact_order_match_flags.append(float(exact_order_match))

        per_query_rows.append(
            {
                "query_index": int(query_index),
                "query_id": str(query_id),
                "top_k": int(top_k),
                "num_qrels_positive_docs": int(len(positive_doc_ids)),
                "ranked_doc_ids": [str(x) for x in pred_doc_ids],
                "ranked_doc_indices": [int(x) for x in pred_doc_indices],
                "exact_topk_doc_ids": [str(x) for x in exact_doc_ids],
                "exact_topk_doc_indices": [int(x) for x in exact_indices.tolist()],
                "first_relevant_rank": int(first_rank) if first_rank is not None else None,
                "top1_hit": bool(top1_hit > 0.0),
                "candidate_contains_relevant": bool(candidate_contains_relevant > 0.0),
                "exact_topk_overlap_count": float(exact_overlap_count),
                "exact_topk_overlap_ratio": float(exact_overlap_count / max(1, int(top_k))),
                "exact_recall_at_k": float(exact_recall_at_k),
                "exact_topk_full_order_match": bool(exact_order_match > 0.0),
                "source_rankings_path": str(ranking_input_path),
            }
        )

        for rank, doc_id in enumerate(pred_doc_ids, start=1):
            topk_rows.append(
                {
                    "query_index": int(query_index),
                    "query_id": str(query_id),
                    "rank": int(rank),
                    "doc_id": str(doc_id),
                    "is_qrels_relevant": bool(str(doc_id) in positive_doc_ids),
                    "first_relevant_rank": int(first_rank) if first_rank is not None else None,
                    "exact_topk_overlap_count": float(exact_overlap_count),
                }
            )

    write_jsonl(rows_out_path, per_query_rows)
    _write_topk_tsv(topk_tsv_path, topk_rows)

    latency_total = _maybe_float(args.latency_total_sec_avg)
    latency_client = _maybe_float(args.latency_client_generate_sec_avg)
    latency_server = _maybe_float(args.latency_server_query_sec_avg)
    latency_recover = _maybe_float(args.latency_client_recover_sec_avg)
    comm_request_bytes = _maybe_float(args.comm_request_bytes_avg)
    comm_response_bytes = _maybe_float(args.comm_response_bytes_avg)
    comm_downstream_bytes = _maybe_float(args.comm_downstream_bytes_avg)
    comm_client_generate_mb = (comm_request_bytes / _MB) if comm_request_bytes is not None else None
    comm_server_query_mb = (comm_response_bytes / _MB) if comm_response_bytes is not None else None
    comm_two_stage_total_mb = None
    if any(x is not None for x in (comm_request_bytes, comm_response_bytes)):
        comm_two_stage_total_mb = float(
            (comm_request_bytes or 0.0) + (comm_response_bytes or 0.0)
        ) / _MB

    notes = [
        "Panther MS summary was reconstructed from external top-k rankings against the repo's MS qrels-aligned bundle.",
        f"ranking_input={str(ranking_input_path)}",
        f"cluster_info_selector_c={int(routing_c)}",
        "latency_total_sec_avg must be a client-observed end-to-end latency if provided; the bridge does not synthesize it from partial stage timings.",
        "Paper-faithful compare-compatible metrics are exposed via avg_exact_recall_at_k and comm_two_stage_total_mb; qrels hit-style fields are retained only for debugging.",
    ]
    for note in list(args.note or []):
        if str(note).strip():
            notes.append(str(note).strip())

    summary = {
        "baseline_slug": "panther",
        "baseline_display_name": "Panther",
        "paper_url": str(cfg.paper_url),
        "code_url": str(cfg.code_url),
        "mode": "ms_aligned_author_code",
        "comparison_axis": "qrels_bundle",
        "run_label": str(args.run_label).strip() or str(summary_stem).replace("_summary", ""),
        "dataset": "ms",
        "num_docs": int(len(runtime["doc_ids"])),
        "num_clusters": int(cluster_info.get("num_clusters", 0)) if cluster_info is not None else None,
        "num_queries": int(len(per_query_rows)),
        "top_k": int(top_k),
        "cost_reporting_mode": "per_query_avg",
        "latency_total_sec_avg": latency_total,
        "latency_client_generate_sec_avg": latency_client,
        "latency_server_query_sec_avg": latency_server,
        "latency_client_recover_sec_avg": latency_recover,
        "comm_request_bytes_avg": comm_request_bytes,
        "comm_response_bytes_avg": comm_response_bytes,
        "comm_downstream_bytes_avg": comm_downstream_bytes,
        "comm_client_generate_query_mb": comm_client_generate_mb,
        "comm_server_query_mb": comm_server_query_mb,
        "comm_two_stage_total_mb": comm_two_stage_total_mb,
        "mean_first_relevant_rank": float(np.mean(np.asarray(first_ranks, dtype=np.float64))) if first_ranks else None,
        "top1_hit_rate": float(np.mean(np.asarray(top1_hits, dtype=np.float64))) if top1_hits else 0.0,
        "strict_hit_rate_at_1": float(np.mean(np.asarray(top1_hits, dtype=np.float64))) if top1_hits else 0.0,
        "candidate_cover_rate": float(np.mean(np.asarray(candidate_cover_flags, dtype=np.float64))) if candidate_cover_flags else 0.0,
        "exact_topk_overlap_mean": float(np.mean(np.asarray(exact_overlap_counts, dtype=np.float64))) if exact_overlap_counts else 0.0,
        "avg_exact_recall_at_k": float(np.mean(np.asarray(exact_recall_at_k_list, dtype=np.float64))) if exact_recall_at_k_list else 0.0,
        "candidate_overlap_mean": float(np.mean(np.asarray(exact_overlap_counts, dtype=np.float64))) if exact_overlap_counts else 0.0,
        "exact_topk_order_match_rate": float(np.mean(np.asarray(exact_order_match_flags, dtype=np.float64))) if exact_order_match_flags else 0.0,
        "source_log": str(args.source_log).strip() or str(ranking_input_path),
        "source_rankings_path": str(ranking_input_path),
        "rows_jsonl": str(rows_out_path),
        "topk_tsv": str(topk_tsv_path),
        "notes": notes,
    }
    save_json(summary_path, summary)
    print(f"[saved] {rows_out_path}")
    print(f"[saved] {topk_tsv_path}")
    print(f"[saved] {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Panther MS author-code bridge assets and evaluate returned rankings.")
    parser.add_argument("--export-ms-assets", action="store_true", help="Export the MS qrels-aligned bundle into a Panther author-bridge package.")
    parser.add_argument("--build-summary-from-rankings", action="store_true", help="Evaluate an external Panther ranking output and write rows_jsonl + summary.json.")
    parser.add_argument("--write-openpanther-text", action="store_true", help="Also emit best-effort OpenPanther-style *.txt inputs next to the bridge manifest.")
    parser.add_argument("--stage-into-openpanther", action="store_true", help="Copy the generated OpenPanther-style txt files into external/OpenPanther/experimental/panther/dataset/ and refresh the bridge config header.")
    parser.add_argument("--dataset-slug", type=str, default="", help="Optional slug used under results/repro_workflows/panther/ms_author_bridge/.")
    parser.add_argument("--routing-c", type=int, default=0, help="Which cluster_info_c* snapshot to use when exporting or evaluating.")
    parser.add_argument("--query-limit", type=int, default=-1, help="Optional evaluation/export query limit. Defaults to PantherConfig.default_query_limit.")
    parser.add_argument("--top-k", type=int, default=0, help="Optional top-k override. Defaults to PantherConfig.default_top_k.")
    parser.add_argument("--max-points-per-cluster", type=int, default=0, help="Panther bridge partition size used when exporting a protocol-compatible ptoc matrix. Defaults to PantherConfig.openpanther_default_max_points_per_cluster.")
    parser.add_argument("--selected-clusters", type=int, default=0, help="Number of Panther bridge clusters retrieved in the first stage. Defaults to ceil(candidate_budget / max_points_per_cluster).")
    parser.add_argument("--summary-stem", type=str, default="", help="Output summary stem under results/repro_workflows/panther/.")
    parser.add_argument("--run-label", type=str, default="", help="Optional run label stored in the generated summary.")
    parser.add_argument("--rankings-jsonl", type=str, default="", help="Path to external ranking rows in JSONL format.")
    parser.add_argument("--rankings-json", type=str, default="", help="Path to external ranking rows in JSON format.")
    parser.add_argument("--rankings-tsv", type=str, default="", help="Path to external ranking rows in TSV format.")
    parser.add_argument("--rankings-npy", type=str, default="", help="Path to external ranking rows in NPY format.")
    parser.add_argument("--latency-total-sec-avg", type=float, default=None, help="Optional total latency override stored in the summary.")
    parser.add_argument("--latency-client-generate-sec-avg", type=float, default=None, help="Optional client-generate latency override.")
    parser.add_argument("--latency-server-query-sec-avg", type=float, default=None, help="Optional server-query latency override.")
    parser.add_argument("--latency-client-recover-sec-avg", type=float, default=None, help="Optional client-recover latency override.")
    parser.add_argument("--comm-request-bytes-avg", type=float, default=None, help="Optional request-bytes override stored in the summary.")
    parser.add_argument("--comm-response-bytes-avg", type=float, default=None, help="Optional response-bytes override stored in the summary.")
    parser.add_argument("--comm-downstream-bytes-avg", type=float, default=None, help="Optional downstream-bytes override stored in the summary.")
    parser.add_argument("--source-log", type=str, default="", help="Optional source log path stored in the summary.")
    parser.add_argument("--note", action="append", default=[], help="Extra note appended to the generated summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not bool(args.export_ms_assets) and not bool(args.build_summary_from_rankings):
        raise SystemExit("choose --export-ms-assets and/or --build-summary-from-rankings")
    project_root = Path(__file__).resolve().parents[3]
    cfg = PantherConfig(project_root=project_root)
    if bool(args.export_ms_assets):
        export_ms_assets(cfg, args)
    if bool(args.build_summary_from_rankings):
        build_summary_from_rankings(cfg, args)


if __name__ == "__main__":
    main()
