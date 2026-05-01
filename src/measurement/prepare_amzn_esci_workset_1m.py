from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.lib.format import open_memmap

from shared.e5_dual_encoder import E5DualEncoder
from shared.gpu_accel import topk_cosine_similarity_matrix as gpu_topk_cosine_similarity_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
PAPERFAITHFUL_SUFFIX = "_paperfaithful_mainline"
MODEL_NAME = "intfloat/e5-large-v2"
DEFAULT_WORKSET_NAME = "amzn_esci_us_workset_1000000"
DEFAULT_QUERY_TAG = "amzn_esci_us_natural_20260427"
DEFAULT_OUTPUT_REPORT = RESULTS_DIR / "prepare_amzn_esci_workset_1m_report_20260427.json"
TEXT_JOINER = "\n\n"
POSITIVE_LABELS = {"E", "S"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an Amazon ESCI US 1M workset that matches the existing paperfaithful "
            "mainline asset contract: docs/doc_ids/meta/corpus + queries/query_ids/gt_topk/qrels."
        )
    )
    parser.add_argument("--examples-parquet", required=True)
    parser.add_argument("--products-parquet", required=True)
    parser.add_argument("--sources-csv", default="")
    parser.add_argument("--workset-name", default=DEFAULT_WORKSET_NAME)
    parser.add_argument("--query-tag", default=DEFAULT_QUERY_TAG)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--locale", default="us")
    parser.add_argument("--num-docs", type=int, default=1_000_000)
    parser.add_argument("--num-eval-queries", type=int, default=128)
    parser.add_argument("--num-calib-queries", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--passage-encode-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--doc-batch-size", type=int, default=512)
    parser.add_argument("--score-doc-chunk", type=int, default=16384)
    parser.add_argument("--score-query-batch", type=int, default=32)
    parser.add_argument("--exact-k", type=int, default=1000)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--small-version-only", action="store_true", default=True)
    parser.add_argument("--disable-small-version-only", dest="small_version_only", action="store_false")
    parser.add_argument("--report-json", default=str(DEFAULT_OUTPUT_REPORT))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _with_suffix_before_ext(stem: Path, ext: str) -> Path:
    return Path(str(stem) + PAPERFAITHFUL_SUFFIX + ext)


def build_workset_paths(workset_name: str) -> dict[str, Path]:
    docs_stem = DATA_DIR / f"docs_{workset_name}"
    doc_ids_stem = DATA_DIR / f"doc_ids_{workset_name}"
    meta_stem = DATA_DIR / f"meta_{workset_name}"
    corpus_stem = RAW_DIR / f"corpus_{workset_name}"
    queries_stem = DATA_DIR / f"queries_{workset_name}"
    query_ids_stem = DATA_DIR / f"query_ids_{workset_name}"
    gt_topk_stem = DATA_DIR / f"gt_topk_{workset_name}"
    queries_meta_stem = DATA_DIR / f"queries_{workset_name}"
    split_meta_path = DATA_DIR / f"queries_{workset_name}_split{PAPERFAITHFUL_SUFFIX}.meta.json"
    calib_queries_stem = DATA_DIR / f"queries_{workset_name}_calibration"
    calib_query_ids_stem = DATA_DIR / f"query_ids_{workset_name}_calibration"
    strict_qrels_stem = RAW_DIR / f"qrels_{workset_name}"
    relaxed_qrels_stem = RAW_DIR / f"qrels_{workset_name}_relaxed"
    return {
        "docs": _with_suffix_before_ext(docs_stem, ".npy"),
        "doc_ids": _with_suffix_before_ext(doc_ids_stem, ".npy"),
        "meta": _with_suffix_before_ext(meta_stem, ".json"),
        "corpus": _with_suffix_before_ext(corpus_stem, ".jsonl"),
        "queries": _with_suffix_before_ext(queries_stem, ".npy"),
        "query_ids": _with_suffix_before_ext(query_ids_stem, ".npy"),
        "gt_topk": _with_suffix_before_ext(gt_topk_stem, ".npy"),
        "queries_meta": _with_suffix_before_ext(queries_meta_stem, ".meta.json"),
        "split_meta": split_meta_path,
        "calib_queries": _with_suffix_before_ext(calib_queries_stem, ".npy"),
        "calib_query_ids": _with_suffix_before_ext(calib_query_ids_stem, ".npy"),
        "strict_qrels": _with_suffix_before_ext(strict_qrels_stem, ".tsv"),
        "relaxed_qrels": _with_suffix_before_ext(relaxed_qrels_stem, ".tsv"),
        "queries_jsonl": _with_suffix_before_ext(RAW_DIR / f"queries_{workset_name}", ".jsonl"),
        "calib_queries_jsonl": RAW_DIR / f"queries_{workset_name}_calibration{PAPERFAITHFUL_SUFFIX}.jsonl",
    }


def _stable_hash_key(text: str, seed: int) -> str:
    payload = f"{seed}::{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_order_key(text: str, seed: int) -> tuple[str, str]:
    return (_stable_hash_key(text, seed), text)


def _ensure_parent(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text


def _compose_product_text(title: str, bullet_point: str, description: str) -> str:
    pieces = []
    if title:
        pieces.append(title)
    if bullet_point:
        pieces.append(bullet_point)
    if description:
        pieces.append(description)
    return TEXT_JOINER.join(piece for piece in pieces if piece).strip()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_tsv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query_id", "doc_id"])
        writer.writerows(rows)


def _require_pyarrow():
    try:
        import pyarrow.dataset as ds  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "pyarrow is required to read the official ESCI parquet files. "
            "Please install it first, e.g. `python -m pip install pyarrow`."
        ) from exc
    return ds


def _resolve_columns(names: list[str], *candidates: str) -> str:
    lowered = {str(name).lower(): str(name) for name in names}
    for candidate in candidates:
        chosen = lowered.get(str(candidate).lower())
        if chosen:
            return chosen
    raise RuntimeError(f"missing required column; candidates={candidates}, available={names}")


def _build_example_schema(example_names: list[str]) -> dict[str, str]:
    schema = list(example_names)
    return {
        "query": _resolve_columns(schema, "query", "query_text"),
        "product_id": _resolve_columns(schema, "product_id", "product_asin", "asin"),
        "locale": _resolve_columns(schema, "product_locale", "query_locale", "locale"),
        "label": _resolve_columns(schema, "esci_label", "label"),
        "small_version": _resolve_columns(schema, "small_version", "is_small_version"),
        "split": _resolve_columns(schema, "split", "data_split"),
    }


def _build_product_schema(product_names: list[str]) -> dict[str, str]:
    schema = list(product_names)
    return {
        "product_id": _resolve_columns(schema, "product_id", "product_asin", "asin"),
        "locale": _resolve_columns(schema, "product_locale", "locale"),
        "title": _resolve_columns(schema, "product_title", "title"),
        "bullet_point": _resolve_columns(schema, "product_bullet_point", "bullet_point", "bullets"),
        "description": _resolve_columns(schema, "product_description", "description"),
    }


def _iter_batches(dataset, columns: list[str], batch_size: int):
    scanner = dataset.scanner(columns=columns, batch_size=int(batch_size))
    for batch in scanner.to_batches():
        data = {name: batch.column(i).to_pylist() for i, name in enumerate(batch.schema.names)}
        size = int(batch.num_rows)
        for row_idx in range(size):
            yield {name: data[name][row_idx] for name in batch.schema.names}


def scan_examples_for_support(
    *,
    ds,
    examples_path: Path,
    locale: str,
    seed: int,
    small_version_only: bool,
) -> tuple[dict[str, int], dict[str, dict], dict]:
    dataset = ds.dataset(str(examples_path), format="parquet")
    schema = _build_example_schema(dataset.schema.names)
    product_support: dict[str, int] = {}
    query_stats: dict[str, dict] = {}
    rows_scanned = 0
    locale_rows = 0
    positive_rows = 0
    for row in _iter_batches(
        dataset,
        [
            schema["query"],
            schema["product_id"],
            schema["locale"],
            schema["label"],
            schema["small_version"],
            schema["split"],
        ],
        batch_size=4096,
    ):
        rows_scanned += 1
        row_locale = _clean_text(row[schema["locale"]]).lower()
        if row_locale != locale:
            continue
        if small_version_only and str(row[schema["small_version"]]).strip() not in {"1", "true", "True"}:
            continue
        locale_rows += 1
        label = _clean_text(row[schema["label"]]).upper()
        product_id = _clean_text(row[schema["product_id"]])
        query_text = _clean_text(row[schema["query"]])
        if not product_id or not query_text:
            continue
        stat = query_stats.setdefault(
            query_text,
            {
                "query_text": query_text,
                "total_rows": 0,
                "positive_rows": 0,
                "positive_product_ids": set(),
                "splits": set(),
            },
        )
        stat["total_rows"] += 1
        split_value = _clean_text(row[schema["split"]]).lower()
        if split_value:
            stat["splits"].add(split_value)
        if label in POSITIVE_LABELS:
            positive_rows += 1
            product_support[product_id] = int(product_support.get(product_id, 0)) + 1
            stat["positive_rows"] += 1
            stat["positive_product_ids"].add(product_id)
    finalized_query_stats: dict[str, dict] = {}
    for query_text, stat in query_stats.items():
        finalized_query_stats[query_text] = {
            "query_text": str(query_text),
            "total_rows": int(stat["total_rows"]),
            "positive_rows": int(stat["positive_rows"]),
            "positive_product_ids": sorted(str(x) for x in stat["positive_product_ids"]),
            "splits": sorted(str(x) for x in stat["splits"]),
            "stable_key": _stable_hash_key(str(query_text), int(seed)),
        }
    summary = {
        "rows_scanned": int(rows_scanned),
        "rows_after_locale_and_small_version": int(locale_rows),
        "positive_rows_after_filter": int(positive_rows),
        "supported_product_count": int(len(product_support)),
        "unique_query_text_count": int(len(finalized_query_stats)),
    }
    return product_support, finalized_query_stats, summary


def _collect_us_fill_product_ids(
    *,
    ds,
    products_path: Path,
    locale: str,
    excluded_product_ids: set[str],
) -> list[str]:
    dataset = ds.dataset(str(products_path), format="parquet")
    schema = _build_product_schema(dataset.schema.names)
    extra_ids: set[str] = set()
    for row in _iter_batches(
        dataset,
        [schema["product_id"], schema["locale"]],
        batch_size=8192,
    ):
        row_locale = _clean_text(row[schema["locale"]]).lower()
        if row_locale != locale:
            continue
        product_id = _clean_text(row[schema["product_id"]])
        if not product_id or product_id in excluded_product_ids:
            continue
        extra_ids.add(product_id)
    return sorted(str(x) for x in extra_ids)


def select_product_ids(
    *,
    product_support: dict[str, int],
    num_docs: int,
    ds,
    products_path: Path,
    locale: str,
) -> tuple[list[str], dict]:
    ranked = sorted(
        ((str(pid), int(count)) for pid, count in product_support.items() if int(count) > 0),
        key=lambda item: (-int(item[1]), str(item[0])),
    )
    positive_ids = [str(pid) for pid, _count in ranked]
    if len(positive_ids) >= int(num_docs):
        return (
            positive_ids[: int(num_docs)],
            {
                "positive_supported_product_count": int(len(positive_ids)),
                "fill_from_products_count": 0,
            },
        )
    need_fill = int(num_docs) - int(len(positive_ids))
    extra_ids = _collect_us_fill_product_ids(
        ds=ds,
        products_path=products_path,
        locale=locale,
        excluded_product_ids=set(positive_ids),
    )
    if len(extra_ids) < int(need_fill):
        raise RuntimeError(
            "not enough remaining US product ids in ESCI products parquet to build the requested corpus: "
            f"need_fill={int(need_fill)} available_fill={int(len(extra_ids))}"
        )
    selected_ids = list(positive_ids) + [str(x) for x in extra_ids[: int(need_fill)]]
    return (
        selected_ids,
        {
            "positive_supported_product_count": int(len(positive_ids)),
            "fill_from_products_count": int(need_fill),
        },
    )


def materialize_docs(
    *,
    ds,
    products_path: Path,
    locale: str,
    selected_product_ids: list[str],
    paths: dict[str, Path],
    batch_size: int,
    encoder: E5DualEncoder | None,
    doc_batch_size: int,
    passage_encode_batch_size: int,
    max_length: int,
    dry_run: bool,
) -> dict:
    dataset = ds.dataset(str(products_path), format="parquet")
    schema = _build_product_schema(dataset.schema.names)
    selected_set = set(str(x) for x in selected_product_ids)
    seen: set[str] = set()
    doc_rows: list[dict] = []
    docs_memmap = None
    write_index = 0
    doc_ids: list[str] = []
    product_title_nonempty = 0
    bullet_nonempty = 0
    description_nonempty = 0
    if not dry_run:
        _ensure_parent([paths["docs"], paths["doc_ids"], paths["corpus"]])
        docs_memmap = open_memmap(
            paths["docs"],
            mode="w+",
            dtype=np.float32,
            shape=(int(len(selected_product_ids)), 1024),
        )
        corpus_f = open(paths["corpus"], "w", encoding="utf-8")
    else:
        corpus_f = None

    try:
        for row in _iter_batches(
            dataset,
            [
                schema["product_id"],
                schema["locale"],
                schema["title"],
                schema["bullet_point"],
                schema["description"],
            ],
            batch_size=int(batch_size),
        ):
            row_locale = _clean_text(row[schema["locale"]]).lower()
            if row_locale != locale:
                continue
            product_id = _clean_text(row[schema["product_id"]])
            if product_id not in selected_set or product_id in seen:
                continue
            title = _clean_text(row[schema["title"]])
            bullet_point = _clean_text(row[schema["bullet_point"]])
            description = _clean_text(row[schema["description"]])
            text = _compose_product_text(title, bullet_point, description)
            if not text:
                continue
            seen.add(product_id)
            product_title_nonempty += int(bool(title))
            bullet_nonempty += int(bool(bullet_point))
            description_nonempty += int(bool(description))
            doc_id = f"amzn_us_{product_id}"
            doc_rows.append(
                {
                    "doc_id": str(doc_id),
                    "source_doc_id": str(product_id),
                    "product_id": str(product_id),
                    "locale": str(locale),
                    "text": str(text),
                }
            )
            if len(doc_rows) < int(doc_batch_size):
                continue
            if not dry_run:
                raw, normalized = encoder.encode_passages(
                    [row["text"] for row in doc_rows],
                    batch_size=int(max(1, min(int(passage_encode_batch_size), len(doc_rows)))),
                    max_length=int(max_length),
                    progress_name=None,
                )
                del raw
                end = write_index + int(normalized.shape[0])
                docs_memmap[write_index:end] = normalized
                for local_offset, doc_row in enumerate(doc_rows):
                    doc_ids.append(str(doc_row["doc_id"]))
                    corpus_f.write(json.dumps(doc_row, ensure_ascii=False) + "\n")
                write_index = int(end)
            else:
                for doc_row in doc_rows:
                    doc_ids.append(str(doc_row["doc_id"]))
                write_index += int(len(doc_rows))
            doc_rows = []
        if doc_rows:
            if not dry_run:
                raw, normalized = encoder.encode_passages(
                    [row["text"] for row in doc_rows],
                    batch_size=int(max(1, min(int(passage_encode_batch_size), len(doc_rows)))),
                    max_length=int(max_length),
                    progress_name=None,
                )
                del raw
                end = write_index + int(normalized.shape[0])
                docs_memmap[write_index:end] = normalized
                for doc_row in doc_rows:
                    doc_ids.append(str(doc_row["doc_id"]))
                    corpus_f.write(json.dumps(doc_row, ensure_ascii=False) + "\n")
                write_index = int(end)
            else:
                for doc_row in doc_rows:
                    doc_ids.append(str(doc_row["doc_id"]))
                write_index += int(len(doc_rows))
    finally:
        if corpus_f is not None:
            corpus_f.close()
        if docs_memmap is not None:
            del docs_memmap

    if int(write_index) != int(len(selected_product_ids)):
        raise RuntimeError(
            "product parquet did not materialize the full selected corpus: "
            f"materialized={int(write_index)} selected={int(len(selected_product_ids))}"
        )
    if not dry_run:
        np.save(paths["doc_ids"], np.asarray(doc_ids, dtype=object))
    return {
        "materialized_docs": int(write_index),
        "product_title_nonempty": int(product_title_nonempty),
        "bullet_nonempty": int(bullet_nonempty),
        "description_nonempty": int(description_nonempty),
    }


def select_queries(
    *,
    query_stats: dict[str, dict],
    selected_product_ids: list[str],
    num_eval: int,
    num_calib: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    selected_set = set(str(x) for x in selected_product_ids)
    eligible = []
    for stat in query_stats.values():
        positive_overlap = [pid for pid in stat["positive_product_ids"] if pid in selected_set]
        if not positive_overlap:
            continue
        eligible.append(
            {
                "query_text": str(stat["query_text"]),
                "positive_overlap_count": int(len(positive_overlap)),
                "total_rows": int(stat["total_rows"]),
                "positive_rows": int(stat["positive_rows"]),
                "positive_product_ids": positive_overlap,
                "splits": list(stat["splits"]),
                "stable_key": str(stat["stable_key"]),
            }
        )
    eligible.sort(
        key=lambda row: (
            -int(row["positive_overlap_count"]),
            -int(row["positive_rows"]),
            -int(row["total_rows"]),
            _stable_order_key(str(row["query_text"]), int(seed)),
        )
    )
    needed = int(num_eval) + int(num_calib)
    if len(eligible) < needed:
        raise RuntimeError(f"not enough eligible queries after selected-product overlap filter: {len(eligible)} < {needed}")
    chosen = eligible[:needed]
    calib = []
    eval_rows = []
    for idx, row in enumerate(chosen):
        query_id = f"amzn_us_q{idx + 1:06d}"
        packed = {
            "query_id": str(query_id),
            "raw_query_id": str(query_id),
            "text": str(row["query_text"]),
            "source_query_id": str(query_id),
            "query_source_family": "real",
            "query_source_detail": "amzn_esci_us_natural",
            "positive_overlap_count": int(row["positive_overlap_count"]),
            "positive_rows": int(row["positive_rows"]),
            "total_rows": int(row["total_rows"]),
            "splits": list(row["splits"]),
            "reference_mode": "exact_embedding_topk",
        }
        if idx < int(num_calib):
            packed["split_role"] = "calibration"
            packed["split_rank"] = int(idx + 1)
            calib.append(packed)
        else:
            packed["split_role"] = "evaluation"
            packed["split_rank"] = int(idx + 1 - int(num_calib))
            eval_rows.append(packed)
    summary = {
        "eligible_query_count": int(len(eligible)),
        "selected_query_count": int(len(chosen)),
        "selected_calibration_queries": int(len(calib)),
        "selected_evaluation_queries": int(len(eval_rows)),
    }
    return calib, eval_rows, summary


def encode_queries(
    *,
    encoder: E5DualEncoder,
    rows: list[dict],
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.zeros((0, 1024), dtype=np.float32), np.asarray([], dtype=object)
    raw, normalized = encoder.encode_queries(
        [str(row["text"]) for row in rows],
        batch_size=int(batch_size),
        max_length=int(max_length),
        progress_name=None,
    )
    del raw
    ids = np.asarray([str(row["query_id"]) for row in rows], dtype=object)
    return normalized, ids


def _merge_topk(
    current_scores: np.ndarray,
    current_indices: np.ndarray,
    new_scores: np.ndarray,
    new_indices: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    merged_scores = np.concatenate([current_scores, new_scores], axis=1)
    merged_indices = np.concatenate([current_indices, new_indices], axis=1)
    keep = np.argpartition(merged_scores, kth=-int(k), axis=1)[:, -int(k):]
    kept_scores = np.take_along_axis(merged_scores, keep, axis=1)
    kept_indices = np.take_along_axis(merged_indices, keep, axis=1)
    order = np.argsort(-kept_scores, axis=1)
    return (
        np.take_along_axis(kept_scores, order, axis=1),
        np.take_along_axis(kept_indices, order, axis=1),
    )


def compute_exact_topk(
    *,
    docs_path: Path,
    query_emb: np.ndarray,
    exact_k: int,
    score_doc_chunk: int,
    score_query_batch: int,
) -> np.ndarray:
    docs = np.load(docs_path, mmap_mode="r")
    if int(query_emb.shape[0]) == 0:
        return np.zeros((0, int(exact_k)), dtype=np.int32)

    gpu_scores, gpu_indices = gpu_topk_cosine_similarity_matrix(
        np.asarray(query_emb, dtype=np.float32),
        docs,
        top_k=int(exact_k),
        cache_x_if_large=False,
        cache_y_if_large=True,
        y_chunk_rows=int(score_doc_chunk),
    )
    if gpu_indices is not None:
        return np.asarray(gpu_indices, dtype=np.int32)

    best_scores = np.full((int(query_emb.shape[0]), int(exact_k)), -np.inf, dtype=np.float32)
    best_indices = np.full((int(query_emb.shape[0]), int(exact_k)), -1, dtype=np.int32)
    for doc_start in range(0, int(docs.shape[0]), int(score_doc_chunk)):
        doc_end = min(int(docs.shape[0]), doc_start + int(score_doc_chunk))
        doc_block = np.asarray(docs[doc_start:doc_end], dtype=np.float32)
        for query_start in range(0, int(query_emb.shape[0]), int(score_query_batch)):
            query_end = min(int(query_emb.shape[0]), query_start + int(score_query_batch))
            query_block = np.asarray(query_emb[query_start:query_end], dtype=np.float32)
            scores = query_block @ doc_block.T
            local_k = int(min(int(exact_k), scores.shape[1]))
            local_pick = np.argpartition(scores, kth=-local_k, axis=1)[:, -local_k:]
            local_scores = np.take_along_axis(scores, local_pick, axis=1)
            local_indices = local_pick.astype(np.int32) + int(doc_start)
            merged_scores, merged_indices = _merge_topk(
                best_scores[query_start:query_end],
                best_indices[query_start:query_end],
                local_scores.astype(np.float32),
                local_indices.astype(np.int32),
                int(exact_k),
            )
            best_scores[query_start:query_end] = merged_scores
            best_indices[query_start:query_end] = merged_indices
    return best_indices.astype(np.int32)


def write_query_artifacts(
    *,
    paths: dict[str, Path],
    calib_rows: list[dict],
    eval_rows: list[dict],
    calib_emb: np.ndarray,
    eval_emb: np.ndarray,
    calib_ids: np.ndarray,
    eval_ids: np.ndarray,
    exact_topk_eval: np.ndarray,
    eval_k: int,
    exact_k: int,
    query_tag: str,
    workset_name: str,
    query_summary: dict,
    example_summary: dict,
    doc_summary: dict,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    np.save(paths["calib_queries"], np.asarray(calib_emb, dtype=np.float32))
    np.save(paths["calib_query_ids"], np.asarray(calib_ids, dtype=object))
    np.save(paths["queries"], np.asarray(eval_emb, dtype=np.float32))
    np.save(paths["query_ids"], np.asarray(eval_ids, dtype=object))
    np.save(paths["gt_topk"], np.asarray(exact_topk_eval[:, : int(eval_k)], dtype=np.int32))

    _ensure_parent(
        [
            paths["queries_jsonl"],
            paths["calib_queries_jsonl"],
            paths["queries_meta"],
            paths["split_meta"],
            paths["strict_qrels"],
            paths["relaxed_qrels"],
        ]
    )
    with open(paths["calib_queries_jsonl"], "w", encoding="utf-8") as f:
        for row in calib_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(paths["queries_jsonl"], "w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    doc_ids = [str(x) for x in np.load(paths["doc_ids"], allow_pickle=True).tolist()]
    strict_rows: list[tuple[str, str]] = []
    relaxed_rows: list[tuple[str, str]] = []
    for query_row, topk_indices in zip(eval_rows, exact_topk_eval.tolist()):
        query_id = str(query_row["query_id"])
        for local_idx in topk_indices[: int(eval_k)]:
            strict_rows.append((query_id, str(doc_ids[int(local_idx)])))
        for local_idx in topk_indices[: int(exact_k)]:
            relaxed_rows.append((query_id, str(doc_ids[int(local_idx)])))
    _write_tsv(paths["strict_qrels"], strict_rows)
    _write_tsv(paths["relaxed_qrels"], relaxed_rows)

    _write_json(
        paths["queries_meta"],
        {
            "workset_name": str(workset_name),
            "query_tag": str(query_tag),
            "selection_policy": "amzn_esci_us_overlap_rank_then_exact_embedding_topk",
            "query_evaluation_track1_only": False,
            "query_fixed_bundle_mode": True,
            "query_fixed_bundle_label": str(query_tag),
            "query_random_bundle_mode": False,
            "reference_mode": "exact_embedding_topk",
            "num_queries": int(len(eval_rows)),
            "query_ids": [str(row["query_id"]) for row in eval_rows],
            "raw_query_ids": [str(row["raw_query_id"]) for row in eval_rows],
            "num_calibration_pool_queries": int(len(calib_rows)),
            "num_evaluation_pool_queries": int(len(eval_rows)),
            "strict_qrels_pairs_selected": int(len(strict_rows)),
            "exact_k": int(exact_k),
            "eval_k": int(eval_k),
            "strict_qrels_path": str(paths["strict_qrels"]),
            "relaxed_qrels_path": str(paths["relaxed_qrels"]),
            "evaluation_queries_jsonl_path": str(paths["queries_jsonl"]),
            "calibration_queries_jsonl_path": str(paths["calib_queries_jsonl"]),
            "example_filter_summary": example_summary,
            "doc_materialization_summary": doc_summary,
            "query_selection_summary": query_summary,
        },
    )
    _write_json(
        paths["split_meta"],
        {
            "protocol_version": "amzn_esci_us_overlap_rank_then_exact_embedding_topk_v1",
            "num_queries_total_candidate_pool": int(query_summary["eligible_query_count"]),
            "num_queries_real_candidate_pool": int(query_summary["eligible_query_count"]),
            "num_queries_calibration": int(len(calib_rows)),
            "num_queries_evaluation": int(len(eval_rows)),
            "split_overlap_count": 0,
            "selection_summary": {
                "mode": "amzn_esci_us_overlap_rank_then_exact_embedding_topk_v1",
                "bundle_label": str(query_tag),
                "bundle_mode": "natural",
                "reference_mode": "exact_embedding_topk",
                "fixed_k": None,
                "eval_k": int(eval_k),
                "exact_k": int(exact_k),
            },
            "full_queries_jsonl_path": str(paths["queries_jsonl"]),
            "calibration_queries_jsonl_path": str(paths["calib_queries_jsonl"]),
            "evaluation_queries_jsonl_path": str(paths["queries_jsonl"]),
            "query_evaluation_track1_only": False,
            "query_fixed_bundle_mode": True,
            "query_fixed_bundle_label": str(query_tag),
            "query_random_bundle_mode": False,
        },
    )


def build_meta_payload(
    *,
    workset_name: str,
    locale: str,
    num_docs: int,
    query_tag: str,
    model_name: str,
    examples_path: Path,
    products_path: Path,
    doc_summary: dict,
    example_summary: dict,
    selected_product_ids: list[str],
    product_selection_summary: dict,
) -> dict:
    return {
        "pipeline": "prepare_amzn_esci_workset_1m",
        "workset_name": str(workset_name),
        "dataset_name": "amazon_science_esci",
        "locale": str(locale),
        "num_docs": int(num_docs),
        "embedding_model": str(model_name),
        "embedding_dim": 1024,
        "examples_parquet_path": str(examples_path),
        "products_parquet_path": str(products_path),
        "query_tag": str(query_tag),
        "selected_product_count": int(len(selected_product_ids)),
        "product_selection_summary": product_selection_summary,
        "doc_materialization_summary": doc_summary,
        "example_filter_summary": example_summary,
    }


def main() -> None:
    args = parse_args()
    examples_path = Path(args.examples_parquet).resolve()
    products_path = Path(args.products_parquet).resolve()
    report_path = Path(args.report_json).resolve()
    locale = str(args.locale).lower()
    workset_name = str(args.workset_name).strip()
    query_tag = str(args.query_tag).strip()
    if not workset_name:
        raise ValueError("workset-name must be non-empty")
    if not query_tag:
        raise ValueError("query-tag must be non-empty")
    if int(args.eval_k) <= 0 or int(args.exact_k) < int(args.eval_k):
        raise ValueError("exact-k must be >= eval-k and eval-k must be positive")

    ds = _require_pyarrow()
    product_support, query_stats, example_summary = scan_examples_for_support(
        ds=ds,
        examples_path=examples_path,
        locale=locale,
        seed=int(args.seed),
        small_version_only=bool(args.small_version_only),
    )
    selected_product_ids, product_selection_summary = select_product_ids(
        product_support=product_support,
        num_docs=int(args.num_docs),
        ds=ds,
        products_path=products_path,
        locale=locale,
    )
    paths = build_workset_paths(workset_name)

    if bool(args.dry_run):
        report = {
            "mode": "dry_run",
            "workset_name": str(workset_name),
            "query_tag": str(query_tag),
            "locale": str(locale),
            "example_summary": example_summary,
            "selected_product_count": int(len(selected_product_ids)),
            "product_selection_summary": product_selection_summary,
            "selected_product_preview": selected_product_ids[:20],
        }
        _write_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    encoder = E5DualEncoder(str(args.model_name), log_prefix="amzn-esci-e5")
    doc_summary = materialize_docs(
        ds=ds,
        products_path=products_path,
        locale=locale,
        selected_product_ids=selected_product_ids,
        paths=paths,
        batch_size=2048,
        encoder=encoder,
        doc_batch_size=int(args.doc_batch_size),
        passage_encode_batch_size=int(args.passage_encode_batch_size),
        max_length=int(args.max_length),
        dry_run=False,
    )
    calib_rows, eval_rows, query_summary = select_queries(
        query_stats=query_stats,
        selected_product_ids=selected_product_ids,
        num_eval=int(args.num_eval_queries),
        num_calib=int(args.num_calib_queries),
        seed=int(args.seed),
    )
    calib_emb, calib_ids = encode_queries(
        encoder=encoder,
        rows=calib_rows,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
    )
    eval_emb, eval_ids = encode_queries(
        encoder=encoder,
        rows=eval_rows,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
    )
    exact_topk_eval = compute_exact_topk(
        docs_path=paths["docs"],
        query_emb=eval_emb,
        exact_k=int(args.exact_k),
        score_doc_chunk=int(args.score_doc_chunk),
        score_query_batch=int(args.score_query_batch),
    )
    write_query_artifacts(
        paths=paths,
        calib_rows=calib_rows,
        eval_rows=eval_rows,
        calib_emb=calib_emb,
        eval_emb=eval_emb,
        calib_ids=calib_ids,
        eval_ids=eval_ids,
        exact_topk_eval=exact_topk_eval,
        eval_k=int(args.eval_k),
        exact_k=int(args.exact_k),
        query_tag=query_tag,
        workset_name=workset_name,
        query_summary=query_summary,
        example_summary=example_summary,
        doc_summary=doc_summary,
        dry_run=False,
    )
    meta_payload = build_meta_payload(
        workset_name=workset_name,
        locale=locale,
        num_docs=int(args.num_docs),
        query_tag=query_tag,
        model_name=str(args.model_name),
        examples_path=examples_path,
        products_path=products_path,
        doc_summary=doc_summary,
        example_summary=example_summary,
        selected_product_ids=selected_product_ids,
        product_selection_summary=product_selection_summary,
    )
    _write_json(paths["meta"], meta_payload)
    report = {
        "mode": "materialized",
        "workset_name": str(workset_name),
        "query_tag": str(query_tag),
        "paths": {key: str(value) for key, value in paths.items()},
        "meta": meta_payload,
        "product_selection_summary": product_selection_summary,
        "query_summary": query_summary,
        "example_summary": example_summary,
        "doc_summary": doc_summary,
    }
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
