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
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

from measurement.prepare_amzn_esci_workset_1m import (
    _write_json,
    build_workset_paths,
    compute_exact_topk,
)
from offline.cluster_offline_method4_balanced_spherical import train_step1_spherical_prototypes
from offline.cluster_method_utils import normalize_vec
from shared.e5_dual_encoder import E5DualEncoder

try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "datasets is required for FIQA-2018 preparation. "
        "Install project requirements first."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_DATASET_ID = "Hyukkyu/beir-fiqa-2018"
DEFAULT_REAL_WORKSET = "fiqa2018_workset_real"
DEFAULT_PARENT100K_WORKSET = "fiqa2018_workset_100000"
DEFAULT_SYNTH1M_WORKSET = "fiqa2018_workset_1000000"
DEFAULT_QUERY_TAG = "fiqa2018_beir_natural_20260428"
DEFAULT_REPORT_JSON = RESULTS_DIR / "prepare_fiqa2018_semantic_worksets_20260428.json"
DEFAULT_BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
QUERY_INSTRUCTION = "Given a finance question, retrieve relevant answers and passages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare FIQA-2018 semantic worksets under the repo's paperfaithful contract: "
            "load real BEIR FIQA docs/queries, encode the real corpus once, expand to a "
            "balanced 100k parent, then expand further to a cluster-ordered 1M workset."
        )
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--real-workset-name", default=DEFAULT_REAL_WORKSET)
    parser.add_argument("--parent100k-workset-name", default=DEFAULT_PARENT100K_WORKSET)
    parser.add_argument("--synth1m-workset-name", default=DEFAULT_SYNTH1M_WORKSET)
    parser.add_argument("--query-tag", default=DEFAULT_QUERY_TAG)
    parser.add_argument("--model-name", default="intfloat/e5-large-v2")
    parser.add_argument("--source-mode", choices=("auto", "hf", "beir_zip"), default="auto")
    parser.add_argument("--beir-url", default=DEFAULT_BEIR_URL)
    parser.add_argument("--num-clusters", type=int, default=100)
    parser.add_argument("--parent-cluster-size", type=int, default=1000)
    parser.add_argument("--synth1m-cluster-size", type=int, default=10000)
    parser.add_argument("--num-eval-queries", type=int, default=128)
    parser.add_argument("--num-calib-queries", type=int, default=128)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--passage-batch-size", type=int, default=8)
    parser.add_argument("--doc-write-batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--neighbor-k", type=int, default=32)
    parser.add_argument("--exact-k", type=int, default=1000)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--score-doc-chunk", type=int, default=16384)
    parser.add_argument("--score-query-batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--step1-max-iter", type=int, default=60)
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    return parser.parse_args()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text


def _stable_hash(text: str, seed: int) -> str:
    payload = f"{int(seed)}::{str(text)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (arr / norms).astype(np.float32, copy=False)


def _compose_fiqa_doc_text(row: dict[str, Any]) -> str:
    title = _clean_text(row.get("title", row.get("doc_title", "")))
    text = _clean_text(row.get("text", row.get("contents", row.get("doc_text", ""))))
    if title and text:
        return f"{title}\n\n{text}"
    if title:
        return title
    return text


def _corpus_row_to_doc_id(row: dict[str, Any], idx: int) -> str:
    raw = row.get("_id", row.get("id", row.get("doc_id", f"fiqa_doc_{idx}")))
    return str(raw)


def _query_row_to_query_id(row: dict[str, Any], idx: int) -> str:
    raw = row.get("_id", row.get("id", row.get("query_id", f"fiqa_q_{idx}")))
    return str(raw)


def _load_hf_rows(dataset_id: str, *, config_name: str, split_name: str) -> list[dict[str, Any]]:
    attempts = [
        {"path": dataset_id, "name": config_name, "split": split_name},
        {"path": dataset_id, "name": config_name, "split": "train"},
        {"path": dataset_id, "split": split_name},
    ]
    last_exc: Exception | None = None
    for kwargs in attempts:
        try:
            ds = load_dataset(**kwargs)
            return [dict(row) for row in ds]
        except Exception as exc:  # pragma: no cover - exercised only with remote data variations
            last_exc = exc
    raise RuntimeError(
        f"failed to load HF dataset split path={dataset_id} config={config_name} split={split_name}"
    ) from last_exc


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(str(url), timeout=60) as response:
        with tempfile.NamedTemporaryFile(delete=False, dir=str(dst.parent), suffix=".part") as tmp:
            shutil.copyfileobj(response, tmp)
            tmp_path = Path(tmp.name)
    tmp_path.replace(dst)


def _ensure_beir_fiqa_dir(beir_url: str) -> Path:
    cache_dir = RAW_DIR / "fiqa2018_beir"
    archive_path = cache_dir / "fiqa.zip"
    extract_dir = cache_dir / "extracted"
    if not archive_path.exists():
        print(f"[fiqa2018] downloading BEIR fiqa.zip from {str(beir_url)}", flush=True)
        _download_file(str(beir_url), archive_path)
    if not extract_dir.exists():
        print(f"[fiqa2018] extracting {str(archive_path)}", flush=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)

    candidates = [extract_dir] + [path for path in extract_dir.rglob("*") if path.is_dir()]
    for candidate in candidates:
        if (candidate / "corpus.jsonl").exists() and (candidate / "queries.jsonl").exists():
            return candidate
    raise RuntimeError(f"unable to locate corpus.jsonl and queries.jsonl under extracted BEIR dir: {extract_dir}")


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(dict(json.loads(text)))
    return rows


def _load_beir_qrels_query_ids(base_dir: Path) -> set[str]:
    qrels_path = base_dir / "qrels" / "test.tsv"
    if not qrels_path.exists():
        return set()
    with qrels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {str(row["query-id"]) for row in reader if row.get("query-id")}


def _load_beir_rows(beir_url: str, *, config_name: str) -> list[dict[str, Any]]:
    base_dir = _ensure_beir_fiqa_dir(str(beir_url))
    if str(config_name) == "corpus":
        return _load_jsonl_rows(base_dir / "corpus.jsonl")
    if str(config_name) == "queries":
        rows = _load_jsonl_rows(base_dir / "queries.jsonl")
        allowed_query_ids = _load_beir_qrels_query_ids(base_dir)
        if allowed_query_ids:
            rows = [row for row in rows if str(_query_row_to_query_id(row, 0)) in allowed_query_ids]
        return rows
    raise ValueError(f"unsupported BEIR config_name: {config_name}")


def _load_source_rows(
    dataset_id: str,
    *,
    config_name: str,
    split_name: str,
    source_mode: str,
    beir_url: str,
) -> list[dict[str, Any]]:
    if str(source_mode) == "hf":
        print(f"[fiqa2018] loading HF rows config={config_name} split={split_name}", flush=True)
        return _load_hf_rows(str(dataset_id), config_name=str(config_name), split_name=str(split_name))
    if str(source_mode) == "beir_zip":
        print(f"[fiqa2018] loading BEIR zip rows config={config_name}", flush=True)
        return _load_beir_rows(str(beir_url), config_name=str(config_name))

    try:
        print(f"[fiqa2018] loading HF rows config={config_name} split={split_name}", flush=True)
        return _load_hf_rows(str(dataset_id), config_name=str(config_name), split_name=str(split_name))
    except Exception as exc:
        print(
            f"[fiqa2018] HF load failed for config={config_name} split={split_name}; "
            f"falling back to BEIR zip: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return _load_beir_rows(str(beir_url), config_name=str(config_name))


def _select_queries(
    query_rows: list[dict[str, Any]],
    *,
    num_calib: int,
    num_eval: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ranked = []
    for idx, row in enumerate(query_rows):
        query_id = _query_row_to_query_id(row, idx)
        text = _clean_text(row.get("text", row.get("query", "")))
        if not query_id or not text:
            continue
        ranked.append(
            {
                "query_id": str(query_id),
                "text": str(text),
                "stable_key": _stable_hash(str(query_id), int(seed)),
            }
        )
    ranked.sort(key=lambda row: (row["stable_key"], row["query_id"]))
    need_total = int(num_calib) + int(num_eval)
    if len(ranked) < need_total:
        raise RuntimeError(f"not enough FIQA queries: have={len(ranked)} need={need_total}")

    selected = ranked[:need_total]
    calib_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        packed = {
            "query_id": str(row["query_id"]),
            "raw_query_id": str(row["query_id"]),
            "source_query_id": str(row["query_id"]),
            "text": str(row["text"]),
            "query_source_family": "beir",
            "query_source_detail": "fiqa_2018_original",
            "reference_mode": "exact_embedding_topk",
        }
        if idx < int(num_calib):
            packed["split_role"] = "calibration"
            packed["split_rank"] = int(idx + 1)
            calib_rows.append(packed)
        else:
            packed["split_role"] = "evaluation"
            packed["split_rank"] = int(idx + 1 - int(num_calib))
            eval_rows.append(packed)
    return (
        calib_rows,
        eval_rows,
        {
            "eligible_query_count": int(len(ranked)),
            "selected_query_count": int(len(selected)),
            "selected_calibration_queries": int(len(calib_rows)),
            "selected_evaluation_queries": int(len(eval_rows)),
            "selection_policy": "stable_hash_first_n",
        },
    )


def _encode_query_rows(
    encoder: E5DualEncoder,
    rows: list[dict[str, Any]],
    *,
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
    return np.asarray(normalized, dtype=np.float32), ids


def _build_neighbor_lists(cluster_docs: np.ndarray, neighbor_k: int) -> np.ndarray:
    if int(cluster_docs.shape[0]) <= 1:
        return np.zeros((int(cluster_docs.shape[0]), 1), dtype=np.int32)
    sims = np.asarray(cluster_docs @ cluster_docs.T, dtype=np.float32)
    np.fill_diagonal(sims, -np.inf)
    order = np.argsort(-sims, axis=1).astype(np.int32)
    topk = int(min(max(1, int(neighbor_k)), int(cluster_docs.shape[0] - 1)))
    return order[:, :topk]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_query_artifacts_generic(
    *,
    paths: dict[str, Path],
    calib_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    calib_emb: np.ndarray,
    eval_emb: np.ndarray,
    calib_ids: np.ndarray,
    eval_ids: np.ndarray,
    exact_topk_eval: np.ndarray,
    eval_k: int,
    exact_k: int,
    query_tag: str,
    workset_name: str,
    query_summary: dict[str, Any],
    doc_summary: dict[str, Any],
    dataset_name: str,
) -> None:
    np.save(paths["calib_queries"], np.asarray(calib_emb, dtype=np.float32))
    np.save(paths["calib_query_ids"], np.asarray(calib_ids, dtype=object))
    np.save(paths["queries"], np.asarray(eval_emb, dtype=np.float32))
    np.save(paths["query_ids"], np.asarray(eval_ids, dtype=object))
    np.save(paths["gt_topk"], np.asarray(exact_topk_eval[:, : int(eval_k)], dtype=np.int32))

    _write_jsonl(paths["calib_queries_jsonl"], calib_rows)
    _write_jsonl(paths["queries_jsonl"], eval_rows)

    doc_ids = [str(x) for x in np.load(paths["doc_ids"], allow_pickle=True).tolist()]
    strict_rows: list[tuple[str, str]] = []
    relaxed_rows: list[tuple[str, str]] = []
    for query_row, topk_indices in zip(eval_rows, exact_topk_eval.tolist()):
        qid = str(query_row["query_id"])
        for local_idx in topk_indices[: int(eval_k)]:
            strict_rows.append((qid, str(doc_ids[int(local_idx)])))
        for local_idx in topk_indices[: int(exact_k)]:
            relaxed_rows.append((qid, str(doc_ids[int(local_idx)])))

    paths["strict_qrels"].parent.mkdir(parents=True, exist_ok=True)
    with paths["strict_qrels"].open("w", encoding="utf-8") as f:
        f.write("query_id\tdoc_id\n")
        for qid, did in strict_rows:
            f.write(f"{qid}\t{did}\n")
    with paths["relaxed_qrels"].open("w", encoding="utf-8") as f:
        f.write("query_id\tdoc_id\n")
        for qid, did in relaxed_rows:
            f.write(f"{qid}\t{did}\n")

    _write_json(
        paths["queries_meta"],
        {
            "workset_name": str(workset_name),
            "dataset_name": str(dataset_name),
            "query_tag": str(query_tag),
            "selection_policy": "fiqa2018_beir_exact_embedding_topk",
            "reference_mode": "exact_embedding_topk",
            "num_queries": int(len(eval_rows)),
            "num_calibration_pool_queries": int(len(calib_rows)),
            "num_evaluation_pool_queries": int(len(eval_rows)),
            "strict_qrels_pairs_selected": int(len(strict_rows)),
            "exact_k": int(exact_k),
            "eval_k": int(eval_k),
            "strict_qrels_path": str(paths["strict_qrels"]),
            "relaxed_qrels_path": str(paths["relaxed_qrels"]),
            "evaluation_queries_jsonl_path": str(paths["queries_jsonl"]),
            "calibration_queries_jsonl_path": str(paths["calib_queries_jsonl"]),
            "query_selection_summary": query_summary,
            "doc_materialization_summary": doc_summary,
        },
    )
    _write_json(
        paths["split_meta"],
        {
            "protocol_version": "fiqa2018_beir_exact_embedding_topk_v1",
            "num_queries_total_candidate_pool": int(query_summary["eligible_query_count"]),
            "num_queries_real_candidate_pool": int(query_summary["eligible_query_count"]),
            "num_queries_calibration": int(len(calib_rows)),
            "num_queries_evaluation": int(len(eval_rows)),
            "selection_summary": {
                "mode": "fiqa2018_beir_exact_embedding_topk_v1",
                "bundle_label": str(query_tag),
                "bundle_mode": "natural",
                "reference_mode": "exact_embedding_topk",
                "eval_k": int(eval_k),
                "exact_k": int(exact_k),
            },
            "full_queries_jsonl_path": str(paths["queries_jsonl"]),
            "calibration_queries_jsonl_path": str(paths["calib_queries_jsonl"]),
            "evaluation_queries_jsonl_path": str(paths["queries_jsonl"]),
            "query_fixed_bundle_mode": True,
            "query_fixed_bundle_label": str(query_tag),
            "query_random_bundle_mode": False,
        },
    )


def _save_workset_docs_only(
    *,
    paths: dict[str, Path],
    docs: np.ndarray,
    doc_ids: list[str],
    corpus_rows: list[dict[str, Any]],
    meta_payload: dict[str, Any],
) -> None:
    paths["docs"].parent.mkdir(parents=True, exist_ok=True)
    paths["corpus"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["docs"], np.asarray(docs, dtype=np.float32))
    np.save(paths["doc_ids"], np.asarray(doc_ids, dtype=object))
    _write_jsonl(paths["corpus"], corpus_rows)
    _write_json(paths["meta"], meta_payload)


def _materialize_real_fiqa(
    *,
    dataset_id: str,
    source_mode: str,
    beir_url: str,
    encoder: E5DualEncoder,
    model_name: str,
    workset_name: str,
    query_tag: str,
    num_calib_queries: int,
    num_eval_queries: int,
    query_batch_size: int,
    passage_batch_size: int,
    doc_write_batch_size: int,
    max_length: int,
    exact_k: int,
    eval_k: int,
    score_doc_chunk: int,
    score_query_batch: int,
    seed: int,
) -> tuple[dict[str, Path], dict[str, Any]]:
    corpus_rows_raw = _load_source_rows(
        dataset_id,
        config_name="corpus",
        split_name="corpus",
        source_mode=str(source_mode),
        beir_url=str(beir_url),
    )
    query_rows_raw = _load_source_rows(
        dataset_id,
        config_name="queries",
        split_name="queries",
        source_mode=str(source_mode),
        beir_url=str(beir_url),
    )
    print(
        f"[fiqa2018] loaded rows corpus={int(len(corpus_rows_raw))} queries={int(len(query_rows_raw))}",
        flush=True,
    )

    workset_paths = build_workset_paths(str(workset_name))
    docs_mm = open_memmap(
        workset_paths["docs"],
        mode="w+",
        dtype=np.float32,
        shape=(int(len(corpus_rows_raw)), 1024),
    )
    doc_ids: list[str] = []
    corpus_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    write_index = 0

    for idx, row in enumerate(corpus_rows_raw):
        doc_id = _corpus_row_to_doc_id(row, idx)
        text = _compose_fiqa_doc_text(row)
        if not text:
            continue
        batch_rows.append(
            {
                "doc_id": str(doc_id),
                "source_doc_id": str(doc_id),
                "title": _clean_text(row.get("title", row.get("doc_title", ""))),
                "text": str(text),
            }
        )
        if len(batch_rows) < int(doc_write_batch_size):
            continue
        raw, normalized = encoder.encode_passages(
            [str(item["text"]) for item in batch_rows],
            batch_size=int(max(1, min(int(passage_batch_size), len(batch_rows)))),
            max_length=int(max_length),
            progress_name=None,
        )
        del raw
        end = int(write_index + normalized.shape[0])
        docs_mm[write_index:end] = np.asarray(normalized, dtype=np.float32)
        for item in batch_rows:
            doc_ids.append(str(item["doc_id"]))
            corpus_rows.append(item)
        write_index = int(end)
        batch_rows = []
    if batch_rows:
        raw, normalized = encoder.encode_passages(
            [str(item["text"]) for item in batch_rows],
            batch_size=int(max(1, min(int(passage_batch_size), len(batch_rows)))),
            max_length=int(max_length),
            progress_name=None,
        )
        del raw
        end = int(write_index + normalized.shape[0])
        docs_mm[write_index:end] = np.asarray(normalized, dtype=np.float32)
        for item in batch_rows:
            doc_ids.append(str(item["doc_id"]))
            corpus_rows.append(item)
        write_index = int(end)
    del docs_mm

    if int(write_index) != int(len(doc_ids)):
        raise RuntimeError("FIQA real workset docs/doc_ids write mismatch")
    real_docs = _normalize_rows(
        np.asarray(np.load(workset_paths["docs"], mmap_mode="r")[: int(write_index)], dtype=np.float32)
    )
    np.save(workset_paths["docs"], np.asarray(real_docs, dtype=np.float32))
    np.save(workset_paths["doc_ids"], np.asarray(doc_ids, dtype=object))
    _write_jsonl(workset_paths["corpus"], corpus_rows)

    calib_rows, eval_rows, query_summary = _select_queries(
        query_rows_raw,
        num_calib=int(num_calib_queries),
        num_eval=int(num_eval_queries),
        seed=int(seed),
    )
    calib_emb, calib_ids = _encode_query_rows(
        encoder,
        calib_rows,
        batch_size=int(query_batch_size),
        max_length=int(max_length),
    )
    eval_emb, eval_ids = _encode_query_rows(
        encoder,
        eval_rows,
        batch_size=int(query_batch_size),
        max_length=int(max_length),
    )
    exact_topk_eval = compute_exact_topk(
        docs_path=workset_paths["docs"],
        query_emb=eval_emb,
        exact_k=int(exact_k),
        score_doc_chunk=int(score_doc_chunk),
        score_query_batch=int(score_query_batch),
    )
    _write_query_artifacts_generic(
        paths=workset_paths,
        calib_rows=calib_rows,
        eval_rows=eval_rows,
        calib_emb=calib_emb,
        eval_emb=eval_emb,
        calib_ids=calib_ids,
        eval_ids=eval_ids,
        exact_topk_eval=exact_topk_eval,
        eval_k=int(eval_k),
        exact_k=int(exact_k),
        query_tag=str(query_tag),
        workset_name=str(workset_name),
        query_summary=query_summary,
        doc_summary={"materialized_docs": int(len(doc_ids)), "synthetic_docs": 0},
        dataset_name="fiqa_2018",
    )
    _write_json(
        workset_paths["meta"],
        {
            "pipeline": "prepare_fiqa_2018_semantic_worksets",
            "dataset_name": "fiqa_2018",
            "workset_name": str(workset_name),
            "num_docs": int(len(doc_ids)),
            "embedding_model": str(model_name),
            "embedding_dim": 1024,
            "query_tag": str(query_tag),
            "source_dataset_id": str(dataset_id),
            "doc_materialization_summary": {"materialized_docs": int(len(doc_ids)), "synthetic_docs": 0},
            "query_selection_summary": query_summary,
        },
    )
    return workset_paths, {
        "num_real_docs": int(len(doc_ids)),
        "num_queries_available": int(len(query_rows_raw)),
        "num_eval_queries": int(len(eval_rows)),
        "num_calib_queries": int(len(calib_rows)),
    }


def _build_balanced_parent100k(
    *,
    real_paths: dict[str, Path],
    output_workset_name: str,
    query_tag: str,
    num_clusters: int,
    target_cluster_size: int,
    neighbor_k: int,
    exact_k: int,
    eval_k: int,
    score_doc_chunk: int,
    score_query_batch: int,
    seed: int,
    step1_max_iter: int,
) -> tuple[dict[str, Path], dict[str, Any]]:
    parent_paths = build_workset_paths(str(output_workset_name))
    real_docs = _normalize_rows(np.asarray(np.load(real_paths["docs"]), dtype=np.float32))
    real_doc_ids = [str(x) for x in np.load(real_paths["doc_ids"], allow_pickle=True).tolist()]
    real_corpus_rows = []
    with real_paths["corpus"].open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                real_corpus_rows.append(json.loads(text))
    if len(real_doc_ids) != int(real_docs.shape[0]) or len(real_corpus_rows) != int(real_docs.shape[0]):
        raise RuntimeError("real FIQA docs/doc_ids/corpus size mismatch")

    step1 = train_step1_spherical_prototypes(
        docs=real_docs,
        num_clusters=int(num_clusters),
        rng_seed=int(seed),
        max_iter=int(step1_max_iter),
    )
    labels = np.asarray(step1["labels"], dtype=np.int32)
    centers = np.asarray(step1["prototypes"], dtype=np.float32)
    cluster_members = [np.where(labels == int(cid))[0].astype(np.int32) for cid in range(int(num_clusters))]
    if any(int(len(idx)) <= 0 for idx in cluster_members):
        raise RuntimeError("step1 clustering produced an empty FIQA cluster; aborting to avoid degenerate synthesis")

    total_docs = int(num_clusters) * int(target_cluster_size)
    docs_mm = open_memmap(
        parent_paths["docs"],
        mode="w+",
        dtype=np.float32,
        shape=(int(total_docs), int(real_docs.shape[1])),
    )
    out_doc_ids: list[str] = []
    out_corpus_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    cursor = 0
    overflow_dropped = 0

    for cid, member_idx in enumerate(cluster_members):
        member_docs = np.asarray(real_docs[member_idx], dtype=np.float32)
        center = normalize_vec(centers[int(cid)])
        sims = np.asarray(member_docs @ center.reshape(-1, 1), dtype=np.float32).reshape(-1)
        order = np.argsort(-sims).astype(np.int32)
        keep_real = member_idx[order[: int(min(len(order), target_cluster_size))]]
        overflow_dropped += int(max(0, len(order) - int(target_cluster_size)))

        cluster_docs = np.asarray(real_docs[keep_real], dtype=np.float32)
        cluster_doc_ids = [real_doc_ids[int(i)] for i in keep_real.tolist()]
        cluster_rows = [dict(real_corpus_rows[int(i)]) for i in keep_real.tolist()]
        neighbors = _build_neighbor_lists(cluster_docs, int(neighbor_k))

        for local_i, doc_row in enumerate(cluster_rows):
            docs_mm[int(cursor)] = cluster_docs[int(local_i)]
            out_doc_ids.append(str(cluster_doc_ids[int(local_i)]))
            out_corpus_rows.append(doc_row)
            cursor += 1

        synth_needed = int(target_cluster_size) - int(cluster_docs.shape[0])
        for synth_rank in range(int(synth_needed)):
            if int(cluster_docs.shape[0]) == 1:
                base_local = 0
                nbr_local = 0
                lam = 0.5
                candidate = np.asarray(cluster_docs[0], dtype=np.float32)
            else:
                base_local = int(rng.integers(0, int(cluster_docs.shape[0])))
                nbr_local = int(neighbors[base_local, int(rng.integers(0, neighbors.shape[1]))])
                lam = float(rng.uniform(0.10, 0.90))
                candidate = normalize_vec(
                    (1.0 - float(lam)) * cluster_docs[base_local] + float(lam) * cluster_docs[nbr_local]
                )
            base_row = dict(cluster_rows[int(base_local)])
            base_source = str(base_row.get("source_doc_id", cluster_doc_ids[int(base_local)]))
            synth_doc_id = f"fiqa_parent100k_c{int(cid):03d}_{int(synth_rank):05d}_{base_source}"
            base_row["doc_id"] = str(synth_doc_id)
            base_row["source_doc_id"] = str(base_source)
            base_row["text"] = (
                f"{str(base_row.get('text', ''))}\n\n"
                f"[synthetic_fiqa_parent100k cluster={int(cid)} lambda={float(lam):.4f}]"
            )
            base_row["synthetic_from_parent_workset"] = True
            base_row["synthetic_parent_doc_id"] = str(cluster_doc_ids[int(base_local)])
            base_row["synthetic_neighbor_doc_id"] = str(cluster_doc_ids[int(nbr_local)])
            base_row["synthetic_cluster_id"] = int(cid)
            base_row["synthetic_mix_lambda"] = float(lam)
            docs_mm[int(cursor)] = np.asarray(candidate, dtype=np.float32)
            out_doc_ids.append(str(synth_doc_id))
            out_corpus_rows.append(base_row)
            cursor += 1
    del docs_mm

    if int(cursor) != int(total_docs):
        raise RuntimeError(f"parent100k row count mismatch: wrote={cursor} expected={total_docs}")

    np.save(parent_paths["doc_ids"], np.asarray(out_doc_ids, dtype=object))
    _write_jsonl(parent_paths["corpus"], out_corpus_rows)

    calib_rows = []
    eval_rows = []
    with real_paths["calib_queries_jsonl"].open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                calib_rows.append(json.loads(text))
    with real_paths["queries_jsonl"].open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                eval_rows.append(json.loads(text))
    calib_emb = np.asarray(np.load(real_paths["calib_queries"]), dtype=np.float32)
    eval_emb = np.asarray(np.load(real_paths["queries"]), dtype=np.float32)
    calib_ids = np.asarray(np.load(real_paths["calib_query_ids"], allow_pickle=True), dtype=object)
    eval_ids = np.asarray(np.load(real_paths["query_ids"], allow_pickle=True), dtype=object)

    exact_topk_eval = compute_exact_topk(
        docs_path=parent_paths["docs"],
        query_emb=eval_emb,
        exact_k=int(exact_k),
        score_doc_chunk=int(score_doc_chunk),
        score_query_batch=int(score_query_batch),
    )
    real_query_meta = json.loads(real_paths["queries_meta"].read_text(encoding="utf-8"))
    query_summary = dict(real_query_meta.get("query_selection_summary") or {})
    query_summary["parent100k_reuses_real_query_bundle"] = True

    _write_query_artifacts_generic(
        paths=parent_paths,
        calib_rows=calib_rows,
        eval_rows=eval_rows,
        calib_emb=calib_emb,
        eval_emb=eval_emb,
        calib_ids=calib_ids,
        eval_ids=eval_ids,
        exact_topk_eval=exact_topk_eval,
        eval_k=int(eval_k),
        exact_k=int(exact_k),
        query_tag=str(query_tag),
        workset_name=str(output_workset_name),
        query_summary=query_summary,
        doc_summary={
            "materialized_docs": int(total_docs),
            "real_parent_docs": int(real_docs.shape[0]),
            "synthetic_docs": int(total_docs - int(real_docs.shape[0]) + overflow_dropped),
            "cluster_ordered_balanced_parent": True,
        },
        dataset_name="fiqa_2018",
    )
    _write_json(
        parent_paths["meta"],
        {
            "pipeline": "prepare_fiqa_2018_semantic_worksets",
            "dataset_name": "fiqa_2018",
            "workset_name": str(output_workset_name),
            "num_docs": int(total_docs),
            "embedding_dim": int(real_docs.shape[1]),
            "num_clusters": int(num_clusters),
            "target_cluster_size": int(target_cluster_size),
            "parent_workset_name": str(real_paths["meta"].stem),
            "synthetic_mode": "balanced_cluster_ordered_parent100k_from_real_fiqa",
            "real_docs_used": int(real_docs.shape[0] - overflow_dropped),
            "overflow_real_docs_dropped": int(overflow_dropped),
        },
    )
    return parent_paths, {
        "num_docs": int(total_docs),
        "overflow_real_docs_dropped": int(overflow_dropped),
    }


def _build_synth1m_from_parent100k(
    *,
    parent_paths: dict[str, Path],
    parent_workset_name: str,
    output_workset_name: str,
    query_tag: str,
    num_clusters: int,
    parent_cluster_size: int,
    synth_cluster_size: int,
    neighbor_k: int,
    exact_k: int,
    eval_k: int,
    score_doc_chunk: int,
    score_query_batch: int,
    seed: int,
) -> tuple[dict[str, Path], dict[str, Any]]:
    synth_paths = build_workset_paths(str(output_workset_name))
    parent_docs = _normalize_rows(np.asarray(np.load(parent_paths["docs"]), dtype=np.float32))
    parent_doc_ids = [str(x) for x in np.load(parent_paths["doc_ids"], allow_pickle=True).tolist()]
    parent_corpus_rows = []
    with parent_paths["corpus"].open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                parent_corpus_rows.append(json.loads(text))

    expected_parent = int(num_clusters) * int(parent_cluster_size)
    if int(parent_docs.shape[0]) != int(expected_parent):
        raise RuntimeError(
            f"parent100k size mismatch for synth1m: rows={int(parent_docs.shape[0])} expected={expected_parent}"
        )

    total_docs = int(num_clusters) * int(synth_cluster_size)
    docs_mm = open_memmap(
        synth_paths["docs"],
        mode="w+",
        dtype=np.float32,
        shape=(int(total_docs), int(parent_docs.shape[1])),
    )
    out_doc_ids: list[str] = []
    rng = np.random.default_rng(int(seed))
    cursor = 0

    for cid in range(int(num_clusters)):
        start = int(cid * int(parent_cluster_size))
        end = int(start + int(parent_cluster_size))
        cluster_docs = np.asarray(parent_docs[start:end], dtype=np.float32)
        cluster_doc_ids = parent_doc_ids[start:end]
        cluster_rows = [dict(row) for row in parent_corpus_rows[start:end]]
        neighbors = _build_neighbor_lists(cluster_docs, int(neighbor_k))

        for local_i, doc_row in enumerate(cluster_rows):
            docs_mm[int(cursor)] = cluster_docs[int(local_i)]
            out_doc_ids.append(str(cluster_doc_ids[int(local_i)]))
            cursor += 1

        synth_needed = int(synth_cluster_size) - int(parent_cluster_size)
        for synth_rank in range(int(synth_needed)):
            base_local = int(rng.integers(0, int(cluster_docs.shape[0])))
            nbr_local = int(neighbors[base_local, int(rng.integers(0, neighbors.shape[1]))])
            lam = float(rng.uniform(0.10, 0.90))
            candidate = normalize_vec(
                (1.0 - float(lam)) * cluster_docs[base_local] + float(lam) * cluster_docs[nbr_local]
            )
            base_row = dict(cluster_rows[int(base_local)])
            base_source = str(base_row.get("source_doc_id", cluster_doc_ids[int(base_local)]))
            synth_doc_id = f"fiqa_syn1m_c{int(cid):03d}_{int(synth_rank):05d}_{base_source}"
            base_row["doc_id"] = str(synth_doc_id)
            base_row["source_doc_id"] = str(base_source)
            base_row["text"] = (
                f"{str(base_row.get('text', ''))}\n\n"
                f"[synthetic_fiqa_1m cluster={int(cid)} lambda={float(lam):.4f}]"
            )
            base_row["synthetic_from_parent_workset"] = True
            base_row["synthetic_parent_doc_id"] = str(cluster_doc_ids[int(base_local)])
            base_row["synthetic_neighbor_doc_id"] = str(cluster_doc_ids[int(nbr_local)])
            base_row["synthetic_cluster_id"] = int(cid)
            base_row["synthetic_mix_lambda"] = float(lam)
            docs_mm[int(cursor)] = np.asarray(candidate, dtype=np.float32)
            out_doc_ids.append(str(synth_doc_id))
            cursor += 1
    del docs_mm

    if int(cursor) != int(total_docs):
        raise RuntimeError(f"synth1m row count mismatch: wrote={cursor} expected={total_docs}")

    np.save(synth_paths["doc_ids"], np.asarray(out_doc_ids, dtype=object))

    calib_rows = []
    eval_rows = []
    with parent_paths["calib_queries_jsonl"].open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                calib_rows.append(json.loads(text))
    with parent_paths["queries_jsonl"].open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                eval_rows.append(json.loads(text))
    calib_emb = np.asarray(np.load(parent_paths["calib_queries"]), dtype=np.float32)
    eval_emb = np.asarray(np.load(parent_paths["queries"]), dtype=np.float32)
    calib_ids = np.asarray(np.load(parent_paths["calib_query_ids"], allow_pickle=True), dtype=object)
    eval_ids = np.asarray(np.load(parent_paths["query_ids"], allow_pickle=True), dtype=object)

    exact_topk_eval = compute_exact_topk(
        docs_path=synth_paths["docs"],
        query_emb=eval_emb,
        exact_k=int(exact_k),
        score_doc_chunk=int(score_doc_chunk),
        score_query_batch=int(score_query_batch),
    )
    parent_query_meta = json.loads(parent_paths["queries_meta"].read_text(encoding="utf-8"))
    query_summary = dict(parent_query_meta.get("query_selection_summary") or {})
    query_summary["synth1m_reuses_parent100k_query_bundle"] = True

    _write_query_artifacts_generic(
        paths=synth_paths,
        calib_rows=calib_rows,
        eval_rows=eval_rows,
        calib_emb=calib_emb,
        eval_emb=eval_emb,
        calib_ids=calib_ids,
        eval_ids=eval_ids,
        exact_topk_eval=exact_topk_eval,
        eval_k=int(eval_k),
        exact_k=int(exact_k),
        query_tag=str(query_tag),
        workset_name=str(output_workset_name),
        query_summary=query_summary,
        doc_summary={
            "materialized_docs": int(total_docs),
            "real_parent_docs": int(parent_docs.shape[0]),
            "synthetic_docs": int(total_docs - int(parent_docs.shape[0])),
            "cluster_ordered_balanced_parent": True,
            "cluster_ordered_synth1m": True,
        },
        dataset_name="fiqa_2018",
    )
    _write_json(
        synth_paths["meta"],
        {
            "pipeline": "prepare_fiqa_2018_semantic_worksets",
            "dataset_name": "fiqa_2018",
            "workset_name": str(output_workset_name),
            "num_docs": int(total_docs),
            "embedding_dim": int(parent_docs.shape[1]),
            "num_clusters": int(num_clusters),
            "parent_cluster_size": int(parent_cluster_size),
            "target_cluster_size": int(synth_cluster_size),
            "parent_workset_name": str(parent_workset_name),
            "synthetic_mode": "cluster_ordered_semantic_interpolation_from_parent100k",
            "corpus_jsonl_written": False,
        },
    )
    return synth_paths, {"num_docs": int(total_docs)}


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_json).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = E5DualEncoder(str(args.model_name), log_prefix="fiqa2018-e5")
    print(
        f"[fiqa2018] source_mode={str(args.source_mode)} dataset_id={str(args.dataset_id)} beir_url={str(args.beir_url)}",
        flush=True,
    )

    real_paths, real_summary = _materialize_real_fiqa(
        dataset_id=str(args.dataset_id),
        source_mode=str(args.source_mode),
        beir_url=str(args.beir_url),
        encoder=encoder,
        model_name=str(args.model_name),
        workset_name=str(args.real_workset_name),
        query_tag=str(args.query_tag),
        num_calib_queries=int(args.num_calib_queries),
        num_eval_queries=int(args.num_eval_queries),
        query_batch_size=int(args.query_batch_size),
        passage_batch_size=int(args.passage_batch_size),
        doc_write_batch_size=int(args.doc_write_batch_size),
        max_length=int(args.max_length),
        exact_k=int(args.exact_k),
        eval_k=int(args.eval_k),
        score_doc_chunk=int(args.score_doc_chunk),
        score_query_batch=int(args.score_query_batch),
        seed=int(args.seed),
    )

    parent_paths, parent_summary = _build_balanced_parent100k(
        real_paths=real_paths,
        output_workset_name=str(args.parent100k_workset_name),
        query_tag=str(args.query_tag),
        num_clusters=int(args.num_clusters),
        target_cluster_size=int(args.parent_cluster_size),
        neighbor_k=int(args.neighbor_k),
        exact_k=int(args.exact_k),
        eval_k=int(args.eval_k),
        score_doc_chunk=int(args.score_doc_chunk),
        score_query_batch=int(args.score_query_batch),
        seed=int(args.seed),
        step1_max_iter=int(args.step1_max_iter),
    )

    synth_paths, synth_summary = _build_synth1m_from_parent100k(
        parent_paths=parent_paths,
        parent_workset_name=str(args.parent100k_workset_name),
        output_workset_name=str(args.synth1m_workset_name),
        query_tag=str(args.query_tag),
        num_clusters=int(args.num_clusters),
        parent_cluster_size=int(args.parent_cluster_size),
        synth_cluster_size=int(args.synth1m_cluster_size),
        neighbor_k=int(args.neighbor_k),
        exact_k=int(args.exact_k),
        eval_k=int(args.eval_k),
        score_doc_chunk=int(args.score_doc_chunk),
        score_query_batch=int(args.score_query_batch),
        seed=int(args.seed),
    )

    report = {
        "status": "completed",
        "dataset_name": "fiqa_2018",
        "real_workset_name": str(args.real_workset_name),
        "parent100k_workset_name": str(args.parent100k_workset_name),
        "synth1m_workset_name": str(args.synth1m_workset_name),
        "query_tag": str(args.query_tag),
        "source_dataset_id": str(args.dataset_id),
        "real_summary": real_summary,
        "parent100k_summary": parent_summary,
        "synth1m_summary": synth_summary,
        "paths": {
            "real_meta": str(real_paths["meta"]),
            "parent100k_meta": str(parent_paths["meta"]),
            "synth1m_meta": str(synth_paths["meta"]),
            "synth1m_docs": str(synth_paths["docs"]),
            "synth1m_doc_ids": str(synth_paths["doc_ids"]),
            "synth1m_queries_meta": str(synth_paths["queries_meta"]),
        },
    }
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
