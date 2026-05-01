"""
Materialize a Cohere MSMARCO v2.1 workset from a selected global-offset manifest.

This writes the standard paperfaithful workset assets for the *current* env-configured workset:
- WORKSET_DOCS_PATH
- WORKSET_DOC_IDS_PATH
- WORKSET_CORPUS_JSONL_PATH
- WORKSET_META_PATH

To avoid downloading the full 523GB dataset for small/medium experiments, this script fetches
only the required rows from the remote `passages_npy/*.npy` shards via HTTP range requests.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import concurrent.futures
import io
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_url

from shared.config import (
    WORKSET_DOCS_PATH,
    WORKSET_DOC_IDS_PATH,
    WORKSET_CORPUS_JSONL_PATH,
    WORKSET_META_PATH,
    WORKSET_NAME,
)


EPS = 1e-12
DEFAULT_REPO_ID = "CohereLabs/msmarco-v2.1-embed-english-v3"
DEFAULT_REPO_TYPE = "dataset"
DEFAULT_NUM_SHARDS = 60
DEFAULT_MERGE_GAP_ROWS = 32
DEFAULT_HTTP_RETRIES = 5
DEFAULT_HTTP_TIMEOUT_SEC = 180.0
DEFAULT_FETCH_TIMEOUT_SEC = 600.0
DEFAULT_FETCH_WORKERS = 4
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARD_META_CACHE_PATH = (
    PROJECT_ROOT / "data" / "external" / "cohere_msmarco_v21" / "passages_shard_meta.json"
)
DEFAULT_LOCAL_SHARDS_ROOT = (
    PROJECT_ROOT / "data" / "external" / "cohere_msmarco_v21" / "passages_npy"
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return (x / norms).astype(np.float32)


def cohere_offset_doc_id(offset: int) -> str:
    return f"cohere_offset_{int(offset)}"


def cohere_offset_text(offset: int, doc_id: str) -> str:
    return f"[cohere synthetic text unavailable] global_offset={int(offset)} doc_id={str(doc_id)}"


@dataclass
class ShardMeta:
    shard_idx: int
    file_name: str
    url: str
    local_path: str
    num_rows: int
    dim: int
    dtype: str
    data_offset: int
    global_start: int
    global_end_exclusive: int


def _http_range_get(
    url: str,
    start: int,
    end_inclusive: int,
    *,
    timeout_sec: float = DEFAULT_HTTP_TIMEOUT_SEC,
    retries: int = DEFAULT_HTTP_RETRIES,
) -> bytes:
    last_err: Exception | None = None
    for attempt in range(1, int(retries) + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={int(start)}-{int(end_inclusive)}",
                    "Accept-Encoding": "identity",
                    "User-Agent": "codex-cohere-workset-materializer/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=float(timeout_sec)) as resp:
                return resp.read()
        except Exception as exc:
            last_err = exc
            if attempt >= int(retries):
                raise
            sleep_sec = float(min(30, 2 ** (attempt - 1)))
            print(
                f"[retry] range request failed attempt={int(attempt)}/{int(retries)} "
                f"bytes={int(start)}-{int(end_inclusive)} sleep={sleep_sec}s err={exc}",
                flush=True,
            )
            time.sleep(float(sleep_sec))
    if last_err is not None:
        raise last_err
    raise RuntimeError("unreachable range-get failure")


def _parse_npy_header_from_prefix(prefix_bytes: bytes) -> tuple[int, int, str, int]:
    bio = io.BytesIO(prefix_bytes)
    version = np.lib.format.read_magic(bio)
    if tuple(version) == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(bio)
    elif tuple(version) == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(bio)
    else:
        raise RuntimeError(f"unsupported npy version for remote row slicing: {version}")
    if bool(fortran_order):
        raise RuntimeError("fortran-order npy is unsupported for remote row slicing")
    if len(shape) != 2:
        raise RuntimeError(f"expected 2D npy matrix, got shape={shape}")
    return int(shape[0]), int(shape[1]), str(np.dtype(dtype)), int(bio.tell())


def _build_shard_meta(*, repo_id: str, repo_type: str, num_shards: int) -> list[ShardMeta]:
    cache_path = DEFAULT_SHARD_META_CACHE_PATH
    cached_lookup: dict[int, ShardMeta] = {}
    local_root = Path(os.environ.get("COHERE_LOCAL_SHARDS_ROOT", str(DEFAULT_LOCAL_SHARDS_ROOT))).resolve()
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if (
            str(payload.get("repo_id", "")) == str(repo_id)
            and str(payload.get("repo_type", "")) == str(repo_type)
            and int(payload.get("num_shards", -1)) == int(num_shards)
        ):
            cached_rows = payload.get("shards", [])
            metas = []
            for row in cached_rows:
                meta = ShardMeta(
                    shard_idx=int(row["shard_idx"]),
                    file_name=str(row["file_name"]),
                    url=str(row["url"]),
                    local_path=str(
                        row.get(
                            "local_path",
                            str(local_root / f"msmarco_v2.1_doc_segmented_{int(row['shard_idx']):02d}.npy"),
                        )
                    ),
                    num_rows=int(row["num_rows"]),
                    dim=int(row["dim"]),
                    dtype=str(row["dtype"]),
                    data_offset=int(row["data_offset"]),
                    global_start=int(row["global_start"]),
                    global_end_exclusive=int(row["global_end_exclusive"]),
                )
                metas.append(meta)
                cached_lookup[int(meta.shard_idx)] = meta
            if len(metas) == int(num_shards):
                print(f"[shard-meta] loaded cache {cache_path}", flush=True)
                return metas

    metas: list[ShardMeta] = []
    global_cursor = 0
    for shard_idx in range(int(num_shards)):
        file_name = f"passages_npy/msmarco_v2.1_doc_segmented_{int(shard_idx):02d}.npy"
        url = hf_hub_url(repo_id=repo_id, filename=file_name, repo_type=repo_type)
        cached_meta = cached_lookup.get(int(shard_idx))
        if cached_meta is not None:
            if int(cached_meta.global_start) != int(global_cursor):
                raise RuntimeError(
                    f"cached shard meta has inconsistent global_start for shard={int(shard_idx):02d}: "
                    f"{int(cached_meta.global_start)} vs expected {int(global_cursor)}"
                )
            metas.append(cached_meta)
            global_cursor = int(cached_meta.global_end_exclusive)
            print(
                f"[shard-meta] reuse cache shard={int(shard_idx):02d} rows={int(cached_meta.num_rows)} "
                f"global_start={int(cached_meta.global_start)} global_end={int(cached_meta.global_end_exclusive)}",
                flush=True,
            )
            continue
        prefix = _http_range_get(
            url,
            0,
            4095,
            timeout_sec=_env_float("COHERE_HTTP_TIMEOUT_SEC", DEFAULT_HTTP_TIMEOUT_SEC),
            retries=_env_int("COHERE_HTTP_RETRIES", DEFAULT_HTTP_RETRIES),
        )
        num_rows, dim, dtype_str, data_offset = _parse_npy_header_from_prefix(prefix)
        meta = ShardMeta(
            shard_idx=int(shard_idx),
            file_name=str(file_name),
            url=str(url),
            local_path=str((local_root / f"msmarco_v2.1_doc_segmented_{int(shard_idx):02d}.npy").resolve()),
            num_rows=int(num_rows),
            dim=int(dim),
            dtype=str(dtype_str),
            data_offset=int(data_offset),
            global_start=int(global_cursor),
            global_end_exclusive=int(global_cursor + num_rows),
        )
        metas.append(meta)
        global_cursor += int(num_rows)
        print(
            f"[shard-meta] shard={int(shard_idx):02d} rows={int(num_rows)} dim={int(dim)} "
            f"global_start={int(meta.global_start)} global_end={int(meta.global_end_exclusive)}",
            flush=True,
        )
        os.makedirs(cache_path.parent, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "repo_id": str(repo_id),
                    "repo_type": str(repo_type),
                    "num_shards": int(num_shards),
                    "shards": [
                        {
                            "shard_idx": int(meta_row.shard_idx),
                            "file_name": str(meta_row.file_name),
                            "url": str(meta_row.url),
                            "local_path": str(meta_row.local_path),
                            "num_rows": int(meta_row.num_rows),
                            "dim": int(meta_row.dim),
                            "dtype": str(meta_row.dtype),
                            "data_offset": int(meta_row.data_offset),
                            "global_start": int(meta_row.global_start),
                            "global_end_exclusive": int(meta_row.global_end_exclusive),
                        }
                        for meta_row in metas
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    print(f"[shard-meta] saved cache {cache_path}", flush=True)
    return metas


def _load_offsets_txt(path: str) -> list[int]:
    offsets: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = str(line).strip()
            if not text:
                continue
            offsets.append(int(text))
    if len(offsets) <= 0:
        raise RuntimeError(f"offset manifest is empty: {path}")
    return offsets


def _load_manifest_offsets(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    fmt = str(manifest.get("format", "")).strip().lower()
    if fmt != "cohere_offset_union_prefix_v1":
        raise RuntimeError(f"unsupported offset manifest format: {fmt}")
    core_offsets_path = str(manifest.get("core_offsets_path", "")).strip()
    prefix_end_exclusive = int(manifest.get("prefix_end_exclusive", 0))
    if not core_offsets_path:
        raise RuntimeError(f"manifest is missing core_offsets_path: {path}")
    core_offsets = _load_offsets_txt(core_offsets_path)
    selected = set(int(x) for x in core_offsets)
    selected.update(range(int(prefix_end_exclusive)))
    ordered = sorted(int(x) for x in selected)
    expected = int(manifest.get("selected_count_effective", len(ordered)))
    if int(len(ordered)) != int(expected):
        raise RuntimeError(
            f"manifest effective size mismatch: len(ordered)={len(ordered)} vs expected={expected}"
        )
    return ordered


def _load_selected_offsets(path: str) -> list[int]:
    if str(path).endswith(".json"):
        return _load_manifest_offsets(path)
    return _load_offsets_txt(path)


def _find_shard_for_offset(offset: int, metas: list[ShardMeta]) -> ShardMeta:
    lo = 0
    hi = len(metas) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        meta = metas[mid]
        if int(offset) < int(meta.global_start):
            hi = mid - 1
        elif int(offset) >= int(meta.global_end_exclusive):
            lo = mid + 1
        else:
            return meta
    raise RuntimeError(f"offset {int(offset)} is outside all shard ranges")


def _group_rows(rows: list[int], *, max_merge_gap_rows: int) -> list[tuple[int, int, list[int]]]:
    if len(rows) <= 0:
        return []
    rows_sorted = sorted(int(x) for x in rows)
    groups: list[tuple[int, int, list[int]]] = []
    current_start = int(rows_sorted[0])
    current_end = int(rows_sorted[0])
    current_rows = [int(rows_sorted[0])]
    for row in rows_sorted[1:]:
        row = int(row)
        if row - int(current_end) <= int(max_merge_gap_rows) + 1:
            current_end = int(row)
            current_rows.append(int(row))
            continue
        groups.append((int(current_start), int(current_end), list(current_rows)))
        current_start = int(row)
        current_end = int(row)
        current_rows = [int(row)]
    groups.append((int(current_start), int(current_end), list(current_rows)))
    return groups


def _fetch_one_shard(
    *,
    meta: ShardMeta,
    local_rows: list[int],
    shard_to_global_rows: dict[int, int],
    max_merge_gap_rows: int,
    fetch_timeout_sec: float,
    fetch_retries: int,
) -> tuple[int, list[tuple[int, np.ndarray]], int, int]:
    row_size_bytes = int(np.dtype(meta.dtype).itemsize) * int(meta.dim)
    groups = _group_rows(local_rows, max_merge_gap_rows=int(max_merge_gap_rows))
    local_path = str(meta.local_path).strip()
    use_local = bool(local_path) and os.path.exists(local_path)
    print(
        f"[fetch] shard={int(meta.shard_idx):02d} selected_rows={int(len(local_rows))} "
        f"groups={int(len(groups))} source={'local' if use_local else 'remote'}",
        flush=True,
    )
    assignments: list[tuple[int, np.ndarray]] = []
    fetched_bytes = 0
    local_arr = None
    if use_local:
        local_arr = np.load(local_path, mmap_mode="r")
    for run_start, run_end, selected_rows in groups:
        if use_local:
            block = np.asarray(local_arr[int(run_start) : int(run_end + 1)], dtype=np.float32)
            fetched_bytes += int(block.size * np.dtype(np.float32).itemsize)
        else:
            byte_start = int(meta.data_offset + run_start * row_size_bytes)
            byte_end_inclusive = int(meta.data_offset + (run_end + 1) * row_size_bytes - 1)
            payload = _http_range_get(
                meta.url,
                byte_start,
                byte_end_inclusive,
                timeout_sec=float(fetch_timeout_sec),
                retries=int(fetch_retries),
            )
            block = np.frombuffer(payload, dtype=np.dtype(meta.dtype)).reshape(
                int(run_end - run_start + 1),
                int(meta.dim),
            )
            fetched_bytes += int(len(payload))
        for local_row in selected_rows:
            offset = int(shard_to_global_rows[int(local_row)])
            assignments.append(
                (
                    int(offset),
                    np.asarray(block[int(local_row - run_start)], dtype=np.float32),
                )
            )
    return int(meta.shard_idx), assignments, int(len(groups)), int(fetched_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a Cohere MSMARCO v2.1 workset from selected global offsets."
    )
    parser.add_argument("--offsets-path", required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE)
    parser.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS)
    parser.add_argument("--max-merge-gap-rows", type=int, default=DEFAULT_MERGE_GAP_ROWS)
    parser.add_argument("--fetch-workers", type=int, default=DEFAULT_FETCH_WORKERS)
    parser.add_argument(
        "--text-template",
        default="[cohere synthetic text unavailable] global_offset={offset} doc_id={doc_id}",
        help="Synthetic corpus text template; {offset} and {doc_id} are supported.",
    )
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Skip final docs normalization before saving; downstream loaders normalize again.",
    )
    parser.add_argument(
        "--skip-corpus-jsonl",
        action="store_true",
        help="Skip writing the large corpus jsonl; cohere latency pipeline can synthesize texts from doc_ids.",
    )
    args = parser.parse_args()
    http_retries = int(_env_int("COHERE_HTTP_RETRIES", DEFAULT_HTTP_RETRIES))
    http_timeout_sec = float(_env_float("COHERE_HTTP_TIMEOUT_SEC", DEFAULT_HTTP_TIMEOUT_SEC))
    fetch_timeout_sec = float(_env_float("COHERE_FETCH_TIMEOUT_SEC", DEFAULT_FETCH_TIMEOUT_SEC))

    offsets = _load_selected_offsets(args.offsets_path)
    ordered_offsets = [int(x) for x in offsets]
    metas = _build_shard_meta(
        repo_id=str(args.repo_id),
        repo_type=str(args.repo_type),
        num_shards=int(args.num_shards),
    )
    if len(metas) <= 0:
        raise RuntimeError("failed to build any shard metadata")
    dim = int(metas[0].dim)
    if any(int(meta.dim) != dim for meta in metas):
        raise RuntimeError("passage shard dims are inconsistent")

    offset_to_position = {int(offset): int(i) for i, offset in enumerate(ordered_offsets)}
    shard_to_local_rows: dict[int, list[int]] = {}
    shard_to_global_rows: dict[int, dict[int, int]] = {}
    for offset in ordered_offsets:
        meta = _find_shard_for_offset(int(offset), metas)
        local_row = int(offset - meta.global_start)
        shard_to_local_rows.setdefault(int(meta.shard_idx), []).append(int(local_row))
        shard_to_global_rows.setdefault(int(meta.shard_idx), {})[int(local_row)] = int(offset)

    os.makedirs(os.path.dirname(WORKSET_DOCS_PATH), exist_ok=True)
    docs = np.lib.format.open_memmap(
        WORKSET_DOCS_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(int(len(ordered_offsets)), int(dim)),
    )
    fetched_rows = 0
    fetched_groups = 0
    fetched_bytes = 0
    active_jobs = [
        (
            meta,
            [int(x) for x in shard_to_local_rows.get(int(meta.shard_idx), [])],
            shard_to_global_rows.get(int(meta.shard_idx), {}),
        )
        for meta in metas
        if len(shard_to_local_rows.get(int(meta.shard_idx), [])) > 0
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.fetch_workers))) as ex:
        futures = [
            ex.submit(
                _fetch_one_shard,
                meta=meta,
                local_rows=local_rows,
                shard_to_global_rows=global_lookup,
                max_merge_gap_rows=int(args.max_merge_gap_rows),
                fetch_timeout_sec=float(fetch_timeout_sec),
                fetch_retries=int(http_retries),
            )
            for meta, local_rows, global_lookup in active_jobs
        ]
        for fut in concurrent.futures.as_completed(futures):
            shard_idx, assignments, num_groups, shard_bytes = fut.result()
            for offset, vec in assignments:
                pos = int(offset_to_position[int(offset)])
                docs[pos] = np.asarray(vec, dtype=np.float32)
            fetched_rows += int(len(assignments))
            fetched_groups += int(num_groups)
            fetched_bytes += int(shard_bytes)
            print(
                f"[fetch-progress] shard={int(shard_idx):02d} fetched_rows_total={int(fetched_rows)} "
                f"fetched_groups_total={int(fetched_groups)} mb={float(fetched_bytes / float(1024 * 1024)):.2f}",
                flush=True,
            )

    if int(fetched_rows) != int(len(ordered_offsets)):
        raise RuntimeError(
            f"materialization mismatch: fetched_rows={int(fetched_rows)} != selected_offsets={int(len(ordered_offsets))}"
        )

    if not bool(args.skip_normalize):
        docs = normalize_rows(docs)
    doc_ids = np.asarray([cohere_offset_doc_id(offset) for offset in ordered_offsets], dtype="<U48")

    del docs
    np.save(WORKSET_DOC_IDS_PATH, doc_ids)
    if not bool(args.skip_corpus_jsonl):
        corpus_rows = []
        for offset, doc_id in zip(ordered_offsets, doc_ids.tolist()):
            corpus_rows.append(
                {
                    "doc_id": str(doc_id),
                    "source_doc_id": str(doc_id),
                    "text": str(args.text_template).format(offset=int(offset), doc_id=str(doc_id)),
                    "global_offset": int(offset),
                    "cohere_global_offset": int(offset),
                    "synthetic_text": True,
                    "source_repo": str(args.repo_id),
                }
            )
        save_jsonl(WORKSET_CORPUS_JSONL_PATH, corpus_rows)
    save_json(
        WORKSET_META_PATH,
        {
            "pipeline": "materialize_cohere_msmarco_v21_workset_from_offsets",
            "workset_name": str(WORKSET_NAME),
            "repo_id": str(args.repo_id),
            "repo_type": str(args.repo_type),
            "offsets_path": str(args.offsets_path),
            "num_docs": int(len(ordered_offsets)),
            "embedding_dim": int(dim),
            "docs_path": str(WORKSET_DOCS_PATH),
            "doc_ids_path": str(WORKSET_DOC_IDS_PATH),
            "corpus_jsonl_path": (str(WORKSET_CORPUS_JSONL_PATH) if not bool(args.skip_corpus_jsonl) else ""),
            "fetched_rows": int(fetched_rows),
            "fetched_groups": int(fetched_groups),
            "fetched_bytes": int(fetched_bytes),
            "estimated_fetched_mb": float(fetched_bytes / float(1024 * 1024)),
            "http_retries": int(http_retries),
            "http_timeout_sec": float(http_timeout_sec),
            "fetch_timeout_sec": float(fetch_timeout_sec),
            "num_shards_scanned": int(len([1 for rows in shard_to_local_rows.values() if rows])),
            "global_offset_min": int(min(ordered_offsets)),
            "global_offset_max": int(max(ordered_offsets)),
            "synthetic_text": True,
            "saved_normalized": bool(not args.skip_normalize),
            "saved_corpus_jsonl": bool(not args.skip_corpus_jsonl),
        },
    )

    print(
        json.dumps(
            {
                "status": "completed",
                "workset_name": str(WORKSET_NAME),
                "num_docs": int(len(ordered_offsets)),
                "embedding_dim": int(dim),
                "fetched_groups": int(fetched_groups),
                "estimated_fetched_mb": float(fetched_bytes / float(1024 * 1024)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
