"""
Materialize Cohere MSMARCO v2.1 queries into local numpy/jsonl artifacts.

Outputs:
- queries.npy           : float32 normalized query embeddings
- query_ids.npy         : object array of query ids
- queries.jsonl         : local jsonl rows with text / qrels / top1k metadata
- meta.json             : summary
"""

from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import gzip
import json
import os
from typing import Iterable, List

import numpy as np


EPS = 1e-12


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return (x / norms).astype(np.float32)


def _iter_query_rows(path: str) -> Iterable[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _find_query_jsonl_files(root: str) -> List[str]:
    candidates: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            lower = str(name).lower()
            if "queries" not in lower:
                continue
            if lower.endswith(".jsonl") or lower.endswith(".jsonl.gz"):
                candidates.append(os.path.join(dirpath, name))
    return sorted(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize Cohere MSMARCO v2.1 queries into local numpy/jsonl artifacts."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    query_files = _find_query_jsonl_files(args.source_dir)
    if len(query_files) <= 0:
        raise FileNotFoundError(
            f"no query jsonl/jsonl.gz files found under source-dir={args.source_dir}"
        )

    rows_out: List[dict] = []
    emb_rows: List[np.ndarray] = []
    qids: List[str] = []
    seen = set()
    for path in query_files:
        for row in _iter_query_rows(path):
            qid = str(row.get("_id", row.get("query_id", row.get("id", "")))).strip()
            if not qid or qid in seen:
                continue
            emb = row.get("emb")
            if emb is None:
                continue
            emb_np = np.asarray(emb, dtype=np.float32).reshape(-1)
            if emb_np.size <= 0:
                continue
            seen.add(qid)
            qids.append(qid)
            emb_rows.append(emb_np)
            rows_out.append(
                {
                    "query_id": qid,
                    "raw_query_id": qid,
                    "text": str(row.get("text", "")),
                    "trec_year": row.get("trec-year"),
                    "top1k_offsets": row.get("top1k_offsets", []),
                    "top1k_passage_ids": row.get("top1k_passage_ids", []),
                    "top1k_cossim": row.get("top1k_cossim", []),
                    "qrels": row.get("qrels", {}),
                    "source_repo": "CohereLabs/msmarco-v2.1-embed-english-v3",
                }
            )

    if len(emb_rows) <= 0:
        raise RuntimeError("no query embeddings found in source query files")

    queries = normalize_rows(np.asarray(emb_rows, dtype=np.float32))
    query_ids = np.asarray(qids, dtype=object)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "queries.npy"), queries.astype(np.float32))
    np.save(os.path.join(args.out_dir, "query_ids.npy"), query_ids)
    save_jsonl(os.path.join(args.out_dir, "queries.jsonl"), rows_out)
    save_json(
        os.path.join(args.out_dir, "meta.json"),
        {
            "status": "completed",
            "source_dir": str(args.source_dir),
            "num_queries": int(len(query_ids)),
            "embedding_dim": int(queries.shape[1]),
            "query_files": [str(x) for x in query_files],
            "outputs": {
                "queries_npy": os.path.join(args.out_dir, "queries.npy"),
                "query_ids_npy": os.path.join(args.out_dir, "query_ids.npy"),
                "queries_jsonl": os.path.join(args.out_dir, "queries.jsonl"),
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "num_queries": int(len(query_ids)),
                "embedding_dim": int(queries.shape[1]),
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
