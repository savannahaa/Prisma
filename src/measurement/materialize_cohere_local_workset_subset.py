"""
Materialize a smaller Cohere workset by selecting exact global offsets from an existing parent workset.

This is the fast path for nested experiments:
- fetch/materialize the 10M parent once from remote Cohere shards
- derive 1M / 100k / 10k locally from the parent workset

The subset is defined by a target offsets manifest (compressed json or explicit txt), and rows are
selected from the parent workset via `global_offset` / `cohere_global_offset` stored in the parent corpus jsonl.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import json
import os

import numpy as np

from shared.config import (
    WORKSET_DOCS_PATH,
    WORKSET_DOC_IDS_PATH,
    WORKSET_CORPUS_JSONL_PATH,
    WORKSET_META_PATH,
    WORKSET_NAME,
)


def cohere_offset_doc_id(offset: int) -> str:
    return f"cohere_offset_{int(offset)}"


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = str(line).strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


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


def _get_row_offset(row: dict) -> int:
    raw = row.get("global_offset", row.get("cohere_global_offset"))
    if raw is None:
        raise RuntimeError(
            "parent corpus rows are missing `global_offset` / `cohere_global_offset`, "
            "cannot derive exact nested subset"
        )
    return int(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a smaller Cohere workset locally from a parent workset using exact global offsets."
    )
    parser.add_argument("--parent-docs-path", required=True)
    parser.add_argument("--parent-doc-ids-path", required=True)
    parser.add_argument("--parent-corpus-jsonl-path", default="")
    parser.add_argument("--offsets-path", required=True)
    parser.add_argument("--parent-label", default="")
    parser.add_argument(
        "--skip-corpus-jsonl",
        action="store_true",
        help="Skip writing the subset corpus jsonl and derive offsets from doc_ids.",
    )
    args = parser.parse_args()

    ordered_offsets = [int(x) for x in _load_selected_offsets(args.offsets_path)]
    parent_docs = np.load(args.parent_docs_path, mmap_mode="r")
    parent_doc_ids = np.load(args.parent_doc_ids_path, allow_pickle=True)
    parent_corpus_rows: list[dict] = []
    offset_to_parent_idx: dict[int, int] = {}
    if str(args.parent_corpus_jsonl_path).strip() and os.path.exists(args.parent_corpus_jsonl_path):
        parent_corpus_rows = load_jsonl(args.parent_corpus_jsonl_path)
        if int(parent_docs.shape[0]) != int(len(parent_doc_ids)) or int(parent_docs.shape[0]) != int(len(parent_corpus_rows)):
            raise RuntimeError(
                "parent workset size mismatch among docs/doc_ids/corpus: "
                f"docs={int(parent_docs.shape[0])}, ids={int(len(parent_doc_ids))}, corpus={int(len(parent_corpus_rows))}"
            )
        for idx, row in enumerate(parent_corpus_rows):
            offset = _get_row_offset(row)
            offset_to_parent_idx[int(offset)] = int(idx)
    else:
        if int(parent_docs.shape[0]) != int(len(parent_doc_ids)):
            raise RuntimeError(
                "parent workset size mismatch among docs/doc_ids: "
                f"docs={int(parent_docs.shape[0])}, ids={int(len(parent_doc_ids))}"
            )
        for idx, doc_id in enumerate(parent_doc_ids.tolist()):
            text = str(doc_id)
            if not text.startswith("cohere_offset_"):
                raise RuntimeError(
                    "parent corpus jsonl is missing and parent doc ids are not cohere offsets; "
                    f"cannot derive nested subset for doc_id={text}"
                )
            offset_to_parent_idx[int(text.removeprefix("cohere_offset_").split("__rep", 1)[0])] = int(idx)

    missing = [int(x) for x in ordered_offsets if int(x) not in offset_to_parent_idx]
    if missing:
        preview = ",".join(str(x) for x in missing[:10])
        raise RuntimeError(
            f"target subset is not contained in parent workset; missing_offsets={len(missing)} preview={preview}"
        )

    positions = np.asarray([int(offset_to_parent_idx[int(x)]) for x in ordered_offsets], dtype=np.int64)
    subset_docs = np.asarray(parent_docs[positions], dtype=np.float32)
    subset_doc_ids = np.asarray(parent_doc_ids[positions], dtype="<U48")

    np.save(WORKSET_DOCS_PATH, subset_docs.astype(np.float32))
    np.save(WORKSET_DOC_IDS_PATH, subset_doc_ids)
    if not bool(args.skip_corpus_jsonl):
        if parent_corpus_rows:
            subset_corpus_rows = [dict(parent_corpus_rows[int(pos)]) for pos in positions.tolist()]
        else:
            subset_corpus_rows = [
                {
                    "doc_id": str(doc_id),
                    "source_doc_id": str(doc_id),
                    "text": str(doc_id),
                    "global_offset": int(offset),
                    "cohere_global_offset": int(offset),
                    "synthetic_text": True,
                }
                for offset, doc_id in zip(ordered_offsets, subset_doc_ids.tolist())
            ]
        save_jsonl(WORKSET_CORPUS_JSONL_PATH, subset_corpus_rows)
    save_json(
        WORKSET_META_PATH,
        {
            "pipeline": "materialize_cohere_local_workset_subset",
            "workset_name": str(WORKSET_NAME),
            "offsets_path": str(args.offsets_path),
            "num_docs": int(len(ordered_offsets)),
            "parent_label": str(args.parent_label),
            "parent_docs_path": str(args.parent_docs_path),
            "parent_doc_ids_path": str(args.parent_doc_ids_path),
            "parent_corpus_jsonl_path": str(args.parent_corpus_jsonl_path),
            "docs_path": str(WORKSET_DOCS_PATH),
            "doc_ids_path": str(WORKSET_DOC_IDS_PATH),
            "corpus_jsonl_path": (str(WORKSET_CORPUS_JSONL_PATH) if not bool(args.skip_corpus_jsonl) else ""),
            "global_offset_min": int(min(ordered_offsets)),
            "global_offset_max": int(max(ordered_offsets)),
            "subset_from_local_parent": True,
            "saved_corpus_jsonl": bool(not args.skip_corpus_jsonl),
        },
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "pipeline": "materialize_cohere_local_workset_subset",
                "workset_name": str(WORKSET_NAME),
                "num_docs": int(len(ordered_offsets)),
                "parent_label": str(args.parent_label),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
