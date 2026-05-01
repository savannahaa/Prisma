"""
Download the lightweight query side of Cohere MSMARCO v2.1 embeddings.

Why query-only first:
- the full dataset is ~523 GB
- the public release first needs the reusable query-side artifacts
- passages / cluster-info preparation can be handled separately
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
from huggingface_hub import snapshot_download


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Cohere MSMARCO v2.1 query artifacts from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default="CohereLabs/msmarco-v2.1-embed-english-v3",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Local target directory.",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
    )
    parser.add_argument(
        "--include-readme",
        action="store_true",
        default=True,
    )
    args = parser.parse_args()

    allow_patterns = [
        "queries_jsonl/*",
        "queries_parquet/*",
    ]
    if bool(args.include_readme):
        allow_patterns.extend(["README.md", ".gitattributes"])

    os.makedirs(args.local_dir, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=str(args.repo_id),
        repo_type=str(args.repo_type),
        local_dir=str(args.local_dir),
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )

    meta = {
        "status": "completed",
        "repo_id": str(args.repo_id),
        "repo_type": str(args.repo_type),
        "local_dir": str(args.local_dir),
        "snapshot_path": str(snapshot_path),
        "allow_patterns": [str(x) for x in allow_patterns],
    }
    save_json(os.path.join(args.local_dir, "download_meta.json"), meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
