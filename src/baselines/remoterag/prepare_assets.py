from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
from pathlib import Path

from baselines.public_assets import resolve_requested_worksets, resolve_workset_assets, save_json, workset_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a public RemoteRAG asset manifest from ready PRISMA worksets."
    )
    parser.add_argument(
        "--workset-name",
        action="append",
        default=[],
        help="Source PRISMA workset name. Repeat for multiple sizes. If omitted, all ready worksets are discovered.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="prepared_worksets",
        help="Output manifest stem under results/repro_workflows/remoterag/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    result_root = project_root / "results" / "repro_workflows" / "remoterag"
    result_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for workset_name in resolve_requested_worksets(project_root, list(args.workset_name)):
        assets = resolve_workset_assets(project_root, workset_name)
        stats = workset_stats(assets)
        rows.append(
            {
                "workset_name": str(workset_name),
                "num_docs": int(stats["num_docs"]),
                "num_queries": int(stats["num_queries"]),
                "num_clusters": int(stats["num_clusters"]),
                "embedding_dim": int(stats["embedding_dim"]),
                "source_paths": assets.to_dict(),
            }
        )

    rows.sort(key=lambda row: (int(row["num_docs"]), str(row["workset_name"])))
    output_stem = str(args.output_stem).strip() or "prepared_worksets"
    output_path = result_root / f"{output_stem}.json"
    payload = {
        "baseline_slug": "remoterag",
        "mode": "public_mainline_workset_manifest",
        "notes": [
            "RemoteRAG now consumes the same PRISMA workset assets directly by workset name.",
            "This manifest is a public orchestration artifact only; it does not duplicate the large numpy/jsonl assets.",
        ],
        "rows": rows,
    }
    save_json(output_path, payload)
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
