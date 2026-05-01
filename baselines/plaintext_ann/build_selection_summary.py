from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
from pathlib import Path

from baselines.public_assets import resolve_requested_worksets, resolve_workset_assets, workset_stats, save_json
from baselines.plaintext_ann.config import PlaintextANNConfig
from shared.config import (
    PAPERFAITHFUL_MAINLINE_OPTIMAL_C,
    PAPERFAITHFUL_MAINLINE_OPTIMAL_EPSILON,
    PAPERFAITHFUL_MAINLINE_OPTIMAL_FIXED_K,
)


def _parse_positive_int(raw: str | int, default: int) -> int:
    value = int(raw)
    return int(value) if int(value) > 0 else int(default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a public per-size plaintext_ann selection summary directly from ready PRISMA worksets."
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
        default="paperfaithful_mainline_latency_scaling_persize_public_defaults",
        help="Output json stem under results/.",
    )
    parser.add_argument("--fixed-k", type=int, default=0, help="Selected fixed_k. Default uses the locked mainline fixed_k.")
    parser.add_argument("--routing-c", type=int, default=0, help="Selected routing c. Default uses the locked mainline routing c, clamped by each workset's num_clusters.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Selected epsilon. Default uses the locked mainline epsilon.")
    parser.add_argument(
        "--candidate-output-cap",
        type=int,
        default=0,
        help="Selected candidate output cap. Default matches selected fixed_k.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = PlaintextANNConfig(project_root=project_root)
    workset_names = resolve_requested_worksets(project_root, list(args.workset_name))
    fixed_k = _parse_positive_int(args.fixed_k, int(PAPERFAITHFUL_MAINLINE_OPTIMAL_FIXED_K))
    epsilon = float(args.epsilon) if float(args.epsilon) > 0.0 else float(PAPERFAITHFUL_MAINLINE_OPTIMAL_EPSILON)
    output_stem = str(args.output_stem).strip() or "paperfaithful_mainline_latency_scaling_persize_public_defaults"

    rows: list[dict] = []
    for workset_name in workset_names:
        assets = resolve_workset_assets(project_root, workset_name)
        stats = workset_stats(assets)
        selected_routing_c = int(
            max(
                1,
                min(
                    int(stats["num_clusters"]),
                    _parse_positive_int(args.routing_c, int(PAPERFAITHFUL_MAINLINE_OPTIMAL_C)),
                ),
            )
        )
        selected_candidate_output_cap = _parse_positive_int(args.candidate_output_cap, int(fixed_k))
        rows.append(
            {
                "num_docs": int(stats["num_docs"]),
                "num_clusters": int(stats["num_clusters"]),
                "workset_name": str(workset_name),
                "selected_fixed_k": int(fixed_k),
                "selected_routing_c": int(selected_routing_c),
                "selected_epsilon": float(epsilon),
                "selected_candidate_output_cap": int(selected_candidate_output_cap),
                "selected_source": "public_default_locked_mainline_selection",
                "source_paths": assets.to_dict(),
            }
        )

    rows.sort(key=lambda row: (int(row["num_docs"]), str(row["workset_name"])))
    result_path = project_root / "results" / f"{output_stem}.json"
    payload = {
        "baseline_slug": "plaintext_ann",
        "baseline_display_name": str(cfg.baseline_display_name),
        "selection_mode": "public_default_locked_mainline_selection",
        "notes": [
            "This public selection summary does not claim to be a latency-tuned sweep output.",
            "Each row simply locks plaintext_ann to the same fixed_k and epsilon family used by the paperfaithful mainline, with routing c clamped by the actual number of clusters in that workset.",
            "same_pipeline_runner then re-measures the clean no-gate no-perturbation path against these selected points.",
        ],
        "rows": rows,
    }
    save_json(result_path, payload)
    print(f"[saved] {result_path}")


if __name__ == "__main__":
    main()
