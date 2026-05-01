from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import subprocess
import sys
from pathlib import Path

from baselines.common import write_manifest
from baselines.plaintext_ann.config import PlaintextANNConfig
from baselines.plaintext_ann.stages import build_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit or execute the Non-Private PRISMA implementation path scaffold.")
    parser.add_argument("--out", type=str, default="", help="Optional manifest output path.")
    parser.add_argument(
        "--prepare-assets",
        action="store_true",
        help="Build the public plaintext_ann selection summary from ready PRISMA worksets.",
    )
    parser.add_argument(
        "--build-scaling-summary",
        action="store_true",
        help="Build the per-size plaintext ANN scaling summary from the matched PRISMA selection summary.",
    )
    parser.add_argument(
        "--export-comparison-contract",
        action="store_true",
        help="Export plaintext ANN outputs into the shared comparison contract.",
    )
    parser.add_argument("--selection-summary-json", type=str, default="", help="Optional explicit PRISMA latency-scaling summary json.")
    parser.add_argument("--workset-name", action="append", default=[], help="Optional source workset name passed to the public selection-summary builder. Repeat for multiple sizes.")
    parser.add_argument("--sizes", type=str, default="", help="Optional comma-separated size subset.")
    parser.add_argument("--query-limit", type=int, default=0, help="Optional positive evaluation-query limit override.")
    parser.add_argument("--output-prefix", type=str, default="", help="Optional summary / comparison-contract output stem override.")
    parser.set_defaults(reuse_existing=True)
    parser.add_argument("--reuse-existing", dest="reuse_existing", action="store_true", help="Reuse matching existing plaintext ANN summaries. Enabled by default.")
    parser.add_argument("--no-reuse-existing", dest="reuse_existing", action="store_false", help="Force re-measuring the plaintext ANN summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = PlaintextANNConfig(project_root=project_root)
    payload = build_path(cfg)

    out_path = (
        Path(args.out)
        if str(args.out).strip()
        else project_root / "results" / "repro_workflows" / cfg.result_root_name / "implementation_path.json"
    )
    write_manifest(out_path, payload)
    print(f"[saved] {out_path}")

    generated_selection_summary: Path | None = None
    if bool(args.prepare_assets):
        script = project_root / "src" / "baselines" / "plaintext_ann" / "build_selection_summary.py"
        cmd = [sys.executable, str(script)]
        for workset_name in list(args.workset_name):
            if str(workset_name).strip():
                cmd.extend(["--workset-name", str(workset_name).strip()])
        if str(args.output_prefix).strip():
            cmd.extend(["--output-stem", str(args.output_prefix).strip()])
        subprocess.run(cmd, cwd=str(project_root), check=True)
        generated_stem = str(args.output_prefix).strip() or "paperfaithful_mainline_latency_scaling_persize_public_defaults"
        generated_selection_summary = project_root / "results" / f"{generated_stem}.json"

    if bool(args.build_scaling_summary):
        script = project_root / "src" / "baselines" / "plaintext_ann" / "same_pipeline_runner.py"
        cmd = [sys.executable, str(script)]
        if str(args.selection_summary_json).strip():
            cmd.extend(["--selection-summary-json", str(args.selection_summary_json).strip()])
        elif generated_selection_summary is not None:
            cmd.extend(["--selection-summary-json", str(generated_selection_summary)])
        if str(args.sizes).strip():
            cmd.extend(["--sizes", str(args.sizes).strip()])
        if int(args.query_limit) > 0:
            cmd.extend(["--query-limit", str(int(args.query_limit))])
        if str(args.output_prefix).strip():
            cmd.extend(["--output-prefix", str(args.output_prefix).strip()])
        if not bool(args.reuse_existing):
            cmd.append("--no-reuse-existing")
        subprocess.run(cmd, cwd=str(project_root), check=True)

    if bool(args.export_comparison_contract):
        script = project_root / "src" / "baselines" / "plaintext_ann" / "comparison_adapter.py"
        cmd = [sys.executable, str(script)]
        if str(args.output_prefix).strip():
            cmd.extend(["--output-prefix", str(args.output_prefix).strip()])
        subprocess.run(cmd, cwd=str(project_root), check=True)


if __name__ == "__main__":
    main()
