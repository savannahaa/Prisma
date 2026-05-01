from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import os
import subprocess
import sys
from pathlib import Path

from baselines.common import write_manifest
from baselines.tiptoe.config import TiptoeConfig
from baselines.tiptoe.stages import build_path


def _default_bundle_root_name(workset_name: str) -> str:
    safe = str(workset_name).strip().replace(" ", "_")
    return f"baseline_bundle_{safe}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit or execute the Tiptoe implementation path scaffold.")
    parser.add_argument("--out", type=str, default="", help="Optional manifest output path.")
    parser.add_argument(
        "--prepare-assets",
        action="store_true",
        help="Build a public Tiptoe/Panther-compatible bundle from one ready PRISMA workset.",
    )
    parser.add_argument("--workset-name", action="append", default=[], help="Optional source workset name forwarded to the public bundle builder. Pass explicitly when multiple ready worksets exist.")
    parser.add_argument("--bundle-root-name", type=str, default="", help="Optional output bundle directory name under results/.")
    parser.add_argument("--selector-cs", type=str, default="7,10", help="Comma-separated cluster_info_c aliases to materialize inside the bundle.")
    parser.add_argument("--run-ranking-service", action="store_true", help="Run the implemented Tiptoe ranking service after emitting the manifest.")
    parser.add_argument("--run-url-service", action="store_true", help="Run the implemented Tiptoe URL service after emitting the manifest.")
    parser.add_argument("--export-comparison-contract", action="store_true", help="Export Tiptoe outputs into the shared comparison contract.")
    parser.add_argument("--top-k", type=int, default=0, help="Optional top-k override.")
    parser.add_argument("--query-limit", type=int, default=-1, help="Optional query limit override for ranking.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = TiptoeConfig(project_root=project_root)
    payload = build_path(cfg)

    out_path = (
        Path(args.out)
        if str(args.out).strip()
        else project_root / "results" / "repro_workflows" / "tiptoe" / "implementation_path.json"
    )
    write_manifest(out_path, payload)
    print(f"[saved] {out_path}")

    runtime_env = dict(os.environ)
    if bool(args.prepare_assets):
        script = project_root / "src" / "baselines" / "build_public_baseline_bundle.py"
        cmd = [sys.executable, str(script), "--selector-cs", str(args.selector_cs)]
        explicit_worksets = [str(x).strip() for x in list(args.workset_name) if str(x).strip()]
        for workset_name in list(args.workset_name):
            if str(workset_name).strip():
                cmd.extend(["--workset-name", str(workset_name).strip()])
        if str(args.bundle_root_name).strip():
            cmd.extend(["--bundle-root-name", str(args.bundle_root_name).strip()])
        subprocess.run(cmd, cwd=str(project_root), check=True)
        if str(args.bundle_root_name).strip():
            runtime_env["TIPTOE_BUNDLE_ROOT"] = str((project_root / "results" / str(args.bundle_root_name).strip()).resolve())
        elif len(explicit_worksets) == 1:
            runtime_env["TIPTOE_BUNDLE_ROOT"] = str((project_root / "results" / _default_bundle_root_name(explicit_worksets[0])).resolve())

    if bool(args.run_ranking_service):
        script = project_root / "src" / "baselines" / "tiptoe" / "ranking_service.py"
        cmd = [sys.executable, str(script)]
        if int(args.top_k) > 0:
            cmd.extend(["--top-k", str(int(args.top_k))])
        if int(args.query_limit) >= 0:
            cmd.extend(["--query-limit", str(int(args.query_limit))])
        subprocess.run(cmd, cwd=str(project_root), env=runtime_env, check=True)

    if bool(args.run_url_service):
        script = project_root / "src" / "baselines" / "tiptoe" / "url_service.py"
        cmd = [sys.executable, str(script)]
        subprocess.run(cmd, cwd=str(project_root), env=runtime_env, check=True)

    if bool(args.export_comparison_contract):
        script = project_root / "src" / "baselines" / "tiptoe" / "comparison_adapter.py"
        subprocess.run([sys.executable, str(script)], cwd=str(project_root), env=runtime_env, check=True)


if __name__ == "__main__":
    main()
