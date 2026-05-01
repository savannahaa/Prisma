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
from baselines.remoterag.config import RemoteRAGConfig
from baselines.remoterag.stages import build_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit or execute the RemoteRAG implementation path scaffold.")
    parser.add_argument("--out", type=str, default="", help="Optional manifest output path.")
    parser.add_argument(
        "--prepare-assets",
        action="store_true",
        help="Build a public RemoteRAG asset manifest from ready PRISMA worksets.",
    )
    parser.add_argument("--run-module1", action="store_true", help="Run the implemented RemoteRAG Module 1 prototype after emitting the manifest.")
    parser.add_argument("--run-module2", action="store_true", help="Run the implemented RemoteRAG Module 2 prototype after emitting the manifest.")
    parser.add_argument("--build-scaling-summary", action="store_true", help="Build the per-size RemoteRAG scaling summary from Module-2 outputs.")
    parser.add_argument("--export-comparison-contract", action="store_true", help="Export RemoteRAG outputs into the shared comparison contract.")
    parser.add_argument("--size", type=int, default=0, help="Optional module-1 workset size override.")
    parser.add_argument("--workset-name", action="append", default=[], help="Optional explicit PRISMA source workset name. Repeat for multiple worksets when building a scaling summary.")
    parser.add_argument("--sizes", type=str, default="", help="Optional comma-separated size subset for scaling-summary generation.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Optional module-1 epsilon override.")
    parser.add_argument("--target-radius", type=float, default=0.0, help="Optional module-1 target perturbation radius override.")
    parser.add_argument("--query-limit", type=int, default=-1, help="Optional module query limit override.")
    parser.add_argument("--module1-k-prime-cap", type=int, default=0, help="Optional Module 1 k' cap override when Module 2 generates Module 1 outputs.")
    parser.add_argument("--module1-query-limit", type=int, default=-1, help="Optional Module 1 query limit override for scaling-summary generation.")
    parser.add_argument("--quantization-scale", type=int, default=0, help="Optional Module 2 fixed-point scale override.")
    parser.add_argument("--paillier-bits", type=int, default=0, help="Optional Module 2 Paillier modulus bits override.")
    parser.add_argument("--ot-prime-bits", type=int, default=0, help="Optional Module 2 OT group prime bits override.")
    parser.add_argument("--output-prefix", type=str, default="", help="Optional scaling-summary / contract output stem override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = RemoteRAGConfig(project_root=project_root)
    payload = build_path(cfg)

    out_path = (
        Path(args.out)
        if str(args.out).strip()
        else project_root / "results" / "repro_workflows" / cfg.result_root_name / "implementation_path.json"
    )
    write_manifest(out_path, payload)
    print(f"[saved] {out_path}")

    explicit_worksets = [str(x).strip() for x in list(args.workset_name) if str(x).strip()]
    primary_workset = explicit_worksets[0] if explicit_worksets else ""
    if bool(args.prepare_assets):
        prep_script = project_root / "src" / "baselines" / "remoterag" / "prepare_assets.py"
        cmd = [sys.executable, str(prep_script)]
        for workset_name in explicit_worksets:
            cmd.extend(["--workset-name", str(workset_name)])
        subprocess.run(cmd, cwd=str(project_root), check=True)

    if bool(args.run_module1):
        module1_script = project_root / "src" / "baselines" / "remoterag" / "module1_distancedp.py"
        cmd = [sys.executable, str(module1_script)]
        if int(args.size) > 0:
            cmd.extend(["--size", str(int(args.size))])
        if str(primary_workset).strip():
            cmd.extend(["--workset-name", str(primary_workset).strip()])
        if float(args.epsilon) > 0.0:
            cmd.extend(["--epsilon", str(float(args.epsilon))])
        if float(args.target_radius) > 0.0:
            cmd.extend(["--target-radius", str(float(args.target_radius))])
        if int(args.query_limit) >= 0:
            cmd.extend(["--query-limit", str(int(args.query_limit))])
        if bool(args.prepare_assets):
            cmd.append("--prepare-assets")
        subprocess.run(cmd, cwd=str(project_root), check=True)

    if bool(args.run_module2):
        module2_script = project_root / "src" / "baselines" / "remoterag" / "module2_phe_ot.py"
        cmd = [sys.executable, str(module2_script)]
        if int(args.size) > 0:
            cmd.extend(["--size", str(int(args.size))])
        if str(primary_workset).strip():
            cmd.extend(["--workset-name", str(primary_workset).strip()])
        if float(args.epsilon) > 0.0:
            cmd.extend(["--epsilon", str(float(args.epsilon))])
        if float(args.target_radius) > 0.0:
            cmd.extend(["--target-radius", str(float(args.target_radius))])
        if int(args.query_limit) >= 0:
            cmd.extend(["--query-limit", str(int(args.query_limit))])
        if int(args.module1_k_prime_cap) > 0:
            cmd.extend(["--module1-k-prime-cap", str(int(args.module1_k_prime_cap))])
        if int(args.quantization_scale) > 0:
            cmd.extend(["--quantization-scale", str(int(args.quantization_scale))])
        if int(args.paillier_bits) > 0:
            cmd.extend(["--paillier-bits", str(int(args.paillier_bits))])
        if int(args.ot_prime_bits) > 0:
            cmd.extend(["--ot-prime-bits", str(int(args.ot_prime_bits))])
        if bool(args.prepare_assets):
            cmd.append("--prepare-assets")
        subprocess.run(cmd, cwd=str(project_root), check=True)

    if bool(args.build_scaling_summary):
        scaling_script = project_root / "src" / "baselines" / "remoterag" / "scaling_runner.py"
        cmd = [sys.executable, str(scaling_script)]
        if str(args.sizes).strip():
            cmd.extend(["--sizes", str(args.sizes).strip()])
        for workset_name in explicit_worksets:
            cmd.extend(["--workset-name", str(workset_name)])
        if int(args.query_limit) > 0:
            cmd.extend(["--query-limit", str(int(args.query_limit))])
        if int(args.module1_query_limit) > 0:
            cmd.extend(["--module1-query-limit", str(int(args.module1_query_limit))])
        if int(args.module1_k_prime_cap) > 0:
            cmd.extend(["--module1-k-prime-cap", str(int(args.module1_k_prime_cap))])
        if float(args.epsilon) > 0.0:
            cmd.extend(["--epsilon", str(float(args.epsilon))])
        if float(args.target_radius) > 0.0:
            cmd.extend(["--target-radius", str(float(args.target_radius))])
        if int(args.quantization_scale) > 0:
            cmd.extend(["--quantization-scale", str(int(args.quantization_scale))])
        if int(args.paillier_bits) > 0:
            cmd.extend(["--paillier-bits", str(int(args.paillier_bits))])
        if int(args.ot_prime_bits) > 0:
            cmd.extend(["--ot-prime-bits", str(int(args.ot_prime_bits))])
        if str(args.output_prefix).strip():
            cmd.extend(["--output-prefix", str(args.output_prefix).strip()])
        if bool(args.prepare_assets):
            cmd.append("--prepare-assets")
        subprocess.run(cmd, cwd=str(project_root), check=True)

    if bool(args.export_comparison_contract):
        adapter_script = project_root / "src" / "baselines" / "remoterag" / "comparison_adapter.py"
        cmd = [sys.executable, str(adapter_script)]
        if str(args.output_prefix).strip():
            cmd.extend(["--scaling-summary-stem", str(args.output_prefix).strip()])
            cmd.extend(["--output-prefix", str(args.output_prefix).strip()])
        subprocess.run(cmd, cwd=str(project_root), check=True)


if __name__ == "__main__":
    main()
