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
from baselines.panther.config import PantherConfig
from baselines.panther.stages import build_path


def _default_bundle_root_name(workset_name: str) -> str:
    safe = str(workset_name).strip().replace(" ", "_")
    return f"baseline_bundle_{safe}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit or execute the Panther implementation path scaffold.")
    parser.add_argument("--out", type=str, default="", help="Optional manifest output path.")
    parser.add_argument(
        "--prepare-assets",
        action="store_true",
        help="Build a public Tiptoe/Panther-compatible bundle from one ready PRISMA workset.",
    )
    parser.add_argument("--workset-name", action="append", default=[], help="Optional source workset name forwarded to the public bundle builder. Pass explicitly when multiple ready worksets exist.")
    parser.add_argument("--bundle-root-name", type=str, default="", help="Optional output bundle directory name under results/.")
    parser.add_argument("--selector-cs", type=str, default="7,10", help="Comma-separated cluster_info_c aliases to materialize inside the bundle.")
    parser.add_argument("--export-ms-author-assets", action="store_true", help="Export the MS qrels bundle into the Panther author-bridge staging directory.")
    parser.add_argument("--build-ms-summary-from-rankings", action="store_true", help="Evaluate an external Panther ranking output against the MS qrels bundle.")
    parser.add_argument("--export-comparison-contract", action="store_true", help="Export Panther outputs into the shared comparison contract.")
    parser.add_argument("--write-summary-template", action="store_true", help="Write a Panther summary template json next to the result root.")
    parser.add_argument("--write-openpanther-text", action="store_true", help="When exporting MS author assets, also emit best-effort OpenPanther-style *.txt inputs.")
    parser.add_argument("--stage-into-openpanther", action="store_true", help="When exporting MS author assets, also stage the generated txt files and bridge config into external/OpenPanther.")
    parser.add_argument("--dataset", type=str, default="ms", help="Panther dataset/profile label for the comparison scaffold. Use `ms` for mainline and `sift` for parity.")
    parser.add_argument("--summary-json", type=str, default="", help="Optional explicit Panther summary json path.")
    parser.add_argument("--summary-stem", type=str, default="", help="Optional Panther summary stem under results/repro_workflows/panther/.")
    parser.add_argument("--dataset-slug", type=str, default="", help="Optional slug used under results/repro_workflows/panther/ms_author_bridge/.")
    parser.add_argument("--run-label", type=str, default="", help="Optional run label stored in the generated Panther MS summary.")
    parser.add_argument("--rankings-jsonl", type=str, default="", help="Optional JSONL ranking input for the Panther MS bridge.")
    parser.add_argument("--rankings-json", type=str, default="", help="Optional JSON ranking input for the Panther MS bridge.")
    parser.add_argument("--rankings-tsv", type=str, default="", help="Optional TSV ranking input for the Panther MS bridge.")
    parser.add_argument("--rankings-npy", type=str, default="", help="Optional NPY ranking input for the Panther MS bridge.")
    parser.add_argument("--routing-c", type=int, default=0, help="Optional cluster_info_c* selector used by the Panther MS bridge.")
    parser.add_argument("--query-limit", type=int, default=-1, help="Optional query limit override for the Panther MS bridge.")
    parser.add_argument("--top-k", type=int, default=0, help="Optional top-k override for the Panther MS bridge.")
    parser.add_argument("--max-points-per-cluster", type=int, default=0, help="Optional Panther/OpenPanther bridge partition size override.")
    parser.add_argument("--selected-clusters", type=int, default=0, help="Optional Panther/OpenPanther first-stage selected cluster count override.")
    parser.add_argument("--latency-total-sec-avg", type=float, default=None, help="Optional total latency stored in a generated Panther MS summary.")
    parser.add_argument("--latency-client-generate-sec-avg", type=float, default=None, help="Optional client-generate latency stored in a generated Panther MS summary.")
    parser.add_argument("--latency-server-query-sec-avg", type=float, default=None, help="Optional server-query latency stored in a generated Panther MS summary.")
    parser.add_argument("--latency-client-recover-sec-avg", type=float, default=None, help="Optional client-recover latency stored in a generated Panther MS summary.")
    parser.add_argument("--comm-request-bytes-avg", type=float, default=None, help="Optional request-bytes stored in a generated Panther MS summary.")
    parser.add_argument("--comm-response-bytes-avg", type=float, default=None, help="Optional response-bytes stored in a generated Panther MS summary.")
    parser.add_argument("--comm-downstream-bytes-avg", type=float, default=None, help="Optional downstream-bytes stored in a generated Panther MS summary.")
    parser.add_argument("--source-log", type=str, default="", help="Optional source log path stored in a generated Panther MS summary.")
    parser.add_argument("--output-prefix", type=str, default="", help="Optional comparison-contract output stem override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = PantherConfig(project_root=project_root)
    payload = build_path(cfg)

    out_path = (
        Path(args.out)
        if str(args.out).strip()
        else project_root / "results" / "repro_workflows" / "panther" / "implementation_path.json"
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
            runtime_env["PANTHER_BUNDLE_ROOT"] = str((project_root / "results" / str(args.bundle_root_name).strip()).resolve())
        elif len(explicit_worksets) == 1:
            runtime_env["PANTHER_BUNDLE_ROOT"] = str((project_root / "results" / _default_bundle_root_name(explicit_worksets[0])).resolve())

    if bool(args.export_ms_author_assets) or bool(args.build_ms_summary_from_rankings):
        script = project_root / "src" / "baselines" / "panther" / "ms_author_bridge.py"
        cmd = [sys.executable, str(script)]
        if bool(args.export_ms_author_assets):
            cmd.append("--export-ms-assets")
        if bool(args.build_ms_summary_from_rankings):
            cmd.append("--build-summary-from-rankings")
        if bool(args.write_openpanther_text):
            cmd.append("--write-openpanther-text")
        if bool(args.stage_into_openpanther):
            cmd.append("--stage-into-openpanther")
        if str(args.dataset_slug).strip():
            cmd.extend(["--dataset-slug", str(args.dataset_slug).strip()])
        if str(args.summary_stem).strip():
            cmd.extend(["--summary-stem", str(args.summary_stem).strip()])
        if str(args.run_label).strip():
            cmd.extend(["--run-label", str(args.run_label).strip()])
        if str(args.rankings_jsonl).strip():
            cmd.extend(["--rankings-jsonl", str(args.rankings_jsonl).strip()])
        if str(args.rankings_json).strip():
            cmd.extend(["--rankings-json", str(args.rankings_json).strip()])
        if str(args.rankings_tsv).strip():
            cmd.extend(["--rankings-tsv", str(args.rankings_tsv).strip()])
        if str(args.rankings_npy).strip():
            cmd.extend(["--rankings-npy", str(args.rankings_npy).strip()])
        if int(args.routing_c) > 0:
            cmd.extend(["--routing-c", str(int(args.routing_c))])
        if int(args.query_limit) > 0:
            cmd.extend(["--query-limit", str(int(args.query_limit))])
        if int(args.top_k) > 0:
            cmd.extend(["--top-k", str(int(args.top_k))])
        if int(args.max_points_per_cluster) > 0:
            cmd.extend(["--max-points-per-cluster", str(int(args.max_points_per_cluster))])
        if int(args.selected_clusters) > 0:
            cmd.extend(["--selected-clusters", str(int(args.selected_clusters))])
        if args.latency_total_sec_avg is not None:
            cmd.extend(["--latency-total-sec-avg", str(float(args.latency_total_sec_avg))])
        if args.latency_client_generate_sec_avg is not None:
            cmd.extend(["--latency-client-generate-sec-avg", str(float(args.latency_client_generate_sec_avg))])
        if args.latency_server_query_sec_avg is not None:
            cmd.extend(["--latency-server-query-sec-avg", str(float(args.latency_server_query_sec_avg))])
        if args.latency_client_recover_sec_avg is not None:
            cmd.extend(["--latency-client-recover-sec-avg", str(float(args.latency_client_recover_sec_avg))])
        if args.comm_request_bytes_avg is not None:
            cmd.extend(["--comm-request-bytes-avg", str(float(args.comm_request_bytes_avg))])
        if args.comm_response_bytes_avg is not None:
            cmd.extend(["--comm-response-bytes-avg", str(float(args.comm_response_bytes_avg))])
        if args.comm_downstream_bytes_avg is not None:
            cmd.extend(["--comm-downstream-bytes-avg", str(float(args.comm_downstream_bytes_avg))])
        if str(args.source_log).strip():
            cmd.extend(["--source-log", str(args.source_log).strip()])
        subprocess.run(cmd, cwd=str(project_root), env=runtime_env, check=True)

    if bool(args.export_comparison_contract) or bool(args.write_summary_template):
        script = project_root / "src" / "baselines" / "panther" / "comparison_adapter.py"
        cmd = [sys.executable, str(script), "--dataset", str(args.dataset)]
        if str(args.summary_json).strip():
            cmd.extend(["--summary-json", str(args.summary_json).strip()])
        elif str(args.summary_stem).strip():
            cmd.extend(["--summary-stem", str(args.summary_stem).strip()])
        if str(args.output_prefix).strip():
            cmd.extend(["--output-prefix", str(args.output_prefix).strip()])
        if bool(args.write_summary_template):
            cmd.append("--write-template")
        subprocess.run(cmd, cwd=str(project_root), env=runtime_env, check=True)


if __name__ == "__main__":
    main()
