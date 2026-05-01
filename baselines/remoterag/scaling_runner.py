from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from baselines.public_assets import resolve_requested_worksets, resolve_workset_assets, workset_stats
from baselines.remoterag.config import RemoteRAGConfig


def _write_csv(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _parse_sizes(text: str, defaults: tuple[int, ...]) -> list[int]:
    if not str(text).strip():
        return [int(x) for x in defaults]
    values: list[int] = []
    seen: set[int] = set()
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value in seen:
            continue
        seen.add(value)
        values.append(int(value))
    return values


def _normalized_exact_recall(summary: dict) -> float | None:
    top_k = int(summary.get("top_k", 0)) if summary.get("top_k") is not None else 0
    if top_k <= 0:
        return None
    overlap_count = float(summary.get("mean_exact_topk_overlap_count", 0.0))
    return float(overlap_count) / float(top_k)


def _build_scaling_row(*, summary: dict, summary_path: Path, run_label: str, summary_stem: str) -> dict:
    retrieve_counts = dict(summary.get("retrieve_mode_counts", {}))
    total_queries = max(int(summary.get("num_queries", 0)), 1)
    direct_rate = float(retrieve_counts.get("direct_indices", 0)) / float(total_queries)
    ot_rate = float(retrieve_counts.get("ot_k_out_of_kprime", 0)) / float(total_queries)
    lat_client = float(summary.get("time_client_encrypt_query_sec_avg", 0.0))
    lat_server = float(summary.get("time_server_phe_score_sec_avg", 0.0))
    lat_recover = float(summary.get("time_client_decrypt_sort_sec_avg", 0.0))
    return {
        "run_label": str(run_label),
        "summary_stem": str(summary_stem),
        "summary_json": str(summary_path),
        "rows_jsonl": str(summary.get("rows_jsonl", "")),
        "num_docs": int(summary.get("num_docs", 0)) if summary.get("num_docs") is not None else None,
        "num_queries": int(summary.get("num_queries", 0)),
        "top_k": int(summary.get("top_k", 0)) if summary.get("top_k") is not None else None,
        "epsilon": float(summary.get("epsilon", 0.0)) if summary.get("epsilon") is not None else None,
        "avg_exact_recall_at_k": _normalized_exact_recall(summary),
        "exact_topk_order_match_rate": float(summary.get("exact_topk_order_match_rate", 0.0)),
        "candidate_cover_rate": float(summary.get("module1_candidate_cover_rate", 0.0)),
        "direct_retrieve_rate": float(direct_rate),
        "ot_retrieve_rate": float(ot_rate),
        "latency_total_sec_avg": float(lat_client + lat_server + lat_recover),
        "latency_client_generate_sec_avg": float(lat_client),
        "latency_server_query_sec_avg": float(lat_server),
        "latency_client_recover_sec_avg": float(lat_recover),
        "comm_request_bytes_avg": float(summary.get("comm_client_encrypt_query_bytes_avg", 0.0))
        + float(summary.get("comm_retrieve_request_bytes_avg", 0.0)),
        "comm_response_bytes_avg": float(summary.get("comm_server_score_response_bytes_avg", 0.0)),
        "comm_downstream_bytes_avg": float(summary.get("comm_retrieve_response_bytes_avg", 0.0)),
    }


def _per_size_stem(base_prefix: str, num_docs: int) -> str:
    return f"{str(base_prefix).strip()}_n{int(num_docs)}"


def _ensure_module2_summary(
    *,
    project_root: Path,
    num_docs: int,
    workset_name: str,
    cfg: RemoteRAGConfig,
    prepare_assets: bool,
    reuse_existing: bool,
    module1_query_limit: int,
    module1_k_prime_cap: int,
    query_limit: int,
    epsilon: float,
    target_radius: float,
    quantization_scale: int,
    paillier_bits: int,
    ot_prime_bits: int,
    output_stem: str,
) -> Path:
    result_root = project_root / "results" / "repro_workflows" / cfg.result_root_name
    summary_path = result_root / f"{output_stem}.json"
    if reuse_existing and summary_path.exists():
        return summary_path

    script = project_root / "src" / "baselines" / "remoterag" / "module2_phe_ot.py"
    cmd = [
        sys.executable,
        str(script),
        "--size",
        str(int(num_docs)),
        "--output-prefix",
        str(output_stem),
    ]
    if str(workset_name).strip():
        cmd.extend(["--workset-name", str(workset_name).strip()])
    if prepare_assets:
        cmd.append("--prepare-assets")
    if not reuse_existing:
        cmd.append("--no-reuse-existing")
    if int(module1_query_limit) > 0:
        cmd.extend(["--module1-query-limit", str(int(module1_query_limit))])
    if int(module1_k_prime_cap) > 0:
        cmd.extend(["--module1-k-prime-cap", str(int(module1_k_prime_cap))])
    if int(query_limit) > 0:
        cmd.extend(["--query-limit", str(int(query_limit))])
    if float(epsilon) > 0.0:
        cmd.extend(["--epsilon", str(float(epsilon))])
    if float(target_radius) > 0.0:
        cmd.extend(["--target-radius", str(float(target_radius))])
    if int(quantization_scale) > 0:
        cmd.extend(["--quantization-scale", str(int(quantization_scale))])
    if int(paillier_bits) > 0:
        cmd.extend(["--paillier-bits", str(int(paillier_bits))])
    if int(ot_prime_bits) > 0:
        cmd.extend(["--ot-prime-bits", str(int(ot_prime_bits))])
    subprocess.run(cmd, cwd=str(project_root), check=True)
    if not summary_path.exists():
        raise FileNotFoundError(f"RemoteRAG module-2 summary missing after run: {summary_path}")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a per-size RemoteRAG scaling summary from module-2 outputs."
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="",
        help="Optional comma-separated size subset. Default uses the config defaults.",
    )
    parser.add_argument(
        "--workset-name",
        action="append",
        default=[],
        help="Optional explicit PRISMA source workset name. Repeat for multiple worksets. When provided, these worksets define the scaling rows directly.",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=0,
        help="Optional positive Module-2 query limit override.",
    )
    parser.add_argument(
        "--module1-query-limit",
        type=int,
        default=0,
        help="Optional positive Module-1 query limit override.",
    )
    parser.add_argument(
        "--module1-k-prime-cap",
        type=int,
        default=0,
        help="Optional Module-1 k' cap forwarded into Module 2 generation.",
    )
    parser.add_argument("--epsilon", type=float, default=0.0, help="Optional explicit epsilon override.")
    parser.add_argument(
        "--target-radius",
        type=float,
        default=0.0,
        help="Optional target perturbation radius when epsilon is not fixed.",
    )
    parser.add_argument(
        "--quantization-scale",
        type=int,
        default=0,
        help="Optional fixed-point scale override.",
    )
    parser.add_argument("--paillier-bits", type=int, default=0, help="Optional Paillier modulus bits override.")
    parser.add_argument("--ot-prime-bits", type=int, default=0, help="Optional OT group prime bits override.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional scaling summary stem under results/repro_workflows/remoterag/.",
    )
    parser.add_argument(
        "--prepare-assets",
        action="store_true",
        help="Build a public RemoteRAG asset manifest from ready PRISMA worksets before running.",
    )
    parser.set_defaults(reuse_existing=True)
    parser.add_argument("--reuse-existing", dest="reuse_existing", action="store_true", help="Reuse existing per-size module-2 summaries when available. Enabled by default.")
    parser.add_argument("--no-reuse-existing", dest="reuse_existing", action="store_false", help="Force rerunning module-2 for each requested size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = RemoteRAGConfig(project_root=project_root)
    if bool(args.prepare_assets):
        prep_script = project_root / "src" / "baselines" / "remoterag" / "prepare_assets.py"
        cmd = [sys.executable, str(prep_script)]
        for workset_name in list(args.workset_name):
            if str(workset_name).strip():
                cmd.extend(["--workset-name", str(workset_name).strip()])
        subprocess.run(cmd, cwd=str(project_root), check=True)

    requested_worksets = [str(x).strip() for x in list(args.workset_name) if str(x).strip()]
    if requested_worksets:
        workset_rows = []
        for workset_name in resolve_requested_worksets(project_root, requested_worksets):
            stats = workset_stats(resolve_workset_assets(project_root, workset_name))
            workset_rows.append((str(workset_name), int(stats["num_docs"])))
    else:
        sizes = _parse_sizes(str(args.sizes), cfg.default_sizes)
        workset_rows = [("", int(num_docs)) for num_docs in sizes]
    result_root = project_root / "results" / "repro_workflows" / cfg.result_root_name
    output_stem = (
        str(args.output_prefix).strip()
        if str(args.output_prefix).strip()
        else str(cfg.default_scaling_summary_stem)
    )

    rows: list[dict] = []
    for workset_name, num_docs in sorted(workset_rows, key=lambda item: (int(item[1]), str(item[0]))):
        per_size_stem = _per_size_stem(output_stem, int(num_docs))
        if str(workset_name).strip():
            per_size_stem = f"{per_size_stem}_{str(workset_name).strip().replace(' ', '_')}"
        summary_path = _ensure_module2_summary(
            project_root=project_root,
            num_docs=int(num_docs),
            workset_name=str(workset_name),
            cfg=cfg,
            prepare_assets=bool(args.prepare_assets),
            reuse_existing=bool(args.reuse_existing),
            module1_query_limit=int(args.module1_query_limit),
            module1_k_prime_cap=int(args.module1_k_prime_cap),
            query_limit=int(args.query_limit),
            epsilon=float(args.epsilon),
            target_radius=float(args.target_radius),
            quantization_scale=int(args.quantization_scale),
            paillier_bits=int(args.paillier_bits),
            ot_prime_bits=int(args.ot_prime_bits),
            output_stem=per_size_stem,
        )
        summary = _load_json(summary_path)
        rows.append(
            _build_scaling_row(
                summary=summary,
                summary_path=summary_path,
                run_label=f"remoterag_n{int(num_docs)}",
                summary_stem=per_size_stem,
            )
        )
        if str(workset_name).strip():
            rows[-1]["workset_name"] = str(workset_name)

    json_path = result_root / f"{output_stem}.json"
    csv_path = result_root / f"{output_stem}.csv"
    jsonl_path = result_root / f"{output_stem}.jsonl"
    payload = {
        "baseline_slug": "remoterag",
        "baseline_display_name": "RemoteRAG",
        "paper_url": str(cfg.paper_url),
        "sizes": [int(row["num_docs"]) for row in rows if row.get("num_docs") is not None],
        "query_limit_requested": int(args.query_limit),
        "module1_query_limit_requested": int(args.module1_query_limit),
        "module1_k_prime_cap_requested": int(args.module1_k_prime_cap),
        "definition": {
            "client_generate_query": "Paillier-encrypted query encoding for Module 2 plus any retrieval request bytes",
            "server_query": "PHE scoring over Module-1 candidates plus any retrieval-serving work",
            "client_recover_docs": "decrypt and sort scores plus any document recovery bytes",
            "recall_metric": "avg_exact_recall_at_k is normalized exact top-k overlap count divided by top_k",
        },
        "artifacts": {
            "summary_json": str(json_path),
            "summary_csv": str(csv_path),
            "rows_jsonl": str(jsonl_path),
        },
        "rows": rows,
    }
    _save_json(json_path, payload)
    _write_csv(rows, csv_path)
    _write_jsonl(rows, jsonl_path)
    print(f"[saved] {json_path}")
    print(f"[saved] {csv_path}")
    print(f"[saved] {jsonl_path}")


if __name__ == "__main__":
    main()
