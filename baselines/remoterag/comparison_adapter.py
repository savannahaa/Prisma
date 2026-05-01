from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import json
import subprocess
import sys
from pathlib import Path

from baselines.common import ComparisonContractRow, write_contract_rows_csv, write_contract_rows_jsonl
from baselines.remoterag.config import RemoteRAGConfig


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalized_exact_recall(summary: dict) -> float | None:
    top_k = int(summary.get("top_k", 0)) if summary.get("top_k") is not None else 0
    if top_k <= 0:
        return None
    return float(summary.get("mean_exact_topk_overlap_count", 0.0)) / float(top_k)


def _build_row_from_summary(
    *,
    cfg: RemoteRAGConfig,
    summary: dict,
    summary_path: Path,
    comparison_axis: str,
    run_label_override: str = "",
) -> ComparisonContractRow:
    retrieve_counts = dict(summary.get("retrieve_mode_counts", {}))
    total_queries = max(int(summary.get("num_queries", 0)), 1)
    direct_rate = float(retrieve_counts.get("direct_indices", 0)) / float(total_queries)
    ot_rate = float(retrieve_counts.get("ot_k_out_of_kprime", 0)) / float(total_queries)
    lat_client = float(summary.get("time_client_encrypt_query_sec_avg", 0.0))
    lat_server = float(summary.get("time_server_phe_score_sec_avg", 0.0))
    lat_recover = float(summary.get("time_client_decrypt_sort_sec_avg", 0.0))
    return ComparisonContractRow(
        baseline_slug="remoterag",
        baseline_display_name="RemoteRAG",
        paper_url=str(summary.get("paper_url", cfg.paper_url)),
        contract_version="v1",
        comparison_axis=str(comparison_axis),
        run_label=str(run_label_override).strip() if str(run_label_override).strip() else str(summary_path.stem),
        num_docs=int(summary.get("num_docs", 0)) if summary.get("num_docs") is not None else None,
        num_clusters=None,
        num_queries=int(summary.get("num_queries", 0)),
        top_k=int(summary.get("top_k", 0)) if summary.get("top_k") is not None else None,
        latency_total_sec_avg=float(lat_client + lat_server + lat_recover),
        latency_client_generate_sec_avg=float(lat_client),
        latency_server_query_sec_avg=float(lat_server),
        latency_client_recover_sec_avg=float(lat_recover),
        comm_request_bytes_avg=float(summary.get("comm_client_encrypt_query_bytes_avg", 0.0))
        + float(summary.get("comm_retrieve_request_bytes_avg", 0.0)),
        comm_response_bytes_avg=float(summary.get("comm_server_score_response_bytes_avg", 0.0)),
        comm_downstream_bytes_avg=float(summary.get("comm_retrieve_response_bytes_avg", 0.0)),
        mean_first_relevant_rank=None,
        top1_hit_rate=None,
        exact_topk_overlap_mean=_normalized_exact_recall(summary),
        exact_topk_order_match_rate=float(summary.get("exact_topk_order_match_rate", 0.0)),
        candidate_cover_rate=float(summary.get("module1_candidate_cover_rate", 0.0)),
        direct_retrieve_rate=float(direct_rate),
        ot_retrieve_rate=float(ot_rate),
        real_cluster_hit_rate=None,
        source_summary_json=str(summary_path),
        source_rows_jsonl=str(summary.get("rows_jsonl", "")),
        notes=(
            "RemoteRAG row combines PHE query bytes with retrieval-request bytes in comm_request_bytes_avg.",
            "exact_topk_overlap_mean stores normalized exact Recall@k (mean overlap count divided by top_k).",
        ),
    )


def _ensure_remoterag_outputs(project_root: Path, summary_path: Path) -> None:
    if summary_path.exists():
        return
    script = project_root / "src" / "baselines" / "remoterag" / "driver.py"
    subprocess.run([sys.executable, str(script), "--run-module2"], cwd=str(project_root), check=True)


def _ensure_scaling_outputs(project_root: Path, scaling_summary_path: Path) -> None:
    if scaling_summary_path.exists():
        return
    script = project_root / "src" / "baselines" / "remoterag" / "scaling_runner.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-prefix",
            str(scaling_summary_path.stem),
        ],
        cwd=str(project_root),
        check=True,
    )


def build_rows_from_single_summary(
    *,
    cfg: RemoteRAGConfig,
    summary_path: Path,
) -> list[ComparisonContractRow]:
    summary = _load_json(summary_path)
    return [
        _build_row_from_summary(
            cfg=cfg,
            summary=summary,
            summary_path=summary_path,
            comparison_axis="database_size",
        )
    ]


def build_rows_from_scaling_summary(
    *,
    cfg: RemoteRAGConfig,
    scaling_summary_path: Path,
) -> list[ComparisonContractRow]:
    payload = _load_json(scaling_summary_path)
    rows: list[ComparisonContractRow] = []
    for item in payload.get("rows", []):
        if not isinstance(item, dict):
            continue
        summary_path = Path(str(item.get("summary_json", "")).strip())
        if not summary_path.is_absolute():
            summary_path = scaling_summary_path.parent / summary_path
        if not summary_path.exists():
            raise FileNotFoundError(f"RemoteRAG per-size summary missing: {summary_path}")
        summary = _load_json(summary_path)
        rows.append(
            _build_row_from_summary(
                cfg=cfg,
                summary=summary,
                summary_path=summary_path,
                comparison_axis="database_size",
                run_label_override=str(item.get("run_label", "")),
            )
        )
    return sorted(
        rows,
        key=lambda row: (row.num_docs if row.num_docs is not None else -1, row.run_label),
    )


def build_rows(*, cfg: RemoteRAGConfig, summary_path: Path) -> list[ComparisonContractRow]:
    return build_rows_from_single_summary(cfg=cfg, summary_path=summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize RemoteRAG outputs into the shared comparison contract.")
    parser.add_argument(
        "--scaling-summary-stem",
        type=str,
        default="remoterag_scaling_summary",
        help="RemoteRAG scaling summary stem under results/repro_workflows/remoterag/.",
    )
    parser.add_argument(
        "--summary-stem",
        type=str,
        default="",
        help="Optional legacy single RemoteRAG module-2 summary stem under results/repro_workflows/remoterag/.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional explicit single RemoteRAG summary json path.",
    )
    parser.add_argument("--output-prefix", type=str, default="remoterag_comparison_contract", help="Output stem under results/repro_workflows/remoterag/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = RemoteRAGConfig(project_root=project_root)
    result_root = project_root / "results" / "repro_workflows" / cfg.result_root_name

    if str(args.summary_json).strip():
        rows = build_rows_from_single_summary(
            cfg=cfg,
            summary_path=Path(str(args.summary_json).strip()),
        )
    elif str(args.summary_stem).strip():
        summary_path = result_root / f"{str(args.summary_stem).strip()}.json"
        _ensure_remoterag_outputs(project_root, summary_path)
        rows = build_rows_from_single_summary(
            cfg=cfg,
            summary_path=summary_path,
        )
    else:
        scaling_summary_path = result_root / f"{str(args.scaling_summary_stem).strip()}.json"
        _ensure_scaling_outputs(project_root, scaling_summary_path)
        rows = build_rows_from_scaling_summary(
            cfg=cfg,
            scaling_summary_path=scaling_summary_path,
        )

    jsonl_path = result_root / f"{str(args.output_prefix).strip()}.jsonl"
    csv_path = result_root / f"{str(args.output_prefix).strip()}.csv"
    write_contract_rows_jsonl(jsonl_path, rows)
    write_contract_rows_csv(csv_path, rows)
    print(f"[saved] {jsonl_path}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
