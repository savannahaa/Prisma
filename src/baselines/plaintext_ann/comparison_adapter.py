from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

from baselines.common import ComparisonContractRow, write_contract_rows_csv, write_contract_rows_jsonl
from baselines.plaintext_ann.config import PlaintextANNConfig


def _bytes_from_mb(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value) * 1024.0 * 1024.0


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _num_clusters_from_summary(summary: dict) -> int | None:
    direct = summary.get("num_clusters")
    if direct is not None:
        return int(direct)
    input_paths = dict(summary.get("input_paths", {}))
    cluster_info_path = str(input_paths.get("cluster_info_pkl", "")).strip()
    if not cluster_info_path:
        return None
    path = Path(cluster_info_path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        cluster_info = pickle.load(f)
    return int(len(cluster_info.get("chunks", [])))


def _summary_latency_total(summary: dict) -> float:
    if summary.get("latency_total_sec_avg") is not None:
        return float(summary["latency_total_sec_avg"])
    if summary.get("time_total_sec_avg") is not None:
        return float(summary["time_total_sec_avg"])
    if summary.get("time_three_stage_total_sec_avg") is not None:
        return float(summary["time_three_stage_total_sec_avg"])
    return float(summary.get("time_client_generate_query_sec_avg", 0.0)) + float(
        summary.get("time_server_query_sec_avg", 0.0)
    ) + float(summary.get("time_client_recover_docs_sec_avg", 0.0))


def _build_row_from_summary(
    *,
    cfg: PlaintextANNConfig,
    summary: dict,
    summary_path: Path,
    comparison_axis: str,
    num_clusters_override: int | None = None,
    run_label_override: str = "",
) -> ComparisonContractRow:
    notes = [
        "Non-Private PRISMA reuses the same paperfaithful_mainline three-stage latency boundaries, but removes gate and RRDP perturbation.",
        "Communication follows the current mainline definition: request bytes + response bytes only; client rerank is local and not counted.",
        "Server retrieval remains the same FAISS HNSW dense backend as the mainline Track1 path.",
    ]
    if summary.get("measurement_impl") is not None:
        notes.append(f"measurement_impl={summary.get('measurement_impl')}")

    num_clusters = (
        int(num_clusters_override)
        if num_clusters_override is not None
        else _num_clusters_from_summary(summary)
    )
    row = ComparisonContractRow(
        baseline_slug="plaintext_ann",
        baseline_display_name=str(cfg.baseline_display_name),
        paper_url=str(cfg.paper_url),
        contract_version="v1",
        comparison_axis=str(comparison_axis),
        run_label=(
            str(run_label_override).strip()
            if str(run_label_override).strip()
            else summary_path.stem
        ),
        num_docs=int(summary.get("num_docs", 0)) if summary.get("num_docs") is not None else None,
        num_clusters=int(num_clusters) if num_clusters is not None else None,
        num_queries=int(summary.get("num_queries", 0)) if summary.get("num_queries") is not None else None,
        top_k=int(summary.get("top_k", 0)) if summary.get("top_k") is not None else None,
        latency_total_sec_avg=float(_summary_latency_total(summary)),
        latency_client_generate_sec_avg=float(summary.get("time_client_generate_query_sec_avg", 0.0)),
        latency_server_query_sec_avg=float(summary.get("time_server_query_sec_avg", 0.0)),
        latency_client_recover_sec_avg=float(summary.get("time_client_recover_docs_sec_avg", 0.0)),
        comm_request_bytes_avg=_bytes_from_mb(summary.get("comm_client_generate_query_mb_avg")),
        comm_response_bytes_avg=_bytes_from_mb(summary.get("comm_server_query_mb_avg")),
        comm_downstream_bytes_avg=None,
        mean_first_relevant_rank=None,
        top1_hit_rate=None,
        exact_topk_overlap_mean=float(summary.get("avg_exact_recall_at_k", 0.0)),
        exact_topk_order_match_rate=None,
        candidate_cover_rate=None,
        direct_retrieve_rate=1.0,
        ot_retrieve_rate=0.0,
        real_cluster_hit_rate=None,
        source_summary_json=str(summary_path),
        source_rows_jsonl="",
        notes=tuple(str(note) for note in notes),
    )
    return row


def build_rows_from_scaling_summary(
    *,
    cfg: PlaintextANNConfig,
    scaling_summary_path: Path,
) -> list[ComparisonContractRow]:
    payload = _load_json(scaling_summary_path)
    rows: list[ComparisonContractRow] = []
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        summary_path = Path(str(row.get("summary_json", "")).strip())
        if not summary_path.exists():
            raise FileNotFoundError(f"plaintext ANN per-size summary missing: {summary_path}")
        summary = _load_json(summary_path)
        rows.append(
            _build_row_from_summary(
                cfg=cfg,
                summary=summary,
                summary_path=summary_path,
                comparison_axis="database_size",
                num_clusters_override=(
                    int(row["num_clusters"]) if row.get("num_clusters") is not None else None
                ),
                run_label_override=f"plaintext_ann_n{int(row['num_docs'])}",
            )
        )
    return rows


def build_rows_from_single_summary(
    *,
    cfg: PlaintextANNConfig,
    summary_path: Path,
) -> list[ComparisonContractRow]:
    summary = _load_json(summary_path)
    return [
        _build_row_from_summary(
            cfg=cfg,
            summary=summary,
            summary_path=summary_path,
            comparison_axis="single_run",
        )
    ]


def _ensure_outputs(
    *,
    project_root: Path,
    scaling_summary_path: Path,
) -> None:
    if scaling_summary_path.exists():
        return
    script = project_root / "src" / "baselines" / "plaintext_ann" / "same_pipeline_runner.py"
    subprocess.run([sys.executable, str(script)], cwd=str(project_root), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize plaintext ANN outputs into the shared comparison contract."
    )
    parser.add_argument(
        "--scaling-summary-stem",
        type=str,
        default="plaintext_ann_scaling_summary",
        help="Scaling summary stem under results/repro_workflows/plaintext_ann/.",
    )
    parser.add_argument(
        "--scaling-summary-json",
        type=str,
        default="",
        help="Optional explicit plaintext ANN scaling summary json.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional explicit single plaintext ANN summary json.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="plaintext_ann_comparison_contract",
        help="Output stem under results/repro_workflows/plaintext_ann/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = PlaintextANNConfig(project_root=project_root)
    result_root = project_root / "results" / "repro_workflows" / cfg.result_root_name

    if str(args.summary_json).strip():
        rows = build_rows_from_single_summary(
            cfg=cfg,
            summary_path=Path(str(args.summary_json).strip()),
        )
    else:
        scaling_summary_path = (
            Path(str(args.scaling_summary_json).strip())
            if str(args.scaling_summary_json).strip()
            else result_root / f"{str(args.scaling_summary_stem).strip()}.json"
        )
        _ensure_outputs(project_root=project_root, scaling_summary_path=scaling_summary_path)
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
