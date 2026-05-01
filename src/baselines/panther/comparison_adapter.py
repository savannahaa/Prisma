from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import json
from pathlib import Path

from baselines.common import ComparisonContractRow, write_contract_rows_csv, write_contract_rows_jsonl
from baselines.panther.config import PantherConfig


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _first_float(payload: dict, *keys: str) -> float | None:
    for key in keys:
        if key not in payload or payload.get(key) is None:
            continue
        try:
            return float(payload.get(key))
        except Exception:
            continue
    return None


def _first_int(payload: dict, *keys: str) -> int | None:
    for key in keys:
        if key not in payload or payload.get(key) is None:
            continue
        try:
            return int(payload.get(key))
        except Exception:
            try:
                return int(float(payload.get(key)))
            except Exception:
                continue
    return None


def _bytes_value(payload: dict, *, byte_key: str, mb_key: str) -> float | None:
    raw_bytes = _first_float(payload, byte_key)
    if raw_bytes is not None:
        return float(raw_bytes)
    raw_mb = _first_float(payload, mb_key)
    if raw_mb is not None:
        return float(raw_mb) * 1024.0 * 1024.0
    return None


def _dataset_profile(dataset: str) -> str:
    key = str(dataset).strip().lower()
    if key in {"ms", "msmarco", "ms_marco", "ms_qrels", "qrels_bundle", "ms_bundle"}:
        return "ms_qrels"
    return "ann_benchmark"


def _default_summary_path(project_root: Path, cfg: PantherConfig, dataset: str) -> Path:
    result_root = project_root / "results" / "repro_workflows" / "panther"
    profile = _dataset_profile(dataset)
    if profile == "ms_qrels":
        stem = f"{str(cfg.default_summary_stem).strip()}.json"
    else:
        stem = f"{str(cfg.parity_summary_stem).strip()}.json"
    return result_root / stem


def _template_path(result_root: Path, cfg: PantherConfig, dataset: str) -> Path:
    profile = _dataset_profile(dataset)
    if profile == "ms_qrels":
        return result_root / f"{str(cfg.default_summary_stem).strip()}.template.json"
    return result_root / f"{str(cfg.parity_summary_stem).strip()}.template.json"


def _ms_template_payload(cfg: PantherConfig) -> dict:
    return {
        "baseline_slug": "panther",
        "baseline_display_name": "Panther",
        "paper_url": str(cfg.paper_url),
        "code_url": str(cfg.code_url),
        "mode": "ms_aligned_author_code",
        "comparison_axis": "paper_faithful_debug_bridge",
        "run_label": "panther_ms_aligned",
        "dataset": "ms",
        "num_docs": int(cfg.default_size),
        "num_clusters": None,
        "num_queries": int(cfg.default_query_limit),
        "top_k": int(cfg.default_top_k),
        "cost_reporting_mode": "per_query_avg",
        "latency_total_sec_avg": None,
        "latency_client_generate_sec_avg": None,
        "latency_server_query_sec_avg": None,
        "latency_client_recover_sec_avg": None,
        "comm_request_bytes_avg": None,
        "comm_response_bytes_avg": None,
        "comm_downstream_bytes_avg": None,
        "comm_client_generate_query_mb": None,
        "comm_server_query_mb": None,
        "comm_two_stage_total_mb": None,
        "avg_exact_recall_at_k": None,
        "mean_first_relevant_rank": None,
        "top1_hit_rate": None,
        "candidate_cover_rate": None,
        "exact_topk_overlap_mean": None,
        "source_log": "",
        "rows_jsonl": "",
        "notes": [
            "Fill this template with metrics parsed from the Panther bridge/debug run.",
            "For any final comparison, prefer avg_exact_recall_at_k plus communication/time aliases over qrels hit-style diagnostics.",
            "The intended input source here is the repo's MS bridge bundle, which is useful for runtime/semantic debugging but is not itself the final paper-faithful Panther dataset.",
            "Store any top-k doc-id ranking rows in rows_jsonl so the existing evaluation bridge can be reproduced.",
        ],
    }


def _ann_template_payload(cfg: PantherConfig, dataset: str) -> dict:
    return {
        "baseline_slug": "panther",
        "baseline_display_name": "Panther",
        "paper_url": str(cfg.paper_url),
        "code_url": str(cfg.code_url),
        "mode": "official_author_code",
        "comparison_axis": "ann_benchmark_dataset",
        "run_label": f"panther_{str(dataset).strip().lower()}_official",
        "dataset": str(dataset).strip().lower(),
        "num_docs": int(cfg.parity_size),
        "num_clusters": None,
        "num_queries": int(cfg.parity_query_limit),
        "top_k": int(cfg.parity_top_k),
        "latency_total_sec_avg": None,
        "latency_client_generate_sec_avg": None,
        "latency_server_query_sec_avg": None,
        "latency_client_recover_sec_avg": None,
        "comm_request_bytes_avg": None,
        "comm_response_bytes_avg": None,
        "comm_downstream_bytes_avg": None,
        "comm_client_generate_query_mb": None,
        "comm_server_query_mb": None,
        "comm_two_stage_total_mb": None,
        "exact_recall_at_k": None,
        "avg_exact_recall_at_k": None,
        "top1_hit_rate": None,
        "source_log": "",
        "rows_jsonl": "",
        "notes": [
            "Fill this template with metrics parsed from the official OpenPanther run.",
            "For paper-faithful Panther comparison, exact_recall_at_k or avg_exact_recall_at_k should be paired with request/response communication totals only.",
            "For ANN benchmark rows, exact_recall_at_k will still be mapped into exact_topk_overlap_mean in the shared contract.",
        ],
    }


def build_rows(*, cfg: PantherConfig, summary_path: Path) -> list[ComparisonContractRow]:
    summary = _load_json(summary_path)
    dataset = str(summary.get("dataset", cfg.default_dataset)).strip().lower() or str(cfg.default_dataset)
    profile = _dataset_profile(dataset)
    lat_client = _first_float(summary, "latency_client_generate_sec_avg", "time_client_generate_query_sec_avg")
    lat_server = _first_float(summary, "latency_server_query_sec_avg", "time_server_query_sec_avg")
    lat_recover = _first_float(summary, "latency_client_recover_sec_avg", "time_client_recover_docs_sec_avg")
    lat_total = _first_float(summary, "latency_total_sec_avg", "time_total_sec_avg")
    if lat_total is None:
        lat_total = float((lat_client or 0.0) + (lat_server or 0.0) + (lat_recover or 0.0))

    request_bytes = _bytes_value(
        summary,
        byte_key="comm_request_bytes_avg",
        mb_key="comm_request_mb_avg",
    )
    response_bytes = _bytes_value(
        summary,
        byte_key="comm_response_bytes_avg",
        mb_key="comm_response_mb_avg",
    )
    downstream_bytes = _bytes_value(
        summary,
        byte_key="comm_downstream_bytes_avg",
        mb_key="comm_downstream_mb_avg",
    )
    notes = [str(x) for x in summary.get("notes", []) if str(x).strip()]
    mean_first_relevant_rank = None
    top1_hit_rate = None
    exact_topk_overlap_mean = None
    exact_topk_order_match_rate = None
    candidate_cover_rate = _first_float(summary, "candidate_cover_rate")
    comparison_axis = str(summary.get("comparison_axis", "qrels_bundle" if profile == "ms_qrels" else "ann_benchmark_dataset"))

    if profile == "ms_qrels":
        mean_first_relevant_rank = _first_float(summary, "mean_first_relevant_rank")
        top1_hit_rate = _first_float(summary, "top1_hit_rate", "strict_hit_rate_at_1", "top1_strict_hit_rate")
        exact_topk_overlap_mean = _first_float(summary, "exact_topk_overlap_mean", "candidate_overlap_mean")
        exact_topk_order_match_rate = _first_float(summary, "exact_topk_order_match_rate")
        notes.append("Panther MS-aligned rows should be interpreted against the repo's qrels-aligned MS bundle, not the official ANN benchmark tables.")
    else:
        exact_topk_overlap_mean = _first_float(summary, "exact_recall_at_k", "recall_at_k", "exact_topk_overlap_mean")
        top1_hit_rate = _first_float(summary, "top1_hit_rate")
        exact_topk_order_match_rate = _first_float(summary, "exact_topk_order_match_rate")
        notes.append("For Panther ANN benchmark rows, exact_topk_overlap_mean stores exact Recall@k against ground-truth top-k.")

    row = ComparisonContractRow(
        baseline_slug="panther",
        baseline_display_name="Panther",
        paper_url=str(summary.get("paper_url", cfg.paper_url)),
        contract_version="v1",
        comparison_axis=str(comparison_axis),
        run_label=str(summary.get("run_label", summary_path.stem.replace("_summary", ""))),
        num_docs=_first_int(summary, "num_docs"),
        num_clusters=_first_int(summary, "num_clusters"),
        num_queries=_first_int(summary, "num_queries"),
        top_k=_first_int(summary, "top_k"),
        latency_total_sec_avg=lat_total,
        latency_client_generate_sec_avg=lat_client,
        latency_server_query_sec_avg=lat_server,
        latency_client_recover_sec_avg=lat_recover,
        comm_request_bytes_avg=request_bytes,
        comm_response_bytes_avg=response_bytes,
        comm_downstream_bytes_avg=downstream_bytes,
        mean_first_relevant_rank=mean_first_relevant_rank,
        top1_hit_rate=top1_hit_rate,
        exact_topk_overlap_mean=exact_topk_overlap_mean,
        exact_topk_order_match_rate=exact_topk_order_match_rate,
        candidate_cover_rate=candidate_cover_rate,
        direct_retrieve_rate=None,
        ot_retrieve_rate=None,
        real_cluster_hit_rate=_first_float(summary, "real_cluster_hit_rate"),
        source_summary_json=str(summary_path),
        source_rows_jsonl=str(summary.get("rows_jsonl", "")),
        notes=tuple(notes),
    )
    return [row]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Panther outputs into the shared comparison contract.")
    parser.add_argument("--summary-stem", type=str, default="", help="Optional Panther summary stem under results/repro_workflows/panther/.")
    parser.add_argument("--summary-json", type=str, default="", help="Optional explicit Panther summary json path.")
    parser.add_argument("--dataset", type=str, default="ms", help="Dataset/profile label used for default summary/template names. Use `ms` for mainline and `sift` for parity.")
    parser.add_argument("--output-prefix", type=str, default="panther_comparison_contract", help="Output stem under results/repro_workflows/panther/.")
    parser.add_argument("--write-template", action="store_true", help="Write a Panther summary template json and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = PantherConfig(project_root=project_root)
    result_root = project_root / "results" / "repro_workflows" / "panther"
    result_root.mkdir(parents=True, exist_ok=True)

    dataset = str(args.dataset).strip().lower() or str(cfg.default_dataset)
    if bool(args.summary_json):
        summary_path = Path(str(args.summary_json)).resolve()
    elif str(args.summary_stem).strip():
        summary_path = result_root / f"{str(args.summary_stem).strip()}.json"
    else:
        summary_path = _default_summary_path(project_root, cfg, dataset)

    if bool(args.write_template):
        template_path = _template_path(result_root, cfg, dataset)
        if _dataset_profile(dataset) == "ms_qrels":
            payload = _ms_template_payload(cfg)
        else:
            payload = _ann_template_payload(cfg, dataset)
        _save_json(template_path, payload)
        print(f"[saved] {template_path}")
        return

    if not summary_path.exists():
        raise FileNotFoundError(
            f"missing Panther summary json: {summary_path}. "
            "Use --write-template first, then fill the template with metrics parsed from the official OpenPanther run."
        )

    rows = build_rows(cfg=cfg, summary_path=summary_path)
    jsonl_path = result_root / f"{str(args.output_prefix).strip()}.jsonl"
    csv_path = result_root / f"{str(args.output_prefix).strip()}.csv"
    write_contract_rows_jsonl(jsonl_path, rows)
    write_contract_rows_csv(csv_path, rows)
    print(f"[saved] {jsonl_path}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
