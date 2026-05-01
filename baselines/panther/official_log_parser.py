from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

from baselines.panther.common import bridge_root, load_json, save_json, write_jsonl
from baselines.panther.config import PantherConfig
from baselines.panther.ms_author_bridge import build_summary_from_rankings


_MB = 1024.0 * 1024.0


def _read_text(path: Path | None) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _last_match(text: str, pattern: str) -> re.Match[str] | None:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        return None
    return matches[-1]


def _last_float(text: str, pattern: str) -> float | None:
    match = _last_match(text, pattern)
    if match is None:
        return None
    return float(match.group(1))


def _last_int(text: str, pattern: str) -> int | None:
    match = _last_match(text, pattern)
    if match is None:
        return None
    return int(match.group(1))


def _normalize_dataset(value: str) -> str:
    return str(value).strip().lower()


def _parse_topk_ids(text: str) -> tuple[int | None, list[int]]:
    header = _last_match(text, r"(\d+)-NNs IDs:")
    if header is None:
        return None, []
    top_k = int(header.group(1))
    tail = text[header.end() :]
    ids_match = re.search(r"\(([0-9 ]+)\)", tail, flags=re.DOTALL)
    if ids_match is None:
        return top_k, []
    ids = [int(part) for part in ids_match.group(1).strip().split() if part.strip()]
    return top_k, ids[:top_k]


def _parse_common_metrics(client_text: str, server_text: str) -> dict:
    client_total_ms = _last_float(client_text, r"Total time:\s*([0-9.]+)\s*ms")
    server_total_ms = _last_float(server_text, r"Total time:\s*([0-9.]+)\s*ms")
    top_k, topk_ids = _parse_topk_ids(client_text)
    accuracy_match = _last_match(client_text, r"Accuracy:\s*(\d+)\s*/\s*(\d+)\s*=\s*([0-9.]+)")
    accuracy_ratio = None
    if accuracy_match is not None:
        accuracy_ratio = float(accuracy_match.group(3))
    return {
        "top_k": int(top_k) if top_k is not None else None,
        "topk_ids": list(topk_ids),
        "query_index": _last_int(client_text, r"Bridge query index:\s*(\d+)"),
        "dataset_slug": (_last_match(client_text, r"Bridge dataset slug:\s*([^\s]+)") or _last_match(server_text, r"Bridge dataset slug:\s*([^\s]+)")).group(1)
        if (_last_match(client_text, r"Bridge dataset slug:\s*([^\s]+)") or _last_match(server_text, r"Bridge dataset slug:\s*([^\s]+)"))
        else "",
        "latency_total_sec_avg": (client_total_ms / 1000.0) if client_total_ms is not None else None,
        "latency_server_total_sec": (server_total_ms / 1000.0) if server_total_ms is not None else None,
        "distance_time_sec": (_last_float(client_text, r"Distance time:\s*([0-9.]+)\s*ms") or 0.0) / 1000.0,
        "topk_time_sec": (_last_float(client_text, r"Topk time:\s*([0-9.]+)\s*ms") or 0.0) / 1000.0,
        "pir_time_sec": (_last_float(client_text, r"Pir time:\s*([0-9.]+)\s*ms") or 0.0) / 1000.0,
        "comm_request_bytes_avg": ((_last_float(client_text, r"Total comm:\s*([0-9.]+)\s*MB") or 0.0) * _MB),
        "comm_response_bytes_avg": ((_last_float(server_text, r"Total comm:\s*([0-9.]+)\s*MB") or 0.0) * _MB),
        "accuracy_ratio": accuracy_ratio,
    }


def _bridge_default_summary_stem(dataset_slug: str, query_index: int) -> str:
    safe_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", str(dataset_slug).strip() or "bridge")
    return f"panther_{safe_slug}_query{int(query_index)}_summary"


def _official_default_summary_path(result_root: Path, dataset: str) -> Path:
    return result_root / f"panther_{_normalize_dataset(dataset)}_official_summary.json"


def _official_num_docs(cfg: PantherConfig, dataset: str) -> int | None:
    key = _normalize_dataset(dataset)
    mapping = {
        "sift": int(cfg.parity_size),
        "random_sift": int(cfg.parity_size),
        "deep10m": 10_000_000,
        "random_deep1m": 1_000_000,
        "random_deep10m": 10_000_000,
    }
    return mapping.get(key)


def _bridge_ranking_jsonl_path(result_root: Path, dataset_slug: str, query_index: int) -> Path:
    safe_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", str(dataset_slug).strip() or "bridge")
    return result_root / f"panther_{safe_slug}_query{int(query_index)}_raw_rankings.jsonl"


def _bridge_num_docs(cfg: PantherConfig, dataset_slug: str) -> int:
    manifest_path = bridge_root(cfg) / str(dataset_slug) / "manifest.json"
    if manifest_path.exists():
        payload = load_json(manifest_path)
        try:
            return int(payload.get("num_docs", cfg.default_size))
        except Exception:
            return int(cfg.default_size)
    return int(cfg.default_size)


def _build_bridge_summary(
    *,
    cfg: PantherConfig,
    result_root: Path,
    client_log: Path,
    server_log: Path | None,
    dataset_slug: str,
    query_index: int,
    top_k: int,
    topk_ids: list[int],
    summary_stem: str,
    metrics: dict,
) -> Path:
    num_docs = _bridge_num_docs(cfg, dataset_slug)
    valid_topk_ids = [int(x) for x in topk_ids[: int(top_k)] if 0 <= int(x) < int(num_docs)]
    invalid_count = int(min(len(topk_ids), int(top_k)) - len(valid_topk_ids))
    if not valid_topk_ids:
        raise ValueError(
            f"bridge parser did not find any valid doc indices in client log {client_log}; "
            f"top_k={top_k}, num_docs={num_docs}"
        )
    rankings_path = _bridge_ranking_jsonl_path(result_root, dataset_slug, query_index)
    write_jsonl(
        rankings_path,
        [
            {
                "query_index": int(query_index),
                "ranked_doc_indices": list(valid_topk_ids),
            }
        ],
    )
    args = SimpleNamespace(
        routing_c=int(cfg.default_cluster_info_selector_c),
        query_limit=-1,
        top_k=int(top_k),
        summary_stem=str(summary_stem),
        run_label=f"panther_{dataset_slug}_query{int(query_index)}",
        rankings_jsonl=str(rankings_path),
        rankings_json="",
        rankings_tsv="",
        rankings_npy="",
        latency_total_sec_avg=metrics["latency_total_sec_avg"],
        latency_client_generate_sec_avg=None,
        latency_server_query_sec_avg=metrics["latency_server_total_sec"],
        latency_client_recover_sec_avg=None,
        comm_request_bytes_avg=metrics["comm_request_bytes_avg"],
        comm_response_bytes_avg=metrics["comm_response_bytes_avg"],
        comm_downstream_bytes_avg=None,
        source_log=str(client_log),
        note=[
            "Parsed automatically from OpenPanther bridge demo logs.",
            f"bridge_dataset_slug={dataset_slug}",
            f"bridge_query_index={int(query_index)}",
            f"filtered_invalid_doc_indices={int(invalid_count)}",
            f"server_log={str(server_log) if server_log is not None else ''}",
        ],
    )
    build_summary_from_rankings(cfg, args)
    return result_root / f"{summary_stem}.json"


def _build_official_summary(
    *,
    cfg: PantherConfig,
    result_root: Path,
    dataset: str,
    client_log: Path,
    server_log: Path | None,
    summary_path: Path,
    metrics: dict,
) -> Path:
    key = _normalize_dataset(dataset)
    payload = {
        "baseline_slug": "panther",
        "baseline_display_name": "Panther",
        "paper_url": str(cfg.paper_url),
        "code_url": str(cfg.code_url),
        "mode": "official_author_code",
        "comparison_axis": "ann_benchmark_dataset",
        "run_label": f"panther_{key}_official",
        "dataset": key,
        "num_docs": _official_num_docs(cfg, key),
        "num_clusters": None,
        "num_queries": 1,
        "top_k": metrics["top_k"],
        "latency_total_sec_avg": metrics["latency_total_sec_avg"],
        "latency_client_generate_sec_avg": None,
        "latency_server_query_sec_avg": metrics["latency_server_total_sec"],
        "latency_client_recover_sec_avg": None,
        "comm_request_bytes_avg": metrics["comm_request_bytes_avg"],
        "comm_response_bytes_avg": metrics["comm_response_bytes_avg"],
        "comm_downstream_bytes_avg": None,
        "exact_recall_at_k": metrics["accuracy_ratio"],
        "top1_hit_rate": None,
        "source_log": str(client_log),
        "rows_jsonl": "",
        "notes": [
            "Parsed automatically from OpenPanther official demo logs.",
            "This summary currently reflects a single wrapped demo run rather than a full multi-query benchmark sweep.",
            f"server_log={str(server_log) if server_log is not None else ''}",
        ],
    }
    save_json(summary_path, payload)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse OpenPanther demo logs into repo-local summary artifacts.")
    parser.add_argument("--dataset", type=str, required=True, help="Panther dataset key such as sift, bridge, deep10m, random_sift.")
    parser.add_argument("--client-log", type=str, required=True, help="Path to the client log produced by run_panther_official_demo.sh.")
    parser.add_argument("--server-log", type=str, default="", help="Optional server log produced by run_panther_official_demo.sh.")
    parser.add_argument("--dataset-slug", type=str, default="", help="Optional bridge dataset slug override.")
    parser.add_argument("--query-index", type=int, default=-1, help="Optional bridge query index override.")
    parser.add_argument("--summary-stem", type=str, default="", help="Optional summary stem under results/repro_workflows/panther/ for bridge runs.")
    parser.add_argument("--summary-json", type=str, default="", help="Optional explicit summary json output path for official parity runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = PantherConfig(project_root=project_root)
    result_root = project_root / "results" / "repro_workflows" / "panther"
    result_root.mkdir(parents=True, exist_ok=True)

    dataset = _normalize_dataset(args.dataset)
    client_log = Path(str(args.client_log)).resolve()
    server_log = Path(str(args.server_log)).resolve() if str(args.server_log).strip() else None
    client_text = _read_text(client_log)
    server_text = _read_text(server_log)
    metrics = _parse_common_metrics(client_text, server_text)

    if dataset == "bridge":
        dataset_slug = str(args.dataset_slug).strip() or str(metrics["dataset_slug"]).strip() or str(cfg.default_author_input_slug)
        query_index = int(args.query_index) if int(args.query_index) >= 0 else int(metrics["query_index"] or 0)
        top_k = int(metrics["top_k"] or cfg.default_top_k)
        if not metrics["topk_ids"]:
            raise ValueError(f"failed to parse bridge top-k ids from client log: {client_log}")
        summary_stem = str(args.summary_stem).strip() or _bridge_default_summary_stem(dataset_slug, query_index)
        summary_path = _build_bridge_summary(
            cfg=cfg,
            result_root=result_root,
            client_log=client_log,
            server_log=server_log,
            dataset_slug=dataset_slug,
            query_index=query_index,
            top_k=top_k,
            topk_ids=list(metrics["topk_ids"]),
            summary_stem=summary_stem,
            metrics=metrics,
        )
    else:
        summary_path = (
            Path(str(args.summary_json)).resolve()
            if str(args.summary_json).strip()
            else _official_default_summary_path(result_root, dataset)
        )
        summary_path = _build_official_summary(
            cfg=cfg,
            result_root=result_root,
            dataset=dataset,
            client_log=client_log,
            server_log=server_log,
            summary_path=summary_path,
            metrics=metrics,
        )

    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
