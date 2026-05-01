from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ImplementationStage:
    slug: str
    title: str
    status: str
    paper_requirement: str
    repo_bridge: str
    entrypoint: str
    outputs: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImplementationPath:
    slug: str
    display_name: str
    paper_url: str
    summary: str
    current_truth: str
    algorithm_requirements: tuple[str, ...]
    integration_points: tuple[str, ...]
    stages: tuple[ImplementationStage, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonContractRow:
    baseline_slug: str
    baseline_display_name: str
    paper_url: str
    contract_version: str
    comparison_axis: str
    run_label: str
    num_docs: int | None = None
    num_clusters: int | None = None
    num_queries: int | None = None
    top_k: int | None = None
    latency_total_sec_avg: float | None = None
    latency_client_generate_sec_avg: float | None = None
    latency_server_query_sec_avg: float | None = None
    latency_client_recover_sec_avg: float | None = None
    comm_request_bytes_avg: float | None = None
    comm_response_bytes_avg: float | None = None
    comm_downstream_bytes_avg: float | None = None
    mean_first_relevant_rank: float | None = None
    top1_hit_rate: float | None = None
    exact_topk_overlap_mean: float | None = None
    exact_topk_order_match_rate: float | None = None
    candidate_cover_rate: float | None = None
    direct_retrieve_rate: float | None = None
    ot_retrieve_rate: float | None = None
    real_cluster_hit_rate: float | None = None
    source_summary_json: str = ""
    source_rows_jsonl: str = ""
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def write_manifest(path: Path, implementation_path: ImplementationPath) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(implementation_path.to_dict(), f, ensure_ascii=False, indent=2)


def write_contract_rows_jsonl(path: Path, rows: list[ComparisonContractRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def write_contract_rows_csv(path: Path, rows: list[ComparisonContractRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payloads = [row.to_dict() for row in rows]
    fieldnames: list[str] = []
    for payload in payloads:
        for key in payload.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for payload in payloads:
            payload = dict(payload)
            payload["notes"] = " | ".join(str(x) for x in payload.get("notes", ()))
            writer.writerow(payload)
