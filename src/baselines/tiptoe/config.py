from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TiptoeConfig:
    project_root: Path
    paper_url: str = "https://eprint.iacr.org/2023/1438"
    workflow_name: str = "tiptoe"
    bundle_root_name: str = ""
    bundle_root_candidates: tuple[str, ...] = (
        "cohere_baseline_bundle",
        "cohere_baseline_bundle_1000000",
    )
    routing_c: int = 7
    output_prefix: str = "repro_tiptoe_curve_v1"
    implementation_root: str = "src/baselines/tiptoe"
    default_top_k: int = 100
    default_query_limit: int = 120
    default_seed: int = 20260422
    default_quantization_levels: int = 15
    default_token_bytes: int = 118784
    default_score_bytes_per_doc: int = 2
    default_url_group_size: int = 32
