from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlaintextANNConfig:
    project_root: Path
    paper_url: str = ""
    workflow_name: str = "plaintext_ann"
    workset_tag: str = "repro_plaintext_ann_scaling_v1"
    output_prefix: str = "repro_plaintext_ann_scaling_v1"
    pipeline_output_tag: str = "repro_plaintext_ann_adapter_v1"
    implementation_root: str = "src/baselines/plaintext_ann"
    result_root_name: str = "plaintext_ann"
    default_sizes: tuple[int, ...] = (2000, 5000, 10000, 15000, 20000)
    default_scaling_summary_stem: str = "plaintext_ann_scaling_summary"
    default_contract_stem: str = "plaintext_ann_comparison_contract"
    default_selection_summary_glob: str = "paperfaithful_mainline_latency_scaling_persize*.json"
    baseline_display_name: str = "Non-Private PRISMA"
