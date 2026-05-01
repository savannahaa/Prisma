from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RemoteRAGConfig:
    project_root: Path
    paper_url: str = "https://aclanthology.org/2025.findings-acl.197.pdf"
    workflow_name: str = "remoterag"
    workset_tag: str = "repro_remoterag_scaling_v1"
    output_prefix: str = "repro_remoterag_scaling_v1"
    pipeline_output_tag: str = "repro_remoterag_adapter_v1"
    implementation_root: str = "src/baselines/remoterag"
    result_root_name: str = "remoterag"
    default_sizes: tuple[int, ...] = (2000, 5000, 10000, 15000, 20000)
    default_scaling_summary_stem: str = "remoterag_scaling_summary"
    default_contract_stem: str = "remoterag_comparison_contract"
    default_module1_size: int = 10000
    default_module1_target_radius: float = 0.1
    default_module1_query_limit: int = 120
    default_module1_seed: int = 20260421
    default_module2_query_limit: int = 5
    default_module2_seed: int = 20260421
    default_module2_quantization_scale: int = 5000
    default_module2_paillier_bits: int = 512
    default_module2_ot_prime_bits: int = 256
