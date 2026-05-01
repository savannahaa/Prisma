from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PantherConfig:
    project_root: Path
    paper_url: str = "https://dl.acm.org/doi/pdf/10.1145/3719027.3765190"
    code_url: str = "https://zenodo.org/records/17020254"
    workflow_name: str = "panther"
    workset_tag: str = "repro_panther_ms_aligned_v1"
    output_prefix: str = "repro_panther_ms_aligned_v1"
    pipeline_output_tag: str = "repro_panther_adapter_v1"
    implementation_root: str = "src/baselines/panther"
    default_dataset: str = "ms"
    default_summary_stem: str = "panther_ms_aligned_summary"
    bundle_root_name: str = ""
    bundle_root_candidates: tuple[str, ...] = (
        "cohere_baseline_bundle",
        "cohere_baseline_bundle_1000000",
    )
    bridge_root_name: str = "ms_author_bridge"
    default_author_input_slug: str = "msmarco_10k_bridge"
    default_cluster_info_selector_c: int = 10
    openpanther_default_max_points_per_cluster: int = 20
    openpanther_default_pointer_dc_bits: int = 8
    openpanther_default_cluster_dc_bits: int = 5
    openpanther_default_compare_radix: int = 5
    openpanther_default_pir_logt: int = 12
    openpanther_default_pir_fixt: int = 2
    openpanther_default_poly_modulus_degree: int = 4096
    openpanther_default_distance_poly_degree: int = 2048
    default_size: int = 10_000
    default_query_limit: int = 120
    default_top_k: int = 100
    default_num_clusters: int = 10
    parity_dataset: str = "sift"
    parity_summary_stem: str = "panther_sift_official_summary"
    parity_size: int = 1_000_000
    parity_query_limit: int = 10_000
    parity_top_k: int = 5
    parity_num_clusters: int = 20
    external_repo_dir: str = os.environ.get("OPENPANTHER_ROOT", "external/OpenPanther")
