from __future__ import annotations

from baselines.common import ImplementationPath, ImplementationStage
from baselines.panther.config import PantherConfig


def build_path(cfg: PantherConfig) -> ImplementationPath:
    return ImplementationPath(
        slug="panther",
        display_name="Panther",
        paper_url=str(cfg.paper_url),
        summary=(
            "Panther upload pack: retain the MS-aligned author bridge, the OpenPanther workset bridge, "
            "and the summary/contract export layer without vendoring the external OpenPanther repository."
        ),
        current_truth=(
            "The upload-ready snapshot includes the repo-side bridge and normalization code, "
            "can build the public aligned bundle from one ready paperfaithful workset, "
            "and still expects the external OpenPanther checkout to be provided separately."
        ),
        algorithm_requirements=(
            "The repo-side bridge must export bundle assets into the format expected by the external OpenPanther code.",
            "Returned rankings or logs must be normalized back into the repo's shared latency/communication contract.",
            "Public paths must stay repo-local or environment-configurable instead of relying on private absolute paths.",
        ),
        integration_points=(
            "OPENPANTHER_ROOT or external/OpenPanther locates the external author code checkout.",
            "ms_author_bridge.py exports MS-aligned assets and rebuilds summaries from returned rankings.",
            "comparison_adapter.py converts Panther summaries into the shared contract row format.",
        ),
        stages=(
            ImplementationStage(
                slug="aligned_bundle",
                title="Aligned Panther Bundle",
                status="implemented_bridge",
                paper_requirement="Provide the Panther-aligned bundle and any required cluster_info snapshots.",
                repo_bridge="The upload-ready pack can build the public aligned bundle directly from one ready paperfaithful workset.",
                entrypoint="src/baselines/panther/driver.py --prepare-assets --workset-name <workset>",
                outputs=(
                    "results/<bundle_root>/docs.npy",
                    "results/<bundle_root>/evaluation_queries.npy",
                    "results/<bundle_root>/cluster_info_c*.pkl",
                ),
            ),
            ImplementationStage(
                slug="ms_author_bridge",
                title="MS Author Bridge",
                status="implemented_core",
                paper_requirement="Export MS-aligned assets and rebuild summaries from Panther ranking outputs.",
                repo_bridge="The upload-ready pack retains the repo-side author bridge.",
                entrypoint="src/baselines/panther/ms_author_bridge.py",
                outputs=(
                    "results/repro_workflows/panther/ms_author_bridge/manifest.json",
                    "panther_ms_aligned_summary.json",
                    "panther_ms_aligned_rows.jsonl",
                ),
            ),
            ImplementationStage(
                slug="openpanther_bridge",
                title="OpenPanther Workset Bridge",
                status="implemented_core",
                paper_requirement="Stage repo-local Cohere/MS assets into the external OpenPanther runtime layout.",
                repo_bridge="The upload-ready pack retains the repo-side bridge while keeping the author repo external.",
                entrypoint="src/baselines/panther/cohere_workset_bridge.py",
                outputs=(
                    "results/repro_workflows/panther/<dataset_slug>/manifest.json",
                    "external/OpenPanther/experimental/panther/demo/panther_bridge_config.h",
                ),
            ),
            ImplementationStage(
                slug="comparison_contract",
                title="Comparison Contract Bridge",
                status="implemented_core",
                paper_requirement="Normalize Panther summaries into the shared comparison contract.",
                repo_bridge="The upload-ready pack retains the summary template writer and contract adapter.",
                entrypoint="src/baselines/panther/comparison_adapter.py",
                outputs=(
                    "panther_comparison_contract.jsonl",
                    "panther_comparison_contract.csv",
                ),
            ),
        ),
    )
