from __future__ import annotations

from baselines.common import ImplementationPath, ImplementationStage
from baselines.tiptoe.config import TiptoeConfig


def build_path(cfg: TiptoeConfig) -> ImplementationPath:
    return ImplementationPath(
        slug="tiptoe",
        display_name="Tiptoe",
        paper_url=str(cfg.paper_url),
        summary=(
            "Tiptoe upload pack: retain the local aligned-bundle implementation path "
            "with a CKKS-backed ranking stage plus a CKKS-backed URL retrieval stage."
        ),
        current_truth=(
            "The upload-ready snapshot includes the runnable ranking and URL stages, "
            "and now ships a public bundle-construction path from one ready paperfaithful workset."
        ),
        algorithm_requirements=(
            "Client encrypts the query embedding and the server scores only inside the selected cluster.",
            "The URL retrieval stage hides payload selection behind the retained CKKS batch-retrieval path.",
            "Both stages export repo-local summaries that can be normalized into the shared comparison contract.",
        ),
        integration_points=(
            "Bundle assets are read from results/<bundle_root>/ with repo-local paths only.",
            "ranking_service.py produces the ranking outputs consumed by url_service.py.",
            "comparison_adapter.py merges the two stage summaries into the shared three-stage contract row.",
        ),
        stages=(
            ImplementationStage(
                slug="aligned_bundle",
                title="Aligned Bundle Input",
                status="implemented_bridge",
                paper_requirement="Prepare Tiptoe's aligned docs, queries, qrels, and cluster partition assets.",
                repo_bridge="The upload-ready pack can build the public aligned bundle directly from one ready paperfaithful workset.",
                entrypoint="src/baselines/tiptoe/driver.py --prepare-assets --workset-name <workset>",
                outputs=(
                    "results/<tiptoe_bundle_root>/docs.npy",
                    "results/<tiptoe_bundle_root>/evaluation_queries.npy",
                    "results/<tiptoe_bundle_root>/evaluation_qrels.json",
                    "results/<tiptoe_bundle_root>/cluster_info_c*.pkl",
                ),
                next_steps=(
                    "If you keep multiple bundles side by side, set TIPTOE_BUNDLE_ROOT to the one you want to run.",
                ),
            ),
            ImplementationStage(
                slug="ranking_service",
                title="Private Ranking Service",
                status="implemented_core",
                paper_requirement="Run encrypted cluster-local ranking over the chosen Tiptoe cluster.",
                repo_bridge="The upload-ready pack retains the repo-local CKKS ranking implementation.",
                entrypoint="src/baselines/tiptoe/ranking_service.py",
                outputs=(
                    "tiptoe_ranking_service_rankings.jsonl",
                    "tiptoe_ranking_service_summary.json",
                    "tiptoe_ranking_service_preprocess.json",
                ),
            ),
            ImplementationStage(
                slug="url_service",
                title="Private URL Retrieval",
                status="implemented_core",
                paper_requirement="Retrieve payloads through the retained encrypted batch-selection stage.",
                repo_bridge="The upload-ready pack retains the repo-local CKKS URL-service implementation.",
                entrypoint="src/baselines/tiptoe/url_service.py",
                outputs=(
                    "tiptoe_url_service_payloads.jsonl",
                    "tiptoe_url_service_summary.json",
                    "tiptoe_url_service_batches.json",
                ),
            ),
            ImplementationStage(
                slug="comparison_contract",
                title="Comparison Contract Bridge",
                status="implemented_core",
                paper_requirement="Normalize the retained Tiptoe outputs into the shared comparison contract.",
                repo_bridge="The adapter combines ranking and URL summaries into one contract row.",
                entrypoint="src/baselines/tiptoe/comparison_adapter.py",
                outputs=(
                    "tiptoe_comparison_contract.jsonl",
                    "tiptoe_comparison_contract.csv",
                ),
            ),
        ),
    )
