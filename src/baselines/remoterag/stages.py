from __future__ import annotations

from baselines.common import ImplementationPath, ImplementationStage
from baselines.remoterag.config import RemoteRAGConfig


def build_path(cfg: RemoteRAGConfig) -> ImplementationPath:
    return ImplementationPath(
        slug="remoterag",
        display_name="RemoteRAG",
        paper_url=str(cfg.paper_url),
        summary=(
            "RemoteRAG upload pack: retain Module 1 DistanceDP candidate generation, "
            "Module 2 PHE+OT scoring/retrieval, and repo-local scaling/contract export."
        ),
        current_truth=(
            "The upload-ready snapshot includes runnable Module 1 and Module 2 code, "
            "and now consumes ready paperfaithful mainline worksets directly by workset name."
        ),
        algorithm_requirements=(
            "Module 1 perturbs the query embedding with DistanceDP and emits a top-k' candidate set.",
            "Module 2 scores those candidates with the retained PHE path and conditionally switches to OT retrieval.",
            "Per-size outputs are normalized into the same comparison contract used by the mainline and other baselines.",
        ),
        integration_points=(
            "Per-size assets are discovered from repo-local data/ and results/ naming conventions.",
            "module2_phe_ot.py can now generate missing Module 1 outputs on demand.",
            "scaling_runner.py and comparison_adapter.py export per-size summaries and contract rows.",
        ),
        stages=(
            ImplementationStage(
                slug="aligned_scaling_assets",
                title="Per-Size Input Assets",
                status="implemented_bridge",
                paper_requirement="Provide per-size docs, queries, gt_topk, and related workset files.",
                repo_bridge="The upload-ready pack now exposes a public workset-manifest builder and reads mainline worksets directly instead of relying on hidden copied aliases.",
                entrypoint="src/baselines/remoterag/prepare_assets.py --workset-name <workset>",
                outputs=(
                    "results/repro_workflows/remoterag/prepared_worksets.json",
                ),
            ),
            ImplementationStage(
                slug="module1_distancedp",
                title="Module 1 DistanceDP",
                status="implemented_core",
                paper_requirement="Generate perturbed-query top-k' candidate sets.",
                repo_bridge="The upload-ready pack retains a self-contained Module 1 implementation.",
                entrypoint="src/baselines/remoterag/module1_distancedp.py",
                outputs=(
                    "remoterag_module1_distancedp_*.jsonl",
                    "remoterag_module1_distancedp_*.json",
                ),
            ),
            ImplementationStage(
                slug="module2_phe_ot",
                title="Module 2 PHE + OT",
                status="implemented_core",
                paper_requirement="Score Module 1 candidates under PHE and retrieve through direct return or OT.",
                repo_bridge="The upload-ready pack retains the runnable PHE+OT prototype and can generate missing Module 1 outputs.",
                entrypoint="src/baselines/remoterag/module2_phe_ot.py",
                outputs=(
                    "remoterag_module2_phe_ot_*.jsonl",
                    "remoterag_module2_phe_ot_*.json",
                    "remoterag_module2_phe_ot_*_ot_trace.json",
                ),
            ),
            ImplementationStage(
                slug="scaling_export",
                title="Scaling Export Bridge",
                status="implemented_core",
                paper_requirement="Aggregate per-size Module 2 outputs into scaling summaries and contract rows.",
                repo_bridge="The upload-ready pack retains the scaling runner and comparison adapter.",
                entrypoint="src/baselines/remoterag/scaling_runner.py",
                outputs=(
                    "remoterag_scaling_summary.json",
                    "remoterag_comparison_contract.csv",
                ),
            ),
        ),
    )
