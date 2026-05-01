from __future__ import annotations

from baselines.common import ImplementationPath, ImplementationStage
from baselines.plaintext_ann.config import PlaintextANNConfig


def build_path(cfg: PlaintextANNConfig) -> ImplementationPath:
    return ImplementationPath(
        slug="plaintext_ann",
        display_name=str(cfg.baseline_display_name),
        paper_url=str(cfg.paper_url),
        summary=(
            "把主线里的 same-pipeline clean ANN 对照线正式抽成独立 baseline workflow，"
            "用于和 paperfaithful_mainline 对齐比较 recall、latency 与 communication。"
        ),
        current_truth=(
            "仓库里已经有 clean_fixed_budget_matched_total 这条内部控制线，"
            "它本质上就是明文 query + 同一套 Top-c routing + 同一套 FAISS HNSW + "
            "同一套客户端 exact rerank；当前上传副本保留的是独立 baseline scaffold。"
        ),
        algorithm_requirements=(
            "必须复用和主线完全相同的 workset、query split、cluster_info 与三阶段时间边界。",
            "客户端发送明文 query embedding，不做 gate 决策与 RRDP 扰动。",
            "服务器仍走同一套 FAISS HNSW dense backend，客户端仍做同一套 exact rerank。",
        ),
        integration_points=(
            "直接复用 paperfaithful_mainline 的 per-size selection summary 作为 theta*_N 选择来源。",
            "same_pipeline_runner 会在上传副本内直接调用 run_online_pipeline.py 的 clean same-pipeline 配置。",
            "comparison_adapter 会把 per-size 明文 ANN summary 归一化成 shared comparison contract 行。",
        ),
        stages=(
            ImplementationStage(
                slug="aligned_assets",
                title="Aligned Fullflow Assets",
                status="implemented_bridge",
                paper_requirement="使用和主线相同的 docs/query/qrels/cluster_info 构造对齐资产。",
                repo_bridge="上传副本现在可以直接从 ready paperfaithful workset 生成公开 selection summary，再驱动 same-pipeline plaintext ANN 复测。",
                entrypoint="src/baselines/plaintext_ann/driver.py --prepare-assets",
                outputs=(
                    "docs.npy",
                    "queries.npy",
                    "qrels.tsv",
                    "cluster_info_*.pkl",
                ),
                next_steps=(
                    "保持数据库规模、query split 与 cluster_info 版本和主线 selection summary 一致。",
                ),
            ),
            ImplementationStage(
                slug="matched_theta_selection",
                title="Matched Theta Selection",
                status="wired",
                paper_requirement="baseline 必须和主线在相同 theta*_N 下比较，而不是单独重新调参。",
                repo_bridge="直接读取主线 latency-scaling summary 里的 selected_fixed_k / c / epsilon。",
                entrypoint="src/baselines/plaintext_ann/same_pipeline_runner.py",
                outputs=(
                    "plaintext_ann_scaling_summary.json",
                    "plaintext_ann_scaling_summary.csv",
                    "plaintext_ann_scaling_rows.jsonl",
                ),
                next_steps=(
                    "如果后续切换到新的主线 selection summary，只需要把 runner 指向新的 summary json。",
                ),
            ),
            ImplementationStage(
                slug="same_pipeline_plaintext_ann",
                title="Same-Pipeline Plaintext ANN",
                status="implemented_core",
                paper_requirement="全流程对应主线，只移除 privacy gate / perturbation / crypto。",
                repo_bridge="复用当前主线 run_online_pipeline.py，但对外展示为 Plaintext ANN baseline。",
                entrypoint="src/baselines/plaintext_ann/same_pipeline_runner.py",
                outputs=(
                    "clean_fixed_budget_matched_total_*.json",
                    "plaintext_ann_scaling_summary.json",
                ),
                next_steps=(
                    "如需进一步做 c-aware route-union ANN ablation，可在这条 baseline 下增加子模式而不改共享 contract。",
                ),
            ),
            ImplementationStage(
                slug="comparison_contract",
                title="Comparison Contract Bridge",
                status="implemented_core",
                paper_requirement="把明文 ANN 对照线映射成统一的 recall / latency / comm contract 行。",
                repo_bridge="上传副本直接内置 plaintext_ann/comparison_adapter.py。 ",
                entrypoint="src/baselines/plaintext_ann/comparison_adapter.py",
                outputs=(
                    "plaintext_ann_comparison_contract.jsonl",
                    "plaintext_ann_comparison_contract.csv",
                ),
                next_steps=(
                    "如果后续要进统一 baseline 对比图，可直接消费这里导出的 contract。",
                ),
            ),
        ),
    )
