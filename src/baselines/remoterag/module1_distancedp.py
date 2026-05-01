"""
RemoteRAG Module 1 prototype.

当前实现忠实落实了论文的两个关键点：
1) 用 (n, epsilon)-DistanceDP 的生成方式对 query embedding 做扰动：
   r ~ Gamma(n, 1/epsilon), v ~ Uniform(S^{n-1}), e' = e + r v
2) 在 perturbed embedding 上做 top-k' 候选检索，并输出给 Module 2 的候选集合。

说明：
- `k'` 当前采用“经验球冠计数版” Theorem-1 bridge：
  用原 query 的 exact top-k 最差角距 alpha_k，加上 delta_alpha，
  再按实际语料中满足 angle <= alpha_k + delta_alpha 的文档数估计 k'。
- 这比纯文档里的均匀球面闭式假设更贴近当前仓库真实 workset，
  但它不是论文 Lemma 1 / Theorem 1 的严格数值积分版。
"""

from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from baselines.public_assets import resolve_requested_worksets, resolve_workset_assets, workset_stats
from baselines.remoterag.config import RemoteRAGConfig
from client.polar_rdp import sample_uniform_sphere
from server.cluster_retrieval import angular_distances, smallest_k_indices_with_tiebreak
from shared.synthetic_doc_access import load_doc_ids_or_synthetic


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32)


def _normalize_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(x))
    if norm <= 1e-12:
        raise ValueError("query norm too small")
    return (x / norm).astype(np.float32)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_slug(text: str) -> str:
    chars: list[str] = []
    for ch in str(text).strip():
        if ch.isalnum() or ch in {"_", "-"}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_")


def _resolve_workset_paths(
    *,
    cfg: RemoteRAGConfig,
    num_docs: int,
    explicit_workset_name: str,
    prepare_assets: bool,
    reuse_existing: bool,
) -> tuple[dict[str, Path], str, int, int]:
    _ = bool(reuse_existing)
    project_root = cfg.project_root
    if bool(prepare_assets):
        prep_script = project_root / "src" / "baselines" / "remoterag" / "prepare_assets.py"
        cmd = [sys.executable, str(prep_script)]
        if str(explicit_workset_name).strip():
            cmd.extend(["--workset-name", str(explicit_workset_name).strip()])
        subprocess.run(cmd, cwd=str(project_root), check=True)
    if str(explicit_workset_name).strip():
        workset_name = str(explicit_workset_name).strip()
    else:
        candidates = resolve_requested_worksets(project_root, [])
        matched: list[str] = []
        for candidate in candidates:
            stats = workset_stats(resolve_workset_assets(project_root, candidate))
            if int(stats["num_docs"]) == int(num_docs):
                matched.append(str(candidate))
        if not matched:
            raise FileNotFoundError(
                "no ready PRISMA workset matches the requested RemoteRAG size "
                f"{int(num_docs)}. Use --workset-name to choose an explicit source workset."
            )
        if len(matched) > 1:
            raise RuntimeError(
                "multiple ready worksets share the requested RemoteRAG size. "
                "Pass --workset-name explicitly.\n"
                f"Matched worksets: {matched}"
            )
        workset_name = str(matched[0])
    resolved = resolve_workset_assets(project_root, workset_name)
    stats = workset_stats(resolved)
    if int(num_docs) > 0 and int(stats["num_docs"]) != int(num_docs):
        raise ValueError(
            f"requested num_docs={int(num_docs)} but workset {workset_name} has num_docs={int(stats['num_docs'])}"
        )
    return (
        {
            "workset_name": Path(workset_name),
            "docs": resolved.docs,
            "doc_ids": resolved.doc_ids,
            "meta": resolved.meta,
            "corpus": resolved.corpus,
            "queries": resolved.queries,
            "query_ids": resolved.query_ids,
            "gt_topk": resolved.gt_topk,
            "cluster_info_pkl": resolved.cluster_info_pkl,
        },
        str(workset_name),
        int(stats["num_clusters"]),
        int(stats["num_docs"]),
    )


def _load_runtime(paths: dict[str, Path], *, query_limit: int) -> dict:
    docs = _normalize_rows(np.load(paths["docs"]).astype(np.float32))
    queries = _normalize_rows(np.load(paths["queries"]).astype(np.float32))
    query_ids = [str(x) for x in np.load(paths["query_ids"], allow_pickle=True).tolist()]
    gt_topk = np.asarray(np.load(paths["gt_topk"]), dtype=np.int32)
    doc_ids, _doc_ids_are_synthetic = load_doc_ids_or_synthetic(
        paths["doc_ids"],
        num_docs=int(docs.shape[0]),
        allow_synthetic=str(os.environ.get("ALLOW_SYNTHETIC_DOCID_TEXT_FALLBACK", "")).strip().lower() in {"1", "true", "yes", "on"},
        default_prefix=f"{paths['docs'].stem}_doc",
    )

    if int(query_limit) > 0:
        limit = int(min(int(query_limit), int(len(queries))))
        queries = queries[:limit]
        query_ids = query_ids[:limit]
        gt_topk = gt_topk[:limit]
    if len(queries) <= 0:
        raise RuntimeError("no evaluation queries available for RemoteRAG Module 1")
    if queries.shape[0] != gt_topk.shape[0]:
        raise RuntimeError(
            f"query/gt_topk size mismatch: {queries.shape[0]} vs {gt_topk.shape[0]}"
        )
    return {
        "docs": docs,
        "queries": queries,
        "query_ids": query_ids,
        "gt_topk": gt_topk,
        "doc_ids": doc_ids,
    }


def _sample_distancedp(
    *,
    query: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> dict:
    query = _normalize_vec(query)
    dim = int(query.shape[0])
    radius = float(rng.gamma(shape=float(dim), scale=float(1.0 / epsilon)))
    direction = sample_uniform_sphere(dim, rng).astype(np.float32)
    perturbed_query = (query + float(radius) * direction).astype(np.float32)
    delta_alpha_rad = float(math.atan2(float(radius), float(np.linalg.norm(query))))
    return {
        "query": query,
        "perturbed_query": perturbed_query,
        "sampled_radius": float(radius),
        "delta_alpha_rad": float(delta_alpha_rad),
        "direction": direction,
    }


def _doc_ids_for_indices(doc_ids: list[str], indices: list[int]) -> list[str]:
    return [str(doc_ids[int(idx)]) for idx in indices]


def _run_module1(
    *,
    cfg: RemoteRAGConfig,
    num_docs: int,
    explicit_workset_name: str,
    epsilon: float,
    query_limit: int,
    seed: int,
    prepare_assets: bool,
    reuse_existing: bool,
    k_prime_cap: int,
) -> tuple[list[dict], dict]:
    paths, workset_name, num_clusters, actual_num_docs = _resolve_workset_paths(
        cfg=cfg,
        num_docs=int(num_docs),
        explicit_workset_name=str(explicit_workset_name),
        prepare_assets=bool(prepare_assets),
        reuse_existing=bool(reuse_existing),
    )
    runtime = _load_runtime(paths, query_limit=int(query_limit))
    docs = runtime["docs"]
    queries = runtime["queries"]
    query_ids = runtime["query_ids"]
    gt_topk = runtime["gt_topk"]
    doc_ids = runtime["doc_ids"]
    num_all_docs = int(actual_num_docs)
    global_indices = np.arange(num_all_docs, dtype=np.int32)
    top_k = int(gt_topk.shape[1])

    rows: list[dict] = []
    coverage: list[float] = []
    overlap_counts: list[int] = []
    sampled_radii: list[float] = []
    delta_alphas: list[float] = []
    theoretical_k_primes: list[int] = []
    final_k_primes: list[int] = []

    for query_index, (query, qid) in enumerate(zip(queries, query_ids)):
        rng = np.random.default_rng(int(seed) + int(query_index))
        sampled = _sample_distancedp(
            query=np.asarray(query, dtype=np.float32),
            epsilon=float(epsilon),
            rng=rng,
        )
        original_theta = angular_distances(
            np.asarray(sampled["query"], dtype=np.float32),
            docs,
        )
        perturbed_theta = angular_distances(
            np.asarray(sampled["perturbed_query"], dtype=np.float32),
            docs,
        )

        gt_indices = [int(x) for x in np.asarray(gt_topk[query_index], dtype=np.int32).tolist()]
        gt_set = set(int(x) for x in gt_indices)
        original_alpha_k = float(np.max(original_theta[np.asarray(gt_indices, dtype=np.int32)]))
        expanded_alpha_k = float(original_alpha_k + float(sampled["delta_alpha_rad"]))
        theoretical_k_prime = int(
            max(
                int(top_k),
                int(np.count_nonzero(np.asarray(original_theta <= expanded_alpha_k + 1e-12))),
            )
        )
        final_k_prime = int(theoretical_k_prime)
        if int(k_prime_cap) > 0:
            final_k_prime = int(min(int(final_k_prime), int(k_prime_cap)))
            final_k_prime = int(max(int(final_k_prime), int(top_k)))

        order = smallest_k_indices_with_tiebreak(
            distances=np.asarray(perturbed_theta, dtype=np.float64),
            top_k=int(final_k_prime),
            global_indices=global_indices,
        )
        candidate_indices = [int(global_indices[int(pos)]) for pos in order.tolist()]
        candidate_set = set(int(x) for x in candidate_indices)
        overlap_count = int(len(gt_set.intersection(candidate_set)))
        include_original_topk = bool(gt_set.issubset(candidate_set))

        sampled_radii.append(float(sampled["sampled_radius"]))
        delta_alphas.append(float(sampled["delta_alpha_rad"]))
        theoretical_k_primes.append(int(theoretical_k_prime))
        final_k_primes.append(int(final_k_prime))
        overlap_counts.append(int(overlap_count))
        coverage.append(1.0 if include_original_topk else 0.0)

        rows.append(
            {
                "query_index": int(query_index),
                "query_id": str(qid),
                "epsilon": float(epsilon),
                "num_docs": int(actual_num_docs),
                "num_clusters": int(num_clusters),
                "sampled_radius": float(sampled["sampled_radius"]),
                "delta_alpha_rad": float(sampled["delta_alpha_rad"]),
                "delta_alpha_deg": float(np.degrees(sampled["delta_alpha_rad"])),
                "original_alpha_k_rad": float(original_alpha_k),
                "original_alpha_k_deg": float(np.degrees(original_alpha_k)),
                "expanded_alpha_k_rad": float(expanded_alpha_k),
                "expanded_alpha_k_deg": float(np.degrees(expanded_alpha_k)),
                "k": int(top_k),
                "k_prime_theory_empirical_cap_count": int(theoretical_k_prime),
                "k_prime_final": int(final_k_prime),
                "k_prime_capped": bool(int(final_k_prime) != int(theoretical_k_prime)),
                "include_original_exact_topk": bool(include_original_topk),
                "original_exact_topk_overlap_count": int(overlap_count),
                "original_exact_topk_indices": [int(x) for x in gt_indices],
                "original_exact_topk_doc_ids": _doc_ids_for_indices(doc_ids, gt_indices),
                "module1_candidate_indices": [int(x) for x in candidate_indices],
                "module1_candidate_doc_ids": _doc_ids_for_indices(doc_ids, candidate_indices),
                "selection_rule": "theorem1_bridge_empirical_cap_count_alpha_k_plus_delta_alpha",
                "workset_name": str(workset_name),
            }
        )

    summary = {
        "module": "RemoteRAG Module 1",
        "paper_url": str(cfg.paper_url),
        "implementation_note": (
            "Implements DistanceDP perturbation exactly as r~Gamma(n,1/epsilon), "
            "v~Uniform(S^{n-1}), e'=e+r*v; k' currently uses an empirical cap-count "
            "bridge based on Theorem 1 rather than the paper's closed-form uniform-sphere assumption."
        ),
        "workset_name": str(workset_name),
        "num_docs": int(actual_num_docs),
        "num_clusters": int(num_clusters),
        "num_queries": int(len(rows)),
        "embedding_dim": int(docs.shape[1]),
        "k": int(top_k),
        "epsilon": float(epsilon),
        "expected_radius_mean_n_over_epsilon": float(docs.shape[1] / float(epsilon)),
        "seed": int(seed),
        "query_limit_requested": int(query_limit),
        "k_prime_cap": int(k_prime_cap),
        "coverage_exact_topk_rate": float(np.mean(coverage)) if coverage else 0.0,
        "mean_original_exact_topk_overlap_count": float(np.mean(overlap_counts)) if overlap_counts else 0.0,
        "mean_sampled_radius": float(np.mean(sampled_radii)) if sampled_radii else 0.0,
        "median_sampled_radius": float(np.median(sampled_radii)) if sampled_radii else 0.0,
        "mean_delta_alpha_rad": float(np.mean(delta_alphas)) if delta_alphas else 0.0,
        "mean_k_prime_theory_empirical_cap_count": float(np.mean(theoretical_k_primes)) if theoretical_k_primes else 0.0,
        "mean_k_prime_final": float(np.mean(final_k_primes)) if final_k_primes else 0.0,
        "max_k_prime_final": int(max(final_k_primes)) if final_k_primes else 0,
        "min_k_prime_final": int(min(final_k_primes)) if final_k_primes else 0,
        "input_paths": {key: str(value) for key, value in paths.items()},
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RemoteRAG Module 1 DistanceDP prototype on isolated assets.")
    parser.add_argument("--size", type=int, default=0, help="Database size. Default uses config.default_module1_size.")
    parser.add_argument("--workset-name", type=str, default="", help="Optional explicit PRISMA source workset name. Recommended when multiple datasets share the same size.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="DistanceDP epsilon. If <=0, derive from --target-radius.")
    parser.add_argument("--target-radius", type=float, default=0.0, help="Target expected radius r_bar=n/epsilon. Used when --epsilon<=0.")
    parser.add_argument("--query-limit", type=int, default=-1, help="<=0 uses config.default_module1_query_limit.")
    parser.add_argument("--seed", type=int, default=-1, help="RNG seed. <0 uses config.default_module1_seed.")
    parser.add_argument(
        "--prepare-assets",
        action="store_true",
        help="Build a public RemoteRAG asset manifest from ready PRISMA worksets before running.",
    )
    parser.set_defaults(reuse_existing=True)
    parser.add_argument("--reuse-existing", dest="reuse_existing", action="store_true")
    parser.add_argument("--no-reuse-existing", dest="reuse_existing", action="store_false")
    parser.add_argument("--k-prime-cap", type=int, default=0, help="Optional upper cap on final k'. 0 disables cap.")
    parser.add_argument("--output-prefix", type=str, default="", help="Optional output stem override under results/repro_workflows/remoterag/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = RemoteRAGConfig(project_root=project_root)

    num_docs = int(args.size) if int(args.size) > 0 else int(cfg.default_module1_size)
    query_limit = int(args.query_limit) if int(args.query_limit) > 0 else int(cfg.default_module1_query_limit)
    seed = int(args.seed) if int(args.seed) >= 0 else int(cfg.default_module1_seed)
    epsilon = float(args.epsilon)
    target_radius = None
    if epsilon <= 0.0:
        target_radius = (
            float(args.target_radius)
            if float(args.target_radius) > 0.0
            else float(cfg.default_module1_target_radius)
        )
        paths, _, _, _ = _resolve_workset_paths(
            cfg=cfg,
            num_docs=int(num_docs),
            explicit_workset_name=str(args.workset_name),
            prepare_assets=False,
            reuse_existing=bool(args.reuse_existing),
        )
        docs_probe = np.load(paths["docs"], mmap_mode="r")
        actual_dim = int(docs_probe.shape[1])
        epsilon = float(actual_dim / float(target_radius))

    rows, summary = _run_module1(
        cfg=cfg,
        num_docs=int(num_docs),
        explicit_workset_name=str(args.workset_name),
        epsilon=float(epsilon),
        query_limit=int(query_limit),
        seed=int(seed),
        prepare_assets=bool(args.prepare_assets),
        reuse_existing=bool(args.reuse_existing),
        k_prime_cap=int(args.k_prime_cap),
    )
    if target_radius is not None:
        summary["target_radius_requested"] = float(target_radius)

    if str(args.output_prefix).strip():
        stem = str(args.output_prefix).strip()
    else:
        base_stem = f"remoterag_module1_distancedp_n{int(num_docs)}_eps{int(round(float(summary['epsilon'])))}"
        if str(args.workset_name).strip():
            stem = f"{base_stem}_{_safe_slug(str(args.workset_name))}"
        else:
            stem = base_stem
    result_root = project_root / "results" / "repro_workflows" / "remoterag"
    rows_path = result_root / f"{stem}.jsonl"
    summary_path = result_root / f"{stem}.json"
    summary["rows_jsonl"] = str(rows_path)
    summary["summary_json"] = str(summary_path)

    _write_jsonl(rows_path, rows)
    _save_json(summary_path, summary)
    print(f"[saved] {rows_path}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
