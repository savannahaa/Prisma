from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _run_script(script_rel: str, *, extra_args: list[str], env_updates: dict[str, str] | None = None) -> None:
    script_path = SRC_ROOT / script_rel
    if not script_path.exists():
        raise FileNotFoundError(f"missing script: {script_path}")
    env = os.environ.copy()
    if env_updates:
        env.update({str(k): str(v) for k, v in env_updates.items() if str(v) != ""})
    cmd = [sys.executable, str(script_path), *extra_args]
    print(f"[prepare_assets] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)


def _default_target_cluster_size(num_docs: int, num_clusters: int) -> int:
    if int(num_clusters) <= 0:
        raise ValueError("num_clusters must be positive")
    if int(num_docs) % int(num_clusters) != 0:
        raise ValueError(
            f"num_docs={int(num_docs)} must be divisible by num_clusters={int(num_clusters)}"
        )
    return int(num_docs) // int(num_clusters)


def _build_mainline_env(args: argparse.Namespace) -> dict[str, str]:
    target_cluster_size = (
        int(args.target_cluster_size)
        if int(args.target_cluster_size) > 0
        else _default_target_cluster_size(int(args.num_docs), int(args.num_clusters))
    )
    return {
        "PIPELINE_VARIANT": str(args.pipeline_variant),
        "WORKSET_NAMESPACE": str(args.workset_namespace),
        "WORKSET_NAME_OVERRIDE": str(args.workset_name),
        "NUM_WORKSET_DOCS": str(int(args.num_docs)),
        "NUM_CLUSTERS": str(int(args.num_clusters)),
        "TARGET_CLUSTER_SIZE": str(int(target_cluster_size)),
        "CORPUS_JSONL_PATH": str(Path(args.raw_corpus).resolve()),
        "FULL_QUERIES_JSONL_PATH": str(Path(args.raw_queries).resolve()),
        "FULL_QRELS_TSV_PATH": str(Path(args.raw_qrels).resolve()),
        "FORCE_REBUILD_QUERY_WORKSET": "1" if bool(args.force_rebuild_queries) else "0",
        "RMAX_ANCHOR_POLICY": str(args.rmax_anchor_policy),
    }


def _build_optional_workset_env(args: argparse.Namespace) -> dict[str, str]:
    env_updates: dict[str, str] = {
        "PIPELINE_VARIANT": str(args.pipeline_variant),
        "WORKSET_NAMESPACE": str(args.workset_namespace),
    }
    if str(args.workset_name).strip():
        env_updates["WORKSET_NAME_OVERRIDE"] = str(args.workset_name).strip()
    if int(args.num_docs) > 0:
        env_updates["NUM_WORKSET_DOCS"] = str(int(args.num_docs))
    if int(args.num_clusters) > 0:
        env_updates["NUM_CLUSTERS"] = str(int(args.num_clusters))
        target_cluster_size = int(args.target_cluster_size)
        if target_cluster_size <= 0 and int(args.num_docs) > 0 and int(args.num_docs) % int(args.num_clusters) == 0:
            target_cluster_size = _default_target_cluster_size(int(args.num_docs), int(args.num_clusters))
        if target_cluster_size > 0:
            env_updates["TARGET_CLUSTER_SIZE"] = str(int(target_cluster_size))
    return env_updates


def _run_mainline_stage(stage: str, args: argparse.Namespace) -> None:
    env = _build_mainline_env(args)
    if stage == "prepare_docs":
        _run_script("offline/prepare_real_data.py", extra_args=[], env_updates=env)
        return
    if stage == "cluster":
        _run_script("offline/cluster_offline.py", extra_args=[], env_updates=env)
        return
    if stage == "prepare_queries":
        _run_script("offline/prepare_real_queries.py", extra_args=[], env_updates=env)
        return
    if stage == "full":
        _run_script("offline/prepare_real_data.py", extra_args=[], env_updates=env)
        _run_script("offline/cluster_offline.py", extra_args=[], env_updates=env)
        _run_script("offline/prepare_real_queries.py", extra_args=[], env_updates=env)
        return
    raise ValueError(f"unsupported mainline/mse5 stage: {stage}")


def _cohere_script_for_stage(stage: str) -> str:
    mapping = {
        "download_queries": "measurement/download_cohere_msmarco_v21_queries.py",
        "materialize_query_pool": "measurement/materialize_cohere_msmarco_v21_query_pool.py",
        "build_local_manifests": "measurement/build_cohere_msmarco_v21_local_offset_manifests.py",
        "materialize_remote_workset": "measurement/materialize_cohere_msmarco_v21_workset_from_offsets.py",
        "materialize_local_subset": "measurement/materialize_cohere_local_workset_subset.py",
        "build_lightweight_cluster_info": "measurement/build_cohere_lightweight_cluster_info.py",
        "download_shard_chunked": "measurement/download_cohere_shard_chunked.py",
    }
    try:
        return mapping[str(stage)]
    except KeyError as exc:
        raise ValueError(f"unsupported cohere stage: {stage}") from exc


def _dataset_dispatch(dataset: str, stage: str, passthrough: list[str], args: argparse.Namespace) -> None:
    ds = str(dataset).strip().lower()
    if ds in {"mse5", "mainline"}:
        _run_mainline_stage(stage, args)
        return
    if ds == "amzn":
        if stage != "full":
            raise ValueError("amzn currently supports only stage=full")
        _run_script("measurement/prepare_amzn_esci_workset_1m.py", extra_args=passthrough)
        return
    if ds == "fiqa":
        if stage != "full":
            raise ValueError("fiqa currently supports only stage=full")
        _run_script("measurement/prepare_fiqa_2018_semantic_worksets.py", extra_args=passthrough)
        return
    if ds == "cohere":
        _run_script(
            _cohere_script_for_stage(stage),
            extra_args=passthrough,
            env_updates=_build_optional_workset_env(args),
        )
        return
    raise ValueError(f"unsupported dataset: {dataset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dataset-switchable asset-preparation entrypoint for the upload-ready paper-code pack. "
            "Use `mse5` for the repo's generic PRISMA E5/MS-style contract, "
            "`amzn` for Amazon ESCI, `fiqa` for FIQA-2018, and `cohere` for the Cohere MSMARCO v2.1 path."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=(
            "mse5",
            "mainline",
            "amzn",
            "fiqa",
            "cohere",
        ),
    )
    parser.add_argument("--stage", default="full")

    parser.add_argument("--raw-corpus", default="")
    parser.add_argument("--raw-queries", default="")
    parser.add_argument("--raw-qrels", default="")
    parser.add_argument("--num-docs", type=int, default=1_000_000)
    parser.add_argument("--num-clusters", type=int, default=100)
    parser.add_argument("--target-cluster-size", type=int, default=0)
    parser.add_argument("--workset-name", default="mse5_workset_1000000")
    parser.add_argument("--workset-namespace", default="mse5")
    parser.add_argument("--pipeline-variant", default="paperfaithful_mainline")
    parser.add_argument("--rmax-anchor-policy", default="calibration_query_only")
    parser.add_argument("--force-rebuild-queries", action="store_true")

    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Stage-specific arguments passed through to the underlying script. Prefix them with `--`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    passthrough = list(args.script_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if str(args.dataset) in {"mse5", "mainline"}:
        required = {
            "prepare_docs": ("raw_corpus",),
            "cluster": ("raw_corpus", "raw_queries", "raw_qrels"),
            "prepare_queries": ("raw_corpus", "raw_queries", "raw_qrels"),
            "full": ("raw_corpus", "raw_queries", "raw_qrels"),
        }
        missing = [
            field
            for field in required.get(str(args.stage), ())
            if not str(getattr(args, field, "")).strip()
        ]
        if missing:
            raise ValueError(
                "mainline/mse5 stage is missing required arguments: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )

    _dataset_dispatch(str(args.dataset), str(args.stage), passthrough, args)


if __name__ == "__main__":
    main()
