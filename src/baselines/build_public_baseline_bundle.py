from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import pickle
from pathlib import Path

import numpy as np

from baselines.public_assets import (
    copy_file,
    load_qrels_tsv,
    resolve_requested_worksets,
    resolve_workset_assets,
    save_json,
    workset_stats,
)


DEFAULT_SELECTOR_CS = (7, 10)


def _parse_selector_cs(text: str) -> list[int]:
    if not str(text).strip():
        return [int(x) for x in DEFAULT_SELECTOR_CS]
    values: list[int] = []
    seen: set[int] = set()
    for part in str(text).replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        value = max(1, int(token))
        if value in seen:
            continue
        seen.add(value)
        values.append(int(value))
    if not values:
        raise ValueError("selector-cs resolved to an empty list")
    return values


def _default_bundle_root_name(workset_name: str) -> str:
    safe = str(workset_name).strip().replace(" ", "_")
    return f"baseline_bundle_{safe}"


def _copy_cluster_info(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f:
        payload = pickle.load(f)
    with open(dst, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a public qrels-aligned baseline bundle from one ready PRISMA workset."
    )
    parser.add_argument(
        "--workset-name",
        action="append",
        default=[],
        help="Source PRISMA workset name. If omitted, auto-discover ready worksets and require a unique match.",
    )
    parser.add_argument(
        "--bundle-root-name",
        type=str,
        default="",
        help="Output directory name under results/. Default: baseline_bundle_<workset_name>.",
    )
    parser.add_argument(
        "--selector-cs",
        type=str,
        default="7,10",
        help="Comma-separated cluster-info selector c values to materialize inside the bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    requested = resolve_requested_worksets(project_root, list(args.workset_name))
    if len(requested) != 1:
        raise RuntimeError(
            "baseline bundle building requires exactly one source workset. "
            "Pass --workset-name explicitly when multiple ready worksets exist.\n"
            f"Resolved worksets: {requested}"
        )
    workset_name = str(requested[0])
    assets = resolve_workset_assets(project_root, workset_name)
    stats = workset_stats(assets)

    bundle_root_name = (
        str(args.bundle_root_name).strip()
        if str(args.bundle_root_name).strip()
        else _default_bundle_root_name(workset_name)
    )
    selector_cs = _parse_selector_cs(str(args.selector_cs))
    bundle_root = project_root / "results" / bundle_root_name
    bundle_root.mkdir(parents=True, exist_ok=True)

    doc_ids = [str(x) for x in np.load(assets.doc_ids, allow_pickle=True).tolist()]
    query_ids = [str(x) for x in np.load(assets.query_ids, allow_pickle=True).tolist()]
    qrels = load_qrels_tsv(
        assets.strict_qrels,
        allowed_query_ids=set(query_ids),
        allowed_doc_ids=set(doc_ids),
    )

    copy_file(assets.docs, bundle_root / "docs.npy")
    copy_file(assets.doc_ids, bundle_root / "doc_ids.npy")
    copy_file(assets.corpus, bundle_root / "corpus.jsonl")
    copy_file(assets.queries, bundle_root / "evaluation_queries.npy")
    copy_file(assets.query_ids, bundle_root / "evaluation_query_ids.npy")
    save_json(
        bundle_root / "evaluation_qrels.json",
        {str(qid): list(qrels.get(str(qid), [])) for qid in query_ids},
    )
    for selector_c in selector_cs:
        _copy_cluster_info(
            assets.cluster_info_pkl,
            bundle_root / f"cluster_info_c{int(selector_c)}.pkl",
        )

    bundle_meta = {
        "bundle_format": "paperfaithful_public_qrels_aligned_bundle_v1",
        "source_workset_name": str(workset_name),
        "num_docs": int(stats["num_docs"]),
        "num_queries": int(stats["num_queries"]),
        "num_clusters": int(stats["num_clusters"]),
        "embedding_dim": int(stats["embedding_dim"]),
        "selector_cs": [int(x) for x in selector_cs],
        "paths": {
            "docs": str(bundle_root / "docs.npy"),
            "doc_ids": str(bundle_root / "doc_ids.npy"),
            "corpus": str(bundle_root / "corpus.jsonl"),
            "evaluation_queries": str(bundle_root / "evaluation_queries.npy"),
            "evaluation_query_ids": str(bundle_root / "evaluation_query_ids.npy"),
            "evaluation_qrels": str(bundle_root / "evaluation_qrels.json"),
        },
        "source_paths": assets.to_dict(),
        "notes": [
            "This bundle is derived directly from one ready PRISMA workset.",
            "evaluation_qrels.json is filtered to the bundled evaluation query ids and bundled doc ids only.",
            "The same mainline cluster_info snapshot is copied under each requested cluster_info_c*.pkl alias because selector c changes routing policy, not the underlying partition.",
        ],
    }
    save_json(bundle_root / "bundle_meta.json", bundle_meta)
    save_json(bundle_root / "build_manifest.json", bundle_meta)
    print(f"[saved] {bundle_root / 'bundle_meta.json'}")
    print(f"[saved] {bundle_root / 'build_manifest.json'}")


if __name__ == "__main__":
    main()
