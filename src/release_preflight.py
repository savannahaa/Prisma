from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
from pathlib import Path

from baselines.panther.common import bundle_paths, external_repo_root
from baselines.panther.config import PantherConfig
from baselines.tiptoe.common import bundle_paths as tiptoe_bundle_paths
from baselines.tiptoe.config import TiptoeConfig
from shared.config import (
    WORKSET_CLUSTER_INFO_PATH,
    WORKSET_CORPUS_JSONL_PATH,
    WORKSET_DOCS_PATH,
    WORKSET_DOC_IDS_PATH,
    WORKSET_GT_TOPK_PATH,
    WORKSET_QUERIES_PATH,
    WORKSET_QUERY_IDS_PATH,
)


def _emit_status(ok: bool, label: str, path: Path | None, note: str = "") -> None:
    mark = "OK" if ok else "MISSING"
    if path is None:
        print(f"[{mark}] {label}")
    else:
        print(f"[{mark}] {label}: {path}")
    if note:
        print(f"      {note}")


def _check_required(label: str, path: Path) -> bool:
    ok = path.exists()
    _emit_status(ok, label, path)
    return ok


def _check_main_assets() -> bool:
    print("\n== Main Pipeline Assets ==")
    expected = [
        ("workset docs", Path(WORKSET_DOCS_PATH)),
        ("workset doc ids", Path(WORKSET_DOC_IDS_PATH)),
        ("workset corpus jsonl", Path(WORKSET_CORPUS_JSONL_PATH)),
        ("workset queries", Path(WORKSET_QUERIES_PATH)),
        ("workset query ids", Path(WORKSET_QUERY_IDS_PATH)),
        ("workset gt topk", Path(WORKSET_GT_TOPK_PATH)),
        ("offline cluster info", Path(WORKSET_CLUSTER_INFO_PATH)),
    ]
    ok = True
    for label, path in expected:
        ok = _check_required(label, path) and ok
    return ok


def _check_plaintext_ann(selection_summary_json: str) -> bool:
    print("\n== Non-Private PRISMA ==")
    ok = _check_main_assets()
    if str(selection_summary_json).strip():
        ok = _check_required(
            "selection summary json",
            Path(str(selection_summary_json).strip()),
        ) and ok
    else:
        _emit_status(
            True,
            "selection summary json",
            None,
            "Provide --selection-summary-json when running plaintext_ann.",
        )
    return ok


def _check_panther(project_root: Path) -> bool:
    print("\n== Panther ==")
    cfg = PantherConfig(project_root=project_root)
    openpanther_root = external_repo_root(cfg)
    bundle = bundle_paths(cfg, cluster_info_selector_c=int(cfg.default_cluster_info_selector_c))
    required = [
        ("OpenPanther repo root", openpanther_root),
        ("Panther bundle meta", bundle["bundle_meta"]),
        ("Panther docs", bundle["docs"]),
        ("Panther doc ids", bundle["doc_ids"]),
        ("Panther corpus", bundle["corpus"]),
        ("Panther evaluation queries", bundle["evaluation_queries"]),
        ("Panther evaluation query ids", bundle["evaluation_query_ids"]),
        ("Panther evaluation qrels", bundle["evaluation_qrels"]),
        ("Panther cluster info", bundle["cluster_info"]),
    ]
    ok = True
    for label, path in required:
        ok = _check_required(label, path) and ok
    return ok


def _check_tiptoe(project_root: Path, ranking_output_prefix: str) -> bool:
    print("\n== Tiptoe ==")
    cfg = TiptoeConfig(project_root=project_root)
    bundle = tiptoe_bundle_paths(cfg)
    required = [
        ("Tiptoe docs", bundle["docs"]),
        ("Tiptoe doc ids", bundle["doc_ids"]),
        ("Tiptoe corpus", bundle["corpus"]),
        ("Tiptoe evaluation queries", bundle["evaluation_queries"]),
        ("Tiptoe evaluation query ids", bundle["evaluation_query_ids"]),
        ("Tiptoe evaluation qrels", bundle["evaluation_qrels"]),
        ("Tiptoe cluster info", bundle["cluster_info"]),
    ]
    ok = True
    for label, path in required:
        ok = _check_required(label, path) and ok
    if str(ranking_output_prefix).strip():
        stem = str(ranking_output_prefix).strip()
        root = project_root / "results" / "repro_workflows" / "tiptoe"
        optional = [
            ("Tiptoe ranking rows", root / f"{stem}_rankings.jsonl"),
            ("Tiptoe ranking summary", root / f"{stem}_summary.json"),
        ]
        for label, path in optional:
            _check_required(label, path)
    return ok


def _check_remoterag(project_root: Path, module1_output_prefix: str) -> bool:
    print("\n== RemoteRAG ==")
    manifest_path = project_root / "results" / "repro_workflows" / "remoterag" / "prepared_worksets.json"
    _check_required("RemoteRAG prepared-workset manifest", manifest_path)
    if not str(module1_output_prefix).strip():
        _emit_status(
            True,
            "RemoteRAG Module 1 outputs",
            None,
            "Pass --remoterag-module1-prefix to check a specific precomputed Module 1 output pair.",
        )
        return True
    stem = str(module1_output_prefix).strip()
    root = project_root / "results" / "repro_workflows" / "remoterag"
    required = [
        ("RemoteRAG Module 1 rows", root / f"{stem}.jsonl"),
        ("RemoteRAG Module 1 summary", root / f"{stem}.json"),
    ]
    ok = True
    for label, path in required:
        ok = _check_required(label, path) and ok
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the upload-ready repository has the code-adjacent assets needed by the retained entrypoints."
    )
    parser.add_argument(
        "--component",
        type=str,
        default="all",
        choices=("all", "main", "plaintext_ann", "panther", "tiptoe", "remoterag"),
        help="Which component to check.",
    )
    parser.add_argument(
        "--selection-summary-json",
        type=str,
        default="",
        help="Optional plaintext_ann selection summary json to validate.",
    )
    parser.add_argument(
        "--tiptoe-ranking-prefix",
        type=str,
        default="",
        help="Optional Tiptoe ranking stem. Default uses tiptoe_ranking_service.",
    )
    parser.add_argument(
        "--remoterag-module1-prefix",
        type=str,
        default="",
        help="Optional RemoteRAG Module 1 stem to validate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    component = str(args.component).strip().lower()

    checks: list[bool] = []
    if component in {"all", "main"}:
        checks.append(_check_main_assets())
    if component in {"all", "plaintext_ann"}:
        checks.append(_check_plaintext_ann(str(args.selection_summary_json)))
    if component in {"all", "panther"}:
        checks.append(_check_panther(project_root))
    if component in {"all", "tiptoe"}:
        checks.append(_check_tiptoe(project_root, str(args.tiptoe_ranking_prefix)))
    if component in {"all", "remoterag"}:
        checks.append(_check_remoterag(project_root, str(args.remoterag_module1_prefix)))

    ok = all(bool(x) for x in checks) if checks else True
    print("\n== Result ==")
    if ok:
        print("Preflight passed for the requested checks.")
        raise SystemExit(0)
    print("Preflight found missing required assets.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
