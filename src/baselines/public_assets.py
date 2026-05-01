from __future__ import annotations

import json
import os
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PAPERFAITHFUL_SUFFIX_PREFIX = "_paperfaithful_mainline"


@dataclass(frozen=True)
class ResolvedWorksetAssets:
    workset_name: str
    docs: Path
    doc_ids: Path
    meta: Path | None
    corpus: Path
    queries: Path
    query_ids: Path
    gt_topk: Path
    strict_qrels: Path
    relaxed_qrels: Path | None
    queries_jsonl: Path | None
    cluster_info_pkl: Path

    def to_dict(self) -> dict[str, str | None]:
        return {
            "workset_name": str(self.workset_name),
            "docs": str(self.docs),
            "doc_ids": str(self.doc_ids),
            "meta": str(self.meta) if self.meta is not None else None,
            "corpus": str(self.corpus),
            "queries": str(self.queries),
            "query_ids": str(self.query_ids),
            "gt_topk": str(self.gt_topk),
            "strict_qrels": str(self.strict_qrels),
            "relaxed_qrels": str(self.relaxed_qrels) if self.relaxed_qrels is not None else None,
            "queries_jsonl": str(self.queries_jsonl) if self.queries_jsonl is not None else None,
            "cluster_info_pkl": str(self.cluster_info_pkl),
        }


def _resolve_variant_path(root: Path, stem: str, ext: str, *, required: bool) -> Path | None:
    exact = root / f"{stem}{PAPERFAITHFUL_SUFFIX_PREFIX}{ext}"
    if exact.exists():
        return exact.resolve()
    pattern = f"{stem}{PAPERFAITHFUL_SUFFIX_PREFIX}*{ext}"
    matches = sorted(root.glob(pattern))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        joined = "\n".join(str(path) for path in matches)
        raise FileExistsError(
            f"ambiguous asset match for {stem}{PAPERFAITHFUL_SUFFIX_PREFIX}*{ext} under {root}:\n{joined}"
        )
    if required:
        raise FileNotFoundError(f"missing required asset matching {stem}{PAPERFAITHFUL_SUFFIX_PREFIX}*{ext} under {root}")
    return None


def resolve_workset_assets(project_root: Path, workset_name: str) -> ResolvedWorksetAssets:
    workset = str(workset_name).strip()
    if not workset:
        raise ValueError("workset_name is empty")
    data_root = project_root / "data"
    raw_root = data_root / "raw"
    results_root = project_root / "results"
    return ResolvedWorksetAssets(
        workset_name=workset,
        docs=_resolve_variant_path(data_root, f"docs_{workset}", ".npy", required=True),
        doc_ids=_resolve_variant_path(data_root, f"doc_ids_{workset}", ".npy", required=True),
        meta=_resolve_variant_path(data_root, f"meta_{workset}", ".json", required=False),
        corpus=_resolve_variant_path(raw_root, f"corpus_{workset}", ".jsonl", required=True),
        queries=_resolve_variant_path(data_root, f"queries_{workset}", ".npy", required=True),
        query_ids=_resolve_variant_path(data_root, f"query_ids_{workset}", ".npy", required=True),
        gt_topk=_resolve_variant_path(data_root, f"gt_topk_{workset}", ".npy", required=True),
        strict_qrels=_resolve_variant_path(raw_root, f"qrels_{workset}", ".tsv", required=True),
        relaxed_qrels=_resolve_variant_path(raw_root, f"qrels_{workset}_relaxed", ".tsv", required=False),
        queries_jsonl=_resolve_variant_path(raw_root, f"queries_{workset}", ".jsonl", required=False),
        cluster_info_pkl=_resolve_variant_path(results_root, f"cluster_info_{workset}_balanced_spherical", ".pkl", required=True),
    )


def _extract_workset_name(path: Path) -> str | None:
    match = re.match(r"^docs_(.+?)_paperfaithful_mainline(?:_.+)?\.npy$", path.name)
    if match is None:
        return None
    return str(match.group(1))


def discover_ready_worksets(project_root: Path) -> list[str]:
    data_root = project_root / "data"
    if not data_root.exists():
        return []
    worksets: list[str] = []
    seen: set[str] = set()
    for path in sorted(data_root.glob(f"docs_*{PAPERFAITHFUL_SUFFIX_PREFIX}*.npy")):
        workset_name = _extract_workset_name(path)
        if workset_name is None or workset_name in seen:
            continue
        try:
            resolve_workset_assets(project_root, workset_name)
        except Exception:
            continue
        seen.add(workset_name)
        worksets.append(workset_name)
    return worksets


def resolve_requested_worksets(project_root: Path, requested: list[str]) -> list[str]:
    requested_clean = [str(x).strip() for x in requested if str(x).strip()]
    if requested_clean:
        ordered: list[str] = []
        seen: set[str] = set()
        for workset_name in requested_clean:
            if workset_name in seen:
                continue
            resolve_workset_assets(project_root, workset_name)
            seen.add(workset_name)
            ordered.append(workset_name)
        return ordered
    env_name = str(os.environ.get("WORKSET_NAME_OVERRIDE", "")).strip()
    if env_name:
        resolve_workset_assets(project_root, env_name)
        return [env_name]
    discovered = discover_ready_worksets(project_root)
    if discovered:
        return discovered
    raise FileNotFoundError("no ready PRISMA-compatible worksets were discovered under this repo")


def load_cluster_info(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_qrels_tsv(path: Path, *, allowed_query_ids: set[str] | None = None, allowed_doc_ids: set[str] | None = None) -> dict[str, list[str]]:
    qrels: dict[str, list[str]] = {}
    allowed_qids = {str(x) for x in allowed_query_ids} if allowed_query_ids is not None else None
    allowed_docs = {str(x) for x in allowed_doc_ids} if allowed_doc_ids is not None else None
    with open(path, "r", encoding="utf-8") as f:
        header_consumed = False
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if not header_consumed:
                header_consumed = True
                lowered = line.lower()
                if lowered.startswith("query_id\t"):
                    continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            query_id = str(parts[0]).strip()
            doc_id = str(parts[1]).strip()
            if not query_id or not doc_id:
                continue
            if allowed_qids is not None and query_id not in allowed_qids:
                continue
            if allowed_docs is not None and doc_id not in allowed_docs:
                continue
            bucket = qrels.setdefault(query_id, [])
            if doc_id not in bucket:
                bucket.append(doc_id)
    return qrels


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def workset_stats(assets: ResolvedWorksetAssets) -> dict[str, int]:
    docs = np.load(assets.docs, mmap_mode="r")
    queries = np.load(assets.queries, mmap_mode="r")
    cluster_info = load_cluster_info(assets.cluster_info_pkl)
    chunks = list(cluster_info.get("chunks", []))
    return {
        "num_docs": int(docs.shape[0]),
        "embedding_dim": int(docs.shape[1]),
        "num_queries": int(queries.shape[0]),
        "num_clusters": int(cluster_info.get("num_clusters", len(chunks))),
    }
