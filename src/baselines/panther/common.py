from __future__ import annotations

import json
import pickle
import os
import shutil
from pathlib import Path

import numpy as np

from baselines.panther.config import PantherConfig


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32)


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def load_corpus_rows(path: Path) -> list[dict]:
    return load_jsonl(path)


def load_qrels(path: Path) -> dict[str, set[str]]:
    raw = load_json(path)
    return {str(k): {str(x) for x in list(v)} for k, v in raw.items()}


def _is_bundle_root(path: Path, *, routing_c: int) -> bool:
    required = [
        path / "bundle_meta.json",
        path / "docs.npy",
        path / "doc_ids.npy",
        path / "corpus.jsonl",
        path / "evaluation_queries.npy",
        path / "evaluation_query_ids.npy",
        path / "evaluation_qrels.json",
        path / f"cluster_info_c{int(routing_c)}.pkl",
    ]
    return all(p.exists() for p in required)


def _discover_bundle_root(cfg: PantherConfig) -> Path | None:
    results_root = cfg.project_root / "results"
    if not results_root.exists():
        return None

    candidate_names: list[str] = []
    if str(cfg.bundle_root_name).strip():
        candidate_names.append(str(cfg.bundle_root_name).strip())
    candidate_names.extend(str(x).strip() for x in cfg.bundle_root_candidates if str(x).strip())

    seen: set[str] = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        candidate = results_root / name
        if _is_bundle_root(candidate, routing_c=int(cfg.default_cluster_info_selector_c)):
            return candidate.resolve()

    matched_roots: list[Path] = []
    for child in results_root.iterdir():
        if not child.is_dir():
            continue
        if _is_bundle_root(child, routing_c=int(cfg.default_cluster_info_selector_c)):
            matched_roots.append(child.resolve())
    if len(matched_roots) == 1:
        return matched_roots[0]
    return None


def bundle_root(cfg: PantherConfig) -> Path:
    override = str(os.environ.get("PANTHER_BUNDLE_ROOT", "")).strip()
    if override:
        return Path(override).expanduser().resolve()
    discovered = _discover_bundle_root(cfg)
    if discovered is not None:
        return discovered
    raise FileNotFoundError(
        "could not resolve a Panther bundle root under this repo. "
        "Resolution order is: `PANTHER_BUNDLE_ROOT`, configured candidate names, "
        "then a unique matching directory under `results/`. "
        "Place one compatible repo-local bundle under `results/` "
        "or set `PANTHER_BUNDLE_ROOT` explicitly."
    )


def bridge_root(cfg: PantherConfig) -> Path:
    return cfg.project_root / "results" / "repro_workflows" / "panther" / str(cfg.bridge_root_name)


def external_repo_root(cfg: PantherConfig) -> Path:
    raw = Path(str(cfg.external_repo_dir)).expanduser()
    if raw.is_absolute():
        return raw
    return (cfg.project_root / raw).resolve()


def bundle_paths(cfg: PantherConfig, *, cluster_info_selector_c: int | None = None) -> dict[str, Path]:
    root = bundle_root(cfg)
    paths = {
        "root": root,
        "bundle_meta": root / "bundle_meta.json",
        "docs": root / "docs.npy",
        "doc_ids": root / "doc_ids.npy",
        "corpus": root / "corpus.jsonl",
        "evaluation_queries": root / "evaluation_queries.npy",
        "evaluation_query_ids": root / "evaluation_query_ids.npy",
        "evaluation_qrels": root / "evaluation_qrels.json",
    }
    if cluster_info_selector_c is not None and int(cluster_info_selector_c) > 0:
        paths["cluster_info"] = root / f"cluster_info_c{int(cluster_info_selector_c)}.pkl"
    return paths


def ensure_ms_bundle(cfg: PantherConfig, *, cluster_info_selector_c: int | None = None) -> None:
    paths = bundle_paths(cfg, cluster_info_selector_c=cluster_info_selector_c)
    required = [
        paths["bundle_meta"],
        paths["docs"],
        paths["doc_ids"],
        paths["corpus"],
        paths["evaluation_queries"],
        paths["evaluation_query_ids"],
        paths["evaluation_qrels"],
    ]
    cluster_info_path = paths.get("cluster_info")
    if cluster_info_path is not None:
        required.append(cluster_info_path)
    if all(path.exists() for path in required):
        return
    missing = [str(path) for path in required if not path.exists()]
    raise FileNotFoundError(
        "Panther aligned bundle assets are missing from upload_ready_code_20260430. "
        "This upload pack expects a repo-local bundle root. "
        "Auto-discovery checks `PANTHER_BUNDLE_ROOT` first, "
        "then common public bundle names, then a unique matching directory under `results/`. "
        f"Resolved path was {bundle_root(cfg)}. Missing files:\n" + "\n".join(missing)
    )


def load_cluster_info(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_bundle(
    cfg: PantherConfig,
    *,
    query_limit: int | None = None,
    cluster_info_selector_c: int | None = None,
) -> dict:
    ensure_ms_bundle(cfg, cluster_info_selector_c=cluster_info_selector_c)
    paths = bundle_paths(cfg, cluster_info_selector_c=cluster_info_selector_c)
    docs = normalize_rows(np.load(paths["docs"]).astype(np.float32))
    queries = normalize_rows(np.load(paths["evaluation_queries"]).astype(np.float32))
    doc_ids = [str(x) for x in np.load(paths["doc_ids"], allow_pickle=True).tolist()]
    query_ids = [str(x) for x in np.load(paths["evaluation_query_ids"], allow_pickle=True).tolist()]
    if query_limit is not None and int(query_limit) > 0:
        queries = queries[: int(query_limit)]
        query_ids = query_ids[: int(query_limit)]
    qrels = load_qrels(paths["evaluation_qrels"])
    corpus_rows = load_corpus_rows(paths["corpus"])
    bundle_meta = load_json(paths["bundle_meta"])
    cluster_info = None
    if "cluster_info" in paths:
        cluster_info = load_cluster_info(paths["cluster_info"])
    return {
        "paths": paths,
        "bundle_meta": bundle_meta,
        "docs": docs,
        "doc_ids": doc_ids,
        "queries": queries,
        "query_ids": query_ids,
        "qrels": qrels,
        "corpus_rows": corpus_rows,
        "cluster_info": cluster_info,
    }


def exact_topk_indices(*, docs: np.ndarray, query: np.ndarray, top_k: int) -> np.ndarray:
    scores = np.asarray(docs, dtype=np.float32) @ np.asarray(query, dtype=np.float32).reshape(-1)
    order = np.argsort(-scores, kind="mergesort")
    return np.asarray(order[: int(top_k)], dtype=np.int32)


def first_relevant_rank(pred_doc_ids: list[str], positive_doc_ids: set[str], cutoff: int) -> int | None:
    positives = {str(x) for x in positive_doc_ids}
    if not positives:
        return None
    for rank, doc_id in enumerate(pred_doc_ids[: int(cutoff)], start=1):
        if str(doc_id) in positives:
            return int(rank)
    return None


def cluster_assignments_from_info(cluster_info: dict, *, num_docs: int) -> tuple[np.ndarray, np.ndarray]:
    assignments = np.full(int(num_docs), -1, dtype=np.int32)
    chunks = list(cluster_info.get("chunks", []))
    for cluster_id, chunk in enumerate(chunks):
        indices = np.asarray(chunk, dtype=np.int32).reshape(-1)
        assignments[indices] = int(cluster_id)
    stash = np.flatnonzero(assignments < 0).astype(np.int32)
    return assignments, stash


def write_matrix_txt(path: Path, matrix: np.ndarray, *, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(matrix), fmt=fmt)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def copy_corpus_jsonl(path: Path, rows: list[dict]) -> None:
    write_jsonl(path, rows)


def copy_json(path: Path, payload: dict | list) -> None:
    save_json(path, payload)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
