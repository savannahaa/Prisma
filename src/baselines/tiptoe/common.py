from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from baselines.tiptoe.config import TiptoeConfig


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32)


def normalize_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(x))
    if norm <= 1e-12:
        raise ValueError("vector norm too small")
    return (x / norm).astype(np.float32)


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_bundle_root(path: Path, *, routing_c: int) -> bool:
    required = [
        path / "docs.npy",
        path / "doc_ids.npy",
        path / "corpus.jsonl",
        path / "evaluation_queries.npy",
        path / "evaluation_query_ids.npy",
        path / "evaluation_qrels.json",
        path / f"cluster_info_c{int(routing_c)}.pkl",
    ]
    return all(p.exists() for p in required)


def _discover_bundle_root(cfg: TiptoeConfig) -> Path | None:
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
        if _is_bundle_root(candidate, routing_c=int(cfg.routing_c)):
            return candidate.resolve()

    matched_roots: list[Path] = []
    for child in results_root.iterdir():
        if not child.is_dir():
            continue
        if _is_bundle_root(child, routing_c=int(cfg.routing_c)):
            matched_roots.append(child.resolve())
    if len(matched_roots) == 1:
        return matched_roots[0]
    return None


def bundle_root(cfg: TiptoeConfig) -> Path:
    override = str(os.environ.get("TIPTOE_BUNDLE_ROOT", "")).strip()
    if override:
        return Path(override).expanduser().resolve()
    discovered = _discover_bundle_root(cfg)
    if discovered is not None:
        return discovered
    raise FileNotFoundError(
        "could not resolve a Tiptoe bundle root under this repo. "
        "Resolution order is: `TIPTOE_BUNDLE_ROOT`, configured candidate names, "
        "then a unique matching directory under `results/`. "
        "Place one compatible repo-local bundle under `results/` "
        "or set `TIPTOE_BUNDLE_ROOT` explicitly."
    )


def bundle_paths(cfg: TiptoeConfig) -> dict[str, Path]:
    root = bundle_root(cfg)
    return {
        "root": root,
        "docs": root / "docs.npy",
        "doc_ids": root / "doc_ids.npy",
        "corpus": root / "corpus.jsonl",
        "evaluation_queries": root / "evaluation_queries.npy",
        "evaluation_query_ids": root / "evaluation_query_ids.npy",
        "evaluation_qrels": root / "evaluation_qrels.json",
        "cluster_info": root / f"cluster_info_c{int(cfg.routing_c)}.pkl",
    }


def load_qrels(path: Path) -> dict[str, set[str]]:
    raw = json.load(open(path, "r", encoding="utf-8"))
    return {str(k): {str(x) for x in list(v)} for k, v in raw.items()}


def load_corpus_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
