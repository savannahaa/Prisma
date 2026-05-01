"""
从真实query里面挑出适合前2000个工作集的一小批query：
·先找哪些能映射到2000个工作集里
·再算每个query离哪个簇中心更近
·检查准确top-k是否和最近父簇一致，优先选择语义上更近的
输出：
·data/queries_e5_workset_2000.npy
选中的 query embedding。

·data/query_ids_e5_workset_2000.npy
选中的 query id。

·data/gt_topk_e5_workset_2000.npy
对每个 query，在当前 2000 文档上按余弦精确计算出来的 top-k 索引，后面用来算 exact recall。

·data/raw/queries_e5_workset_2000.jsonl
选中的 query 文本和其选择信息。

·data/raw/qrels_e5_workset_2000.tsv
strict qrels，只保留当前工作集里真实正例命中的 query-doc 对。

·data/raw/qrels_e5_workset_2000_relaxed.tsv
relaxed qrels，不是原始标注，而是按当前 embedding 几何定义出的 relaxed positive 集合。

还有一个派生元数据文件：
·data/queries_e5_workset_2000.meta.json
这个路径不在 config.py 里显式写死，而是根据 WORKSET_QUERIES_PATH 自动推出来的。
"""

# Allow running this file directly: `python src/offline/prepare_real_queries.py`
if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import json
import hashlib
import os
import pickle
import re
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np

from shared.config import (
    WORKSET_DOCS_PATH,
    WORKSET_DOC_IDS_PATH,
    WORKSET_CORPUS_JSONL_PATH,
    WORKSET_NAME,
    WORKSET_QUERIES_PATH,
    WORKSET_QUERY_IDS_PATH,
    WORKSET_GT_TOPK_PATH,
    WORKSET_CALIBRATION_QUERIES_PATH,
    WORKSET_CALIBRATION_QUERY_IDS_PATH,
    WORKSET_QRELS_PATH,
    WORKSET_RELAXED_QRELS_PATH,
    WORKSET_QUERIES_JSONL_PATH,
    WORKSET_CALIBRATION_QUERIES_JSONL_PATH,
    WORKSET_QUERY_SPLIT_META_PATH,
    WORKSET_META_PATH,
    WORKSET_CLUSTER_INFO_PATH,
    FULL_QUERIES_JSONL_PATH,
    FULL_QRELS_TSV_PATH,
    QUERY_CALIBRATION_RATIO,
    QUERY_CALIBRATION_MIN_COUNT,
    QUERY_EVALUATION_MIN_COUNT,
    QUERY_SPLIT_SEED,
    NEW_MODEL_NAME,
    BATCH_SIZE,
    MAX_LENGTH,
    EVAL_K,
    FIXED_K,
    ALPHA,
    EPSILON,
    NUM_CLUSTERS,
    TARGET_CLUSTER_SIZE,
    PIPELINE_IS_PAPERFAITHFUL_MAINLINE,
    PAPERFAITHFUL_MAINLINE_TRACK1_ONLY,
    EPS,
)
from shared.cluster_info_contract import assert_cluster_info_contract
from shared.e5_dual_encoder import E5DualEncoder as _E5EncoderBackend
from client.privacy_gate import gate_one_query
from server.cluster_retrieval import filtered_global_hnsw_topk_doc_indices_and_scores


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return str(default)
    text = str(raw).strip()
    return text if text else str(default)


FORCE_REBUILD_QUERY_WORKSET = _env_flag("FORCE_REBUILD_QUERY_WORKSET", False)
QUERY_FAST_MODE = _env_flag(
    "QUERY_FAST_MODE",
    True,
)


def _default_calibration_total() -> int:
    if not PIPELINE_IS_PAPERFAITHFUL_MAINLINE:
        return 0
    if bool(QUERY_FAST_MODE):
        return max(int(NUM_CLUSTERS) * 10, 200)
    return int(NUM_CLUSTERS) * 220


def _default_evaluation_total() -> int:
    if not PIPELINE_IS_PAPERFAITHFUL_MAINLINE:
        return 0
    # 当前主线实验固定只保留 120 条 evaluation queries。
    return 120


QUERY_CALIBRATION_TARGET_TOTAL = _env_int("QUERY_CALIBRATION_TARGET_TOTAL", _default_calibration_total())
QUERY_EVALUATION_TARGET_TOTAL = _env_int("QUERY_EVALUATION_TARGET_TOTAL", _default_evaluation_total())
# paper-faithful 主线下，不再把 evaluation pool 全量吃掉；改为按簇定额构造。
QUERY_CALIBRATION_TARGET_PER_CLUSTER = (
    int(max(1, np.ceil(float(QUERY_CALIBRATION_TARGET_TOTAL) / max(1, int(NUM_CLUSTERS)))))
    if PIPELINE_IS_PAPERFAITHFUL_MAINLINE
    else 0
)
QUERY_EVALUATION_TARGET_PER_CLUSTER = (
    int(max(1, np.ceil(float(QUERY_EVALUATION_TARGET_TOTAL) / max(1, int(NUM_CLUSTERS)))))
    if PIPELINE_IS_PAPERFAITHFUL_MAINLINE
    else 0
)
TARGET_NUM_QUERIES = (
    int(NUM_CLUSTERS) * int(QUERY_EVALUATION_TARGET_PER_CLUSTER)
    if PIPELINE_IS_PAPERFAITHFUL_MAINLINE
    else 0
)
QUERY_EMBEDDING_PREPROCESS = "none_raw_e5"
QUERY_CALIBRATION_SOURCE_POLICY = _env_str(
    "QUERY_CALIBRATION_SOURCE_POLICY",
    "real_only",
).strip().lower()
if QUERY_CALIBRATION_SOURCE_POLICY != "real_only":
    raise ValueError(
        "invalid QUERY_CALIBRATION_SOURCE_POLICY="
        f"{QUERY_CALIBRATION_SOURCE_POLICY}, expected 'real_only'"
    )
if PIPELINE_IS_PAPERFAITHFUL_MAINLINE:
    if bool(QUERY_FAST_MODE):
        QUERY_SELECTION_POLICY = "stable_random_real_eval_real_only_calibration_v3"
        QUERY_SPLIT_PROTOCOL_VERSION = "random_real_eval_disjoint_real_only_calibration_v3"
    else:
        QUERY_SELECTION_POLICY = "cluster_balanced_real_only_eval_real_only_calibration_v1"
        QUERY_SPLIT_PROTOCOL_VERSION = "cluster_balanced_disjoint_real_only_calibration_v1"
else:
    QUERY_SELECTION_POLICY = "cluster_balanced_difficulty_stratified"
    QUERY_SPLIT_PROTOCOL_VERSION = "query_pool_split_v1_disjoint_hash"
QUERY_EVALUATION_RANKING_POLICY = _env_str(
    "QUERY_EVALUATION_RANKING_POLICY",
    "paperfaithful_real_natural" if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else "coverage_first_legacy",
).strip().lower()
if QUERY_EVALUATION_RANKING_POLICY not in {
    "paperfaithful_real_natural",
    "coverage_first_legacy",
}:
    raise ValueError(
        "invalid QUERY_EVALUATION_RANKING_POLICY="
        f"{QUERY_EVALUATION_RANKING_POLICY}, expected one of "
        "['paperfaithful_real_natural', 'coverage_first_legacy']"
    )
QUERY_EVAL_SAMPLING_POLICY = (
    "stable_random_real_with_source_query_id"
    if PIPELINE_IS_PAPERFAITHFUL_MAINLINE
    else "legacy_ranked_selection"
)
QUERY_USE_REAL_QUERY_POOL = _env_flag("QUERY_USE_REAL_QUERY_POOL", True)
QUERY_REAL_POOL_REQUIRE_WORKSET_POSITIVE = _env_flag(
    "QUERY_REAL_POOL_REQUIRE_WORKSET_POSITIVE",
    False,
)
QUERY_REAL_POOL_ENCODE_CAP = _env_int(
    "QUERY_REAL_POOL_ENCODE_CAP",
    0 if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else (400 if bool(QUERY_FAST_MODE) else 0),
)
QUERY_EVALUATION_TRACK1_ONLY = bool(
    PIPELINE_IS_PAPERFAITHFUL_MAINLINE and bool(PAPERFAITHFUL_MAINLINE_TRACK1_ONLY)
)
WORKSET_QUERY_META_PATH = os.path.splitext(WORKSET_QUERIES_PATH)[0] + ".meta.json"

# 真实 query 池扩容策略（论文主图建议：总量 >=300，兼顾稳定性与运行成本）。
AUTO_EXPAND_FULL_QUERY_POOL = _env_flag("AUTO_EXPAND_FULL_QUERY_POOL", not bool(QUERY_FAST_MODE))
# 为 r_max calibration 留出足够 anchor 候选。
# 当前默认至少覆盖 calibration + evaluation 的按簇定额需求，再留少量余量，
# 让 10000/20 簇场景下也能直接运行。
TARGET_FULL_QUERY_POOL_SIZE = max(
    5000,
    int(NUM_CLUSTERS)
    * int(max(QUERY_CALIBRATION_TARGET_PER_CLUSTER + QUERY_EVALUATION_TARGET_PER_CLUSTER, 400)),
)
QUERY_POOL_EXPAND_SEED = 20260409
MSMARCO_QUERIES_DEV_SMALL_URL = (
    "https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/queries.dev.small.tsv"
)
MSMARCO_QUERIES_DEV_SMALL_CACHE = os.path.join(
    os.path.dirname(FULL_QUERIES_JSONL_PATH),
    "queries_msmarco_dev_small.tsv",
)

ENGLISH_STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "more",
    "most",
    "not",
    "of",
    "on",
    "or",
    "our",
    "out",
    "over",
    "s",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "under",
    "until",
    "up",
    "use",
    "using",
    "used",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


class E5QueryEncoder:
    def __init__(self, model_name: str):
        self.backend = _E5EncoderBackend(
            model_name,
            log_prefix="encoder",
        )

    def encode_queries(self, texts: List[str], batch_size: int = 8, max_length: int = 512) -> np.ndarray:
        _, norm = self.backend.encode_queries(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            progress_name="encode-query",
        )
        return norm.astype(np.float32)


def clean_text(text: str) -> str:
    text = str(text).replace("\n", " ").replace("\t", " ").strip()
    return " ".join(text.split())


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return (x / norms).astype(np.float32)


def _gate_track1_diagnostics(
    *,
    query: np.ndarray,
    centers: np.ndarray,
    cluster_r_k: np.ndarray,
    cluster_r_fixed: np.ndarray,
    cluster_r_max: np.ndarray,
) -> dict:
    gate = gate_one_query(
        query=np.asarray(query, dtype=np.float32),
        centers=np.asarray(centers, dtype=np.float32),
        cluster_r_max=np.asarray(cluster_r_max, dtype=np.float32),
        cluster_r_k=np.asarray(cluster_r_k, dtype=np.float32),
        cluster_r_fixed=np.asarray(cluster_r_fixed, dtype=np.float32),
        alpha=float(ALPHA),
        epsilon=float(EPSILON),
        epsilon_min=None,
        epsilon_max=None,
        enforce_epsilon_interval=False,
    )
    return {
        "track": str(gate.get("track", "")),
        "cluster_id": int(gate.get("cluster_id", 0)),
        "r_max": float(gate.get("r_max", 0.0)),
        "r_rdp_bar": float(gate.get("r_rdp_bar", 0.0)),
        "epsilon_used": float(gate.get("epsilon_used", EPSILON)),
        "margin_to_track1": float(gate.get("r_max", 0.0) - gate.get("r_rdp_bar", 0.0)),
    }


def _filter_track1_candidates_by_embedding_index(
    *,
    candidates: List[dict],
    query_emb: np.ndarray,
    centers: np.ndarray,
    cluster_r_k: np.ndarray,
    cluster_r_fixed: np.ndarray,
    cluster_r_max: np.ndarray,
) -> tuple[List[dict], dict]:
    filtered: List[dict] = []
    track_counter: Dict[str, int] = {}
    for item in candidates:
        emb_idx = int(item.get("embedding_index", item.get("idx", -1)))
        if emb_idx < 0 or emb_idx >= int(len(query_emb)):
            continue
        gate_diag = _gate_track1_diagnostics(
            query=np.asarray(query_emb[emb_idx], dtype=np.float32),
            centers=centers,
            cluster_r_k=cluster_r_k,
            cluster_r_fixed=cluster_r_fixed,
            cluster_r_max=cluster_r_max,
        )
        item["selection_gate_track"] = str(gate_diag["track"])
        item["selection_gate_cluster_id"] = int(gate_diag["cluster_id"])
        item["selection_gate_r_max"] = float(gate_diag["r_max"])
        item["selection_gate_r_rdp_bar"] = float(gate_diag["r_rdp_bar"])
        item["selection_gate_margin_to_track1"] = float(gate_diag["margin_to_track1"])
        track = str(gate_diag["track"])
        track_counter[track] = int(track_counter.get(track, 0)) + 1
        if track == "dense_rdp":
            filtered.append(item)
    return filtered, {
        "input_count": int(len(candidates)),
        "track_counts": {str(k): int(v) for k, v in sorted(track_counter.items())},
        "track1_count": int(len(filtered)),
        "dropped_count": int(len(candidates) - len(filtered)),
    }


def _filter_track1_materialized_queries(
    *,
    items: List[dict],
    query_emb: np.ndarray,
    centers: np.ndarray,
    cluster_r_k: np.ndarray,
    cluster_r_fixed: np.ndarray,
    cluster_r_max: np.ndarray,
) -> tuple[List[dict], np.ndarray, dict]:
    filtered_items: List[dict] = []
    filtered_emb_rows: List[np.ndarray] = []
    track_counter: Dict[str, int] = {}
    for idx, item in enumerate(items):
        gate_diag = _gate_track1_diagnostics(
            query=np.asarray(query_emb[idx], dtype=np.float32),
            centers=centers,
            cluster_r_k=cluster_r_k,
            cluster_r_fixed=cluster_r_fixed,
            cluster_r_max=cluster_r_max,
        )
        item["selection_gate_track"] = str(gate_diag["track"])
        item["selection_gate_cluster_id"] = int(gate_diag["cluster_id"])
        item["selection_gate_r_max"] = float(gate_diag["r_max"])
        item["selection_gate_r_rdp_bar"] = float(gate_diag["r_rdp_bar"])
        item["selection_gate_margin_to_track1"] = float(gate_diag["margin_to_track1"])
        track = str(gate_diag["track"])
        track_counter[track] = int(track_counter.get(track, 0)) + 1
        if track != "dense_rdp":
            continue
        filtered_items.append(item)
        filtered_emb_rows.append(np.asarray(query_emb[idx], dtype=np.float32))
    filtered_emb = (
        normalize_rows(np.asarray(filtered_emb_rows, dtype=np.float32))
        if len(filtered_emb_rows) > 0
        else np.zeros((0, int(query_emb.shape[1])), dtype=np.float32)
    )
    return filtered_items, filtered_emb, {
        "input_count": int(len(items)),
        "track_counts": {str(k): int(v) for k, v in sorted(track_counter.items())},
        "track1_count": int(len(filtered_items)),
        "dropped_count": int(len(items) - len(filtered_items)),
    }


def parse_query_row(row: dict) -> Tuple[str, str, Optional[str]]:
    qid = None
    for key in ("query_id", "id", "_id"):
        if key in row:
            qid = str(row[key]).strip()
            break

    qtext = None
    for key in ("text", "query", "contents", "content"):
        if key in row:
            qtext = clean_text(row[key])
            break

    if not qid or not qtext:
        raise ValueError("invalid query row")
    source_query_id = row.get("source_query_id")
    if source_query_id is not None:
        source_query_id = str(source_query_id).strip()
        if len(source_query_id) == 0:
            source_query_id = None
    return qid, qtext, source_query_id


def canonical_query_id(raw_query_id: str, source_query_id: Optional[str]) -> str:
    source_qid = str(source_query_id).strip() if source_query_id is not None else ""
    if source_qid:
        return source_qid
    return str(raw_query_id).strip()


def stable_hash_to_unit_interval(s: str) -> float:
    h = hashlib.sha256(str(s).encode("utf-8")).hexdigest()
    # 取前 16 hex（64bit）即可稳定映射到 [0, 1)。
    n = int(h[:16], 16)
    return float(n / float(2**64))


def split_query_rows_for_protocol(
    all_rows: List[dict],
    *,
    calibration_ratio: float,
    calibration_min_count: int,
    evaluation_min_count: int,
    split_seed: int,
) -> Tuple[List[dict], List[dict], dict]:
    rows = list(all_rows)
    n = int(len(rows))
    if n <= 1:
        raise RuntimeError("not enough queries to split calibration/evaluation pools")

    scored = []
    for row in rows:
        qid = str(row["query_id"])
        score = stable_hash_to_unit_interval(f"{int(split_seed)}::{qid}")
        scored.append((score, qid, row))
    scored.sort(key=lambda x: (float(x[0]), str(x[1])))

    ratio = float(np.clip(float(calibration_ratio), 0.05, 0.95))
    min_cal = int(max(1, int(calibration_min_count)))
    min_eval = int(max(1, int(evaluation_min_count)))
    if min_cal + min_eval > n:
        min_cal = int(max(1, min(min_cal, n // 2)))
        min_eval = int(max(1, n - min_cal))
    desired_cal = int(round(ratio * n))
    desired_cal = int(max(min_cal, min(desired_cal, n - min_eval)))

    calibration_rows = [x[2] for x in scored[:desired_cal]]
    evaluation_rows = [x[2] for x in scored[desired_cal:]]

    cal_qids = {str(r["query_id"]) for r in calibration_rows}
    eval_qids = {str(r["query_id"]) for r in evaluation_rows}
    overlap = sorted(cal_qids & eval_qids)
    if overlap:
        raise RuntimeError(f"calibration/evaluation split overlap detected: {overlap[:3]}")
    if len(evaluation_rows) == 0:
        raise RuntimeError("empty evaluation pool after split")
    if len(calibration_rows) == 0:
        raise RuntimeError("empty calibration pool after split")

    meta = {
        "protocol_version": "query_pool_split_v1_disjoint_hash",
        "split_seed": int(split_seed),
        "calibration_ratio": float(ratio),
        "calibration_min_count": int(calibration_min_count),
        "evaluation_min_count": int(evaluation_min_count),
        "num_queries_total": int(n),
        "num_queries_calibration": int(len(calibration_rows)),
        "num_queries_evaluation": int(len(evaluation_rows)),
        "split_overlap_count": int(len(overlap)),
        "split_overlap_query_ids": overlap,
    }
    return calibration_rows, evaluation_rows, meta


def source_doc_id_aliases(doc_id: str) -> List[str]:
    s = str(doc_id).strip()
    if len(s) == 0:
        return []
    aliases = [s]
    if s.startswith("d") and s[1:].isdigit():
        aliases.append(s[1:])
    elif s.startswith("D") and s[1:].isdigit():
        aliases.append(s[1:])
        aliases.append(f"d{s[1:]}")
    elif s.isdigit():
        aliases.append(f"d{s}")

    out = []
    seen = set()
    for x in aliases:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_msmarco_queries_tsv(path: str) -> Dict[str, str]:
    q = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            qid = str(parts[0]).strip()
            text = clean_text(parts[1])
            if not qid or not text:
                continue
            q.setdefault(qid, text)
    return q


def load_qrels_query_ids(path: str) -> List[str]:
    ids = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 1:
                continue
            qid = str(parts[0]).strip()
            if not qid or qid in seen:
                continue
            seen.add(qid)
            ids.append(qid)
    return ids


def ensure_full_query_pool_size(
    *,
    full_queries_jsonl_path: str,
    qrels_tsv_path: str,
    target_size: int,
    source_url: str,
    cache_tsv_path: str,
    seed: int,
) -> dict:
    os.makedirs(os.path.dirname(full_queries_jsonl_path), exist_ok=True)
    existing_rows = load_jsonl(full_queries_jsonl_path)
    if len(existing_rows) >= int(target_size):
        return {
            "expanded": False,
            "reason": "already_large_enough",
            "target_size": int(target_size),
            "existing_size_before": int(len(existing_rows)),
            "existing_size_after": int(len(existing_rows)),
            "added": 0,
            "source_url": source_url,
            "cache_tsv_path": cache_tsv_path,
        }

    if not os.path.exists(cache_tsv_path):
        print(
            "query pool expansion: downloading msmarco dev small queries to cache: "
            f"{cache_tsv_path}"
        )
        urllib.request.urlretrieve(source_url, cache_tsv_path)

    msmarco_queries = load_msmarco_queries_tsv(cache_tsv_path)
    qrels_query_ids = set(load_qrels_query_ids(qrels_tsv_path))

    existing_source_qids = set()
    existing_texts = set()
    max_auto_id = -1
    for row in existing_rows:
        existing_texts.add(clean_text(str(row.get("text", ""))).lower())
        source_qid = row.get("source_query_id")
        if source_qid is not None:
            source_qid = str(source_qid).strip()
            if source_qid:
                existing_source_qids.add(source_qid)
        qid = str(row.get("query_id", "")).strip()
        if qid.startswith("q_real_auto_"):
            try:
                max_auto_id = max(max_auto_id, int(qid.split("_")[-1]))
            except Exception:
                pass

    candidates = []
    for qid in qrels_query_ids:
        text = msmarco_queries.get(str(qid))
        if text is None:
            continue
        text_clean = clean_text(text)
        if len(text_clean) == 0:
            continue
        tok_len = len(text_clean.split())
        if tok_len < 2 or tok_len > 24:
            continue
        if str(qid) in existing_source_qids:
            continue
        if text_clean.lower() in existing_texts:
            continue
        score = stable_hash_to_unit_interval(f"{int(seed)}::{qid}")
        candidates.append((score, str(qid), text_clean))
    candidates.sort(key=lambda x: (float(x[0]), str(x[1])))

    needed = int(max(0, int(target_size) - int(len(existing_rows))))
    selected = candidates[:needed]

    now_utc = datetime.now(timezone.utc).isoformat()
    next_auto_id = int(max_auto_id + 1)
    added_rows = []
    for _, source_qid, text in selected:
        row = {
            "query_id": f"q_real_auto_{next_auto_id}",
            "text": text,
            "source": "real_query_crawled_from_public_dataset",
            "source_dataset": "msmarco_dev_small",
            "source_url": source_url,
            "source_query_id": str(source_qid),
            "fetched_at_utc": now_utc,
            "query_pool_policy": "msmarco_dev_small_qrels_hash_sample",
            "query_pool_seed": int(seed),
        }
        next_auto_id += 1
        added_rows.append(row)

    all_rows = list(existing_rows) + added_rows
    save_jsonl(full_queries_jsonl_path, all_rows)
    return {
        "expanded": bool(len(added_rows) > 0),
        "reason": "expanded_from_msmarco_dev_small",
        "target_size": int(target_size),
        "existing_size_before": int(len(existing_rows)),
        "existing_size_after": int(len(all_rows)),
        "added": int(len(added_rows)),
        "source_url": source_url,
        "cache_tsv_path": cache_tsv_path,
        "candidate_pool_size": int(len(candidates)),
        "qrels_query_id_count": int(len(qrels_query_ids)),
        "msmarco_query_count": int(len(msmarco_queries)),
    }


def load_qrels_map(path: str) -> Dict[str, List[str]]:
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise RuntimeError("qrels.tsv is empty")

    first = lines[0].split("\t")
    if first == ["query_id", "doc_id"]:
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            qid, did = parts
            qrels.setdefault(qid, []).append(did)
        return qrels

    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            qid = str(parts[0])
            did = str(parts[2])
            qrels.setdefault(qid, []).append(did)
    return qrels


def cosine_scores(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    docs = np.asarray(docs, dtype=np.float64)
    q_norm = float(np.linalg.norm(query))
    if q_norm <= EPS:
        raise ValueError("query norm too small")
    numer = np.sum(docs * query[None, :], axis=1, dtype=np.float64)
    d_norms = np.sqrt(np.sum(docs * docs, axis=1, dtype=np.float64))
    denom = np.maximum(q_norm * d_norms, EPS)
    return (numer / denom).astype(np.float64)


def topk_indices_by_cosine(query: np.ndarray, docs: np.ndarray, top_k: int) -> np.ndarray:
    sims = cosine_scores(query, docs)
    all_indices = np.arange(len(sims), dtype=np.int32)
    order = np.lexsort((all_indices, -sims))
    return all_indices[order[:top_k]].astype(np.int32)


def relaxed_positive_indices_by_cosine(query: np.ndarray, docs: np.ndarray, top_k: int) -> np.ndarray:
    sims = cosine_scores(query, docs)
    topk = topk_indices_by_cosine(query, docs, top_k=top_k)
    kth_score = float(sims[int(topk[-1])])
    return np.where(sims >= kth_score - 1e-8)[0].astype(np.int32)


def fingerprint_doc_ids(doc_ids: np.ndarray) -> str:
    joined = "\n".join(str(x) for x in doc_ids.tolist())
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def fingerprint_cluster_query_context(cluster_info: dict) -> str:
    h = hashlib.sha256()
    centers = normalize_rows(np.asarray(cluster_info.get("centers", []), dtype=np.float32))
    chunks = list(cluster_info.get("chunks", []))
    h.update(f"centers_shape={tuple(int(x) for x in centers.shape)};".encode("utf-8"))
    h.update(centers.tobytes())
    h.update(f"num_chunks={int(len(chunks))};".encode("utf-8"))
    for cid, chunk in enumerate(chunks):
        arr = np.asarray(chunk, dtype=np.int32).reshape(-1)
        h.update(f"chunk={int(cid)};size={int(arr.size)};".encode("utf-8"))
        h.update(arr.tobytes())
    for key in ("num_clusters", "eval_k", "fixed_k", "target_cluster_size"):
        h.update(f"{key}={cluster_info.get(key, None)};".encode("utf-8"))
    return h.hexdigest()


def nearest_doc_angular_dist(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    q = np.asarray(queries, dtype=np.float32)
    d = np.asarray(docs, dtype=np.float32)
    sims = (q @ d.T).astype(np.float32)
    max_sim = np.max(sims, axis=1).astype(np.float32)
    max_sim = np.clip(max_sim, -1.0, 1.0)
    theta = np.arccos(max_sim).astype(np.float32)
    return theta


def assign_difficulty_bucket(scores: np.ndarray) -> List[str]:
    x = np.asarray(scores, dtype=np.float64)
    if len(x) == 0:
        return []
    if len(x) < 3:
        return ["mid"] * len(x)
    q33, q66 = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0])
    buckets = []
    for s in x.tolist():
        if s <= q33:
            buckets.append("easy")
        elif s <= q66:
            buckets.append("mid")
        else:
            buckets.append("hard")
    return buckets


def reorder_candidates_for_difficulty_balance(items: List[dict]) -> List[dict]:
    if len(items) <= 1:
        return list(items)
    groups = {"easy": [], "mid": [], "hard": [], "unknown": []}
    for item in sorted(
        items,
        key=lambda c: (
            float(c["semantic_score"]),
            str(c["query_id"]),
        ),
    ):
        bucket = str(item.get("difficulty_bucket", "unknown"))
        groups[bucket if bucket in groups else "unknown"].append(item)

    ordered = []
    while True:
        progressed = False
        for key in ("mid", "easy", "hard", "unknown"):
            if groups[key]:
                ordered.append(groups[key].pop(0))
                progressed = True
        if not progressed:
            break
    return ordered


def allocate_balanced_cluster_quotas(
    cluster_to_size: Dict[int, int],
    target_total: int,
) -> Dict[int, int]:
    active = [cid for cid, sz in cluster_to_size.items() if int(sz) > 0]
    quotas = {int(cid): 0 for cid in cluster_to_size.keys()}
    if target_total <= 0 or len(active) == 0:
        return quotas

    cap_total = int(sum(int(cluster_to_size[cid]) for cid in active))
    if target_total >= cap_total:
        for cid in active:
            quotas[int(cid)] = int(cluster_to_size[cid])
        return quotas

    base = int(target_total // len(active))
    for cid in active:
        quotas[int(cid)] = int(min(int(cluster_to_size[cid]), base))

    assigned = int(sum(quotas.values()))
    while assigned < int(target_total):
        progressed = False
        cid_order = sorted(
            active,
            key=lambda c: (
                -(int(cluster_to_size[c]) - int(quotas[int(c)])),
                int(c),
            ),
        )
        for cid in cid_order:
            cid = int(cid)
            if int(quotas[cid]) < int(cluster_to_size[cid]):
                quotas[cid] += 1
                assigned += 1
                progressed = True
                if assigned >= int(target_total):
                    break
        if not progressed:
            break
    return quotas


def _candidate_majority_cluster(cluster_ids: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(cluster_ids, dtype=np.int32), return_counts=True)
    if len(vals) == 0:
        return -1
    order = np.lexsort((vals, -counts))
    return int(vals[int(order[0])])


def annotate_query_candidates(
    *,
    query_rows: List[dict],
    query_emb: np.ndarray,
    docs: np.ndarray,
    centers: np.ndarray,
    chunks: List[np.ndarray],
    overlap_doc_indices_by_cluster: List[np.ndarray] | None,
    doc_index_to_cluster: np.ndarray,
    eval_k: int,
    fixed_k: int,
) -> List[dict]:
    if len(query_rows) != len(query_emb):
        raise RuntimeError("query_rows/query_emb length mismatch in annotate_query_candidates")

    query_emb = normalize_rows(np.asarray(query_emb, dtype=np.float32))
    docs = normalize_rows(np.asarray(docs, dtype=np.float32))
    centers = normalize_rows(np.asarray(centers, dtype=np.float32))

    center_sims = np.clip(query_emb @ centers.T, -1.0, 1.0).astype(np.float32)
    center_dmat = np.arccos(center_sims).astype(np.float32)
    nearest_cluster = np.argmin(center_dmat, axis=1).astype(np.int32)
    nearest_center = center_dmat[np.arange(len(query_emb), dtype=np.int32), nearest_cluster]
    nearest_doc = nearest_doc_angular_dist(query_emb, docs=docs)
    semantic_score = (nearest_center + 0.35 * nearest_doc).astype(np.float32)

    global_sims = np.clip(query_emb @ docs.T, -1.0, 1.0).astype(np.float32)
    topk_local = np.argpartition(-global_sims, kth=int(eval_k) - 1, axis=1)[:, : int(eval_k)]
    topk_scores = np.take_along_axis(global_sims, topk_local, axis=1)
    topk_order = np.argsort(-topk_scores, axis=1)
    gt_topk_idx = np.take_along_axis(topk_local, topk_order, axis=1).astype(np.int32)
    gt_topk_scores = np.take_along_axis(global_sims, gt_topk_idx, axis=1).astype(np.float32)
    gt_topk_clusters = doc_index_to_cluster[gt_topk_idx]
    top1_idx = gt_topk_idx[:, 0].astype(np.int32)
    top1_cluster = doc_index_to_cluster[top1_idx].astype(np.int32)
    single_cluster_exact_consistent = np.all(gt_topk_clusters == nearest_cluster[:, None], axis=1)

    local_cover_top300_ratio = np.zeros(len(query_rows), dtype=np.float32)
    local_cover_top300_full = np.zeros(len(query_rows), dtype=bool)
    local_cover_top500_ratio = np.zeros(len(query_rows), dtype=np.float32)
    local_cover_top500_full = np.zeros(len(query_rows), dtype=bool)
    ideal_track1_rmax = np.zeros(len(query_rows), dtype=np.float32)
    topc_overlap_active = bool(
        overlap_doc_indices_by_cluster is not None and len(overlap_doc_indices_by_cluster) == len(chunks)
    )
    support_indices_by_cluster = (
        [np.asarray(vals, dtype=np.int32).reshape(-1) for vals in overlap_doc_indices_by_cluster]
        if topc_overlap_active
        else [np.asarray(chunk, dtype=np.int32).reshape(-1) for chunk in chunks]
    )
    support_sets = {
        int(cid): set(int(x) for x in np.asarray(vals, dtype=np.int32).tolist())
        for cid, vals in enumerate(support_indices_by_cluster)
    }
    for cid, chunk in enumerate(chunks):
        mask = np.where(nearest_cluster == int(cid))[0].astype(np.int32)
        if len(mask) == 0:
            continue
        support_idx = np.asarray(support_indices_by_cluster[int(cid)], dtype=np.int32).reshape(-1)
        chunk_set = support_sets[int(cid)]

        for row_pos, query_pos in enumerate(mask.tolist()):
            gt_set = set(int(x) for x in gt_topk_idx[int(query_pos)].tolist())
            top_fixed_global, top_fixed_theta, _search_meta = filtered_global_hnsw_topk_doc_indices_and_scores(
                query_for_server=np.asarray(query_emb[int(query_pos)], dtype=np.float32),
                docs=docs,
                allowed_doc_indices=support_idx,
                fixed_k=int(fixed_k),
            )
            top300_set = set(int(x) for x in np.asarray(top_fixed_global, dtype=np.int32).tolist())
            cover300 = float(len(gt_set & top300_set) / max(1, int(eval_k)))
            cover500 = float(len(gt_set & chunk_set) / max(1, int(eval_k)))
            local_cover_top300_ratio[int(query_pos)] = cover300
            local_cover_top500_ratio[int(query_pos)] = cover500
            local_cover_top300_full[int(query_pos)] = bool(gt_set.issubset(top300_set))
            local_cover_top500_full[int(query_pos)] = bool(gt_set.issubset(chunk_set))
            if bool(local_cover_top300_full[int(query_pos)]) and len(top_fixed_theta) > 0:
                gt_worst = float(np.min(gt_topk_scores[int(query_pos)]))
                theta_gt = float(np.arccos(np.clip(gt_worst, -1.0, 1.0)))
                theta_fixed = float(np.max(np.asarray(top_fixed_theta, dtype=np.float64)))
                half_margin = max((theta_fixed - theta_gt) / 2.0, 0.0)
                ideal_track1_rmax[int(query_pos)] = float(np.tan(half_margin))

    items: List[dict] = []
    for idx, row in enumerate(query_rows):
        source_family = str(row.get("query_source_family", "real"))
        items.append(
            {
                "idx": int(idx),
                "query_id": str(row["query_id"]),
                "raw_query_id": str(row.get("raw_query_id", row["query_id"])),
                "source_query_id": (
                    str(row["source_query_id"]) if row.get("source_query_id") is not None else None
                ),
                "text": str(row["text"]),
                "query_source_family": source_family,
                "query_source_detail": str(
                    row.get(
                        "query_source_detail",
                        "real_query_pool" if source_family == "real" else source_family,
                    )
                ),
                "target_cluster_id": (
                    int(row["target_cluster_id"])
                    if row.get("target_cluster_id") is not None
                    else None
                ),
                "source_doc_id": (
                    str(row["source_doc_id"]) if row.get("source_doc_id") is not None else None
                ),
                "source_passage_doc_id": (
                    str(row["source_passage_doc_id"])
                    if row.get("source_passage_doc_id") is not None
                    else None
                ),
                "nearest_cluster_id": int(nearest_cluster[int(idx)]),
                "nearest_center_theta": float(nearest_center[int(idx)]),
                "nearest_doc_theta": float(nearest_doc[int(idx)]),
                "semantic_score": float(semantic_score[int(idx)]),
                "top1_cluster_id": int(top1_cluster[int(idx)]),
                "top5_majority_cluster_id": int(_candidate_majority_cluster(gt_topk_clusters[int(idx)])),
                "single_cluster_exact_consistent": bool(single_cluster_exact_consistent[int(idx)]),
                "local_cover_top300_ratio": float(local_cover_top300_ratio[int(idx)]),
                "local_cover_top300_full": bool(local_cover_top300_full[int(idx)]),
                "local_cover_top500_ratio": float(local_cover_top500_ratio[int(idx)]),
                "local_cover_top500_full": bool(local_cover_top500_full[int(idx)]),
                "ideal_track1_rmax": float(ideal_track1_rmax[int(idx)]),
            }
        )
    return items


def sort_cluster_candidates_for_calibration(items: List[dict]) -> List[dict]:
    return sorted(
        items,
        key=lambda c: (
            0 if bool(c.get("local_cover_top300_full")) else 1,
            0 if bool(c.get("local_cover_top500_full")) else 1,
            0 if str(c.get("query_source_family", "other")) == "real" else 1,
            -float(c.get("ideal_track1_rmax", 0.0)),
            float(c.get("semantic_score", 0.0)),
            str(c.get("query_id", "")),
        ),
    )


def sort_cluster_candidates_for_evaluation(items: List[dict]) -> List[dict]:
    if str(QUERY_EVALUATION_RANKING_POLICY) == "paperfaithful_real_natural":
        # 论文主线默认使用真实 query 自然采样，不再优先挑“主簇即可覆盖 top-k”的 easy query。
        # 这里保持稳定 hash 顺序，避免把边界/跨簇 query 系统性压到后面。
        return sorted(
            items,
            key=lambda c: (
                0 if str(c.get("query_source_family", "other")) == "real" else 1,
                stable_hash_to_unit_interval(f"{int(QUERY_SPLIT_SEED)}::{str(c.get('query_id', ''))}"),
                str(c.get("query_id", "")),
            ),
        )

    return sorted(
        items,
        key=lambda c: (
            0 if str(c.get("query_source_family", "other")) == "real" else 1,
            0 if bool(c.get("local_cover_top500_full")) else 1,
            0 if bool(c.get("single_cluster_exact_consistent")) else 1,
            float(c.get("semantic_score", 0.0)),
            str(c.get("query_id", "")),
        ),
    )


def select_cluster_balanced_query_sets(
    *,
    cluster_to_candidates: Dict[int, List[dict]],
    calibration_target_per_cluster: int,
    evaluation_target_per_cluster: int,
) -> Tuple[List[dict], List[dict], dict]:
    calibration_selected: List[dict] = []
    evaluation_selected: List[dict] = []
    selection_summary = {
        "calibration_target_per_cluster": int(calibration_target_per_cluster),
        "evaluation_target_per_cluster": int(evaluation_target_per_cluster),
        "per_cluster": [],
    }

    for cid in range(int(NUM_CLUSTERS)):
        items = list(cluster_to_candidates.get(int(cid), []))
        if len(items) == 0:
            raise RuntimeError(f"cluster {cid} has no query candidates")

        calibration_ranked = sort_cluster_candidates_for_calibration(items)
        evaluation_ranked = sort_cluster_candidates_for_evaluation(items)
        used_qids = set()
        cal_items: List[dict] = []
        eval_items: List[dict] = []

        for item in calibration_ranked:
            if len(cal_items) >= int(calibration_target_per_cluster):
                break
            qid = str(item["query_id"])
            if qid in used_qids:
                continue
            used_qids.add(qid)
            cal_items.append(item)

        for item in evaluation_ranked:
            if len(eval_items) >= int(evaluation_target_per_cluster):
                break
            qid = str(item["query_id"])
            if qid in used_qids:
                continue
            used_qids.add(qid)
            eval_items.append(item)

        if len(cal_items) < int(calibration_target_per_cluster):
            raise RuntimeError(
                f"cluster {cid} calibration queries insufficient: "
                f"need {calibration_target_per_cluster}, got {len(cal_items)}"
            )
        if len(eval_items) < int(evaluation_target_per_cluster):
            raise RuntimeError(
                f"cluster {cid} evaluation queries insufficient: "
                f"need {evaluation_target_per_cluster}, got {len(eval_items)}"
            )

        calibration_selected.extend(cal_items)
        evaluation_selected.extend(eval_items)
        selection_summary["per_cluster"].append(
            {
                "cluster_id": int(cid),
                "num_candidates_total": int(len(items)),
                "num_real_candidates": int(
                    len([x for x in items if str(x.get("query_source_family")) == "real"])
                ),
                "num_calibration_selected": int(len(cal_items)),
                "num_evaluation_selected": int(len(eval_items)),
                "num_calibration_real": int(
                    len([x for x in cal_items if str(x.get("query_source_family")) == "real"])
                ),
                "num_evaluation_real": int(
                    len([x for x in eval_items if str(x.get("query_source_family")) == "real"])
                ),
                "calibration_positive_rmax_ratio": float(
                    np.mean([float(x.get("ideal_track1_rmax", 0.0)) > 0.0 for x in cal_items])
                ),
                "evaluation_positive_rmax_ratio": float(
                    np.mean([float(x.get("ideal_track1_rmax", 0.0)) > 0.0 for x in eval_items])
                ),
            }
        )

    return calibration_selected, evaluation_selected, selection_summary


def select_fast_query_sets(
    *,
    candidate_items: List[dict],
    calibration_target_total: int,
    evaluation_target_total: int,
) -> Tuple[List[dict], List[dict], dict]:
    items = list(candidate_items)
    if len(items) == 0:
        raise RuntimeError("no query candidates for fast selection")

    effective_eval = int(min(int(evaluation_target_total), len(items)))
    if effective_eval <= 0:
        raise RuntimeError("fast selection has no room for evaluation queries")
    effective_cal = int(min(int(calibration_target_total), max(0, len(items) - effective_eval)))

    calibration_ranked = sort_cluster_candidates_for_calibration(items)
    used_qids = set()
    calibration_selected: List[dict] = []
    for item in calibration_ranked:
        if len(calibration_selected) >= int(effective_cal):
            break
        qid = str(item["query_id"])
        if qid in used_qids:
            continue
        used_qids.add(qid)
        calibration_selected.append(item)

    evaluation_ranked = sort_cluster_candidates_for_evaluation(items)
    evaluation_selected: List[dict] = []
    for item in evaluation_ranked:
        if len(evaluation_selected) >= int(effective_eval):
            break
        qid = str(item["query_id"])
        if qid in used_qids:
            continue
        used_qids.add(qid)
        evaluation_selected.append(item)

    if len(evaluation_selected) < int(effective_eval):
        raise RuntimeError(
            f"fast selection evaluation insufficient: need {effective_eval}, got {len(evaluation_selected)}"
        )

    selection_summary = {
        "mode": "fast_global_selection",
        "calibration_target_total": int(calibration_target_total),
        "evaluation_target_total": int(evaluation_target_total),
        "calibration_selected_total": int(len(calibration_selected)),
        "evaluation_selected_total": int(len(evaluation_selected)),
        "candidate_total": int(len(items)),
        "calibration_cluster_counts": count_items_by_cluster(calibration_selected, "nearest_cluster_id"),
        "evaluation_cluster_counts": count_items_by_cluster(evaluation_selected, "nearest_cluster_id"),
        "calibration_source_counts": count_items_by_source(calibration_selected),
        "evaluation_source_counts": count_items_by_source(evaluation_selected),
    }
    return calibration_selected, evaluation_selected, selection_summary


def _stable_query_priority(item: dict) -> tuple:
    qid = str(item.get("query_id", ""))
    return (
        stable_hash_to_unit_interval(f"{int(QUERY_SPLIT_SEED)}::{qid}"),
        str(qid),
    )


def select_random_real_evaluation_set(
    *,
    real_candidates: List[dict],
    target_total: int,
    require_source_query_id: bool = True,
) -> Tuple[List[dict], dict]:
    real_items = [
        dict(x) for x in real_candidates if str(x.get("query_source_family", "real")) == "real"
    ]
    if len(real_items) == 0:
        raise RuntimeError("no real candidates available for random evaluation selection")

    items = list(real_items)
    if bool(require_source_query_id):
        items = [
            x
            for x in items
            if x.get("source_query_id") is not None and str(x.get("source_query_id")).strip()
        ]
    if len(items) == 0:
        raise RuntimeError(
            "no sourced real candidates available for random evaluation selection"
        )

    items_sorted = sorted(items, key=_stable_query_priority)
    effective_total = int(min(int(target_total), len(items_sorted)))
    if effective_total <= 0:
        raise RuntimeError("random evaluation selection target_total <= 0")

    selected = list(items_sorted[:effective_total])
    summary = {
        "mode": "real_only_stable_random_eval",
        "sampling_policy": str(QUERY_EVAL_SAMPLING_POLICY),
        "selection_seed": int(QUERY_SPLIT_SEED),
        "require_source_query_id": bool(require_source_query_id),
        "candidate_total_real": int(len(real_items)),
        "candidate_total_real_with_source_query_id": int(len(items_sorted)),
        "selected_total": int(len(selected)),
        "selected_cluster_counts": count_items_by_cluster(selected, "nearest_cluster_id"),
        "selected_source_counts": count_items_by_source(selected),
    }
    return selected, summary


def select_calibration_only_fast(
    *,
    candidate_items: List[dict],
    target_total: int,
    forbidden_qids: set[str] | None = None,
) -> Tuple[List[dict], dict]:
    items = list(candidate_items)
    if len(items) == 0 or int(target_total) <= 0:
        return [], {
            "mode": "fast_calibration_only",
            "target_total": int(max(0, int(target_total))),
            "selected_total": 0,
        }

    ranked = sort_cluster_candidates_for_calibration(items)
    forbidden = set() if forbidden_qids is None else set(str(x) for x in forbidden_qids)
    selected: List[dict] = []
    used_qids = set(forbidden)
    for item in ranked:
        if len(selected) >= int(target_total):
            break
        qid = str(item["query_id"])
        if qid in used_qids:
            continue
        used_qids.add(qid)
        selected.append(item)

    summary = {
        "mode": "fast_calibration_only",
        "target_total": int(target_total),
        "selected_total": int(len(selected)),
        "candidate_total": int(len(items)),
        "selected_cluster_counts": count_items_by_cluster(selected, "nearest_cluster_id"),
        "selected_source_counts": count_items_by_source(selected),
    }
    return selected, summary


def select_calibration_balanced_after_eval(
    *,
    cluster_to_candidates: Dict[int, List[dict]],
    target_total: int,
    excluded_qids: set[str] | None = None,
) -> Tuple[List[dict], dict]:
    forbidden = set() if excluded_qids is None else set(str(x) for x in excluded_qids)
    target_total = int(max(0, int(target_total)))
    if target_total <= 0:
        return [], {
            "mode": "cluster_balanced_calibration_after_eval",
            "target_total": 0,
            "selected_total": 0,
            "per_cluster_target": 0,
            "per_cluster": [],
        }

    per_cluster_target = int(max(1, np.ceil(float(target_total) / max(1, int(NUM_CLUSTERS)))))
    selected: List[dict] = []
    used_qids = set(forbidden)
    leftovers: List[dict] = []
    per_cluster = []

    for cid in range(int(NUM_CLUSTERS)):
        items = list(cluster_to_candidates.get(int(cid), []))
        ranked = sort_cluster_candidates_for_calibration(items)
        local_selected = []
        local_leftover = []
        for item in ranked:
            qid = str(item["query_id"])
            if qid in used_qids:
                continue
            if len(local_selected) < int(per_cluster_target):
                used_qids.add(qid)
                local_selected.append(item)
            else:
                local_leftover.append(item)
        selected.extend(local_selected)
        leftovers.extend(local_leftover)
        per_cluster.append(
            {
                "cluster_id": int(cid),
                "num_candidates_total": int(len(items)),
                "num_selected": int(len(local_selected)),
                "num_leftover": int(len(local_leftover)),
                "num_real_selected": int(
                    len([x for x in local_selected if str(x.get("query_source_family")) == "real"])
                ),
            }
        )

    if len(selected) < int(target_total):
        ranked_leftovers = sort_cluster_candidates_for_calibration(leftovers)
        for item in ranked_leftovers:
            if len(selected) >= int(target_total):
                break
            qid = str(item["query_id"])
            if qid in used_qids:
                continue
            used_qids.add(qid)
            selected.append(item)
    elif len(selected) > int(target_total):
        selected = selected[: int(target_total)]

    summary = {
        "mode": "cluster_balanced_calibration_after_eval",
        "target_total": int(target_total),
        "selected_total": int(len(selected)),
        "per_cluster_target": int(per_cluster_target),
        "selected_cluster_counts": count_items_by_cluster(selected, "nearest_cluster_id"),
        "selected_source_counts": count_items_by_source(selected),
        "per_cluster": per_cluster,
    }
    return selected, summary


def count_items_by_cluster(items: List[dict], field: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        key = str(int(item.get(field, -1)))
        out[key] = int(out.get(key, 0)) + 1
    return out


def count_items_by_source(items: List[dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        key = str(item.get("query_source_family", "unknown"))
        out[key] = int(out.get(key, 0)) + 1
    return out


def get_workset_positive_doc_ids_for_query(
    *,
    canonical_query_id: str,
    qrels_map: Dict[str, List[str]],
    source_docid_to_workset_docids: Dict[str, List[str]],
) -> List[str]:
    out: List[str] = []
    seen = set()
    for source_did in qrels_map.get(str(canonical_query_id), []):
        for alias in source_doc_id_aliases(str(source_did)):
            for workset_did in source_docid_to_workset_docids.get(alias, []):
                if workset_did in seen:
                    continue
                seen.add(workset_did)
                out.append(str(workset_did))
    return out


def _load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _swap_workset_name_in_path(path: str, source_workset_name: str) -> str:
    current_workset_name = str(WORKSET_NAME).strip()
    source_workset_name = str(source_workset_name).strip()
    if not current_workset_name:
        raise RuntimeError("WORKSET_NAME is empty; cannot derive shared query bundle paths")
    if not source_workset_name:
        raise RuntimeError("source_workset_name is empty; cannot derive shared query bundle paths")
    if current_workset_name not in str(path):
        raise RuntimeError(
            f"cannot swap workset name in path without current token '{current_workset_name}': {path}"
        )
    return str(path).replace(current_workset_name, source_workset_name)


def _current_query_bundle_signature_fields() -> dict:
    fields = {
        "query_bundle_mode": "independent",
        "query_shared_source_workset_name": "",
        "query_shared_source_eval_jsonl_mtime_ns": -1,
        "query_shared_source_calibration_jsonl_mtime_ns": -1,
        "query_shared_source_split_meta_mtime_ns": -1,
    }
    try:
        workset_meta = _load_json_file(WORKSET_META_PATH)
    except Exception:
        return fields
    if str(workset_meta.get("build_mode", "")).strip().lower() != "nested_master_balanced_round_robin":
        return fields
    nested_master = dict(workset_meta.get("nested_master", {}))
    source_workset_name = str(nested_master.get("source_workset_name", "")).strip()
    if not source_workset_name or source_workset_name == str(WORKSET_NAME):
        return fields
    source_eval_jsonl = _swap_workset_name_in_path(WORKSET_QUERIES_JSONL_PATH, source_workset_name)
    source_calibration_jsonl = _swap_workset_name_in_path(
        WORKSET_CALIBRATION_QUERIES_JSONL_PATH,
        source_workset_name,
    )
    source_split_meta = _swap_workset_name_in_path(WORKSET_QUERY_SPLIT_META_PATH, source_workset_name)
    fields.update(
        {
            "query_bundle_mode": "nested_shared_from_master",
            "query_shared_source_workset_name": str(source_workset_name),
            "query_shared_source_eval_jsonl_mtime_ns": (
                int(os.stat(source_eval_jsonl).st_mtime_ns)
                if os.path.exists(source_eval_jsonl)
                else -1
            ),
            "query_shared_source_calibration_jsonl_mtime_ns": (
                int(os.stat(source_calibration_jsonl).st_mtime_ns)
                if os.path.exists(source_calibration_jsonl)
                else -1
            ),
            "query_shared_source_split_meta_mtime_ns": (
                int(os.stat(source_split_meta).st_mtime_ns)
                if os.path.exists(source_split_meta)
                else -1
            ),
        }
    )
    return fields


def _resolve_shared_query_bundle_spec(*, require_existing: bool) -> Optional[dict]:
    signature = _current_query_bundle_signature_fields()
    if str(signature.get("query_bundle_mode", "independent")) != "nested_shared_from_master":
        return None
    source_workset_name = str(signature.get("query_shared_source_workset_name", "")).strip()
    if not source_workset_name:
        return None
    spec = {
        "mode": "nested_shared_from_master",
        "source_workset_name": str(source_workset_name),
        "evaluation_queries_jsonl_path": _swap_workset_name_in_path(
            WORKSET_QUERIES_JSONL_PATH,
            source_workset_name,
        ),
        "calibration_queries_jsonl_path": _swap_workset_name_in_path(
            WORKSET_CALIBRATION_QUERIES_JSONL_PATH,
            source_workset_name,
        ),
        "source_split_meta_path": _swap_workset_name_in_path(
            WORKSET_QUERY_SPLIT_META_PATH,
            source_workset_name,
        ),
    }
    if bool(require_existing):
        missing = [
            str(path)
            for path in (
                spec["evaluation_queries_jsonl_path"],
                spec["calibration_queries_jsonl_path"],
                spec["source_split_meta_path"],
            )
            if not os.path.exists(path)
        ]
        if missing:
            raise FileNotFoundError(
                "shared master query bundle is not ready for nested child workset: "
                + ", ".join(missing)
            )
    return spec


def _count_queries_with_source_query_id(rows: List[dict]) -> int:
    return int(len([row for row in rows if row.get("source_query_id") is not None]))


def _count_queries_canonicalized_from_source(rows: List[dict]) -> int:
    return int(
        len(
            [
                row
                for row in rows
                if str(row.get("query_id", "")) != str(row.get("raw_query_id", row.get("query_id", "")))
            ]
        )
    )


def _annotate_fixed_query_rows(
    *,
    source_rows: List[dict],
    encoder,
    docs: np.ndarray,
    centers: np.ndarray,
    chunks: List[np.ndarray],
    overlap_doc_indices_by_cluster: List[np.ndarray] | None,
    doc_index_to_cluster: np.ndarray,
    eval_k: int,
    fixed_k: int,
) -> Tuple[List[dict], np.ndarray]:
    normalized_rows: List[dict] = []
    for row in source_rows:
        query_id = str(row.get("query_id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not query_id or not text:
            continue
        source_query_id = row.get("source_query_id")
        selection_target_cluster_id = row.get(
            "selection_target_cluster_id",
            row.get("selection_nearest_cluster_id", row.get("target_cluster_id")),
        )
        normalized_rows.append(
            {
                "query_id": str(query_id),
                "raw_query_id": str(row.get("raw_query_id", query_id)),
                "source_query_id": (
                    str(source_query_id).strip()
                    if source_query_id is not None and str(source_query_id).strip()
                    else None
                ),
                "text": str(text),
                "query_source_family": str(row.get("query_source_family", "real")),
                "query_source_detail": str(
                    row.get(
                        "query_source_detail",
                        row.get("source", row.get("query_source_family", "real")),
                    )
                ),
                "target_cluster_id": (
                    int(selection_target_cluster_id)
                    if selection_target_cluster_id is not None
                    else None
                ),
                "source_doc_id": row.get(
                    "selection_source_doc_id",
                    row.get("source_doc_id"),
                ),
                "source_passage_doc_id": row.get(
                    "selection_source_passage_doc_id",
                    row.get("source_passage_doc_id"),
                ),
            }
        )
    if len(normalized_rows) == 0:
        return [], np.zeros((0, int(docs.shape[1])), dtype=np.float32)
    query_emb = encoder.encode_queries(
        [row["text"] for row in normalized_rows],
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    query_emb = normalize_rows(query_emb)
    items = annotate_query_candidates(
        query_rows=normalized_rows,
        query_emb=query_emb,
        docs=docs,
        centers=centers,
        chunks=chunks,
        overlap_doc_indices_by_cluster=overlap_doc_indices_by_cluster,
        doc_index_to_cluster=doc_index_to_cluster,
        eval_k=int(eval_k),
        fixed_k=int(fixed_k),
    )
    for item in items:
        item["embedding_source"] = "shared_bundle"
        item["embedding_index"] = int(item["idx"])
    return items, query_emb


def current_query_generation_signature(*, cluster_query_cache_key: str) -> dict:
    return {
        "selection_policy": str(QUERY_SELECTION_POLICY),
        "split_protocol_version": str(QUERY_SPLIT_PROTOCOL_VERSION),
        "query_evaluation_track1_only": bool(QUERY_EVALUATION_TRACK1_ONLY),
        "query_fast_mode": bool(QUERY_FAST_MODE),
        "query_use_real_query_pool": bool(QUERY_USE_REAL_QUERY_POOL),
        "query_real_pool_require_workset_positive": bool(QUERY_REAL_POOL_REQUIRE_WORKSET_POSITIVE),
        "query_real_pool_encode_cap": int(QUERY_REAL_POOL_ENCODE_CAP),
        "query_calibration_source_policy": str(QUERY_CALIBRATION_SOURCE_POLICY),
        "query_eval_sampling_policy": str(QUERY_EVAL_SAMPLING_POLICY),
        "query_evaluation_ranking_policy": str(QUERY_EVALUATION_RANKING_POLICY),
        "query_calibration_target_total": int(QUERY_CALIBRATION_TARGET_TOTAL),
        "query_evaluation_target_total": int(QUERY_EVALUATION_TARGET_TOTAL),
        "query_pool_auto_expand": bool(AUTO_EXPAND_FULL_QUERY_POOL),
        "query_pool_target_size": int(TARGET_FULL_QUERY_POOL_SIZE),
        "eval_k": int(EVAL_K),
        "fixed_k": int(FIXED_K),
        "num_clusters": int(NUM_CLUSTERS),
        "target_cluster_size": int(TARGET_CLUSTER_SIZE),
        "cluster_query_cache_key": str(cluster_query_cache_key),
        **_current_query_bundle_signature_fields(),
    }


def main():
    required = [
        WORKSET_DOCS_PATH,
        WORKSET_DOC_IDS_PATH,
        WORKSET_CORPUS_JSONL_PATH,
        FULL_QUERIES_JSONL_PATH,
        FULL_QRELS_TSV_PATH,
        WORKSET_META_PATH,
        WORKSET_CLUSTER_INFO_PATH,
    ]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing required file: {path}")

    docs = normalize_rows(np.load(WORKSET_DOCS_PATH).astype(np.float32))
    doc_ids = np.load(WORKSET_DOC_IDS_PATH, allow_pickle=True)

    with open(WORKSET_CLUSTER_INFO_PATH, "rb") as f:
        cluster_info = pickle.load(f)
    cluster_contract = assert_cluster_info_contract(
        cluster_info,
        expected_eval_k=int(EVAL_K),
        expected_fixed_k=int(FIXED_K),
        expected_num_clusters=int(NUM_CLUSTERS),
        expected_target_cluster_size=int(TARGET_CLUSTER_SIZE),
    )
    print(f"cluster_info contract: {cluster_contract['signature']}")
    cluster_query_cache_key = fingerprint_cluster_query_context(cluster_info)
    centers = normalize_rows(np.asarray(cluster_info["centers"], dtype=np.float32))
    cluster_r_k = np.asarray(cluster_info["cluster_r_k"], dtype=np.float32)
    cluster_r_fixed = np.asarray(cluster_info["cluster_r_fixed"], dtype=np.float32)
    cluster_r_max = np.asarray(cluster_info.get("cluster_r_max", []), dtype=np.float32)
    overlap_doc_indices_by_cluster = [
        np.asarray(vals, dtype=np.int32).reshape(-1)
        for vals in cluster_info.get("cluster_topc_overlap_doc_indices", cluster_info.get("chunks", []))
    ]
    docid_to_parent_cluster = {
        str(did): int(cid)
        for did, cid in cluster_info["docid_to_parent_cluster"].items()
    }

    current_doc_fingerprint = fingerprint_doc_ids(doc_ids)
    current_query_sig = current_query_generation_signature(
        cluster_query_cache_key=cluster_query_cache_key,
    )
    if (not FORCE_REBUILD_QUERY_WORKSET) and os.path.exists(WORKSET_QUERY_META_PATH):
        try:
            meta = json.load(open(WORKSET_QUERY_META_PATH, "r", encoding="utf-8"))
            if (
                meta.get("doc_ids_sha256") == current_doc_fingerprint
                and meta.get("query_generation_signature") == current_query_sig
            ):
                cached_queries = np.load(WORKSET_QUERIES_PATH)
                print("reuse cached query workset, shape=", cached_queries.shape)
                return
        except Exception:
            pass

    workset_corpus_rows = load_jsonl(WORKSET_CORPUS_JSONL_PATH)
    workset_docid_set = set(str(x) for x in doc_ids.tolist())
    source_docid_to_workset_docids: Dict[str, List[str]] = {}
    for row in workset_corpus_rows:
        did = str(row.get("doc_id", "")).strip()
        if not did:
            continue
        source_did = str(row.get("source_doc_id", did))
        for alias in source_doc_id_aliases(source_did):
            source_docid_to_workset_docids.setdefault(alias, []).append(did)

    qrels_map = load_qrels_map(FULL_QRELS_TSV_PATH)
    encoder = E5QueryEncoder(NEW_MODEL_NAME)
    shared_query_bundle_spec = _resolve_shared_query_bundle_spec(require_existing=False)
    selection_policy_effective = str(QUERY_SELECTION_POLICY)
    split_protocol_version_effective = str(QUERY_SPLIT_PROTOCOL_VERSION)
    real_track1_filter_summary = {
        "enabled": False,
        "input_count": 0,
        "track1_count": 0,
        "dropped_count": 0,
        "track_counts": {},
    }
    query_pool_expand_info = {
        "expanded": False,
        "reason": "disabled",
        "target_size": int(TARGET_FULL_QUERY_POOL_SIZE),
    }
    all_query_rows: List[dict] = []
    num_all_queries_with_source_query_id = 0
    num_all_queries_canonicalized_from_source = 0
    real_query_emb = np.zeros((0, int(docs.shape[1])), dtype=np.float32)
    real_candidates: List[dict] = []
    final_candidates: List[dict] = []
    calibration_candidates: List[dict] = []
    query_emb = np.zeros((0, int(docs.shape[1])), dtype=np.float32)
    calibration_query_emb = np.zeros((0, int(docs.shape[1])), dtype=np.float32)
    balanced_selection_summary: dict = {}

    doc_index_to_cluster = np.full(len(doc_ids), -1, dtype=np.int32)
    for cid, chunk in enumerate(cluster_info["chunks"]):
        chunk_idx = np.asarray(chunk, dtype=np.int32)
        doc_index_to_cluster[chunk_idx] = int(cid)

    if shared_query_bundle_spec is not None:
        shared_query_bundle_spec = _resolve_shared_query_bundle_spec(require_existing=True)
        shared_eval_rows = load_jsonl(shared_query_bundle_spec["evaluation_queries_jsonl_path"])
        shared_calibration_rows = load_jsonl(shared_query_bundle_spec["calibration_queries_jsonl_path"])
        final_candidates, query_emb = _annotate_fixed_query_rows(
            source_rows=shared_eval_rows,
            encoder=encoder,
            docs=docs,
            centers=centers,
            chunks=list(cluster_info["chunks"]),
            overlap_doc_indices_by_cluster=overlap_doc_indices_by_cluster,
            doc_index_to_cluster=doc_index_to_cluster,
            eval_k=int(EVAL_K),
            fixed_k=int(FIXED_K),
        )
        calibration_candidates, calibration_query_emb = _annotate_fixed_query_rows(
            source_rows=shared_calibration_rows,
            encoder=encoder,
            docs=docs,
            centers=centers,
            chunks=list(cluster_info["chunks"]),
            overlap_doc_indices_by_cluster=overlap_doc_indices_by_cluster,
            doc_index_to_cluster=doc_index_to_cluster,
            eval_k=int(EVAL_K),
            fixed_k=int(FIXED_K),
        )
        all_query_rows = list(shared_eval_rows) + list(shared_calibration_rows)
        num_all_queries_with_source_query_id = _count_queries_with_source_query_id(all_query_rows)
        num_all_queries_canonicalized_from_source = _count_queries_canonicalized_from_source(
            all_query_rows
        )
        real_candidates = list(final_candidates) + list(calibration_candidates)
        selection_policy_effective = "nested_shared_query_bundle_from_master"
        split_protocol_version_effective = "nested_shared_query_bundle_v1"
        query_pool_expand_info = {
            "expanded": False,
            "reason": "nested_shared_query_bundle",
            "target_size": int(TARGET_FULL_QUERY_POOL_SIZE),
            "source_workset_name": str(shared_query_bundle_spec["source_workset_name"]),
        }
        balanced_selection_summary = {
            "mode": "nested_shared_query_bundle",
            "source_workset_name": str(shared_query_bundle_spec["source_workset_name"]),
            "source_split_meta_path": str(shared_query_bundle_spec["source_split_meta_path"]),
            "num_source_evaluation_rows": int(len(shared_eval_rows)),
            "num_source_calibration_rows": int(len(shared_calibration_rows)),
            "num_materialized_evaluation_rows": int(len(final_candidates)),
            "num_materialized_calibration_rows": int(len(calibration_candidates)),
        }
        print(
            "shared query bundle reuse:",
            {
                "source_workset_name": str(shared_query_bundle_spec["source_workset_name"]),
                "evaluation_rows": int(len(shared_eval_rows)),
                "calibration_rows": int(len(shared_calibration_rows)),
                "materialized_eval": int(len(final_candidates)),
                "materialized_calibration": int(len(calibration_candidates)),
            },
        )
    else:
        if bool(AUTO_EXPAND_FULL_QUERY_POOL):
            query_pool_expand_info = ensure_full_query_pool_size(
                full_queries_jsonl_path=FULL_QUERIES_JSONL_PATH,
                qrels_tsv_path=FULL_QRELS_TSV_PATH,
                target_size=int(TARGET_FULL_QUERY_POOL_SIZE),
                source_url=MSMARCO_QUERIES_DEV_SMALL_URL,
                cache_tsv_path=MSMARCO_QUERIES_DEV_SMALL_CACHE,
                seed=int(QUERY_POOL_EXPAND_SEED),
            )
            print(
                "query pool expansion: "
                f"expanded={query_pool_expand_info.get('expanded')}, "
                f"before={query_pool_expand_info.get('existing_size_before')}, "
                f"after={query_pool_expand_info.get('existing_size_after')}, "
                f"added={query_pool_expand_info.get('added')}"
            )

        query_rows_raw = load_jsonl(FULL_QUERIES_JSONL_PATH)
        seen_canonical_qids = set()
        for row in query_rows_raw:
            try:
                raw_qid, qtext, source_query_id = parse_query_row(row)
                qid = canonical_query_id(raw_query_id=raw_qid, source_query_id=source_query_id)
                if qid in seen_canonical_qids:
                    continue
                seen_canonical_qids.add(qid)
                all_query_rows.append(
                    {
                        "query_id": qid,
                        "raw_query_id": raw_qid,
                        "text": qtext,
                        "source_query_id": source_query_id,
                    }
                )
            except Exception:
                continue

        if len(all_query_rows) == 0:
            raise RuntimeError("no valid query rows in FULL_QUERIES_JSONL_PATH")

        if bool(QUERY_REAL_POOL_REQUIRE_WORKSET_POSITIVE) or int(QUERY_REAL_POOL_ENCODE_CAP) > 0:
            filtered_query_rows = []
            for row in all_query_rows:
                positive_doc_ids = get_workset_positive_doc_ids_for_query(
                    canonical_query_id=str(row["query_id"]),
                    qrels_map=qrels_map,
                    source_docid_to_workset_docids=source_docid_to_workset_docids,
                )
                row["precheck_positive_doc_ids_in_workset"] = list(positive_doc_ids)
                row["precheck_num_positive_docs_in_workset"] = int(len(positive_doc_ids))
                if bool(QUERY_REAL_POOL_REQUIRE_WORKSET_POSITIVE) and len(positive_doc_ids) == 0:
                    continue
                filtered_query_rows.append(row)
            all_query_rows = filtered_query_rows

        if int(QUERY_REAL_POOL_ENCODE_CAP) > 0 and len(all_query_rows) > int(QUERY_REAL_POOL_ENCODE_CAP):
            all_query_rows = sorted(
                all_query_rows,
                key=lambda r: (
                    -int(r.get("precheck_num_positive_docs_in_workset", 0)),
                    stable_hash_to_unit_interval(str(r["query_id"])),
                    str(r["query_id"]),
                ),
            )[: int(QUERY_REAL_POOL_ENCODE_CAP)]

        if len(all_query_rows) == 0:
            raise RuntimeError("no real query rows left after fast prefiltering")

        num_all_queries_with_source_query_id = _count_queries_with_source_query_id(all_query_rows)
        num_all_queries_canonicalized_from_source = _count_queries_canonicalized_from_source(
            all_query_rows
        )

        if bool(QUERY_USE_REAL_QUERY_POOL) and len(all_query_rows) > 0:
            real_query_texts = [row["text"] for row in all_query_rows]
            real_query_emb = encoder.encode_queries(
                real_query_texts,
                batch_size=BATCH_SIZE,
                max_length=MAX_LENGTH,
            )
            real_query_emb = normalize_rows(real_query_emb)
            real_candidates = annotate_query_candidates(
                query_rows=all_query_rows,
                query_emb=real_query_emb,
                docs=docs,
                centers=centers,
                chunks=list(cluster_info["chunks"]),
                overlap_doc_indices_by_cluster=overlap_doc_indices_by_cluster,
                doc_index_to_cluster=doc_index_to_cluster,
                eval_k=int(EVAL_K),
                fixed_k=int(FIXED_K),
            )
            for item in real_candidates:
                item["embedding_source"] = "real"
                item["embedding_index"] = int(item["idx"])
            if bool(QUERY_EVALUATION_TRACK1_ONLY):
                real_candidates, real_track1_filter_summary = _filter_track1_candidates_by_embedding_index(
                    candidates=real_candidates,
                    query_emb=real_query_emb,
                    centers=centers,
                    cluster_r_k=cluster_r_k,
                    cluster_r_fixed=cluster_r_fixed,
                    cluster_r_max=cluster_r_max,
                )
            else:
                real_track1_filter_summary = {
                    "enabled": False,
                    "input_count": int(len(real_candidates)),
                    "track1_count": int(len(real_candidates)),
                    "dropped_count": 0,
                    "track_counts": {},
                }

        if bool(PIPELINE_IS_PAPERFAITHFUL_MAINLINE):
            cluster_to_candidates: Dict[int, List[dict]] = {int(cid): [] for cid in range(int(NUM_CLUSTERS))}
            for item in real_candidates:
                cluster_to_candidates[int(item["nearest_cluster_id"])].append(item)

            for cid in range(int(NUM_CLUSTERS)):
                items = list(cluster_to_candidates.get(int(cid), []))
                if len(items) == 0:
                    cluster_to_candidates[int(cid)] = []
                    continue
                difficulty_buckets = assign_difficulty_bucket(
                    np.asarray([float(x["semantic_score"]) for x in items], dtype=np.float32)
                )
                for item, bucket in zip(items, difficulty_buckets):
                    item["difficulty_bucket"] = str(bucket)
                cluster_to_candidates[int(cid)] = items
            print(
                "query candidate counts by cluster:",
                {
                    int(cid): {
                        "total": int(len(cluster_to_candidates[int(cid)])),
                        "real": int(
                            len(
                                [
                                    x
                                    for x in cluster_to_candidates[int(cid)]
                                    if str(x.get("query_source_family")) == "real"
                                ]
                            )
                        ),
                    }
                    for cid in range(int(NUM_CLUSTERS))
                },
            )

            if bool(QUERY_FAST_MODE):
                final_candidates, eval_selection_summary = select_random_real_evaluation_set(
                    real_candidates=list(real_candidates),
                    target_total=int(TARGET_NUM_QUERIES),
                    require_source_query_id=True,
                )
                excluded_eval_qids = {str(x["query_id"]) for x in final_candidates}
                calibration_cluster_to_candidates: Dict[int, List[dict]] = {
                    int(cid): [] for cid in range(int(NUM_CLUSTERS))
                }
                calibration_real_only = str(QUERY_CALIBRATION_SOURCE_POLICY) == "real_only"
                for cid in range(int(NUM_CLUSTERS)):
                    for item in cluster_to_candidates.get(int(cid), []):
                        if str(item["query_id"]) in excluded_eval_qids:
                            continue
                        if calibration_real_only and str(item.get("query_source_family", "")) != "real":
                            continue
                        calibration_cluster_to_candidates[int(cid)].append(item)
                calibration_candidates, calibration_selection_summary = select_calibration_balanced_after_eval(
                    cluster_to_candidates=calibration_cluster_to_candidates,
                    target_total=int(QUERY_CALIBRATION_TARGET_TOTAL),
                    excluded_qids=excluded_eval_qids,
                )
                balanced_selection_summary = {
                    "mode": "fast_random_real_eval_real_only_calibration",
                    "evaluation": dict(eval_selection_summary),
                    "calibration": dict(calibration_selection_summary),
                }
            else:
                calibration_candidates, final_candidates, balanced_selection_summary = (
                    select_cluster_balanced_query_sets(
                        cluster_to_candidates=cluster_to_candidates,
                        calibration_target_per_cluster=int(QUERY_CALIBRATION_TARGET_PER_CLUSTER),
                        evaluation_target_per_cluster=int(QUERY_EVALUATION_TARGET_PER_CLUSTER),
                    )
                )
        else:
            query_candidates = list(real_candidates)
            difficulty_buckets = assign_difficulty_bucket(
                np.asarray([float(c["semantic_score"]) for c in query_candidates], dtype=np.float32)
            )
            for item, bucket in zip(query_candidates, difficulty_buckets):
                item["difficulty_bucket"] = str(bucket)

            cluster_to_candidates = {}
            for item in query_candidates:
                cid = int(item["nearest_cluster_id"])
                cluster_to_candidates.setdefault(cid, []).append(item)
            for cid in list(cluster_to_candidates.keys()):
                cluster_to_candidates[cid] = reorder_candidates_for_difficulty_balance(
                    cluster_to_candidates[cid]
                )

            target_num_queries = (
                int(len(query_candidates))
                if int(TARGET_NUM_QUERIES) <= 0
                else int(min(int(TARGET_NUM_QUERIES), len(query_candidates)))
            )
            if target_num_queries <= 0:
                raise RuntimeError("no query selected into current query workset")

            cluster_sizes = {int(cid): int(len(items)) for cid, items in cluster_to_candidates.items()}
            cluster_quotas = allocate_balanced_cluster_quotas(
                cluster_to_size=cluster_sizes,
                target_total=int(target_num_queries),
            )
            final_candidates = []
            selected_query_ids = set()
            for cid in sorted(cluster_to_candidates.keys()):
                quota = int(cluster_quotas.get(int(cid), 0))
                if quota <= 0:
                    continue
                for item in cluster_to_candidates[cid][:quota]:
                    qid = str(item["query_id"])
                    if qid in selected_query_ids:
                        continue
                    selected_query_ids.add(qid)
                    final_candidates.append(item)
            calibration_candidates = []
            balanced_selection_summary = {
                "fallback_mode": "legacy_cluster_balanced_difficulty_stratified",
                "selection_cluster_quotas": {str(k): int(v) for k, v in cluster_quotas.items()},
            }

        if len(final_candidates) == 0:
            raise RuntimeError("no query selected into current query workset")

        def _materialize_query_emb(items: List[dict]) -> np.ndarray:
            emb_rows = []
            for item in items:
                emb_idx = int(item.get("embedding_index", item["idx"]))
                emb_rows.append(real_query_emb[emb_idx])
            if len(emb_rows) == 0:
                return np.zeros((0, int(docs.shape[1])), dtype=np.float32)
            return normalize_rows(np.asarray(emb_rows, dtype=np.float32))

        calibration_query_emb = _materialize_query_emb(calibration_candidates)
        query_emb = _materialize_query_emb(final_candidates)

    if bool(QUERY_EVALUATION_TRACK1_ONLY):
        final_candidates, query_emb, selected_track1_filter_summary = _filter_track1_materialized_queries(
            items=final_candidates,
            query_emb=query_emb,
            centers=centers,
            cluster_r_k=cluster_r_k,
            cluster_r_fixed=cluster_r_fixed,
            cluster_r_max=cluster_r_max,
        )
        if int(QUERY_EVALUATION_TARGET_TOTAL) > 0 and len(final_candidates) != int(
            QUERY_EVALUATION_TARGET_TOTAL
        ):
            raise RuntimeError(
                "track1-only evaluation query selection failed to preserve the fixed query count: "
                f"selected={len(final_candidates)} vs target={int(QUERY_EVALUATION_TARGET_TOTAL)}. "
                "Please rebuild the experiment query bundle with a larger eligible pool."
            )
    else:
        selected_track1_filter_summary = {
            "enabled": False,
            "input_count": int(len(final_candidates)),
            "track1_count": int(len(final_candidates)),
            "dropped_count": 0,
            "track_counts": {},
        }

    if len(final_candidates) == 0:
        raise RuntimeError("no query selected into current query workset")

    calibration_query_ids = np.asarray([str(x["query_id"]) for x in calibration_candidates], dtype=object)
    query_ids = np.asarray([str(x["query_id"]) for x in final_candidates], dtype=object)

    calibration_output_rows = []
    for rank, item in enumerate(calibration_candidates, start=1):
        calibration_output_rows.append(
            {
                "query_id": str(item["query_id"]),
                "raw_query_id": str(item.get("raw_query_id", item["query_id"])),
                "source_query_id": (
                    str(item["source_query_id"]) if item.get("source_query_id") is not None else None
                ),
                "text": str(item["text"]),
                "query_id_namespace": "canonical_source_query_id_or_raw_query_id",
                "split_role": "calibration",
                "split_rank": int(rank),
                "split_policy": split_protocol_version_effective,
                "query_source_family": str(item.get("query_source_family", "real")),
                "query_source_detail": str(item.get("query_source_detail", "")),
                "selection_target_cluster_id": int(item["nearest_cluster_id"]),
                "selection_local_cover_top300_full": bool(item.get("local_cover_top300_full", False)),
                "selection_ideal_track1_rmax": float(item.get("ideal_track1_rmax", 0.0)),
            }
        )
    save_jsonl(WORKSET_CALIBRATION_QUERIES_JSONL_PATH, calibration_output_rows)

    num_eval_queries_with_source_query_id = int(
        len([r for r in final_candidates if r.get("source_query_id") is not None])
    )
    num_eval_queries_canonicalized_from_source = int(
        len(
            [
                r
                for r in final_candidates
                if str(r["query_id"]) != str(r.get("raw_query_id", r["query_id"]))
            ]
        )
    )
    num_calibration_queries_with_source_query_id = int(
        len([r for r in calibration_candidates if r.get("source_query_id") is not None])
    )
    num_calibration_queries_canonicalized_from_source = int(
        len(
            [
                r
                for r in calibration_candidates
                if str(r["query_id"]) != str(r.get("raw_query_id", r["query_id"]))
            ]
        )
    )

    split_meta = {
        "protocol_version": split_protocol_version_effective,
        "split_seed": int(QUERY_SPLIT_SEED),
        "num_queries_total_candidate_pool": int(len(real_candidates)),
        "num_queries_real_candidate_pool": int(len(real_candidates)),
        "num_queries_calibration": int(len(calibration_candidates)),
        "num_queries_evaluation": int(len(final_candidates)),
        "split_overlap_count": 0,
        "calibration_cluster_counts": count_items_by_cluster(calibration_candidates, "nearest_cluster_id"),
        "evaluation_cluster_counts": count_items_by_cluster(final_candidates, "nearest_cluster_id"),
        "calibration_source_counts": count_items_by_source(calibration_candidates),
        "evaluation_source_counts": count_items_by_source(final_candidates),
        "selection_summary": balanced_selection_summary,
    }
    save_json(
        WORKSET_QUERY_SPLIT_META_PATH,
        {
            **split_meta,
            "full_queries_jsonl_path": FULL_QUERIES_JSONL_PATH,
            "calibration_queries_jsonl_path": WORKSET_CALIBRATION_QUERIES_JSONL_PATH,
            "evaluation_queries_jsonl_path": WORKSET_QUERIES_JSONL_PATH,
            "num_all_queries_with_source_query_id": int(num_all_queries_with_source_query_id),
            "num_all_queries_canonicalized_from_source": int(num_all_queries_canonicalized_from_source),
            "query_fast_mode": bool(QUERY_FAST_MODE),
            "query_use_real_query_pool": bool(QUERY_USE_REAL_QUERY_POOL),
            "query_real_pool_require_workset_positive": bool(QUERY_REAL_POOL_REQUIRE_WORKSET_POSITIVE),
            "query_real_pool_encode_cap": int(QUERY_REAL_POOL_ENCODE_CAP),
            "query_calibration_source_policy": str(QUERY_CALIBRATION_SOURCE_POLICY),
            "query_eval_sampling_policy": str(QUERY_EVAL_SAMPLING_POLICY),
            "query_evaluation_ranking_policy": str(QUERY_EVALUATION_RANKING_POLICY),
            "query_evaluation_track1_only": bool(QUERY_EVALUATION_TRACK1_ONLY),
            "real_track1_filter_summary": {
                **dict(real_track1_filter_summary),
                "enabled": bool(QUERY_EVALUATION_TRACK1_ONLY),
            },
            "selected_track1_filter_summary": {
                **dict(selected_track1_filter_summary),
                "enabled": bool(QUERY_EVALUATION_TRACK1_ONLY),
            },
            "query_pool_auto_expand": bool(AUTO_EXPAND_FULL_QUERY_POOL),
            "query_pool_target_size": int(TARGET_FULL_QUERY_POOL_SIZE),
            "query_pool_expand_seed": int(QUERY_POOL_EXPAND_SEED),
            "query_pool_expand_info": query_pool_expand_info,
        },
    )

    qid_alias_to_query_id: Dict[str, str] = {}
    for item in final_candidates:
        qid = str(item["query_id"])
        qid_alias_to_query_id.setdefault(qid, qid)
        raw_query_id = str(item.get("raw_query_id", "")).strip()
        if raw_query_id:
            qid_alias_to_query_id.setdefault(raw_query_id, qid)
        source_query_id = item.get("source_query_id")
        if source_query_id is not None:
            source_query_id = str(source_query_id).strip()
            if source_query_id:
                qid_alias_to_query_id.setdefault(source_query_id, qid)

    strict_overlap_by_qid = {}
    matched_qrels_rows = []
    for qid_in_qrels, pos_doc_ids in qrels_map.items():
        qid = qid_alias_to_query_id.get(str(qid_in_qrels))
        if qid is None:
            continue
        kept_doc_ids = []
        for source_did in pos_doc_ids:
            hit = []
            for alias in source_doc_id_aliases(source_did):
                hit.extend(source_docid_to_workset_docids.get(alias, []))
            kept_doc_ids.extend(hit)
        kept_doc_ids = sorted(set(did for did in kept_doc_ids if did in workset_docid_set))
        if not kept_doc_ids:
            continue

        parent_clusters = sorted({docid_to_parent_cluster[did] for did in kept_doc_ids})
        strict_overlap_by_qid[qid] = {
            "num_positive_docs_in_workset": int(len(kept_doc_ids)),
            "positive_doc_ids_in_workset": kept_doc_ids,
            "parent_clusters": parent_clusters,
        }
        for did in kept_doc_ids:
            matched_qrels_rows.append({"query_id": qid, "doc_id": did})
    matched_qrels_rows.sort(key=lambda r: (str(r["query_id"]), str(r["doc_id"])))

    final_rows = []
    for rank, item in enumerate(final_candidates, start=1):
        strict_info = strict_overlap_by_qid.get(str(item["query_id"]))
        if strict_info is None:
            strict_info = {
                "num_positive_docs_in_workset": 0,
                "positive_doc_ids_in_workset": [],
                "parent_clusters": [],
            }
        final_rows.append(
            {
                "query_id": str(item["query_id"]),
                "raw_query_id": str(item.get("raw_query_id", item["query_id"])),
                "source_query_id": (
                    str(item["source_query_id"])
                    if item.get("source_query_id") is not None
                    else None
                ),
                "text": str(item["text"]),
                "source": str(item.get("query_source_detail", item.get("query_source_family", "unknown"))),
                "query_source_family": str(item.get("query_source_family", "unknown")),
                "query_id_namespace": "canonical_source_query_id_or_raw_query_id",
                "num_positive_docs_in_workset": int(strict_info["num_positive_docs_in_workset"]),
                "positive_doc_ids_in_workset": list(strict_info["positive_doc_ids_in_workset"]),
                "parent_clusters": list(strict_info["parent_clusters"]),
                "is_anchor_query": bool(str(item["query_id"]) in strict_overlap_by_qid),
                "query_embedding_preprocess": QUERY_EMBEDDING_PREPROCESS,
                "selection_rank": int(rank),
                "selection_policy": selection_policy_effective,
                "selection_nearest_cluster_id": int(item["nearest_cluster_id"]),
                "selection_target_cluster_id": int(item["nearest_cluster_id"]),
                "selection_top1_cluster_id": int(item["top1_cluster_id"]),
                "selection_top5_majority_cluster_id": int(item["top5_majority_cluster_id"]),
                "selection_difficulty_bucket": str(item.get("difficulty_bucket", "unknown")),
                "selection_semantic_score": float(item["semantic_score"]),
                "selection_nearest_center_theta": float(item["nearest_center_theta"]),
                "selection_nearest_doc_theta": float(item["nearest_doc_theta"]),
                "selection_single_cluster_exact_consistent": bool(
                    item["single_cluster_exact_consistent"]
                ),
                "selection_local_cover_top300_ratio": float(item.get("local_cover_top300_ratio", 0.0)),
                "selection_local_cover_top300_full": bool(item.get("local_cover_top300_full", False)),
                "selection_local_cover_top500_ratio": float(item.get("local_cover_top500_ratio", 0.0)),
                "selection_local_cover_top500_full": bool(item.get("local_cover_top500_full", False)),
                "selection_ideal_track1_rmax": float(item.get("ideal_track1_rmax", 0.0)),
                "selection_source_doc_id": item.get("source_doc_id"),
                "selection_source_passage_doc_id": item.get("source_passage_doc_id"),
            }
        )

    target_num_queries = int(len(final_candidates))
    strict_rows_selected = [r for r in matched_qrels_rows if str(r["query_id"]) in set(query_ids.tolist())]
    if len(strict_rows_selected) == 0:
        print(
            "[warn] strict qrels overlap is empty for selected query workset; "
            "strict metrics may be uninformative."
        )

    gt_topk = []
    relaxed_qrels_rows = []
    for q in query_emb:
        gt_topk.append(topk_indices_by_cosine(q, docs, EVAL_K))
        relaxed_qrels_rows.append(relaxed_positive_indices_by_cosine(q, docs, EVAL_K))
    gt_topk = np.asarray(gt_topk, dtype=np.int32)

    np.save(WORKSET_QUERIES_PATH, query_emb)
    np.save(WORKSET_QUERY_IDS_PATH, query_ids)
    np.save(WORKSET_GT_TOPK_PATH, gt_topk)
    np.save(WORKSET_CALIBRATION_QUERIES_PATH, calibration_query_emb.astype(np.float32))
    np.save(WORKSET_CALIBRATION_QUERY_IDS_PATH, calibration_query_ids)
    save_jsonl(WORKSET_QUERIES_JSONL_PATH, final_rows)

    os.makedirs(os.path.dirname(WORKSET_QRELS_PATH), exist_ok=True)
    with open(WORKSET_QRELS_PATH, "w", encoding="utf-8") as f:
        f.write("query_id\tdoc_id\n")
        for row in strict_rows_selected:
            f.write(f"{row['query_id']}\t{row['doc_id']}\n")

    with open(WORKSET_RELAXED_QRELS_PATH, "w", encoding="utf-8") as f:
        f.write("query_id\tdoc_id\n")
        for qid, relaxed_idx in zip(query_ids.tolist(), relaxed_qrels_rows):
            for idx in relaxed_idx.tolist():
                f.write(f"{qid}\t{str(doc_ids[int(idx)])}\n")

    selected_cluster_counts: Dict[str, int] = {}
    selected_difficulty_counts: Dict[str, int] = {}
    for row in final_rows:
        cid_key = str(int(row.get("selection_nearest_cluster_id", -1)))
        selected_cluster_counts[cid_key] = int(selected_cluster_counts.get(cid_key, 0)) + 1
        dkey = str(row.get("selection_difficulty_bucket", "unknown"))
        selected_difficulty_counts[dkey] = int(selected_difficulty_counts.get(dkey, 0)) + 1
    selected_source_counts = count_items_by_source(final_candidates)
    calibration_source_counts = count_items_by_source(calibration_candidates)
    calibration_cluster_counts = count_items_by_cluster(calibration_candidates, "nearest_cluster_id")
    if str(selection_policy_effective) == "nested_shared_query_bundle_from_master":
        selection_cluster_quotas = dict(selected_cluster_counts)
    else:
        selection_cluster_quotas = (
            (
                {str(cid): int(QUERY_EVALUATION_TARGET_PER_CLUSTER) for cid in range(int(NUM_CLUSTERS))}
                if not bool(QUERY_FAST_MODE)
                else dict(
                    balanced_selection_summary.get("evaluation", {}).get("selected_cluster_counts", {})
                    if isinstance(balanced_selection_summary.get("evaluation", {}), dict)
                    else {}
                )
            )
            if bool(PIPELINE_IS_PAPERFAITHFUL_MAINLINE)
            else dict(balanced_selection_summary.get("selection_cluster_quotas", {}))
        )
    strict_selected_count = int(len([r for r in final_rows if bool(r.get("is_anchor_query"))]))
    single_cluster_exact_consistent_selected_count = int(
        len([r for r in final_rows if bool(r.get("selection_single_cluster_exact_consistent"))])
    )

    save_json(
        WORKSET_QUERY_META_PATH,
        {
            "doc_ids_sha256": current_doc_fingerprint,
            "cluster_query_cache_key": str(cluster_query_cache_key),
            "query_generation_signature": current_query_sig,
            "num_queries": int(len(query_emb)),
            "query_ids": [str(x) for x in query_ids.tolist()],
            "query_id_namespace": "canonical_source_query_id_or_raw_query_id",
            "raw_query_ids": [str(r.get("raw_query_id", r["query_id"])) for r in final_rows],
            "source_query_ids": [
                (str(r["source_query_id"]) if r.get("source_query_id") is not None else None)
                for r in final_rows
            ],
            "query_embedding_preprocess": QUERY_EMBEDDING_PREPROCESS,
            "query_adapter_used": False,
            "query_split_meta_path": WORKSET_QUERY_SPLIT_META_PATH,
            "calibration_queries_jsonl_path": WORKSET_CALIBRATION_QUERIES_JSONL_PATH,
            "calibration_queries_emb_path": WORKSET_CALIBRATION_QUERIES_PATH,
            "calibration_query_ids_path": WORKSET_CALIBRATION_QUERY_IDS_PATH,
            "evaluation_queries_jsonl_path": WORKSET_QUERIES_JSONL_PATH,
            "calibration_eval_disjoint": True,
            "query_pool_auto_expand": bool(AUTO_EXPAND_FULL_QUERY_POOL),
            "query_pool_target_size": int(TARGET_FULL_QUERY_POOL_SIZE),
            "query_pool_expand_seed": int(QUERY_POOL_EXPAND_SEED),
            "query_pool_expand_info": query_pool_expand_info,
            "query_fast_mode": bool(QUERY_FAST_MODE),
            "query_use_real_query_pool": bool(QUERY_USE_REAL_QUERY_POOL),
            "query_real_pool_require_workset_positive": bool(QUERY_REAL_POOL_REQUIRE_WORKSET_POSITIVE),
            "query_real_pool_encode_cap": int(QUERY_REAL_POOL_ENCODE_CAP),
            "query_calibration_source_policy": str(QUERY_CALIBRATION_SOURCE_POLICY),
            "query_eval_sampling_policy": str(QUERY_EVAL_SAMPLING_POLICY),
            "query_evaluation_ranking_policy": str(QUERY_EVALUATION_RANKING_POLICY),
            "num_all_queries_with_source_query_id": int(num_all_queries_with_source_query_id),
            "num_all_queries_canonicalized_from_source": int(num_all_queries_canonicalized_from_source),
            "num_real_query_candidates": int(len(real_candidates)),
            "num_calibration_pool_queries": int(len(calibration_candidates)),
            "num_calibration_pool_queries_embedded": int(len(calibration_query_ids)),
            "num_calibration_pool_queries_with_source_query_id": int(
                num_calibration_queries_with_source_query_id
            ),
            "num_calibration_pool_queries_canonicalized_from_source": int(
                num_calibration_queries_canonicalized_from_source
            ),
            "calibration_pool_cluster_counts": calibration_cluster_counts,
            "calibration_pool_source_counts": calibration_source_counts,
            "num_evaluation_pool_queries": int(len(final_candidates)),
            "num_evaluation_pool_queries_with_source_query_id": int(
                num_eval_queries_with_source_query_id
            ),
            "num_evaluation_pool_queries_canonicalized_from_source": int(
                num_eval_queries_canonicalized_from_source
            ),
            "evaluation_pool_cluster_counts": selected_cluster_counts,
            "evaluation_pool_source_counts": selected_source_counts,
            "num_selected_queries_with_source_query_id": int(
                len([r for r in final_rows if r.get("source_query_id") is not None])
            ),
            "num_selected_queries_canonicalized_from_source": int(
                len([r for r in final_rows if str(r["query_id"]) != str(r.get("raw_query_id", r["query_id"]))])
            ),
            "strict_mode": bool(len(final_rows) > 0 and all(bool(r.get("is_anchor_query")) for r in final_rows)),
            "selection_target_num_queries_config": int(TARGET_NUM_QUERIES),
            "selection_effective_target_num_queries": int(target_num_queries),
            "selection_policy": selection_policy_effective,
            "selection_cluster_quotas": selection_cluster_quotas,
            "selection_cluster_counts_selected": selected_cluster_counts,
            "selection_difficulty_counts_selected": selected_difficulty_counts,
            "selection_source_counts_selected": selected_source_counts,
            "selection_summary": balanced_selection_summary,
            "strict_overlap_query_count_total_all_candidates": int(strict_selected_count),
            "strict_overlap_query_count_total": int(strict_selected_count),
            "strict_overlap_query_count_selected": int(strict_selected_count),
            "strict_qrels_pairs_selected": int(len(strict_rows_selected)),
            "semantic_fillup_query_count_selected": int(
                len([r for r in final_rows if not bool(r.get("is_anchor_query"))])
            ),
            "single_cluster_exact_consistent_query_count_selected": int(
                single_cluster_exact_consistent_selected_count
            ),
            "avg_selection_nearest_center_theta": float(
                np.mean([r["selection_nearest_center_theta"] for r in final_rows])
            ),
            "avg_selection_nearest_doc_theta": float(
                np.mean([r["selection_nearest_doc_theta"] for r in final_rows])
            ),
            "strict_qrels_path": WORKSET_QRELS_PATH,
            "relaxed_qrels_path": WORKSET_RELAXED_QRELS_PATH,
            "query_source_policy": selection_policy_effective,
        },
    )

    print("query workset generated (paperfaithful mainline)")
    print(
        "query generation mode:",
        {
            "fast_mode": bool(QUERY_FAST_MODE),
            "use_real_query_pool": bool(QUERY_USE_REAL_QUERY_POOL),
            "real_pool_require_workset_positive": bool(QUERY_REAL_POOL_REQUIRE_WORKSET_POSITIVE),
            "real_pool_encode_cap": int(QUERY_REAL_POOL_ENCODE_CAP),
            "calibration_source_policy": str(QUERY_CALIBRATION_SOURCE_POLICY),
            "eval_sampling_policy": str(QUERY_EVAL_SAMPLING_POLICY),
            "evaluation_ranking_policy": str(QUERY_EVALUATION_RANKING_POLICY),
        },
    )
    print(
        "full query pool: "
        f"size={len(all_query_rows)}, "
        f"auto_expand={bool(AUTO_EXPAND_FULL_QUERY_POOL)}, "
        f"target={int(TARGET_FULL_QUERY_POOL_SIZE)}"
    )
    print(
        "selected split pools: "
        f"calibration={len(calibration_candidates)}, evaluation={len(final_candidates)}, disjoint=True"
    )
    print("num_queries:", int(len(query_emb)))
    print("target_num_queries:", int(target_num_queries))
    print("num_real_query_candidates:", int(len(real_candidates)))
    print("num_strict_selected:", int(strict_selected_count))
    print("num_non_strict_selected:", int(len([r for r in final_rows if not bool(r.get("is_anchor_query"))])))
    print(
        "num_single_cluster_exact_consistent_selected:",
        int(single_cluster_exact_consistent_selected_count),
    )
    print("selected_cluster_counts:", selected_cluster_counts)
    print("selected_source_counts:", selected_source_counts)
    print("calibration_cluster_counts:", calibration_cluster_counts)
    print("calibration_source_counts:", calibration_source_counts)
    print("selected_difficulty_counts:", selected_difficulty_counts)
    print(
        "num_selected_canonicalized_from_source:",
        int(len([r for r in final_rows if str(r["query_id"]) != str(r.get("raw_query_id", r["query_id"]))])),
    )
    print("WORKSET_QUERIES_PATH:", WORKSET_QUERIES_PATH)
    print("WORKSET_CALIBRATION_QUERIES_JSONL_PATH:", WORKSET_CALIBRATION_QUERIES_JSONL_PATH)
    print("WORKSET_CALIBRATION_QUERIES_PATH:", WORKSET_CALIBRATION_QUERIES_PATH)
    print("WORKSET_CALIBRATION_QUERY_IDS_PATH:", WORKSET_CALIBRATION_QUERY_IDS_PATH)
    print("WORKSET_QUERY_SPLIT_META_PATH:", WORKSET_QUERY_SPLIT_META_PATH)
    print("WORKSET_QUERY_IDS_PATH:", WORKSET_QUERY_IDS_PATH)
    print("WORKSET_GT_TOPK_PATH:", WORKSET_GT_TOPK_PATH)
    print("WORKSET_QRELS_PATH:", WORKSET_QRELS_PATH)
    print("WORKSET_RELAXED_QRELS_PATH:", WORKSET_RELAXED_QRELS_PATH)


if __name__ == "__main__":
    main()
