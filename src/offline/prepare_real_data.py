"""
离线工作集构建（论文口径：docs-only query-free passage）。

流程：
1) 从 raw/corpus.jsonl 切分 passage-like retrieval units（局部完整语义块）。
2) 只基于文档 embedding 做 query-free 几何分布保持采样，严格不使用 query/qrels。
3) 禁用 adapter，仅用原生 E5 passage 编码。
4) 输出 docs/doc_ids/corpus/meta 四个工作集文件。
"""

# Allow running this file directly: `python src/offline/prepare_real_data.py`
if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import hashlib
import json
import os
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
from numpy.lib.format import open_memmap
from sklearn.cluster import MiniBatchKMeans

from shared.config import (
    CORPUS_JSONL_PATH,
    EPS,
    MAX_LENGTH,
    NEW_MODEL_NAME,
    NUM_CLUSTERS,
    NUM_WORKSET_DOCS,
    PASSAGE_CACHE_DOCS_PATH,
    PASSAGE_CACHE_DOC_IDS_PATH,
    PASSAGE_CACHE_META_PATH,
    PASSAGE_CACHE_PARTIAL_PATH,
    PASSAGE_CACHE_PROGRESS_PATH,
    PASSAGE_MAX_SENTENCES,
    PASSAGE_MAX_TOKENS,
    PASSAGE_MIN_SENTENCES,
    PASSAGE_MIN_TOKENS,
    PASSAGE_OVERLAP_TOKENS,
    PASSAGE_SENTENCE_STRIDE,
    PASSAGE_TARGET_TOKENS,
    WORKSET_COARSE_GROUPS,
    WORKSET_NEAR_DUP_HAMMING_MAX_GLOBAL,
    WORKSET_NEAR_DUP_HAMMING_MAX_WITHIN_SOURCE,
    WORKSET_PREENCODE_MULTIPLIER,
    WORKSET_SOURCE_DOC_MAX_PASSAGES,
    WORKSET_BUILD_MODE,
    WORKSET_CORPUS_JSONL_PATH,
    WORKSET_DOC_IDS_PATH,
    WORKSET_DOCS_PATH,
    WORKSET_META_PATH,
    TARGET_CLUSTER_SIZE,
    RANDOM_STATE,
)
from shared.e5_dual_encoder import E5DualEncoder

FORCE_REBUILD_WORKSET = True
PASSAGE_CACHE_BATCH_SIZE = 16
PASSAGE_CACHE_MAX_LENGTH = min(MAX_LENGTH, 256)
# 先做轻量预筛，再仅编码预筛后的候选。
# 说明：
# - 这里保留 multiplier 仅给非主线分支使用；
# - paperfaithful 主线默认从 100000 条候选 retrieval units 起步；
# - 若 partial memmap 已经继续编码到更长前缀，则直接复用该前缀。
GEOMETRY_PREENCODE_MULTIPLIER = int(WORKSET_PREENCODE_MULTIPLIER)
GEOMETRY_COARSE_CLUSTERS = int(WORKSET_COARSE_GROUPS)
GEOMETRY_KMEANS_MAX_ITER = 120
GEOMETRY_KMEANS_BATCH_SIZE = 4096
PAPERFAITHFUL_CANDIDATE_POOL_MIN = 100000
CLUSTER_CORE_SWEEP_SEED_START = 0
CLUSTER_CORE_SWEEP_SEED_END = 12
CLUSTER_CORE_SWEEP_N_INIT = 5


def clean_text(text: str) -> str:
    if text is None:
        return ""
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


def try_load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return (x / norms).astype(np.float32)


def fingerprint_ids(ids: List[str]) -> str:
    joined = "\n".join(str(x) for x in ids)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", clean_text(text).lower())


def parse_doc_row(row: dict) -> Tuple[str, str]:
    did = None
    for key in ("doc_id", "id", "_id"):
        if key in row:
            did = str(row[key])
            break

    text = None
    for key in ("text", "contents", "content", "body"):
        if key in row:
            text = clean_text(row[key])
            break

    if not did or not text:
        return None, None
    return did, text


def split_text_into_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    # 先把步骤号和编号列表折叠成普通断句边界，再做句级切分。
    normalized = re.sub(r"(?i)\bstep\s+\d+[.:)]?\s*", ". ", text)
    normalized = re.sub(r"(?<![A-Za-z])\d+[.)]?\s+", ". ", normalized)
    parts = re.split(r"(?<=[.!?;])\s+", normalized)
    sentences = []
    for part in parts:
        s = clean_text(part).strip(" -;:,")
        if s:
            sentences.append(s)
    return sentences


def split_document_into_units(doc_id: str, text: str) -> List[dict]:
    text = clean_text(text)
    if not text:
        return []

    sentences = split_text_into_sentences(text)
    if not sentences:
        sentences = [text]

    sent_tokens = [max(1, len(tokenize(s))) for s in sentences]
    n = len(sentences)
    min_sentences = max(1, int(PASSAGE_MIN_SENTENCES))
    max_sentences = max(min_sentences, int(PASSAGE_MAX_SENTENCES))
    min_tokens = max(8, int(PASSAGE_MIN_TOKENS))
    target_tokens = max(min_tokens, int(PASSAGE_TARGET_TOKENS))
    max_tokens = max(target_tokens, int(PASSAGE_MAX_TOKENS))
    overlap_tokens = int(max(0, PASSAGE_OVERLAP_TOKENS))
    # 禁止 stride=1 的高密滑窗：默认至少跨 2 句推进。
    min_advance_sentences = max(2, int(PASSAGE_SENTENCE_STRIDE))

    candidates: List[str] = []
    seen_texts = set()
    start = 0
    while start < n:
        end = start
        tok_sum = 0
        while end < n and (end - start) < max_sentences:
            next_tok = int(sent_tokens[end])
            if tok_sum + next_tok > max_tokens and (end - start) >= min_sentences:
                break
            tok_sum += next_tok
            end += 1
            enough_sent = (end - start) >= min_sentences
            enough_tok = tok_sum >= min_tokens
            hit_target = tok_sum >= target_tokens
            if enough_sent and enough_tok and hit_target:
                break

        # 兜底：保证至少有一个句子，且尽量满足 min_sentences。
        if end <= start:
            end = min(n, start + 1)
            tok_sum = int(sent_tokens[start])
        while end < n and (end - start) < min_sentences and tok_sum + int(sent_tokens[end]) <= max_tokens:
            tok_sum += int(sent_tokens[end])
            end += 1

        chunk_text = " ".join(sentences[start:end]).strip()
        if chunk_text and chunk_text not in seen_texts:
            seen_texts.add(chunk_text)
            candidates.append(chunk_text)

        if end >= n:
            break

        # 以 token overlap 回退确定下一窗口起点，避免过密滑窗。
        back_tok = 0
        next_start = end
        j = end - 1
        while j >= start and back_tok < overlap_tokens:
            back_tok += int(sent_tokens[j])
            j -= 1
        next_start = max(j + 1, start + min_advance_sentences)
        if next_start >= end:
            next_start = min(n, start + min_advance_sentences)
        if next_start <= start:
            next_start = min(n, start + min_advance_sentences)
        start = int(next_start)

    if len(candidates) == 0:
        candidates = [text]

    units: List[dict] = []
    for unit_idx, unit_text in enumerate(candidates):
        units.append(
            {
                "doc_id": f"{doc_id}::p{unit_idx}",
                "source_doc_id": doc_id,
                "text": unit_text,
            }
        )
    return units


def extract_ordered_doc_rows(corpus_rows: List[dict]) -> List[dict]:
    ordered_rows: List[dict] = []
    seen_source_ids = set()
    duplicate_ids = []

    for row in corpus_rows:
        did, text = parse_doc_row(row)
        if not did or not text:
            continue
        if did in seen_source_ids:
            duplicate_ids.append(did)
            continue

        units = split_document_into_units(doc_id=did, text=text)
        ordered_rows.extend(units)
        seen_source_ids.add(did)

    if duplicate_ids:
        dup_examples = sorted(set(duplicate_ids))[:10]
        raise RuntimeError(
            "corpus.jsonl 中出现重复 doc_id，无法按 retrieval-unit 建立索引；"
            f"重复数量={len(duplicate_ids)}，示例={dup_examples}"
        )

    if len(ordered_rows) == 0:
        raise RuntimeError("corpus.jsonl 中没有可用 retrieval units，无法建立 passage cache。")

    return ordered_rows


def _token_hash64(token: str) -> int:
    h = hashlib.sha1(token.encode("utf-8")).digest()[:8]
    return int.from_bytes(h, byteorder="big", signed=False)


def simhash64(text: str) -> int:
    toks = tokenize(text)
    if not toks:
        return 0
    vec = [0] * 64
    for tok in toks:
        hv = _token_hash64(tok)
        for b in range(64):
            if (hv >> b) & 1:
                vec[b] += 1
            else:
                vec[b] -= 1
    out = 0
    for b, v in enumerate(vec):
        if v >= 0:
            out |= (1 << b)
    return int(out)


def hamming64(a: int, b: int) -> int:
    return int((int(a) ^ int(b)).bit_count())


def deduplicate_units_docs_only(rows: List[dict]) -> Tuple[List[dict], dict]:
    """
    两层 docs-only 去重：
    1) source_doc 内部近重复去重；
    2) 全局近重复去重。
    """
    if len(rows) == 0:
        return [], {
            "num_before": 0,
            "num_after_within_source": 0,
            "num_after_global": 0,
            "removed_within_source": 0,
            "removed_global": 0,
            "within_source_dup_rate": 0.0,
            "global_dup_rate": 0.0,
        }

    by_source: Dict[str, List[dict]] = {}
    source_order: List[str] = []
    for r in rows:
        sid = str(r.get("source_doc_id", r.get("doc_id", "")))
        if sid not in by_source:
            by_source[sid] = []
            source_order.append(sid)
        by_source[sid].append(r)

    kept_within: List[dict] = []
    removed_within = 0
    src_thr = int(max(0, WORKSET_NEAR_DUP_HAMMING_MAX_WITHIN_SOURCE))
    for sid in source_order:
        bucket = by_source[sid]
        kept_hashes: List[int] = []
        for r in bucket:
            txt = str(r.get("text", ""))
            h = simhash64(txt)
            is_dup = any(hamming64(h, prev_h) <= src_thr for prev_h in kept_hashes)
            if is_dup:
                removed_within += 1
                continue
            rr = dict(r)
            rr["_simhash64"] = int(h)
            kept_within.append(rr)
            kept_hashes.append(int(h))

    # 全局去重：先按高位分桶，再做 hamming 校验。
    glb_thr = int(max(0, WORKSET_NEAR_DUP_HAMMING_MAX_GLOBAL))
    prefix_bits = 16
    buckets: Dict[int, List[int]] = {}
    kept_global: List[dict] = []
    removed_global = 0
    for r in kept_within:
        h = int(r["_simhash64"])
        key = int(h >> (64 - prefix_bits))
        cand_indices = buckets.get(key, [])
        is_dup = False
        for idx in cand_indices:
            prev_h = int(kept_global[idx]["_simhash64"])
            if hamming64(h, prev_h) <= glb_thr:
                is_dup = True
                break
        if is_dup:
            removed_global += 1
            continue
        buckets.setdefault(key, []).append(len(kept_global))
        kept_global.append(r)

    # 清理内部字段
    cleaned = []
    for r in kept_global:
        rr = dict(r)
        rr.pop("_simhash64", None)
        cleaned.append(rr)

    num_before = int(len(rows))
    num_after_within = int(len(kept_within))
    num_after_global = int(len(cleaned))
    stats = {
        "num_before": num_before,
        "num_after_within_source": num_after_within,
        "num_after_global": num_after_global,
        "removed_within_source": int(removed_within),
        "removed_global": int(removed_global),
        "within_source_dup_rate": float(removed_within / max(1, num_before)),
        "global_dup_rate": float(removed_global / max(1, num_after_within)),
        "within_source_hamming_threshold": int(src_thr),
        "global_hamming_threshold": int(glb_thr),
    }
    return cleaned, stats


def cache_meta_matches_expected(cache_meta: dict, expected_num_docs: int, expected_doc_ids_sha256: str) -> bool:
    if not isinstance(cache_meta, dict):
        return False
    if cache_meta.get("pipeline") != "selected_e5_passage_pool_cache":
        return False
    if int(cache_meta.get("num_docs", -1)) != int(expected_num_docs):
        return False
    if cache_meta.get("doc_ids_sha256") != expected_doc_ids_sha256:
        return False
    if cache_meta.get("model_name") != NEW_MODEL_NAME:
        return False
    prefix = cache_meta.get("embedding_prefix", cache_meta.get("encoding_prefix"))
    return prefix == "passage: "


class E5PassageEncoder:
    def __init__(self, model_name: str):
        self.backend = E5DualEncoder(
            model_name,
            log_prefix="passage-encoder",
        )

    def encode_passages(self, texts: List[str], batch_size: int, max_length: int) -> np.ndarray:
        _raw, norm = self.backend.encode_passages(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            progress_name="encode-selected-passage",
        )
        return norm.astype(np.float32)


def _cache_candidates() -> List[Tuple[str, str, str]]:
    return [(PASSAGE_CACHE_DOCS_PATH, PASSAGE_CACHE_DOC_IDS_PATH, PASSAGE_CACHE_META_PATH)]


def try_load_paperfaithful_partial_prefix_pool(
    dedup_rows: List[dict],
) -> Tuple[List[dict], np.ndarray, np.ndarray, dict] | None:
    """
    若 paperfaithful 主线的大池编码尚未完成，但 partial memmap 已经有足够长的有效前缀，
    直接复用该前缀作为 docs-only 候选池，避免重新编码。
    """
    if str(WORKSET_BUILD_MODE) != "paperfaithful_docs_only_mainline":
        return None
    progress = try_load_json(PASSAGE_CACHE_PROGRESS_PATH)
    if progress is None or (not os.path.exists(PASSAGE_CACHE_PARTIAL_PATH)):
        return None

    try:
        expected_doc_ids_sha256 = fingerprint_ids([str(r["doc_id"]) for r in dedup_rows])
        next_index = int(progress.get("next_index", -1))
        total = int(progress.get("num_docs", -1))
        dim = int(progress.get("dim", -1))
    except Exception:
        return None

    if progress.get("pipeline") != "selected_e5_passage_pool_cache":
        return None
    if progress.get("model_name") != NEW_MODEL_NAME:
        return None
    if progress.get("doc_ids_sha256") != expected_doc_ids_sha256:
        return None
    if int(total) != int(len(dedup_rows)):
        return None
    if int(next_index) < int(max(NUM_WORKSET_DOCS, PAPERFAITHFUL_CANDIDATE_POOL_MIN)):
        return None

    docs_partial = np.load(PASSAGE_CACHE_PARTIAL_PATH, mmap_mode="r")
    if docs_partial.shape != (int(total), int(dim)):
        return None

    prefix_rows = list(dedup_rows[: int(next_index)])
    prefix_docs = normalize_rows(np.asarray(docs_partial[: int(next_index)], dtype=np.float32))
    prefix_doc_ids = np.asarray([str(r["doc_id"]) for r in prefix_rows], dtype=object)
    summary = {
        "preselect_mode": "paperfaithful_partial_prefix_candidate_pool",
        "num_total_units_before_preselect": int(len(dedup_rows)),
        "num_total_units_after_dedup": int(len(dedup_rows)),
        "num_preselected_units": int(len(prefix_rows)),
        "partial_progress_next_index": int(next_index),
        "partial_progress_total_docs": int(total),
        "partial_progress_dim": int(dim),
        "candidate_pool_embedding_source": PASSAGE_CACHE_PARTIAL_PATH,
        "preencode_uses_full_dedup_pool": False,
    }
    return prefix_rows, prefix_docs, prefix_doc_ids, summary


def build_full_passage_cache(ordered_doc_rows: List[dict], encoder: E5PassageEncoder) -> Tuple[np.ndarray, np.ndarray]:
    doc_ids = [str(row["doc_id"]) for row in ordered_doc_rows]
    texts = [str(row["text"]) for row in ordered_doc_rows]
    total = len(texts)
    doc_ids_sha256 = fingerprint_ids(doc_ids)

    progress = try_load_json(PASSAGE_CACHE_PROGRESS_PATH)
    resume_ok = False
    next_index = 0
    dim = None

    if progress is not None and os.path.exists(PASSAGE_CACHE_PARTIAL_PATH):
        try:
            resume_ok = (
                progress.get("pipeline") == "selected_e5_passage_pool_cache"
                and progress.get("model_name") == NEW_MODEL_NAME
                and progress.get("doc_ids_sha256") == doc_ids_sha256
                and int(progress.get("num_docs", -1)) == total
                and int(progress.get("next_index", -1)) >= 0
            )
            if resume_ok:
                next_index = int(progress["next_index"])
                dim = int(progress["dim"])
                if next_index > total:
                    resume_ok = False
        except Exception:
            resume_ok = False

    if resume_ok:
        docs_partial = np.load(PASSAGE_CACHE_PARTIAL_PATH, mmap_mode="r+")
        if docs_partial.shape != (total, dim):
            resume_ok = False
            next_index = 0
            dim = None

    if not resume_ok:
        print("=== 开始编码 selected passage cache ===")
        first_batch = encoder.encode_passages(
            texts[:PASSAGE_CACHE_BATCH_SIZE],
            batch_size=PASSAGE_CACHE_BATCH_SIZE,
            max_length=PASSAGE_CACHE_MAX_LENGTH,
        )
        dim = int(first_batch.shape[1])
        docs_partial = open_memmap(
            PASSAGE_CACHE_PARTIAL_PATH,
            mode="w+",
            dtype=np.float32,
            shape=(total, dim),
        )
        docs_partial[: len(first_batch)] = first_batch.astype(np.float32)
        docs_partial.flush()
        next_index = int(len(first_batch))
        save_json(
            PASSAGE_CACHE_PROGRESS_PATH,
            {
                "pipeline": "selected_e5_passage_pool_cache",
                "model_name": NEW_MODEL_NAME,
                "encoding_prefix": "passage: ",
                "num_docs": int(total),
                "dim": int(dim),
                "next_index": int(next_index),
                "doc_ids_sha256": doc_ids_sha256,
            },
        )
        print(f"passage cache progress       : {next_index}/{total}")
    else:
        print("=== 检测到可续跑的 selected passage cache，继续编码 ===")
        print(f"passage cache resume         : {next_index}/{total}")

    while next_index < total:
        batch_texts = texts[next_index : next_index + PASSAGE_CACHE_BATCH_SIZE]
        batch_emb = encoder.encode_passages(
            batch_texts,
            batch_size=PASSAGE_CACHE_BATCH_SIZE,
            max_length=PASSAGE_CACHE_MAX_LENGTH,
        )
        docs_partial[next_index : next_index + len(batch_emb)] = batch_emb.astype(np.float32)
        next_index += len(batch_emb)
        docs_partial.flush()
        save_json(
            PASSAGE_CACHE_PROGRESS_PATH,
            {
                "pipeline": "selected_e5_passage_pool_cache",
                "model_name": NEW_MODEL_NAME,
                "encoding_prefix": "passage: ",
                "num_docs": int(total),
                "dim": int(dim),
                "next_index": int(next_index),
                "doc_ids_sha256": doc_ids_sha256,
            },
        )
        print(f"passage cache progress       : {next_index}/{total}")

    docs_partial.flush()
    del docs_partial

    np.save(PASSAGE_CACHE_DOC_IDS_PATH, np.asarray(doc_ids, dtype=object))
    shutil.copyfile(PASSAGE_CACHE_PARTIAL_PATH, PASSAGE_CACHE_DOCS_PATH)
    save_json(
        PASSAGE_CACHE_META_PATH,
        {
            "pipeline": "selected_e5_passage_pool_cache",
            "model_name": NEW_MODEL_NAME,
            "encoding_prefix": "passage: ",
            "retrieval_unit": "one longer passage-like contiguous chunk (multi-sentence) split from corpus.jsonl",
            "num_docs": int(total),
            "dim": int(dim),
            "doc_ids_sha256": doc_ids_sha256,
            "source_corpus_jsonl": CORPUS_JSONL_PATH,
            "chunking": {
                "min_tokens": int(PASSAGE_MIN_TOKENS),
                "target_tokens": int(PASSAGE_TARGET_TOKENS),
                "max_tokens": int(PASSAGE_MAX_TOKENS),
                "min_sentences": int(PASSAGE_MIN_SENTENCES),
                "max_sentences": int(PASSAGE_MAX_SENTENCES),
                "sentence_stride": int(PASSAGE_SENTENCE_STRIDE),
                "overlap_tokens": int(PASSAGE_OVERLAP_TOKENS),
            },
            "note": "该缓存服务当前已选中的工作集候选；编码阶段不使用 adapter。",
        },
    )

    if os.path.exists(PASSAGE_CACHE_PROGRESS_PATH):
        try:
            os.remove(PASSAGE_CACHE_PROGRESS_PATH)
        except OSError:
            pass
    if os.path.exists(PASSAGE_CACHE_PARTIAL_PATH):
        try:
            os.remove(PASSAGE_CACHE_PARTIAL_PATH)
        except OSError:
            pass

    full_docs = normalize_rows(np.load(PASSAGE_CACHE_DOCS_PATH).astype(np.float32))
    full_doc_ids = np.load(PASSAGE_CACHE_DOC_IDS_PATH, allow_pickle=True)
    return full_docs, full_doc_ids


def ensure_full_passage_cache(ordered_doc_rows: List[dict], encoder: E5PassageEncoder) -> Tuple[np.ndarray, np.ndarray]:
    expected_doc_ids = [str(row["doc_id"]) for row in ordered_doc_rows]
    expected_doc_ids_sha256 = fingerprint_ids(expected_doc_ids)
    expected_num_docs = len(expected_doc_ids)
    for docs_path, doc_ids_path, meta_path in _cache_candidates():
        cache_meta = try_load_json(meta_path)
        if not (os.path.exists(docs_path) and os.path.exists(doc_ids_path)):
            continue
        if not cache_meta_matches_expected(
            cache_meta=cache_meta,
            expected_num_docs=expected_num_docs,
            expected_doc_ids_sha256=expected_doc_ids_sha256,
        ):
            continue
        full_docs = np.load(docs_path).astype(np.float32)
        full_doc_ids = np.load(doc_ids_path, allow_pickle=True)
        if len(full_doc_ids) != expected_num_docs or full_docs.shape[0] != expected_num_docs:
            continue
        loaded_doc_ids_sha256 = fingerprint_ids([str(x) for x in full_doc_ids.tolist()])
        if loaded_doc_ids_sha256 != expected_doc_ids_sha256:
            continue
        print("=== 检测到与当前候选池一致的 selected passage cache，直接复用 ===")
        print("passage cache docs shape    =", full_docs.shape)
        print("passage cache doc_ids len   =", len(full_doc_ids))
        return normalize_rows(full_docs), full_doc_ids

    print("=== 当前 docs.npy 与本次候选池不一致，将重新建立 ===")
    print(f"expected candidate passages  : {expected_num_docs}")
    full_docs, full_doc_ids = build_full_passage_cache(
        ordered_doc_rows=ordered_doc_rows,
        encoder=encoder,
    )
    print("passage cache docs shape    =", full_docs.shape)
    print("passage cache doc_ids len   =", len(full_doc_ids))
    return full_docs, full_doc_ids


def preselect_rows_round_robin(ordered_doc_rows: List[dict], target_size: int) -> Tuple[List[dict], dict]:
    target_size = int(target_size)
    if len(ordered_doc_rows) < target_size:
        raise RuntimeError(
            f"retrieval units 总数不足：need={target_size}, got={len(ordered_doc_rows)}"
        )

    source_to_rows: Dict[str, List[dict]] = {}
    source_order: List[str] = []
    for row in ordered_doc_rows:
        did = str(row.get("doc_id", "")).strip()
        if not did:
            continue
        source_did = str(row.get("source_doc_id", did))
        if source_did not in source_to_rows:
            source_to_rows[source_did] = []
            source_order.append(source_did)
        source_to_rows[source_did].append(row)

    selected_rows: List[dict] = []
    round_idx = 0
    rounds_used = 0
    while len(selected_rows) < target_size:
        rounds_used += 1
        added_this_round = 0
        for source_did in source_order:
            bucket = source_to_rows[source_did]
            if round_idx < len(bucket):
                selected_rows.append(bucket[round_idx])
                added_this_round += 1
                if len(selected_rows) >= target_size:
                    break
        if added_this_round == 0:
            break
        round_idx += 1

    if len(selected_rows) < target_size:
        raise RuntimeError(
            f"round-robin 预筛失败：need={target_size}, got={len(selected_rows)}"
        )
    selected_rows = selected_rows[:target_size]

    summary = {
        "preselect_mode": "source_round_robin_natural_order",
        "num_total_units_before_preselect": int(len(ordered_doc_rows)),
        "num_unique_source_docs_before_preselect": int(len(source_to_rows)),
        "preselect_round_robin_rounds_used": int(rounds_used),
        "num_preselected_units": int(len(selected_rows)),
    }
    return selected_rows, summary


def allocate_integer_quota_proportional(counts: np.ndarray, target_size: int) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1:
        raise ValueError("counts must be 1-D")
    if int(counts.sum()) < int(target_size):
        raise ValueError("counts sum smaller than target_size")
    if int(target_size) <= 0:
        return np.zeros_like(counts, dtype=np.int32)

    expected = counts.astype(np.float64) * (float(target_size) / float(counts.sum()))
    quota = np.floor(expected).astype(np.int64)
    quota = np.minimum(quota, counts)
    fractional = expected - np.floor(expected)

    remaining = int(target_size - int(quota.sum()))
    while remaining > 0:
        candidates = np.where(quota < counts)[0]
        if len(candidates) == 0:
            break
        order = sorted(
            candidates.tolist(),
            key=lambda cid: (
                -float(fractional[cid]),
                -int(counts[cid] - quota[cid]),
                int(cid),
            ),
        )
        for cid in order:
            if remaining <= 0:
                break
            if quota[cid] >= counts[cid]:
                continue
            quota[cid] += 1
            remaining -= 1

    if int(quota.sum()) != int(target_size):
        raise RuntimeError(
            "quota allocation failed: "
            f"sum={int(quota.sum())}, target={int(target_size)}"
        )
    return quota.astype(np.int32)


def select_rows_geometry_preserving(
    ordered_doc_rows: List[dict],
    full_docs: np.ndarray,
    full_doc_ids: np.ndarray,
    target_size: int,
    random_state: int,
) -> Tuple[List[dict], dict]:
    target_size = int(target_size)
    num_docs = int(len(ordered_doc_rows))
    if num_docs < target_size:
        raise RuntimeError(
            f"retrieval units 总数不足：need={target_size}, got={num_docs}"
        )
    if int(full_docs.shape[0]) != num_docs or int(len(full_doc_ids)) != num_docs:
        raise RuntimeError(
            "passage cache 与候选池长度不一致："
            f"rows={num_docs}, docs={full_docs.shape[0]}, ids={len(full_doc_ids)}"
        )

    row_doc_ids = [str(r.get("doc_id", "")) for r in ordered_doc_rows]
    cache_doc_ids = [str(x) for x in full_doc_ids.tolist()]
    if row_doc_ids != cache_doc_ids:
        raise RuntimeError("passage cache doc_id 顺序与候选池不一致，无法安全采样")

    source_cap = int(max(0, int(WORKSET_SOURCE_DOC_MAX_PASSAGES)))
    source_pick_counts: Dict[str, int] = {}
    if num_docs == target_size:
        selected_indices = np.arange(num_docs, dtype=np.int32)
        labels = np.zeros(num_docs, dtype=np.int32)
        coarse_k = 1
        centers = np.mean(np.asarray(full_docs, dtype=np.float32), axis=0, keepdims=True).astype(np.float32)
    else:
        coarse_k = int(
            min(
                GEOMETRY_COARSE_CLUSTERS,
                max(4, int(round(np.sqrt(float(num_docs) / 2.0)))),
            )
        )
        coarse_k = int(min(coarse_k, num_docs))
        kmeans = MiniBatchKMeans(
            n_clusters=int(coarse_k),
            random_state=int(random_state),
            batch_size=int(min(GEOMETRY_KMEANS_BATCH_SIZE, max(256, coarse_k * 64))),
            max_iter=int(GEOMETRY_KMEANS_MAX_ITER),
            n_init=5,
        )
        labels = kmeans.fit_predict(np.asarray(full_docs, dtype=np.float32)).astype(np.int32)
        centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)

        counts = np.bincount(labels, minlength=coarse_k).astype(np.int64)
        quotas = allocate_integer_quota_proportional(counts=counts, target_size=target_size)

        picked = []
        picked_set = set()
        for cid in range(coarse_k):
            cluster_indices = np.where(labels == int(cid))[0]
            need = int(quotas[cid])
            if need <= 0:
                continue
            cluster_vecs = np.asarray(full_docs[cluster_indices], dtype=np.float32)
            centroid = np.asarray(centers[int(cid)], dtype=np.float32).reshape(1, -1)
            dists = np.linalg.norm(cluster_vecs - centroid, axis=1).astype(np.float32)
            order = cluster_indices[np.argsort(dists, kind="mergesort")].astype(np.int32)
            chosen_cnt = 0
            for idx in order.tolist():
                if idx in picked_set:
                    continue
                sid = str(ordered_doc_rows[int(idx)].get("source_doc_id", ordered_doc_rows[int(idx)].get("doc_id")))
                if source_cap > 0 and int(source_pick_counts.get(sid, 0)) >= source_cap:
                    continue
                picked.append(int(idx))
                picked_set.add(int(idx))
                source_pick_counts[sid] = int(source_pick_counts.get(sid, 0) + 1)
                chosen_cnt += 1
                if chosen_cnt >= need:
                    break

        if len(picked) != target_size:
            # 补样：仍按各 coarse group 到中心距离优先，且尽量遵守 source cap。
            all_order = []
            for cid in range(coarse_k):
                cluster_indices = np.where(labels == int(cid))[0]
                cluster_vecs = np.asarray(full_docs[cluster_indices], dtype=np.float32)
                centroid = np.asarray(centers[int(cid)], dtype=np.float32).reshape(1, -1)
                dists = np.linalg.norm(cluster_vecs - centroid, axis=1).astype(np.float32)
                order = cluster_indices[np.argsort(dists, kind="mergesort")].astype(np.int32)
                all_order.extend(order.tolist())
            for idx in all_order:
                if len(picked) >= target_size:
                    break
                if int(idx) in picked_set:
                    continue
                sid = str(ordered_doc_rows[int(idx)].get("source_doc_id", ordered_doc_rows[int(idx)].get("doc_id")))
                if source_cap > 0 and int(source_pick_counts.get(sid, 0)) >= source_cap:
                    continue
                picked.append(int(idx))
                picked_set.add(int(idx))
                source_pick_counts[sid] = int(source_pick_counts.get(sid, 0) + 1)

            # 若 source cap 过严仍不足，最后允许突破 cap 补齐（保持可行性）。
            if len(picked) < target_size:
                for idx in all_order:
                    if len(picked) >= target_size:
                        break
                    if int(idx) in picked_set:
                        continue
                    sid = str(ordered_doc_rows[int(idx)].get("source_doc_id", ordered_doc_rows[int(idx)].get("doc_id")))
                    picked.append(int(idx))
                    picked_set.add(int(idx))
                    source_pick_counts[sid] = int(source_pick_counts.get(sid, 0) + 1)

        selected_indices = np.asarray(sorted(int(x) for x in picked), dtype=np.int32)
        if len(selected_indices) != target_size:
            raise RuntimeError(
                "geometry sampler 输出长度错误："
                f"need={target_size}, got={len(selected_indices)}"
            )

    selected_rows = [ordered_doc_rows[int(i)] for i in selected_indices.tolist()]
    selected_labels = labels[selected_indices]
    full_hist = np.bincount(labels, minlength=coarse_k).astype(np.float64)
    selected_hist = np.bincount(selected_labels, minlength=coarse_k).astype(np.float64)
    full_hist = full_hist / np.maximum(np.sum(full_hist), 1.0)
    selected_hist = selected_hist / np.maximum(np.sum(selected_hist), 1.0)

    selected_source_ids = [str(r.get("source_doc_id", r.get("doc_id"))) for r in selected_rows]
    source_pick_counts: Dict[str, int] = {}
    for sid in selected_source_ids:
        source_pick_counts[sid] = int(source_pick_counts.get(sid, 0) + 1)
    selected_token_lengths = [len(tokenize(str(r.get("text", "")))) for r in selected_rows]

    source_dominance = (
        float(max(source_pick_counts.values()) / max(1, len(selected_rows)))
        if source_pick_counts
        else 0.0
    )
    coarse_sizes = np.bincount(labels, minlength=coarse_k).astype(np.int32)
    summary = {
        "selection_mode": "docs_only_coarse_group_representative_sampling",
        "num_total_units_before_select": int(num_docs),
        "num_selected_units": int(len(selected_rows)),
        "num_unique_source_docs_selected": int(len(source_pick_counts)),
        "max_units_selected_from_one_source": int(max(source_pick_counts.values()))
        if source_pick_counts
        else 0,
        "selected_token_len_p50": float(np.percentile(selected_token_lengths, 50.0))
        if selected_token_lengths
        else 0.0,
        "selected_token_len_p90": float(np.percentile(selected_token_lengths, 90.0))
        if selected_token_lengths
        else 0.0,
        "coarse_geometry_clusters": int(coarse_k),
        "geometry_hist_l1_distance": float(np.sum(np.abs(selected_hist - full_hist))),
        "geometry_hist_linf_distance": float(np.max(np.abs(selected_hist - full_hist))),
        "source_doc_max_passages_cap": int(source_cap),
        "source_doc_dominance_ratio": float(source_dominance),
        "coarse_group_size_min": int(np.min(coarse_sizes)),
        "coarse_group_size_p50": float(np.percentile(coarse_sizes, 50.0)),
        "coarse_group_size_max": int(np.max(coarse_sizes)),
        "coarse_group_size_distribution": [int(x) for x in coarse_sizes.tolist()],
    }
    return selected_rows, summary


def select_rows_cluster_core_seed_sweep(
    ordered_doc_rows: List[dict],
    full_docs: np.ndarray,
    full_doc_ids: np.ndarray,
    target_size: int,
) -> Tuple[List[dict], dict]:
    """
    docs-only workset 新主线：
    - 在候选池上直接做 NUM_CLUSTERS-way MiniBatchKMeans；
    - 每个簇取最核心的 TARGET_CLUSTER_SIZE 个 docs；
    - 做一个小范围 seed sweep；
    - 仅按 docs-only 的 core density / support 选最佳 workset。
    """
    target_size = int(target_size)
    expected_total = int(NUM_CLUSTERS) * int(TARGET_CLUSTER_SIZE)
    if int(target_size) != int(expected_total):
        raise RuntimeError(
            "cluster-core seed sweep requires exact fixed-cluster setup: "
            f"target_size={target_size}, expected={expected_total}"
        )

    num_docs = int(len(ordered_doc_rows))
    if num_docs < target_size:
        raise RuntimeError(f"candidate pool too small: need={target_size}, got={num_docs}")
    if int(full_docs.shape[0]) != num_docs or int(len(full_doc_ids)) != num_docs:
        raise RuntimeError(
            "candidate pool rows/docs/doc_ids length mismatch: "
            f"rows={num_docs}, docs={full_docs.shape[0]}, ids={len(full_doc_ids)}"
        )

    row_doc_ids = [str(r.get("doc_id", "")) for r in ordered_doc_rows]
    cache_doc_ids = [str(x) for x in full_doc_ids.tolist()]
    if row_doc_ids != cache_doc_ids:
        raise RuntimeError("candidate pool doc_id order mismatch; cannot do safe cluster-core selection")

    docs = normalize_rows(np.asarray(full_docs, dtype=np.float32))
    best = None
    sweep_rows = []
    seed_start = int(CLUSTER_CORE_SWEEP_SEED_START)
    seed_end = int(CLUSTER_CORE_SWEEP_SEED_END)

    for seed in range(seed_start, seed_end):
        kmeans = MiniBatchKMeans(
            n_clusters=int(NUM_CLUSTERS),
            random_state=int(seed),
            batch_size=int(min(GEOMETRY_KMEANS_BATCH_SIZE, max(1024, int(NUM_CLUSTERS) * 256))),
            max_iter=int(GEOMETRY_KMEANS_MAX_ITER),
            n_init=int(CLUSTER_CORE_SWEEP_N_INIT),
        )
        labels = kmeans.fit_predict(docs).astype(np.int32)
        centers = normalize_rows(np.asarray(kmeans.cluster_centers_, dtype=np.float32))
        cluster_sizes = np.bincount(labels, minlength=int(NUM_CLUSTERS)).astype(np.int32)
        if int(np.min(cluster_sizes)) < int(TARGET_CLUSTER_SIZE):
            sweep_rows.append(
                {
                    "seed": int(seed),
                    "feasible": False,
                    "cluster_sizes": [int(x) for x in cluster_sizes.tolist()],
                }
            )
            continue

        selected_blocks = []
        core_similarity_all = []
        selected_centers = []
        selected_margin_all = []
        per_cluster = []
        for cid in range(int(NUM_CLUSTERS)):
            cluster_idx = np.where(labels == int(cid))[0].astype(np.int32)
            center = centers[int(cid)]
            sims = (docs[cluster_idx] @ center).astype(np.float32)
            order_local = np.argsort(-sims, kind="mergesort")[: int(TARGET_CLUSTER_SIZE)]
            chosen_idx = cluster_idx[order_local].astype(np.int32)
            chosen_docs = docs[chosen_idx]

            selected_blocks.append(chosen_idx)
            core_similarity_all.extend(sims[order_local].astype(np.float64).tolist())

            selected_center = normalize_rows(
                np.mean(chosen_docs, axis=0, keepdims=True).astype(np.float32)
            )[0]
            selected_centers.append(selected_center)

            if int(NUM_CLUSTERS) > 1:
                other_centers = np.delete(centers, int(cid), axis=0)
                margin = (chosen_docs @ center) - np.max(chosen_docs @ other_centers.T, axis=1)
                selected_margin_all.extend(np.asarray(margin, dtype=np.float64).tolist())

            per_cluster.append(
                {
                    "cluster_id": int(cid),
                    "candidate_support": int(len(cluster_idx)),
                    "selected_core_similarity_mean": float(np.mean(sims[order_local])),
                    "selected_core_similarity_p10": float(np.percentile(sims[order_local], 10.0)),
                }
            )

        selected_centers = np.asarray(selected_centers, dtype=np.float32)
        selected_center_sep = []
        for i in range(int(NUM_CLUSTERS)):
            for j in range(i + 1, int(NUM_CLUSTERS)):
                selected_center_sep.append(
                    float(np.linalg.norm(selected_centers[i] - selected_centers[j]))
                )

        mean_core_similarity = float(np.mean(core_similarity_all)) if core_similarity_all else 0.0
        support_min = int(np.min(cluster_sizes))
        support_p50 = float(np.percentile(cluster_sizes, 50.0))
        support_max = int(np.max(cluster_sizes))
        margin_mean = float(np.mean(selected_margin_all)) if selected_margin_all else 0.0
        center_sep_mean = float(np.mean(selected_center_sep)) if selected_center_sep else 0.0

        row = {
            "seed": int(seed),
            "feasible": True,
            "cluster_sizes": [int(x) for x in cluster_sizes.tolist()],
            "support_min": int(support_min),
            "support_p50": float(support_p50),
            "support_max": int(support_max),
            "mean_core_similarity": float(mean_core_similarity),
            "mean_selected_margin": float(margin_mean),
            "selected_center_separation_mean": float(center_sep_mean),
            "per_cluster": per_cluster,
            "selected_indices": np.concatenate(selected_blocks, axis=0).astype(np.int32),
        }
        sweep_rows.append(row)
        rank_key = (
            float(mean_core_similarity),
            int(support_min),
            float(support_p50),
            float(margin_mean),
        )
        if best is None or rank_key > best["rank_key"]:
            best = {
                "rank_key": rank_key,
                "row": row,
            }

    feasible_rows = [r for r in sweep_rows if bool(r.get("feasible", False))]
    if best is None or not feasible_rows:
        raise RuntimeError(
            "cluster-core seed sweep found no feasible candidate workset under "
            f"NUM_CLUSTERS={int(NUM_CLUSTERS)}, TARGET_CLUSTER_SIZE={int(TARGET_CLUSTER_SIZE)}"
        )

    selected_indices = np.asarray(best["row"]["selected_indices"], dtype=np.int32)
    selected_rows = [ordered_doc_rows[int(i)] for i in selected_indices.tolist()]
    selected_source_ids = [str(r.get("source_doc_id", r.get("doc_id"))) for r in selected_rows]
    source_pick_counts: Dict[str, int] = {}
    for sid in selected_source_ids:
        source_pick_counts[sid] = int(source_pick_counts.get(sid, 0) + 1)
    top_rows = sorted(
        feasible_rows,
        key=lambda r: (
            float(r["mean_core_similarity"]),
            int(r["support_min"]),
            float(r["support_p50"]),
            float(r["mean_selected_margin"]),
        ),
        reverse=True,
    )[:5]

    summary = {
        "selection_mode": "docs_only_cluster_core_seed_sweep",
        "seed_sweep_start": int(seed_start),
        "seed_sweep_end_exclusive": int(seed_end),
        "num_candidate_units": int(num_docs),
        "num_selected_units": int(len(selected_rows)),
        "num_unique_source_docs_selected": int(len(source_pick_counts)),
        "max_units_selected_from_one_source": int(max(source_pick_counts.values()))
        if source_pick_counts
        else 0,
        "selected_seed": int(best["row"]["seed"]),
        "selected_seed_rank_key": [
            float(best["rank_key"][0]),
            int(best["rank_key"][1]),
            float(best["rank_key"][2]),
            float(best["rank_key"][3]),
        ],
        "selected_mean_core_similarity": float(best["row"]["mean_core_similarity"]),
        "selected_mean_selected_margin": float(best["row"]["mean_selected_margin"]),
        "selected_center_separation_mean": float(best["row"]["selected_center_separation_mean"]),
        "selected_support_min": int(best["row"]["support_min"]),
        "selected_support_p50": float(best["row"]["support_p50"]),
        "selected_support_max": int(best["row"]["support_max"]),
        "selected_cluster_sizes_before_core_pick": [
            int(x) for x in best["row"]["cluster_sizes"]
        ],
        "top_seed_rows": [
            {
                "seed": int(r["seed"]),
                "mean_core_similarity": float(r["mean_core_similarity"]),
                "mean_selected_margin": float(r["mean_selected_margin"]),
                "selected_center_separation_mean": float(r["selected_center_separation_mean"]),
                "support_min": int(r["support_min"]),
                "support_p50": float(r["support_p50"]),
                "support_max": int(r["support_max"]),
                "cluster_sizes": [int(x) for x in r["cluster_sizes"]],
            }
            for r in top_rows
        ],
    }
    return selected_rows, summary


def summarize_embedding_geometry(x: np.ndarray, sample_pairs: int = 20000, seed: int = 42) -> dict:
    x = np.asarray(x, dtype=np.float32)
    if len(x) <= 0:
        return {
            "norm_min": 0.0,
            "norm_p50": 0.0,
            "norm_mean": 0.0,
            "norm_p95": 0.0,
            "norm_max": 0.0,
            "pairwise_cos_min": 0.0,
            "pairwise_cos_p50": 0.0,
            "pairwise_cos_mean": 0.0,
            "pairwise_cos_p95": 0.0,
            "pairwise_cos_max": 0.0,
            "pairwise_cos_num_pairs": 0,
        }
    norms = np.linalg.norm(x, axis=1).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    n = int(len(x))
    m = int(min(int(sample_pairs), max(1, n * (n - 1) // 2)))
    if n <= 1:
        pairs = np.asarray([], dtype=np.float32)
    else:
        i = rng.integers(0, n, size=m, endpoint=False)
        j = rng.integers(0, n, size=m, endpoint=False)
        mask = i != j
        if not np.any(mask):
            pairs = np.asarray([], dtype=np.float32)
        else:
            i = i[mask]
            j = j[mask]
            pairs = np.sum(x[i] * x[j], axis=1).astype(np.float64)
    return {
        "norm_min": float(np.min(norms)),
        "norm_p50": float(np.percentile(norms, 50.0)),
        "norm_mean": float(np.mean(norms)),
        "norm_p95": float(np.percentile(norms, 95.0)),
        "norm_max": float(np.max(norms)),
        "pairwise_cos_min": float(np.min(pairs)) if len(pairs) > 0 else 0.0,
        "pairwise_cos_p50": float(np.percentile(pairs, 50.0)) if len(pairs) > 0 else 0.0,
        "pairwise_cos_mean": float(np.mean(pairs)) if len(pairs) > 0 else 0.0,
        "pairwise_cos_p95": float(np.percentile(pairs, 95.0)) if len(pairs) > 0 else 0.0,
        "pairwise_cos_max": float(np.max(pairs)) if len(pairs) > 0 else 0.0,
        "pairwise_cos_num_pairs": int(len(pairs)),
    }


def _stable_hash_preselect_rows(rows: List[dict], target_size: int) -> List[dict]:
    if target_size >= len(rows):
        return list(rows)
    scored = []
    for r in rows:
        did = str(r.get("doc_id", ""))
        sid = str(r.get("source_doc_id", did))
        h = hashlib.sha1(f"{sid}::{did}".encode("utf-8")).hexdigest()
        scored.append((h, did, r))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in scored[: int(target_size)]]


def _paperfaithful_preencode_size(target_size: int, total_rows: int) -> int:
    """
    paperfaithful 主线保持固定前缀候选池口径：
    - 至少覆盖目标工作集大小；
    - 默认从 100000 条候选 retrieval units 起步；
    - 若 dedup 后不足 100000，则退化为可用全量；
    - 不再走旧的 20000 候选池口径。
    """
    return int(min(int(total_rows), max(int(target_size), int(PAPERFAITHFUL_CANDIDATE_POOL_MIN))))


def build_workset_from_corpus(corpus_rows: List[dict]):
    ordered_doc_rows = extract_ordered_doc_rows(corpus_rows)
    dedup_rows, dedup_stats = deduplicate_units_docs_only(ordered_doc_rows)
    target_size = int(NUM_WORKSET_DOCS)
    candidate_pool = try_load_paperfaithful_partial_prefix_pool(dedup_rows)
    if candidate_pool is not None:
        preselected_rows, full_docs, full_doc_ids, preselect_summary = candidate_pool
    else:
        if str(WORKSET_BUILD_MODE) == "paperfaithful_docs_only_mainline":
            preencode_size = _paperfaithful_preencode_size(
                target_size=int(target_size),
                total_rows=int(len(dedup_rows)),
            )
            preselected_rows = list(dedup_rows[: int(preencode_size)])
            preselect_summary = {
                "preselect_mode": "paperfaithful_partial_prefix_candidate_pool",
                "num_total_units_before_preselect": int(len(ordered_doc_rows)),
                "num_total_units_after_dedup": int(len(dedup_rows)),
                "num_preselected_units": int(len(preselected_rows)),
                "fixed_candidate_pool_min": int(PAPERFAITHFUL_CANDIDATE_POOL_MIN),
                "preencode_uses_full_dedup_pool": bool(preencode_size == len(dedup_rows)),
            }
        else:
            preencode_size = int(
                min(
                    len(dedup_rows),
                    max(target_size, target_size * int(GEOMETRY_PREENCODE_MULTIPLIER)),
                )
            )
            preselected_rows = _stable_hash_preselect_rows(dedup_rows, preencode_size)
            preselect_summary = {
                "preselect_mode": "docs_only_hash_stable_preencode_pool",
                "num_total_units_before_preselect": int(len(ordered_doc_rows)),
                "num_total_units_after_dedup": int(len(dedup_rows)),
                "num_preselected_units": int(len(preselected_rows)),
                "preencode_uses_full_dedup_pool": bool(preencode_size == len(dedup_rows)),
            }

        encoder = E5PassageEncoder(NEW_MODEL_NAME)
        full_docs, full_doc_ids = ensure_full_passage_cache(
            ordered_doc_rows=preselected_rows,
            encoder=encoder,
        )

    if str(WORKSET_BUILD_MODE) == "paperfaithful_docs_only_mainline":
        candidate_embedding_path = (
            str(preselect_summary.get("candidate_pool_embedding_source"))
            if preselect_summary.get("candidate_pool_embedding_source")
            else PASSAGE_CACHE_DOCS_PATH
        )
    else:
        candidate_embedding_path = PASSAGE_CACHE_DOCS_PATH

    print(f"=== workset 构造模式：{WORKSET_BUILD_MODE} ===")
    print(f"candidate retrieval units    : {len(ordered_doc_rows)}")
    print(f"after dedup units            : {len(dedup_rows)}")
    print(f"preselected for encoding     : {len(preselected_rows)}")
    print(
        "=== 编码策略："
        + "禁用 adapter，仅 passage 前缀"
        + " ==="
    )

    if str(WORKSET_BUILD_MODE) == "paperfaithful_docs_only_mainline":
        selected_rows, selection_summary = select_rows_cluster_core_seed_sweep(
            ordered_doc_rows=preselected_rows,
            full_docs=full_docs,
            full_doc_ids=full_doc_ids,
            target_size=target_size,
        )
    else:
        selected_rows, selection_summary = select_rows_geometry_preserving(
            ordered_doc_rows=preselected_rows,
            full_docs=full_docs,
            full_doc_ids=full_doc_ids,
            target_size=target_size,
            random_state=int(RANDOM_STATE),
        )

    full_doc_id_to_index = {
        str(did): int(i)
        for i, did in enumerate(full_doc_ids.tolist())
    }
    selected_indices = np.asarray(
        [full_doc_id_to_index[str(r["doc_id"])] for r in selected_rows],
        dtype=np.int32,
    )
    subset_doc_ids = np.asarray([str(r["doc_id"]) for r in selected_rows], dtype=object)
    subset_doc_vecs = normalize_rows(np.asarray(full_docs[selected_indices], dtype=np.float32))
    subset_rows = selected_rows

    print(f"selected retrieval units     : {len(subset_rows)}")
    print(f"selected unique source docs  : {selection_summary['num_unique_source_docs_selected']}")

    source_counts: Dict[str, int] = {}
    for row in subset_rows:
        sid = str(row.get("source_doc_id", row.get("doc_id", "")))
        source_counts[sid] = int(source_counts.get(sid, 0) + 1)
    source_count_values = np.asarray(list(source_counts.values()), dtype=np.int32) if source_counts else np.asarray([], dtype=np.int32)
    source_count_hist = {}
    if len(source_count_values) > 0:
        uniq, cnt = np.unique(source_count_values, return_counts=True)
        source_count_hist = {str(int(u)): int(c) for u, c in zip(uniq.tolist(), cnt.tolist())}
    source_stats = {
        "num_source_docs_covered": int(len(source_counts)),
        "source_doc_count_min": int(np.min(source_count_values)) if len(source_count_values) > 0 else 0,
        "source_doc_count_p50": float(np.percentile(source_count_values, 50.0)) if len(source_count_values) > 0 else 0.0,
        "source_doc_count_p90": float(np.percentile(source_count_values, 90.0)) if len(source_count_values) > 0 else 0.0,
        "source_doc_count_max": int(np.max(source_count_values)) if len(source_count_values) > 0 else 0,
        "source_doc_dominance_ratio": float(np.max(source_count_values) / max(1, len(subset_rows))) if len(source_count_values) > 0 else 0.0,
        "source_doc_count_distribution": source_count_hist,
    }
    embedding_geometry_stats = summarize_embedding_geometry(
        subset_doc_vecs,
        sample_pairs=20000,
        seed=int(RANDOM_STATE),
    )

    np.save(WORKSET_DOCS_PATH, subset_doc_vecs)
    np.save(WORKSET_DOC_IDS_PATH, subset_doc_ids)
    save_jsonl(WORKSET_CORPUS_JSONL_PATH, subset_rows)

    if str(WORKSET_BUILD_MODE) == "paperfaithful_docs_only_mainline":
        pipeline_name = "paperfaithful_docs_only_mainline"
        subset_strategy = (
            "paper-faithful docs-only long-passage workset build over a fixed docs-only prefix candidate pool; "
            "run fixed-capacity docs-only seed sweep under the current num_clusters/target_cluster_size "
            "without using queries/qrels."
        )
    else:
        pipeline_name = "query_free_long_passage_geometry_preserving_sampling"
        subset_strategy = (
            "docs-only query-free geometry-preserving sampling over deduplicated long passages; "
            "coarse clustering + proportional quota + representative (near-centroid) picking."
        )

    meta = {
        "pipeline": str(pipeline_name),
        "build_mode": str(WORKSET_BUILD_MODE),
        "new_model_name": NEW_MODEL_NAME,
        "num_output_docs": int(len(subset_doc_ids)),
        "dim": int(subset_doc_vecs.shape[1]),
        "query_dependency": "none_docs_only",
        "selection_inputs": {"corpus_path": CORPUS_JSONL_PATH},
        "adapter_used_for_passage_encoding": False,
        "subset_strategy": str(subset_strategy),
        "target_offline_partition": {
            "num_clusters": int(NUM_CLUSTERS),
            "target_cluster_size": int(TARGET_CLUSTER_SIZE),
        },
        "preselect_summary": preselect_summary,
        "dedup_summary": dedup_stats,
        "selection_summary": selection_summary,
        "candidate_pool_embedding_path": str(candidate_embedding_path),
        "source_doc_summary": source_stats,
        "embedding_geometry_summary": embedding_geometry_stats,
        "chunking": {
            "min_tokens": int(PASSAGE_MIN_TOKENS),
            "target_tokens": int(PASSAGE_TARGET_TOKENS),
            "max_tokens": int(PASSAGE_MAX_TOKENS),
            "min_sentences": int(PASSAGE_MIN_SENTENCES),
            "max_sentences": int(PASSAGE_MAX_SENTENCES),
            "sentence_stride": int(PASSAGE_SENTENCE_STRIDE),
            "overlap_tokens": int(PASSAGE_OVERLAP_TOKENS),
        },
        "cluster_infos": [],
        "output_docs_path": WORKSET_DOCS_PATH,
        "output_doc_ids_path": WORKSET_DOC_IDS_PATH,
        "output_corpus_path": WORKSET_CORPUS_JSONL_PATH,
        "passage_cache_docs_path_used": PASSAGE_CACHE_DOCS_PATH,
        "passage_cache_doc_ids_path_used": PASSAGE_CACHE_DOC_IDS_PATH,
        "full_retrieval_unit": "one longer, contiguous passage-like chunk split from a cleaned corpus row",
        "note": (
            "工作集选择阶段严格不使用 query/qrels；仅依赖 docs embedding 的离线几何分布保持采样，"
            "并包含 docs-only 两层去重与 source_doc 贡献上限约束。"
        ),
    }
    save_json(WORKSET_META_PATH, meta)

    print(f"=== {int(NUM_WORKSET_DOCS)} 条 E5 工作集已生成 ===")
    print("WORKSET_DOCS_PATH         =", WORKSET_DOCS_PATH)
    print("WORKSET_DOC_IDS_PATH      =", WORKSET_DOC_IDS_PATH)
    print("WORKSET_CORPUS_JSONL_PATH =", WORKSET_CORPUS_JSONL_PATH)
    print("WORKSET_META_PATH         =", WORKSET_META_PATH)
    print("shape                     =", subset_doc_vecs.shape)


def workset_doc_cache_exists() -> bool:
    required = [
        WORKSET_DOCS_PATH,
        WORKSET_DOC_IDS_PATH,
        WORKSET_CORPUS_JSONL_PATH,
        WORKSET_META_PATH,
    ]
    return all(os.path.exists(p) for p in required)


def main():
    if not os.path.exists(CORPUS_JSONL_PATH):
        raise FileNotFoundError(f"未找到必要文件：{CORPUS_JSONL_PATH}")

    print(f"=== WORKSET_BUILD_MODE = {WORKSET_BUILD_MODE}（forced）===")
    if workset_doc_cache_exists() and (not FORCE_REBUILD_WORKSET):
        print("=== 检测到已存在的工作集缓存，直接复用，不重新构造 ===")
        cached_docs = np.load(WORKSET_DOCS_PATH)
        print("cached docs shape =", cached_docs.shape)
        return

    corpus_rows = load_jsonl(CORPUS_JSONL_PATH)
    build_workset_from_corpus(corpus_rows)


if __name__ == "__main__":
    main()
