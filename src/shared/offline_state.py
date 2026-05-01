"""
在线阶段离线状态装载工具。

当前上传版仍使用单进程脚本模拟 client/server 往返，但这里把在线前
已经存在的离线资产显式拆成：
1) client offline state：query / qrels / routing 所需元数据；
2) server offline state：docs / doc_ids / doc_texts / cluster payload；
3) shared cluster state：同一个 offline cluster artifact 中，client 和
   server 都会读取的只读聚类信息。

这样 run_online_pipeline.py 不再把双方离线资产混在一个加载块里。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import pickle
from typing import Any, Optional

import numpy as np

from shared.cluster_info_contract import assert_cluster_info_contract
from shared.config import (
    EPS,
    EVAL_K,
    FIXED_K,
    NUM_CLUSTERS,
    TARGET_CLUSTER_SIZE,
    WORKSET_CLUSTER_INFO_PATH,
    WORKSET_CORPUS_JSONL_PATH,
    WORKSET_DOCS_PATH,
    WORKSET_DOC_IDS_PATH,
    WORKSET_GT_TOPK_PATH,
    WORKSET_QRELS_PATH,
    WORKSET_QUERIES_PATH,
    WORKSET_QUERY_IDS_PATH,
    WORKSET_RELAXED_QRELS_PATH,
)
from shared.synthetic_doc_access import LazySyntheticTexts, load_doc_ids_or_synthetic


@dataclass(frozen=True)
class SharedClusterOfflineState:
    cluster_info: dict
    cluster_info_contract: dict
    centers: np.ndarray
    cluster_r_k: np.ndarray
    cluster_r_fixed: np.ndarray
    cluster_r_max: np.ndarray
    chunks: list
    overlap_doc_indices_by_cluster: list[np.ndarray]
    top_k: int
    fixed_k: int


@dataclass(frozen=True)
class ClientOfflineState:
    queries: np.ndarray
    query_ids: np.ndarray
    gt_topk: Optional[np.ndarray]
    strict_qrels_map: dict
    relaxed_qrels_map: dict
    reference_metrics_available: bool
    cluster: SharedClusterOfflineState


@dataclass(frozen=True)
class ServerOfflineState:
    docs: np.ndarray
    doc_ids: Any
    doc_texts: Any
    doc_ids_are_synthetic: bool
    cluster: SharedClusterOfflineState


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, float(EPS))
    return (x / norms).astype(np.float32)


def _load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_qrels_tsv(path: str):
    qrels_map = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        if header != ["query_id", "doc_id"]:
            raise ValueError("qrels.tsv header must be query_id\\tdoc_id")
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, did = line.split("\t")
            qrels_map.setdefault(qid, set()).add(did)
    return qrels_map


def _require_existing(paths) -> None:
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing required file: {path}")


def _load_shared_cluster_offline_state() -> SharedClusterOfflineState:
    _require_existing([WORKSET_CLUSTER_INFO_PATH])

    with open(WORKSET_CLUSTER_INFO_PATH, "rb") as f:
        cluster_info = pickle.load(f)

    cluster_info_contract = assert_cluster_info_contract(
        cluster_info,
        expected_eval_k=int(EVAL_K),
        expected_fixed_k=int(FIXED_K),
        expected_num_clusters=int(NUM_CLUSTERS),
        expected_target_cluster_size=int(TARGET_CLUSTER_SIZE),
    )

    centers = np.asarray(cluster_info["centers"], dtype=np.float32)
    cluster_r_k = np.asarray(cluster_info["cluster_r_k"], dtype=np.float32)
    cluster_r_fixed = np.asarray(cluster_info["cluster_r_fixed"], dtype=np.float32)
    cluster_r_max = np.asarray(cluster_info.get("cluster_r_max", []), dtype=np.float32)
    if len(cluster_r_max) != len(centers):
        raise RuntimeError(
            "cluster_info 缺少有效 cluster_r_max，请先重新运行 offline 聚类："
            f"len(cluster_r_max)={len(cluster_r_max)}, len(centers)={len(centers)}"
        )

    chunks = cluster_info["chunks"]
    overlap_doc_indices_by_cluster = [
        np.asarray(x, dtype=np.int32).reshape(-1)
        for x in cluster_info.get("cluster_topc_overlap_doc_indices", chunks)
    ]

    return SharedClusterOfflineState(
        cluster_info=cluster_info,
        cluster_info_contract=cluster_info_contract,
        centers=centers,
        cluster_r_k=cluster_r_k,
        cluster_r_fixed=cluster_r_fixed,
        cluster_r_max=cluster_r_max,
        chunks=chunks,
        overlap_doc_indices_by_cluster=overlap_doc_indices_by_cluster,
        top_k=int(cluster_info["eval_k"]),
        fixed_k=int(cluster_info["fixed_k"]),
    )


def load_online_offline_states(
    *,
    query_limit: int,
    allow_missing_reference: bool,
    allow_synthetic_docid_text_fallback: bool,
) -> tuple[ClientOfflineState, ServerOfflineState]:
    shared_cluster = _load_shared_cluster_offline_state()

    client_required = [
        WORKSET_QUERIES_PATH,
        WORKSET_QUERY_IDS_PATH,
    ]
    server_required = [WORKSET_DOCS_PATH]
    if not bool(allow_synthetic_docid_text_fallback):
        server_required.append(WORKSET_DOC_IDS_PATH)

    reference_required = [
        WORKSET_GT_TOPK_PATH,
        WORKSET_QRELS_PATH,
        WORKSET_RELAXED_QRELS_PATH,
    ]
    if not bool(allow_missing_reference):
        client_required.extend(reference_required)

    _require_existing(client_required)
    _require_existing(server_required)

    docs = np.load(WORKSET_DOCS_PATH, mmap_mode="r")
    doc_ids, doc_ids_are_synthetic = load_doc_ids_or_synthetic(
        WORKSET_DOC_IDS_PATH,
        num_docs=int(docs.shape[0]),
        allow_synthetic=bool(allow_synthetic_docid_text_fallback),
        default_prefix=f"{os.path.basename(WORKSET_DOCS_PATH).replace('.npy', '')}_doc",
    )
    if os.path.exists(WORKSET_CORPUS_JSONL_PATH):
        corpus_rows = _load_jsonl(WORKSET_CORPUS_JSONL_PATH)
        doc_texts = [str(row["text"]) for row in corpus_rows]
    else:
        if bool(doc_ids_are_synthetic):
            doc_texts = LazySyntheticTexts(doc_ids=doc_ids)
        else:
            doc_texts = [str(did) for did in doc_ids.tolist()]

    queries = _normalize_rows(np.load(WORKSET_QUERIES_PATH).astype(np.float32))
    query_ids = np.load(WORKSET_QUERY_IDS_PATH, allow_pickle=True)
    effective_query_limit = int(len(queries)) if int(query_limit) <= 0 else int(min(int(query_limit), len(queries)))
    if effective_query_limit <= 0:
        raise ValueError("no query available")
    queries = queries[:effective_query_limit]
    query_ids = query_ids[:effective_query_limit]

    reference_metrics_available = all(os.path.exists(path) for path in reference_required)
    if reference_metrics_available:
        gt_topk = np.load(WORKSET_GT_TOPK_PATH).astype(np.int32)[:effective_query_limit]
        strict_qrels_map = _load_qrels_tsv(WORKSET_QRELS_PATH)
        relaxed_qrels_map = _load_qrels_tsv(WORKSET_RELAXED_QRELS_PATH)
    else:
        gt_topk = None
        strict_qrels_map = {}
        relaxed_qrels_map = {}

    client_state = ClientOfflineState(
        queries=queries,
        query_ids=query_ids,
        gt_topk=gt_topk,
        strict_qrels_map=strict_qrels_map,
        relaxed_qrels_map=relaxed_qrels_map,
        reference_metrics_available=bool(reference_metrics_available),
        cluster=shared_cluster,
    )
    server_state = ServerOfflineState(
        docs=docs,
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        doc_ids_are_synthetic=bool(doc_ids_are_synthetic),
        cluster=shared_cluster,
    )
    return client_state, server_state
