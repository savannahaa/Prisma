"""
Tiptoe private ranking service with a real HE-backed query flow.

这版实现保留 Tiptoe 的 cluster-local ranking 控制流，同时把“加密查询 -> 服务器内簇打分
-> 客户端解密排序”切到真实 TenSEAL CKKS：
1) 客户端在本地根据公开 cluster centroids 选最近簇；
2) 客户端用 CKKS 加密 query embedding；
3) ranking service 只在目标簇内对明文文档矩阵做密文 matmul；
4) 客户端解密完整簇内分数并导出 top-k doc_ids 供 URL service 使用。

说明：
- 这已经是“真实 HE 请求/响应”路径，不再是之前的量化分数 emulator。
- 它仍然不是论文原始 LHE/Underhood 工程本体，但协议核心已经落到真实密码学后端。
"""

from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import pickle
import numpy as np
from pathlib import Path

from baselines.crypto.tenseal_ckks import (
    build_runtime,
    decrypt_vector_response,
    encrypt_vector_request,
    server_matmul_response,
)
from baselines.tiptoe.common import (
    bundle_paths,
    load_corpus_rows,
    load_qrels,
    normalize_rows,
    normalize_vec,
    save_json,
    write_jsonl,
)
from baselines.tiptoe.config import TiptoeConfig


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _cluster_centers_from_info(cluster_info: dict, docs: np.ndarray) -> np.ndarray:
    centers = cluster_info.get("centers")
    if centers is not None:
        return normalize_rows(np.asarray(centers, dtype=np.float32))
    chunks = [np.asarray(chunk, dtype=np.int32).reshape(-1) for chunk in cluster_info["chunks"]]
    computed = np.stack([np.mean(docs[chunk], axis=0) for chunk in chunks], axis=0)
    return normalize_rows(computed.astype(np.float32))


def _first_relevant_rank(ranked_doc_ids: list[str], relevant: set[str]) -> int:
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if str(doc_id) in relevant:
            return int(rank)
    return 0


def _load_bundle(cfg: TiptoeConfig) -> dict:
    paths = bundle_paths(cfg)
    required = [
        paths["docs"],
        paths["doc_ids"],
        paths["corpus"],
        paths["evaluation_queries"],
        paths["evaluation_query_ids"],
        paths["evaluation_qrels"],
        paths["cluster_info"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Tiptoe aligned bundle assets are missing from upload_ready_code_20260430. "
            "Auto-discovery checks `TIPTOE_BUNDLE_ROOT` first, then common public bundle names, "
            "then a unique matching directory under `results/`. "
            "Please provide the precomputed Tiptoe bundle first. Missing files:\n"
            + "\n".join(missing)
        )
    docs = normalize_rows(np.load(paths["docs"]).astype(np.float32))
    evaluation_queries = normalize_rows(np.load(paths["evaluation_queries"]).astype(np.float32))
    evaluation_query_ids = [str(x) for x in np.load(paths["evaluation_query_ids"], allow_pickle=True).tolist()]
    doc_ids = [str(x) for x in np.load(paths["doc_ids"], allow_pickle=True).tolist()]
    corpus_rows = load_corpus_rows(paths["corpus"])
    qrels = load_qrels(paths["evaluation_qrels"])
    cluster_info = _load_pickle(paths["cluster_info"])
    chunks = [np.asarray(chunk, dtype=np.int32).reshape(-1) for chunk in cluster_info["chunks"]]
    centers = _cluster_centers_from_info(cluster_info, docs)
    return {
        "paths": paths,
        "docs": docs,
        "queries": evaluation_queries,
        "query_ids": evaluation_query_ids,
        "doc_ids": doc_ids,
        "corpus_rows": corpus_rows,
        "qrels": qrels,
        "cluster_info": cluster_info,
        "chunks": chunks,
        "centers": centers,
    }


def _preprocess_manifest(cfg: TiptoeConfig, runtime: dict, he_runtime) -> dict:
    chunks = runtime["chunks"]
    docs = runtime["docs"]
    cluster_doc_counts = [int(len(chunk)) for chunk in chunks]
    cluster_shards = []
    for cluster_id, chunk in enumerate(chunks):
        shard = np.asarray(docs[np.asarray(chunk, dtype=np.int32)], dtype=np.float32)
        cluster_shards.append(
            {
                "cluster_id": int(cluster_id),
                "num_docs": int(shard.shape[0]),
                "embedding_dim": int(shard.shape[1]),
                "server_plaintext_matrix_shape": [int(shard.shape[1]), int(shard.shape[0])],
                "server_plaintext_dtype": "float32",
                "token_independent_preprocessing": True,
            }
        )
    return {
        "bundle_root": str(runtime["paths"]["root"]),
        "routing_c": int(cfg.routing_c),
        "num_clusters": int(len(chunks)),
        "cluster_doc_counts": cluster_doc_counts,
        "cluster_shards": cluster_shards,
        "he_library": "tenseal",
        "he_scheme": "ckks",
        "public_context_bytes_once": int(he_runtime.public_context_bytes),
        "private_context_bytes_once": int(he_runtime.private_context_bytes),
        "setup_time_sec_once": float(he_runtime.setup_time_sec),
        "client_independent_preprocessing": True,
    }


def _rank_queries(
    *,
    cfg: TiptoeConfig,
    runtime: dict,
    top_k: int,
    query_limit: int,
    he_runtime,
) -> tuple[list[dict], dict, dict]:
    docs = runtime["docs"]
    queries = runtime["queries"]
    query_ids = runtime["query_ids"]
    doc_ids = runtime["doc_ids"]
    qrels = runtime["qrels"]
    chunks = runtime["chunks"]
    centers = runtime["centers"]
    corpus_rows = runtime["corpus_rows"]

    if int(query_limit) > 0:
        queries = queries[: int(query_limit)]
        query_ids = query_ids[: int(query_limit)]

    top_k = int(min(int(top_k), int(len(doc_ids))))
    rows: list[dict] = []
    first_ranks: list[int] = []
    top1_hits: list[float] = []
    chosen_cluster_sizes: list[int] = []
    encrypt_times: list[float] = []
    server_times: list[float] = []
    decrypt_times: list[float] = []
    request_bytes: list[int] = []
    response_bytes: list[int] = []

    for query_index, (query, qid) in enumerate(zip(queries, query_ids)):
        query = normalize_vec(query)
        cluster_scores = np.asarray(centers @ query, dtype=np.float64)
        cluster_id = int(np.argmax(cluster_scores))
        cluster_indices = np.asarray(chunks[int(cluster_id)], dtype=np.int32)
        chosen_cluster_sizes.append(int(len(cluster_indices)))
        cluster_doc_matrix = np.asarray(docs[cluster_indices], dtype=np.float64).T

        request = encrypt_vector_request(runtime=he_runtime, vector=query)
        response = server_matmul_response(
            runtime=he_runtime,
            request_blob=request["request_blob"],
            plaintext_matrix=cluster_doc_matrix,
        )
        recovered = decrypt_vector_response(
            runtime=he_runtime,
            response_blob=response["response_blob"],
            expected_length=int(len(cluster_indices)),
        )
        decrypted_scores = np.asarray(recovered["values"], dtype=np.float64)
        order_local = np.argsort(-decrypted_scores, kind="mergesort")[: int(top_k)]
        chosen_global = np.asarray(cluster_indices[order_local], dtype=np.int32)
        ranked_doc_ids = [str(doc_ids[int(idx)]) for idx in chosen_global.tolist()]
        ranked_rows = [corpus_rows[int(idx)] for idx in chosen_global.tolist()]
        relevant = qrels.get(str(qid), set())
        first_rank = _first_relevant_rank(ranked_doc_ids, relevant)
        top1_hit = 1.0 if (ranked_doc_ids and ranked_doc_ids[0] in relevant) else 0.0
        first_ranks.append(int(first_rank) if first_rank > 0 else int(top_k + 1))
        top1_hits.append(float(top1_hit))
        encrypt_times.append(float(request["client_encrypt_sec"]))
        server_times.append(float(response["server_compute_sec"]))
        decrypt_times.append(float(recovered["client_decrypt_sec"]))
        request_bytes.append(int(request["request_bytes"]))
        response_bytes.append(int(response["response_bytes"]))

        rows.append(
            {
                "query_index": int(query_index),
                "query_id": str(qid),
                "chosen_cluster_id": int(cluster_id),
                "chosen_cluster_size": int(len(cluster_indices)),
                "cluster_score_proxy": float(cluster_scores[int(cluster_id)]),
                "encrypted_query_token_bytes": int(request["request_bytes"]),
                "encrypted_scores_bytes": int(response["response_bytes"]),
                "ranking_backend": "tiptoe_tenseal_ckks_cluster_local_encrypted_scoring",
                "time_client_encrypt_query_sec": float(request["client_encrypt_sec"]),
                "time_server_rank_cluster_sec": float(response["server_compute_sec"]),
                "time_client_decrypt_scores_sec": float(recovered["client_decrypt_sec"]),
                "public_context_bytes_once": int(request["public_context_bytes_once"]),
                "top_k": int(top_k),
                "ranked_doc_ids": ranked_doc_ids,
                "ranked_doc_indices": [int(x) for x in chosen_global.tolist()],
                "ranked_source_doc_ids": [str(row.get("source_doc_id", row.get("doc_id", ""))) for row in ranked_rows],
                "ranked_text_preview": [str(row.get("text", ""))[:160] for row in ranked_rows[:5]],
                "first_relevant_rank": int(first_rank),
                "top1_hit": bool(top1_hit > 0.0),
            }
        )

    summary = {
        "module": "Tiptoe Ranking Service",
        "paper_url": str(cfg.paper_url),
        "implementation_note": (
            "Implements Tiptoe's cluster-local ranking service with a real TenSEAL CKKS "
            "encrypted-query / encrypted-score flow over the chosen cluster."
        ),
        "bundle_root": str(runtime["paths"]["root"]),
        "num_docs": int(len(doc_ids)),
        "routing_c": int(cfg.routing_c),
        "num_clusters": int(len(chunks)),
        "num_queries": int(len(rows)),
        "top_k": int(top_k),
        "mean_chosen_cluster_size": float(np.mean(chosen_cluster_sizes)) if chosen_cluster_sizes else 0.0,
        "mean_first_relevant_rank": float(np.mean(first_ranks)) if first_ranks else 0.0,
        "top1_hit_rate": float(np.mean(top1_hits)) if top1_hits else 0.0,
        "mean_encrypted_query_token_bytes": float(np.mean([row["encrypted_query_token_bytes"] for row in rows])) if rows else 0.0,
        "mean_encrypted_scores_bytes": float(np.mean([row["encrypted_scores_bytes"] for row in rows])) if rows else 0.0,
        "time_client_encrypt_query_sec_avg": float(np.mean(encrypt_times)) if encrypt_times else 0.0,
        "time_server_rank_cluster_sec_avg": float(np.mean(server_times)) if server_times else 0.0,
        "time_client_decrypt_scores_sec_avg": float(np.mean(decrypt_times)) if decrypt_times else 0.0,
        "public_context_bytes_once": int(he_runtime.public_context_bytes),
        "private_context_bytes_once": int(he_runtime.private_context_bytes),
        "setup_time_sec_once": float(he_runtime.setup_time_sec),
        "ranking_backend": "tiptoe_tenseal_ckks_cluster_local_encrypted_scoring",
    }
    preprocess = _preprocess_manifest(cfg, runtime, he_runtime)
    return rows, summary, preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tiptoe's private ranking service with real CKKS encrypted scoring.")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k ranking length. <=0 uses config.default_top_k.")
    parser.add_argument("--query-limit", type=int, default=-1, help="Evaluation query limit. <=0 uses config.default_query_limit.")
    parser.add_argument("--output-prefix", type=str, default="", help="Optional output stem override under results/repro_workflows/tiptoe/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = TiptoeConfig(project_root=project_root)
    runtime = _load_bundle(cfg)

    top_k = int(args.top_k) if int(args.top_k) > 0 else int(cfg.default_top_k)
    query_limit = int(args.query_limit) if int(args.query_limit) > 0 else int(cfg.default_query_limit)
    he_runtime = build_runtime(required_slots=max(int(runtime["docs"].shape[1]), int(max(len(chunk) for chunk in runtime["chunks"]))))
    rows, summary, preprocess = _rank_queries(
        cfg=cfg,
        runtime=runtime,
        top_k=int(top_k),
        query_limit=int(query_limit),
        he_runtime=he_runtime,
    )

    stem = str(args.output_prefix).strip() or "tiptoe_ranking_service"
    result_root = project_root / "results" / "repro_workflows" / "tiptoe"
    rows_path = result_root / f"{stem}_rankings.jsonl"
    summary_path = result_root / f"{stem}_summary.json"
    preprocess_path = result_root / f"{stem}_preprocess.json"
    summary["rankings_jsonl"] = str(rows_path)
    summary["summary_json"] = str(summary_path)
    summary["preprocess_json"] = str(preprocess_path)

    write_jsonl(rows_path, rows)
    save_json(summary_path, summary)
    save_json(preprocess_path, preprocess)
    print(f"[saved] {rows_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {preprocess_path}")


if __name__ == "__main__":
    main()
