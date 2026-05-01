"""
Tiptoe URL service with a real HE-backed batch retrieval path.

这版实现保留 Tiptoe 的 URL retrieval 控制流，同时把单 batch 取回切到真实 CKKS：
1) 读取 ranking service 的 top-k doc IDs；
2) 为每个文档构造 metadata payload（这里用 synthetic URL + text snippet）；
3) 客户端对 batch selector 做 CKKS 加密，请求目标 batch；
4) 服务器在密文上返回加密 batch payload indices，客户端解密并恢复 URLs。

说明：
- 当前语料没有真实 URL，因此继续用 `tiptoe://msmarco/<source_doc_id>` 作为可追踪替代 URL。
- 这不是论文中的专用 PIR 实现，但已经是“真实加密 batch retrieval”而非明文模拟。
"""

from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from baselines.crypto.tenseal_ckks import (
    build_runtime,
    decrypt_rounded_int_response,
    encrypt_vector_request,
    server_matmul_response,
)
from baselines.tiptoe.common import bundle_paths, load_corpus_rows, save_json, write_jsonl
from baselines.tiptoe.config import TiptoeConfig


_MB = 1024.0 * 1024.0


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ranking_stem() -> str:
    return "tiptoe_ranking_service"


def _ensure_ranking_outputs(project_root: Path, ranking_rows_path: Path, ranking_summary_path: Path) -> None:
    if ranking_rows_path.exists() and ranking_summary_path.exists():
        return
    raise FileNotFoundError(
        "Tiptoe ranking outputs are missing, and ranking_service.py was removed from "
        "upload_ready_code_20260430. Please provide the precomputed ranking outputs first."
    )


def _synthetic_url(source_doc_id: str) -> str:
    return f"tiptoe://msmarco/{str(source_doc_id)}"


def _group_batches(payloads: list[dict], group_size: int) -> tuple[dict[str, int], list[list[dict]]]:
    doc_to_batch: dict[str, int] = {}
    batches: list[list[dict]] = []
    current: list[dict] = []
    batch_id = 0
    for payload in payloads:
        current.append(payload)
        doc_to_batch[str(payload["doc_id"])] = int(batch_id)
        if len(current) >= int(group_size):
            batches.append(current)
            current = []
            batch_id += 1
    if current:
        batches.append(current)
    return doc_to_batch, batches


def _build_payload_table(cfg: TiptoeConfig) -> tuple[list[dict], dict[str, int], list[list[dict]]]:
    paths = bundle_paths(cfg)
    corpus_rows = load_corpus_rows(paths["corpus"])
    payloads: list[dict] = []
    for row in corpus_rows:
        doc_id = str(row.get("doc_id", ""))
        source_doc_id = str(row.get("source_doc_id", doc_id))
        text = str(row.get("text", ""))
        payloads.append(
            {
                "doc_id": str(doc_id),
                "source_doc_id": str(source_doc_id),
                "url": _synthetic_url(source_doc_id),
                "snippet": text[:200],
                "payload_bytes": len(text[:200].encode("utf-8")) + len(source_doc_id.encode("utf-8")) + 24,
            }
        )
    doc_to_batch, batches = _group_batches(payloads, int(cfg.default_url_group_size))
    return payloads, doc_to_batch, batches


def _build_batch_index_matrix(batches: list[list[dict]], group_size: int) -> tuple[np.ndarray, list[int]]:
    matrix_rows: list[list[float]] = []
    valid_counts: list[int] = []
    for batch in batches:
        batch_indices = [int(item["_payload_index"]) for item in batch]
        valid_count = int(len(batch_indices))
        pad_value = int(batch_indices[0]) if batch_indices else 0
        padded = batch_indices + [pad_value] * max(0, int(group_size) - valid_count)
        matrix_rows.append([float(x) for x in padded[: int(group_size)]])
        valid_counts.append(int(valid_count))
    return np.asarray(matrix_rows, dtype=np.float64), valid_counts


def _run_url_service(
    *,
    cfg: TiptoeConfig,
    ranking_rows_path: Path,
    ranking_summary_path: Path,
) -> tuple[list[dict], dict, dict]:
    ranking_rows = _load_jsonl(ranking_rows_path)
    ranking_summary = _load_json(ranking_summary_path)
    payloads, doc_to_batch, batches = _build_payload_table(cfg)
    payloads = [{**row, "_payload_index": int(index)} for index, row in enumerate(payloads)]
    payload_by_doc_id = {str(row["doc_id"]): row for row in payloads}
    indexed_batches = [
        [payload_by_doc_id[str(item["doc_id"])] for item in batch]
        for batch in batches
    ]
    batch_index_matrix, valid_counts = _build_batch_index_matrix(indexed_batches, int(cfg.default_url_group_size))
    he_runtime = build_runtime(required_slots=max(int(len(batches)), int(cfg.default_url_group_size)))

    rows: list[dict] = []
    pir_batches_used: list[int] = []
    pir_response_bytes: list[int] = []
    request_bytes: list[int] = []
    request_times: list[float] = []
    server_times: list[float] = []
    decrypt_times: list[float] = []
    latency_total_sec_list: list[float] = []
    comm_total_bytes_list: list[int] = []
    max_round_errors: list[float] = []

    for row in ranking_rows:
        ranked_doc_ids = [str(x) for x in row["ranked_doc_ids"]]
        if not ranked_doc_ids:
            continue
        anchor_doc_id = str(ranked_doc_ids[0])
        batch_id = int(doc_to_batch.get(anchor_doc_id, 0))
        batch_payload = list(indexed_batches[int(batch_id)])
        selector = np.zeros(int(len(batches)), dtype=np.float64)
        selector[int(batch_id)] = 1.0
        request = encrypt_vector_request(runtime=he_runtime, vector=selector)
        response = server_matmul_response(
            runtime=he_runtime,
            request_blob=request["request_blob"],
            plaintext_matrix=batch_index_matrix,
        )
        recovered_response = decrypt_rounded_int_response(
            runtime=he_runtime,
            response_blob=response["response_blob"],
            expected_length=int(cfg.default_url_group_size),
        )
        recovered_indices = [
            int(x)
            for x in recovered_response["rounded"][: int(valid_counts[int(batch_id)])].tolist()
        ]
        he_batch_payload = [payloads[int(idx)] for idx in recovered_indices]
        he_lookup = {str(item["doc_id"]): item for item in he_batch_payload}
        recovered = [
            he_lookup[doc_id]
            for doc_id in ranked_doc_ids
            if doc_id in he_lookup
        ]
        recovered_urls = [str(item["url"]) for item in recovered]
        latency_total_sec = float(
            request["client_encrypt_sec"]
            + response["server_compute_sec"]
            + recovered_response["client_decrypt_sec"]
        )
        comm_total_bytes = int(request["request_bytes"]) + int(response["response_bytes"])
        rows.append(
            {
                "query_index": int(row["query_index"]),
                "query_id": str(row["query_id"]),
                "anchor_doc_id": str(anchor_doc_id),
                "pir_batch_id": int(batch_id),
                "pir_request_bytes": int(request["request_bytes"]),
                "pir_response_bytes": int(response["response_bytes"]),
                "time_client_build_pir_sec": float(request["client_encrypt_sec"]),
                "time_server_process_pir_sec": float(response["server_compute_sec"]),
                "time_client_recover_pir_sec": float(recovered_response["client_decrypt_sec"]),
                "latency_total_sec": latency_total_sec,
                "batch_doc_count": int(len(batch_payload)),
                "comm_request_bytes_total": int(request["request_bytes"]),
                "comm_response_bytes_total": int(response["response_bytes"]),
                "comm_total_bytes": int(comm_total_bytes),
                "recovered_doc_ids": [str(item["doc_id"]) for item in recovered],
                "recovered_urls": recovered_urls,
                "recovered_snippets": [str(item["snippet"]) for item in recovered[:5]],
                "pir_backend": "tiptoe_tenseal_ckks_batch_selection",
                "decrypt_round_max_abs_error": float(recovered_response["max_abs_error"]),
            }
        )
        pir_batches_used.append(int(batch_id))
        pir_response_bytes.append(int(response["response_bytes"]))
        request_bytes.append(int(request["request_bytes"]))
        request_times.append(float(request["client_encrypt_sec"]))
        server_times.append(float(response["server_compute_sec"]))
        decrypt_times.append(float(recovered_response["client_decrypt_sec"]))
        latency_total_sec_list.append(float(latency_total_sec))
        comm_total_bytes_list.append(int(comm_total_bytes))
        max_round_errors.append(float(recovered_response["max_abs_error"]))

    batch_manifest = {
        "url_group_size": int(cfg.default_url_group_size),
        "num_payloads": int(len(payloads)),
        "num_batches": int(len(batches)),
        "batch_doc_counts": [int(len(batch)) for batch in batches],
    }
    summary = {
        "module": "Tiptoe URL Service",
        "paper_url": str(cfg.paper_url),
        "implementation_note": (
            "Implements Tiptoe's URL retrieval path as a real TenSEAL CKKS encrypted "
            "batch-selection flow over synthetic URL payload indices."
        ),
        "ranking_rows_jsonl": str(ranking_rows_path),
        "ranking_summary_json": str(ranking_summary_path),
        "num_docs": int(len(payloads)),
        "num_queries": int(len(rows)),
        "url_group_size": int(cfg.default_url_group_size),
        "cost_reporting_mode": "per_query_avg",
        "latency_total_sec_avg": float(np.mean(latency_total_sec_list)) if latency_total_sec_list else 0.0,
        "latency_client_generate_sec_avg": float(np.mean(request_times)) if request_times else 0.0,
        "latency_server_query_sec_avg": float(np.mean(server_times)) if server_times else 0.0,
        "latency_client_recover_sec_avg": float(np.mean(decrypt_times)) if decrypt_times else 0.0,
        "comm_request_bytes_avg": float(np.mean(request_bytes)) if request_bytes else 0.0,
        "comm_response_bytes_avg": float(np.mean(pir_response_bytes)) if pir_response_bytes else 0.0,
        "comm_client_generate_query_mb": (float(np.mean(request_bytes)) / _MB) if request_bytes else 0.0,
        "comm_server_query_mb": (float(np.mean(pir_response_bytes)) / _MB) if pir_response_bytes else 0.0,
        "comm_two_stage_total_mb": (float(np.mean(comm_total_bytes_list)) / _MB) if comm_total_bytes_list else 0.0,
        "time_client_build_pir_sec_avg": float(np.mean(request_times)) if request_times else 0.0,
        "time_server_process_pir_sec_avg": float(np.mean(server_times)) if server_times else 0.0,
        "time_client_recover_pir_sec_avg": float(np.mean(decrypt_times)) if decrypt_times else 0.0,
        "decrypt_round_max_abs_error_max": float(np.max(max_round_errors)) if max_round_errors else 0.0,
        "unique_pir_batches_touched": int(len(set(pir_batches_used))),
        "num_batches_total": int(len(batches)),
        "ranking_top_k": int(ranking_summary.get("top_k", 0)),
        "cost_stage_definition": {
            "latency_total_sec_avg": "per-query end-to-end URL retrieval latency = client PIR request build + server PIR processing + client PIR recovery; excludes upstream ranking stage.",
            "comm_request_bytes_avg": "per-query true client->server online request bytes for the URL-retrieval PIR stage.",
            "comm_response_bytes_avg": "per-query true server->client online response bytes for the URL-retrieval PIR stage.",
            "comm_two_stage_total_mb": "per-query cross-boundary communication total = request bytes + response bytes for the URL-retrieval PIR stage.",
        },
        "public_context_bytes_once": int(he_runtime.public_context_bytes),
        "private_context_bytes_once": int(he_runtime.private_context_bytes),
        "setup_time_sec_once": float(he_runtime.setup_time_sec),
        "pir_backend": "tiptoe_tenseal_ckks_batch_selection",
    }
    return rows, summary, batch_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tiptoe's URL service with real CKKS batch retrieval on ranking outputs.")
    parser.add_argument("--ranking-output-prefix", type=str, default="", help="Optional ranking output stem. Default uses tiptoe_ranking_service.")
    parser.add_argument("--output-prefix", type=str, default="", help="Optional URL-service output stem override under results/repro_workflows/tiptoe/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = TiptoeConfig(project_root=project_root)
    ranking_stem = str(args.ranking_output_prefix).strip() or _ranking_stem()
    result_root = project_root / "results" / "repro_workflows" / "tiptoe"
    ranking_rows_path = result_root / f"{ranking_stem}_rankings.jsonl"
    ranking_summary_path = result_root / f"{ranking_stem}_summary.json"
    _ensure_ranking_outputs(project_root, ranking_rows_path, ranking_summary_path)

    rows, summary, batch_manifest = _run_url_service(
        cfg=cfg,
        ranking_rows_path=ranking_rows_path,
        ranking_summary_path=ranking_summary_path,
    )

    stem = str(args.output_prefix).strip() or "tiptoe_url_service"
    rows_path = result_root / f"{stem}_payloads.jsonl"
    summary_path = result_root / f"{stem}_summary.json"
    batch_manifest_path = result_root / f"{stem}_batches.json"
    summary["payloads_jsonl"] = str(rows_path)
    summary["summary_json"] = str(summary_path)
    summary["batch_manifest_json"] = str(batch_manifest_path)

    write_jsonl(rows_path, rows)
    save_json(summary_path, summary)
    save_json(batch_manifest_path, batch_manifest)
    print(f"[saved] {rows_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {batch_manifest_path}")


if __name__ == "__main__":
    main()
