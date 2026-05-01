from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from baselines.common import ComparisonContractRow, write_contract_rows_csv, write_contract_rows_jsonl
from baselines.tiptoe.common import bundle_root, normalize_rows
from baselines.tiptoe.config import TiptoeConfig


_SECTION_RE = re.compile(r"Running (embedding|URL) queries \(over (\d+)-doc")
_PREPROC_RE = re.compile(r"Preprocessed query to (\d+)-document corpus in: ([^\s]+)")
_ANSWER_RE = re.compile(r"Answered query to (\d+)-document corpus in: ([^\s]+)")
_UPLOAD_RE = re.compile(r"Upload: ([0-9.]+) MB")
_DOWNLOAD_RE = re.compile(r"Download: ([0-9.]+) MB")
_HINT_RE = re.compile(r"(Embeddings|Urls) hint: ([0-9.]+) MB")
_TOTAL_METADATA_RE = re.compile(r"Total metadata: ([0-9.]+) MB")


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


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _seconds_from_duration(raw: str) -> float:
    value = str(raw).strip()
    if value.endswith("µs"):
        return float(value[:-2]) / 1_000_000.0
    if value.endswith("us"):
        return float(value[:-2]) / 1_000_000.0
    if value.endswith("ms"):
        return float(value[:-2]) / 1_000.0
    if value.endswith("s"):
        return float(value[:-1])
    raise ValueError(f"unsupported duration format: {raw}")


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _resolve_bundle_root(cfg: TiptoeConfig, ranking_summary: dict) -> Path:
    raw = str(ranking_summary.get("bundle_root", "")).strip()
    if raw:
        return Path(raw)
    return bundle_root(cfg)


def _compute_exact_metrics(
    *,
    cfg: TiptoeConfig,
    ranking_summary: dict,
    ranking_summary_path: Path,
) -> dict:
    rows_path = Path(str(ranking_summary.get("rankings_jsonl", "")).strip())
    if not rows_path.is_absolute():
        rows_path = ranking_summary_path.parent / rows_path
    if not rows_path.exists():
        raise FileNotFoundError(f"Tiptoe rankings jsonl missing: {rows_path}")

    rows = _load_jsonl(rows_path)
    if not rows:
        return {
            "avg_exact_recall_at_k": None,
            "exact_topk_order_match_rate": None,
            "candidate_cover_rate": None,
            "real_cluster_hit_rate": None,
        }

    bundle_root = _resolve_bundle_root(cfg, ranking_summary)
    docs_path = bundle_root / "docs.npy"
    doc_ids_path = bundle_root / "doc_ids.npy"
    queries_path = bundle_root / "evaluation_queries.npy"
    query_ids_path = bundle_root / "evaluation_query_ids.npy"
    cluster_info_path = bundle_root / f"cluster_info_c{int(cfg.routing_c)}.pkl"
    for required in [docs_path, doc_ids_path, queries_path, query_ids_path]:
        if not required.exists():
            raise FileNotFoundError(f"Tiptoe bundle asset missing: {required}")

    docs = normalize_rows(np.load(docs_path).astype(np.float32))
    queries = normalize_rows(np.load(queries_path).astype(np.float32))
    doc_ids = [str(x) for x in np.load(doc_ids_path, allow_pickle=True).tolist()]
    query_ids = [str(x) for x in np.load(query_ids_path, allow_pickle=True).tolist()]
    query_lookup = {str(qid): int(index) for index, qid in enumerate(query_ids)}

    chunks: list[np.ndarray] | None = None
    if cluster_info_path.exists():
        cluster_info = _load_pickle(cluster_info_path)
        chunks = [
            np.asarray(chunk, dtype=np.int32).reshape(-1)
            for chunk in cluster_info.get("chunks", [])
        ]

    exact_recall_list: list[float] = []
    order_match_list: list[float] = []
    candidate_cover_list: list[float] = []
    real_cluster_hit_list: list[float] = []

    for row in rows:
        qid = str(row.get("query_id", ""))
        query_index = query_lookup.get(qid)
        if query_index is None:
            continue

        predicted_doc_ids = [str(x) for x in row.get("ranked_doc_ids", [])]
        if not predicted_doc_ids:
            continue

        top_k = min(int(len(predicted_doc_ids)), int(ranking_summary.get("top_k", len(predicted_doc_ids))), int(len(doc_ids)))
        if top_k <= 0:
            continue

        query = np.asarray(queries[int(query_index)], dtype=np.float32)
        scores = np.asarray(docs @ query, dtype=np.float64)
        exact_order = np.argsort(-scores, kind="mergesort")[: int(top_k)]
        exact_doc_ids = [str(doc_ids[int(idx)]) for idx in exact_order.tolist()]
        predicted_doc_ids = predicted_doc_ids[: int(top_k)]

        overlap = len(set(predicted_doc_ids) & set(exact_doc_ids))
        exact_recall_list.append(float(overlap) / float(top_k))
        order_match_list.append(1.0 if predicted_doc_ids == exact_doc_ids else 0.0)

        if chunks is not None:
            cluster_id = row.get("chosen_cluster_id")
            if cluster_id is not None and 0 <= int(cluster_id) < len(chunks):
                cluster_set = {
                    int(x) for x in np.asarray(chunks[int(cluster_id)], dtype=np.int32).reshape(-1).tolist()
                }
                candidate_cover_list.append(
                    1.0 if all(int(idx) in cluster_set for idx in exact_order.tolist()) else 0.0
                )
                real_cluster_hit_list.append(
                    1.0 if int(exact_order[0]) in cluster_set else 0.0
                )

    return {
        "avg_exact_recall_at_k": _mean(exact_recall_list),
        "exact_topk_order_match_rate": _mean(order_match_list),
        "candidate_cover_rate": _mean(candidate_cover_list),
        "real_cluster_hit_rate": _mean(real_cluster_hit_list),
    }


def _ensure_local_tiptoe_outputs(project_root: Path, ranking_summary: Path, url_summary: Path) -> None:
    if not ranking_summary.exists():
        script = project_root / "src" / "baselines" / "tiptoe" / "ranking_service.py"
        subprocess.run([sys.executable, str(script)], cwd=str(project_root), check=True)
    if not url_summary.exists():
        script = project_root / "src" / "baselines" / "tiptoe" / "url_service.py"
        subprocess.run([sys.executable, str(script)], cwd=str(project_root), check=True)


def _ensure_official_log(project_root: Path, log_path: Path) -> None:
    if log_path.exists():
        return
    raise FileNotFoundError(
        f"official Tiptoe log missing: {log_path}. "
        "The upload-ready pack does not bundle the original author fake-corpus wrapper; "
        "please pass --official-log explicitly if you want to normalize an official run."
    )


def build_local_rows(
    *,
    cfg: TiptoeConfig,
    ranking_summary_path: Path,
    url_summary_path: Path,
) -> tuple[list[ComparisonContractRow], dict]:
    ranking = _load_json(ranking_summary_path)
    url = _load_json(url_summary_path)
    exact_metrics = _compute_exact_metrics(
        cfg=cfg,
        ranking_summary=ranking,
        ranking_summary_path=ranking_summary_path,
    )
    latency_client_generate = float(ranking.get("time_client_encrypt_query_sec_avg", 0.0)) + float(
        url.get("time_client_build_pir_sec_avg", 0.0)
    )
    latency_server_query = float(ranking.get("time_server_rank_cluster_sec_avg", 0.0)) + float(
        url.get("time_server_process_pir_sec_avg", 0.0)
    )
    latency_client_recover = float(ranking.get("time_client_decrypt_scores_sec_avg", 0.0)) + float(
        url.get("time_client_recover_pir_sec_avg", 0.0)
    )
    latency_total = latency_client_generate + latency_server_query + latency_client_recover
    request_bytes_avg = float(ranking.get("mean_encrypted_query_token_bytes", 0.0)) + float(
        url.get("mean_pir_request_bytes", 0.0)
    )
    response_bytes_avg = float(ranking.get("mean_encrypted_scores_bytes", 0.0))
    downstream_bytes_avg = float(url.get("mean_pir_response_bytes", 0.0))
    row = ComparisonContractRow(
        baseline_slug="tiptoe",
        baseline_display_name="Tiptoe",
        paper_url=str(cfg.paper_url),
        contract_version="v1",
        comparison_axis="qrels_bundle",
        run_label=str(ranking_summary_path.stem).replace("_summary", ""),
        num_docs=int(ranking.get("num_docs", 0)) if ranking.get("num_docs") is not None else None,
        num_clusters=int(ranking.get("num_clusters", cfg.routing_c))
        if ranking.get("num_clusters") is not None
        else int(cfg.routing_c),
        num_queries=int(ranking.get("num_queries", 0)),
        top_k=int(ranking.get("top_k", 0)),
        latency_total_sec_avg=float(latency_total),
        latency_client_generate_sec_avg=float(latency_client_generate),
        latency_server_query_sec_avg=float(latency_server_query),
        latency_client_recover_sec_avg=float(latency_client_recover),
        comm_request_bytes_avg=float(request_bytes_avg),
        comm_response_bytes_avg=float(response_bytes_avg),
        comm_downstream_bytes_avg=float(downstream_bytes_avg),
        mean_first_relevant_rank=float(ranking.get("mean_first_relevant_rank", 0.0)),
        top1_hit_rate=float(ranking.get("top1_hit_rate", 0.0)),
        exact_topk_overlap_mean=exact_metrics["avg_exact_recall_at_k"],
        exact_topk_order_match_rate=exact_metrics["exact_topk_order_match_rate"],
        candidate_cover_rate=exact_metrics["candidate_cover_rate"],
        direct_retrieve_rate=1.0,
        ot_retrieve_rate=0.0,
        real_cluster_hit_rate=exact_metrics["real_cluster_hit_rate"],
        source_summary_json=str(ranking_summary_path),
        source_rows_jsonl=str(ranking.get("rankings_jsonl", "")),
        notes=(
            "Tiptoe row folds ranking-service CKKS and URL-service CKKS into the shared three-stage latency/communication contract.",
            "exact_topk_overlap_mean stores exact Recall@k against the shared bundle's global exact top-k.",
            f"URL summary source: {str(url_summary_path)}",
        ),
    )
    summary = {
        "mode": "local_comparison_path",
        "ranking_summary_json": str(ranking_summary_path),
        "url_summary_json": str(url_summary_path),
        "comparison_axis": "qrels_bundle",
        "latency_total_sec_avg": float(latency_total),
        "latency_client_generate_sec_avg": float(latency_client_generate),
        "latency_server_query_sec_avg": float(latency_server_query),
        "latency_client_recover_sec_avg": float(latency_client_recover),
        "request_bytes_avg": float(request_bytes_avg),
        "response_bytes_avg": float(response_bytes_avg),
        "downstream_bytes_avg": float(downstream_bytes_avg),
        "avg_exact_recall_at_k": exact_metrics["avg_exact_recall_at_k"],
        "exact_topk_order_match_rate": exact_metrics["exact_topk_order_match_rate"],
        "candidate_cover_rate": exact_metrics["candidate_cover_rate"],
        "real_cluster_hit_rate": exact_metrics["real_cluster_hit_rate"],
    }
    return [row], summary


def build_official_rows(
    *,
    cfg: TiptoeConfig,
    log_path: Path,
) -> tuple[list[ComparisonContractRow], dict]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    sections: dict[str, dict[str, list[float] | float | int | None]] = {
        "embedding": {
            "num_docs": None,
            "preproc_sec": [],
            "answer_sec": [],
            "offline_upload_mb": [],
            "offline_download_mb": [],
            "online_upload_mb": [],
            "online_download_mb": [],
            "hint_mb": [],
            "metadata_mb": [],
        },
        "url": {
            "num_docs": None,
            "preproc_sec": [],
            "answer_sec": [],
            "offline_upload_mb": [],
            "offline_download_mb": [],
            "online_upload_mb": [],
            "online_download_mb": [],
            "hint_mb": [],
            "metadata_mb": [],
        },
    }

    current: str | None = None
    pending_phase: tuple[str, str] | None = None
    pass_seen = False

    for raw_line in lines:
        line = str(raw_line).strip()
        if not line:
            continue
        if line == "PASS":
            pass_seen = True
        m = _SECTION_RE.search(line)
        if m:
            current = "embedding" if str(m.group(1)).lower() == "embedding" else "url"
            sections[current]["num_docs"] = int(m.group(2))
            pending_phase = None
            continue
        if current is None:
            continue

        m = _PREPROC_RE.search(line)
        if m:
            sections[current]["num_docs"] = int(m.group(1))
            cast = sections[current]["preproc_sec"]
            assert isinstance(cast, list)
            cast.append(_seconds_from_duration(m.group(2)))
            pending_phase = (current, "offline")
            continue

        m = _ANSWER_RE.search(line)
        if m:
            sections[current]["num_docs"] = int(m.group(1))
            cast = sections[current]["answer_sec"]
            assert isinstance(cast, list)
            cast.append(_seconds_from_duration(m.group(2)))
            pending_phase = (current, "online")
            continue

        m = _UPLOAD_RE.search(line)
        if m and pending_phase is not None:
            section_name, phase = pending_phase
            bucket = "offline_upload_mb" if phase == "offline" else "online_upload_mb"
            cast = sections[section_name][bucket]
            assert isinstance(cast, list)
            cast.append(float(m.group(1)))
            continue

        m = _DOWNLOAD_RE.search(line)
        if m and pending_phase is not None:
            section_name, phase = pending_phase
            bucket = "offline_download_mb" if phase == "offline" else "online_download_mb"
            cast = sections[section_name][bucket]
            assert isinstance(cast, list)
            cast.append(float(m.group(1)))
            continue

        m = _HINT_RE.search(line)
        if m:
            section_name = "embedding" if str(m.group(1)).lower().startswith("embedding") else "url"
            cast = sections[section_name]["hint_mb"]
            assert isinstance(cast, list)
            cast.append(float(m.group(2)))
            continue

        m = _TOTAL_METADATA_RE.search(line)
        if m:
            cast = sections[current]["metadata_mb"]
            assert isinstance(cast, list)
            cast.append(float(m.group(1)))
            continue

    embedding = sections["embedding"]
    url = sections["url"]
    emb_preproc = embedding["preproc_sec"]
    emb_answer = embedding["answer_sec"]
    url_preproc = url["preproc_sec"]
    url_answer = url["answer_sec"]
    assert isinstance(emb_preproc, list)
    assert isinstance(emb_answer, list)
    assert isinstance(url_preproc, list)
    assert isinstance(url_answer, list)

    num_queries = int(min(len(emb_preproc), len(emb_answer), len(url_preproc), len(url_answer)))
    if int(num_queries) <= 0:
        raise RuntimeError(f"could not parse any official Tiptoe query pairs from {log_path}")

    emb_offline_up = embedding["offline_upload_mb"]
    emb_offline_down = embedding["offline_download_mb"]
    emb_online_up = embedding["online_upload_mb"]
    emb_online_down = embedding["online_download_mb"]
    url_offline_up = url["offline_upload_mb"]
    url_offline_down = url["offline_download_mb"]
    url_online_up = url["online_upload_mb"]
    url_online_down = url["online_download_mb"]
    assert isinstance(emb_offline_up, list)
    assert isinstance(emb_offline_down, list)
    assert isinstance(emb_online_up, list)
    assert isinstance(emb_online_down, list)
    assert isinstance(url_offline_up, list)
    assert isinstance(url_offline_down, list)
    assert isinstance(url_online_up, list)
    assert isinstance(url_online_down, list)

    latency_total = [
        emb_preproc[i] + emb_answer[i] + url_preproc[i] + url_answer[i]
        for i in range(num_queries)
    ]
    request_mb = [
        emb_offline_up[i] + emb_online_up[i]
        for i in range(min(num_queries, len(emb_offline_up), len(emb_online_up)))
    ]
    response_mb = [
        emb_offline_down[i] + emb_online_down[i]
        for i in range(min(num_queries, len(emb_offline_down), len(emb_online_down)))
    ]
    downstream_mb = [
        url_offline_up[i] + url_offline_down[i] + url_online_up[i] + url_online_down[i]
        for i in range(min(num_queries, len(url_offline_up), len(url_offline_down), len(url_online_up), len(url_online_down)))
    ]

    row = ComparisonContractRow(
        baseline_slug="tiptoe",
        baseline_display_name="Tiptoe",
        paper_url=str(cfg.paper_url),
        contract_version="v1",
        comparison_axis="single_run",
        run_label="tiptoe_official_fake_corpus",
        num_docs=int(embedding["num_docs"]) if embedding["num_docs"] is not None else None,
        num_clusters=None,
        num_queries=int(num_queries),
        top_k=None,
        latency_total_sec_avg=_mean(latency_total),
        latency_client_generate_sec_avg=_mean(emb_preproc[:num_queries]) + _mean(url_preproc[:num_queries]) if _mean(emb_preproc[:num_queries]) is not None and _mean(url_preproc[:num_queries]) is not None else None,
        latency_server_query_sec_avg=_mean(emb_answer[:num_queries]),
        latency_client_recover_sec_avg=_mean(url_answer[:num_queries]),
        comm_request_bytes_avg=(_mean(request_mb) or 0.0) * 1024.0 * 1024.0,
        comm_response_bytes_avg=(_mean(response_mb) or 0.0) * 1024.0 * 1024.0,
        comm_downstream_bytes_avg=(_mean(downstream_mb) or 0.0) * 1024.0 * 1024.0,
        mean_first_relevant_rank=None,
        top1_hit_rate=None,
        exact_topk_overlap_mean=None,
        exact_topk_order_match_rate=None,
        candidate_cover_rate=None,
        direct_retrieve_rate=1.0 if pass_seen else None,
        ot_retrieve_rate=0.0 if pass_seen else None,
        real_cluster_hit_rate=None,
        source_summary_json="",
        source_rows_jsonl=str(log_path),
        notes=(
            "Official Tiptoe row parsed from the author fake-corpus end-to-end correctness log.",
            "Request/response bytes map embedding-stage communication; downstream bytes map URL-stage communication.",
            f"PASS observed: {pass_seen}",
        ),
    )
    summary = {
        "mode": "official_author_log",
        "paper_url": str(cfg.paper_url),
        "official_log": str(log_path),
        "pass_seen": bool(pass_seen),
        "num_queries": int(num_queries),
        "num_docs": int(embedding["num_docs"]) if embedding["num_docs"] is not None else None,
        "embedding_hint_mb_max": max(embedding["hint_mb"]) if embedding["hint_mb"] else 0.0,
        "url_hint_mb_max": max(url["hint_mb"]) if url["hint_mb"] else 0.0,
        "embedding_metadata_mb_max": max(embedding["metadata_mb"]) if embedding["metadata_mb"] else 0.0,
        "url_metadata_mb_max": max(url["metadata_mb"]) if url["metadata_mb"] else 0.0,
        "latency_total_sec_avg": _mean(latency_total),
        "latency_embedding_preproc_sec_avg": _mean(emb_preproc[:num_queries]),
        "latency_embedding_answer_sec_avg": _mean(emb_answer[:num_queries]),
        "latency_url_preproc_sec_avg": _mean(url_preproc[:num_queries]),
        "latency_url_answer_sec_avg": _mean(url_answer[:num_queries]),
        "comm_embedding_request_mb_avg": _mean(request_mb),
        "comm_embedding_response_mb_avg": _mean(response_mb),
        "comm_url_total_mb_avg": _mean(downstream_mb),
    }
    return [row], summary


def build_rows(
    *,
    cfg: TiptoeConfig,
    ranking_summary_path: Path,
    url_summary_path: Path,
) -> list[ComparisonContractRow]:
    rows, _summary = build_local_rows(
        cfg=cfg,
        ranking_summary_path=ranking_summary_path,
        url_summary_path=url_summary_path,
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Tiptoe outputs into the shared comparison contract.")
    parser.add_argument("--mode", type=str, default="local", choices=["local", "official"], help="Use the in-repo comparison path or the official fake-corpus log path.")
    parser.add_argument("--ranking-stem", type=str, default="tiptoe_ranking_service", help="Tiptoe ranking stem under results/repro_workflows/tiptoe/.")
    parser.add_argument("--url-stem", type=str, default="tiptoe_url_service", help="Tiptoe URL-service stem under results/repro_workflows/tiptoe/.")
    parser.add_argument("--official-log", type=str, default="", help="Optional path to the official Tiptoe fake-corpus log.")
    parser.add_argument("--output-prefix", type=str, default="tiptoe_comparison_contract", help="Output stem under results/repro_workflows/tiptoe/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = TiptoeConfig(project_root=project_root)
    result_root = project_root / "results" / "repro_workflows" / "tiptoe"

    if str(args.mode) == "official":
        log_path = (
            Path(args.official_log)
            if str(args.official_log).strip()
            else result_root / "tiptoe_official_fake_corpus.log"
        )
        _ensure_official_log(project_root, log_path)
        rows, summary = build_official_rows(cfg=cfg, log_path=log_path)
    else:
        ranking_summary_path = result_root / f"{str(args.ranking_stem).strip()}_summary.json"
        url_summary_path = result_root / f"{str(args.url_stem).strip()}_summary.json"
        _ensure_local_tiptoe_outputs(project_root, ranking_summary_path, url_summary_path)
        rows, summary = build_local_rows(
            cfg=cfg,
            ranking_summary_path=ranking_summary_path,
            url_summary_path=url_summary_path,
        )

    prefix = str(args.output_prefix).strip()
    jsonl_path = result_root / f"{prefix}.jsonl"
    csv_path = result_root / f"{prefix}.csv"
    summary_path = result_root / f"{prefix}_summary.json"
    write_contract_rows_jsonl(jsonl_path, rows)
    write_contract_rows_csv(csv_path, rows)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[saved] {jsonl_path}")
    print(f"[saved] {csv_path}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
