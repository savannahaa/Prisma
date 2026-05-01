from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import contextlib
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from baselines.public_assets import resolve_workset_assets
from baselines.plaintext_ann.config import PlaintextANNConfig
from shared.config import (
    PAPERFAITHFUL_MAINLINE_AUDIT_CSV,
    PAPERFAITHFUL_MAINLINE_AUDIT_JSON,
    PAPERFAITHFUL_MAINLINE_AUDIT_PKL,
    PAPERFAITHFUL_MAINLINE_OPTIMAL_ALPHA,
    PAPERFAITHFUL_MAINLINE_OPTIMAL_C,
    PAPERFAITHFUL_MAINLINE_OPTIMAL_EPSILON,
    PAPERFAITHFUL_MAINLINE_OPTIMAL_FIXED_K,
    RESULTS_DIR,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_ONLINE_PIPELINE = PROJECT_ROOT / "src" / "client" / "run_online_pipeline.py"
DEFAULT_TOP_K = 5


def _write_csv(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _parse_sizes(text: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value in seen:
            continue
        seen.add(value)
        values.append(int(value))
    return values


def _selection_from_row(row: dict) -> dict:
    return {
        "fixed_k": int(row["selected_fixed_k"]),
        "routing_c": int(row["selected_routing_c"]),
        "epsilon": float(row["selected_epsilon"]),
        "candidate_output_cap": int(row["selected_candidate_output_cap"]),
        "selection_source": str(row.get("selected_source", "unknown")),
    }


def _looks_like_selection_summary(payload: dict) -> bool:
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return False
    required = {
        "num_docs",
        "num_clusters",
        "workset_name",
        "selected_fixed_k",
        "selected_routing_c",
        "selected_epsilon",
        "selected_candidate_output_cap",
    }
    for row in rows:
        if not isinstance(row, dict):
            return False
        if not required.issubset(set(row.keys())):
            return False
    return True


def _resolve_selection_summary_path(
    project_root: Path,
    cfg: PlaintextANNConfig,
    explicit_json: str,
) -> Path:
    if str(explicit_json).strip():
        path = Path(str(explicit_json).strip())
        if not path.exists():
            raise FileNotFoundError(f"selection summary json not found: {path}")
        return path

    result_root = project_root / "results"
    candidates = sorted(
        result_root.glob(str(cfg.default_selection_summary_glob)),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for candidate in candidates:
        try:
            payload = _load_json(candidate)
        except Exception:
            continue
        if _looks_like_selection_summary(payload):
            return candidate
    raise FileNotFoundError(
        "could not auto-resolve a paperfaithful latency-scaling selection summary under "
        f"{result_root} matching {cfg.default_selection_summary_glob}"
    )


def _workset_paths(workset_name: str) -> dict[str, Path]:
    resolved = resolve_workset_assets(PROJECT_ROOT, str(workset_name))
    results_dir = Path(RESULTS_DIR)
    online_results_pattern = sorted(results_dir.glob(f"online_results_{str(workset_name)}_paperfaithful_mainline*.jsonl"))
    online_summary_pattern = sorted(results_dir.glob(f"online_summary_{str(workset_name)}_paperfaithful_mainline*.json"))
    if len(online_results_pattern) > 1:
        raise FileExistsError(
            "ambiguous online_results match for plaintext_ann workset "
            f"{workset_name}: {[str(path) for path in online_results_pattern]}"
        )
    if len(online_summary_pattern) > 1:
        raise FileExistsError(
            "ambiguous online_summary match for plaintext_ann workset "
            f"{workset_name}: {[str(path) for path in online_summary_pattern]}"
        )
    return {
        "workset_name": Path(workset_name),
        "docs": resolved.docs,
        "doc_ids": resolved.doc_ids,
        "meta": resolved.meta,
        "corpus": resolved.corpus,
        "queries": resolved.queries,
        "queries_meta": None,
        "query_ids": resolved.query_ids,
        "gt_topk": resolved.gt_topk,
        "strict_qrels": resolved.strict_qrels,
        "relaxed_qrels": resolved.relaxed_qrels,
        "cluster_info_pkl": resolved.cluster_info_pkl,
        "online_results": online_results_pattern[0] if online_results_pattern else results_dir / f"online_results_{workset_name}_paperfaithful_mainline.jsonl",
        "online_summary": online_summary_pattern[0] if online_summary_pattern else results_dir / f"online_summary_{workset_name}_paperfaithful_mainline.json",
    }


def _effective_c_for_num_clusters(num_clusters: int) -> int:
    return int(max(1, min(int(num_clusters), int(PAPERFAITHFUL_MAINLINE_OPTIMAL_C))))


def _build_env(*, num_docs: int, num_clusters: int, workset_name: str) -> dict[str, str]:
    if int(num_docs) % int(num_clusters) != 0:
        raise ValueError(f"num_docs={num_docs} must be divisible by num_clusters={num_clusters}")
    env = dict(os.environ)
    env["PIPELINE_VARIANT"] = "paperfaithful_mainline"
    env["NUM_WORKSET_DOCS"] = str(int(num_docs))
    env["NUM_CLUSTERS"] = str(int(num_clusters))
    env["TARGET_CLUSTER_SIZE"] = str(int(num_docs) // int(num_clusters))
    env["WORKSET_NAME_OVERRIDE"] = str(workset_name)
    env["EVAL_K"] = str(int(DEFAULT_TOP_K))
    env["FIXED_K"] = str(int(PAPERFAITHFUL_MAINLINE_OPTIMAL_FIXED_K))
    env["ALPHA"] = str(float(PAPERFAITHFUL_MAINLINE_OPTIMAL_ALPHA))
    env["EPSILON"] = str(float(PAPERFAITHFUL_MAINLINE_OPTIMAL_EPSILON))
    env["PAPERFAITHFUL_MAINLINE_GATE_MODE"] = "auto"
    env["ROUTING_CLUSTER_SELECTION_POLICY"] = "soft_topc_fixed"
    env["ROUTING_FIXED_TOP_C"] = str(int(_effective_c_for_num_clusters(int(num_clusters))))
    env["ROUTING_ENABLE_BOUNDARY_MULTICLUSTER"] = "0"
    return env


def _run_python_script(script_path: Path, env: dict[str, str], label: str) -> None:
    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{label} failed with exit code {exc.returncode}") from exc


@contextlib.contextmanager
def _preserve_shared_audit_files():
    temp_root = Path(tempfile.mkdtemp(prefix="pf_latency_audit_backup_"))
    backups: list[tuple[Path, Path]] = []
    shared_paths = [
        Path(PAPERFAITHFUL_MAINLINE_AUDIT_JSON),
        Path(PAPERFAITHFUL_MAINLINE_AUDIT_CSV),
        Path(PAPERFAITHFUL_MAINLINE_AUDIT_PKL),
    ]
    try:
        for path in shared_paths:
            if path.exists():
                backup = temp_root / path.name
                shutil.copy2(path, backup)
                backups.append((path, backup))
        yield
    finally:
        for original, backup in backups:
            if backup.exists():
                shutil.copy2(backup, original)
        shutil.rmtree(temp_root, ignore_errors=True)


def _sanitize_tag(text: str) -> str:
    keep = []
    for ch in str(text).strip():
        if ch.isalnum() or ch in {"_", "-", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("._-")


def _selection_tag(selection: dict) -> str:
    epsilon = float(selection["epsilon"])
    if abs(epsilon - round(epsilon)) <= 1e-9:
        eps_tag = str(int(round(epsilon)))
    else:
        eps_tag = _sanitize_tag(f"{epsilon:g}")
    return f"k{int(selection['fixed_k'])}_eps{eps_tag}_c{int(selection['routing_c'])}"


def _summary_matches_selected_params(
    summary: dict,
    *,
    selection: dict,
    query_limit: int,
    mtimes: dict[str, int],
) -> bool:
    reference = dict(summary.get("reference_selection", {}))
    file_mtimes = dict(summary.get("input_file_mtime_ns", {}))
    return (
        int(summary.get("query_limit_requested", -1)) == int(query_limit)
        and int(reference.get("fixed_k", -1)) == int(selection["fixed_k"])
        and int(reference.get("routing_c", -1)) == int(selection["routing_c"])
        and abs(float(reference.get("epsilon", 0.0)) - float(selection["epsilon"])) <= 1e-9
        and int(file_mtimes.get("docs", -1)) == int(mtimes["docs"])
        and int(file_mtimes.get("queries", -1)) == int(mtimes["queries"])
        and int(file_mtimes.get("corpus", -1)) == int(mtimes["corpus"])
        and int(file_mtimes.get("cluster_info_pkl", -1)) == int(mtimes["cluster_info_pkl"])
    )


def _clean_fixed_budget_summary_path(paths: dict[str, Path], selection: dict) -> Path:
    return (
        Path(RESULTS_DIR)
        / f"clean_fixed_budget_matched_total_{str(paths['workset_name'])}_{_selection_tag(selection)}{PAPERFAITHFUL_SUFFIX}.json"
    )


def _normalize_clean_online_summary(
    *,
    clean_online_summary: dict,
    paths: dict[str, Path],
    selection: dict,
    query_limit: int,
    mtimes: dict[str, int],
) -> dict:
    summary = dict(clean_online_summary)
    time_client_generate = float(summary.get("time_client_generate_query_sec_avg", 0.0))
    time_server_query = float(summary.get("time_server_query_sec_avg", 0.0))
    time_client_recover = float(summary.get("time_client_recover_docs_sec_avg", 0.0))
    comm_request_mb = float(summary.get("comm_client_generate_query_mb_avg", 0.0))
    comm_response_mb = float(summary.get("comm_server_query_mb_avg", 0.0))

    summary.update(
        {
            "baseline_name": "clean_fixed_budget_matched_total",
            "baseline_slug": "plaintext_ann",
            "baseline_display_name": "Non-Private PRISMA",
            "query_limit_requested": int(query_limit),
            "query_limit_effective": int(summary.get("num_queries", 0)),
            "reference_selection": {
                "fixed_k": int(selection["fixed_k"]),
                "routing_c": int(selection["routing_c"]),
                "epsilon": float(selection["epsilon"]),
                "candidate_output_cap": int(selection["candidate_output_cap"]),
                "selection_source": str(selection.get("selection_source", "unknown")),
            },
            "definition": {
                "client_generate_query": "clean query embedding plus the same Top-c routing metadata as Ours; no privacy gate and no perturbation",
                "server_query": "same FAISS HNSW fixed-budget dense backend as Ours under theta*_N, but with clean query embedding",
                "client_recover_docs": "same local exact rerank over the returned candidate payload",
                "gate": "disabled via same-pipeline skip_gate flag",
                "perturbation": "disabled via forced zero perturb radius",
                "routing_policy": "same soft_topc_fixed routing under selected c",
            },
            "cost_reporting_mode": "per_query_avg",
            "cost_stage_definition": {
                "comm_two_stage_total_mb": "true per-query cross-boundary communication total = client->server request bytes + server->client response bytes",
                "comm_client_generate_query_mb": "per-query client->server request bytes for the clean same-pipeline baseline",
                "comm_server_query_mb": "per-query server->client response bytes for the clean same-pipeline baseline",
            },
            "input_paths": {
                "docs": str(paths["docs"]),
                "doc_ids": str(paths["doc_ids"]),
                "queries": str(paths["queries"]),
                "query_ids": str(paths["query_ids"]),
                "gt_topk": str(paths["gt_topk"]),
                "strict_qrels": str(paths["strict_qrels"]),
                "relaxed_qrels": str(paths["relaxed_qrels"]),
                "corpus": str(paths["corpus"]),
                "cluster_info_pkl": str(paths["cluster_info_pkl"]),
            },
            "input_file_mtime_ns": dict(mtimes),
        }
    )
    summary["time_total_sec_sum"] = float(
        float(summary.get("time_client_generate_query_sec_sum", 0.0))
        + float(summary.get("time_server_query_sec_sum", 0.0))
        + float(summary.get("time_client_recover_docs_sec_sum", 0.0))
    )
    summary["time_total_sec_avg"] = float(time_client_generate + time_server_query + time_client_recover)
    summary["time_three_stage_total_sec_sum"] = float(summary["time_total_sec_sum"])
    summary["time_three_stage_total_sec_avg"] = float(summary["time_total_sec_avg"])
    summary["latency_total_sec_avg"] = float(summary["time_total_sec_avg"])
    summary["comm_two_stage_total_mb_sum"] = float(
        float(summary.get("comm_client_generate_query_mb_sum", 0.0))
        + float(summary.get("comm_server_query_mb_sum", 0.0))
    )
    summary["comm_two_stage_total_mb_avg"] = float(comm_request_mb + comm_response_mb)
    summary["comm_request_bytes_avg"] = float(comm_request_mb * 1024.0 * 1024.0)
    summary["comm_response_bytes_avg"] = float(comm_response_mb * 1024.0 * 1024.0)
    summary["comm_downstream_bytes_avg"] = None
    summary.pop("comm_three_stage_total_mb_sum", None)
    summary.pop("comm_three_stage_total_mb_avg", None)
    summary.pop("comm_client_recover_docs_mb_sum", None)
    summary.pop("comm_client_recover_docs_mb_avg", None)
    return summary


def _measure_clean_fixed_budget_matched_total(
    *,
    num_docs: int,
    num_clusters: int,
    workset_name: str,
    paths: dict[str, Path],
    selection: dict,
    query_limit: int,
    reuse_existing: bool,
) -> tuple[dict, Path]:
    summary_path = _clean_fixed_budget_summary_path(paths, selection)
    mtimes = {
        "docs": int(paths["docs"].stat().st_mtime_ns),
        "queries": int(paths["queries"].stat().st_mtime_ns),
        "corpus": int(paths["corpus"].stat().st_mtime_ns),
        "cluster_info_pkl": int(paths["cluster_info_pkl"].stat().st_mtime_ns),
    }
    if reuse_existing and summary_path.exists():
        summary = _load_json(summary_path)
        if _summary_matches_selected_params(
            summary,
            selection=selection,
            query_limit=query_limit,
            mtimes=mtimes,
        ):
            summary = _normalize_clean_online_summary(
                clean_online_summary=summary,
                paths=paths,
                selection=selection,
                query_limit=query_limit,
                mtimes=mtimes,
            )
            _save_json(summary_path, summary)
            return summary, summary_path

    env = _build_env(
        num_docs=int(num_docs),
        num_clusters=int(num_clusters),
        workset_name=str(workset_name),
    )
    env["FIXED_K"] = str(int(selection["fixed_k"]))
    env["EPSILON"] = str(float(selection["epsilon"]))
    env["ROUTING_FIXED_TOP_C"] = str(int(selection["routing_c"]))
    env["PAPERFAITHFUL_MAINLINE_GATE_MODE"] = "force_dense"
    env["PAPERFAITHFUL_MAINLINE_SKIP_GATE"] = "1"
    env["TRACK1_FORCE_PERTURB_R"] = "0"
    env["TRACK1_FORCE_PERTURB_R_CLIP_TO_RMAX"] = "0"
    env["ONLINE_QUERY_LIMIT"] = str(max(0, int(query_limit)))

    summary_backup = paths["online_summary"].read_bytes() if paths["online_summary"].exists() else None
    results_backup = paths["online_results"].read_bytes() if paths["online_results"].exists() else None
    clean_online_summary: dict | None = None
    try:
        with _preserve_shared_audit_files():
            _run_python_script(
                RUN_ONLINE_PIPELINE,
                env,
                f"run_online_pipeline_clean_baseline[{workset_name}]",
            )
        clean_online_summary = _load_json(paths["online_summary"])
    finally:
        if summary_backup is not None:
            paths["online_summary"].write_bytes(summary_backup)
        if results_backup is not None:
            paths["online_results"].write_bytes(results_backup)

    if clean_online_summary is None:
        raise RuntimeError("clean same-pipeline baseline did not produce an online summary")

    routing = dict(clean_online_summary.get("routing_protocol", {}))
    if not (
        int(clean_online_summary.get("num_docs", -1)) == int(num_docs)
        and int(clean_online_summary.get("fixed_k", -1)) == int(selection["fixed_k"])
        and abs(float(clean_online_summary.get("epsilon", 0.0)) - float(selection["epsilon"])) <= 1e-9
        and int(routing.get("fixed_top_c", -1)) == int(selection["routing_c"])
        and bool(clean_online_summary.get("paperfaithful_mainline_skip_gate", False))
    ):
        raise RuntimeError(
            "clean same-pipeline baseline summary does not match the requested selected point: "
            f"{selection}"
        )

    summary = _normalize_clean_online_summary(
        clean_online_summary=clean_online_summary,
        paths=paths,
        selection=selection,
        query_limit=query_limit,
        mtimes=mtimes,
    )
    _save_json(summary_path, summary)
    return summary, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the same-pipeline plaintext ANN baseline summary from a paperfaithful mainline selection summary."
    )
    parser.add_argument(
        "--selection-summary-json",
        type=str,
        default="",
        help="Optional explicit paperfaithful latency-scaling summary json.",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="",
        help="Optional comma-separated size subset. Default: use every size in the selection summary.",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=0,
        help="Optional positive evaluation-query limit override; 0 keeps the full configured evaluation split.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional output stem under results/repro_workflows/plaintext_ann/.",
    )
    parser.set_defaults(reuse_existing=True)
    parser.add_argument(
        "--reuse-existing",
        dest="reuse_existing",
        action="store_true",
        help="Reuse matching clean baseline summaries when available. Enabled by default.",
    )
    parser.add_argument(
        "--no-reuse-existing",
        dest="reuse_existing",
        action="store_false",
        help="Force re-measuring the plaintext ANN baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = PlaintextANNConfig(project_root=project_root)
    selection_summary_path = _resolve_selection_summary_path(
        project_root=project_root,
        cfg=cfg,
        explicit_json=str(args.selection_summary_json),
    )
    selection_summary = _load_json(selection_summary_path)

    selected_sizes = set(_parse_sizes(str(args.sizes))) if str(args.sizes).strip() else None
    selection_rows = [
        dict(row)
        for row in selection_summary.get("rows", [])
        if isinstance(row, dict)
        and (selected_sizes is None or int(row.get("num_docs", -1)) in selected_sizes)
    ]
    if not selection_rows:
        raise RuntimeError(
            "no matching rows found in selection summary: "
            f"{selection_summary_path}"
        )

    result_root = project_root / "results" / "repro_workflows" / cfg.result_root_name
    output_stem = (
        str(args.output_prefix).strip()
        if str(args.output_prefix).strip()
        else str(cfg.default_scaling_summary_stem)
    )
    json_path = result_root / f"{output_stem}.json"
    csv_path = result_root / f"{output_stem}.csv"
    jsonl_path = result_root / f"{output_stem}.jsonl"

    rows: list[dict] = []
    for row in sorted(selection_rows, key=lambda item: int(item["num_docs"])):
        num_docs = int(row["num_docs"])
        num_clusters = int(row["num_clusters"])
        workset_name = str(row["workset_name"])
        selection = _selection_from_row(row)
        paths = _workset_paths(workset_name)
        summary, summary_path = _measure_clean_fixed_budget_matched_total(
            num_docs=int(num_docs),
            num_clusters=int(num_clusters),
            workset_name=str(workset_name),
            paths=paths,
            selection=selection,
            query_limit=int(args.query_limit),
            reuse_existing=bool(args.reuse_existing),
        )
        rows.append(
            {
                "num_docs": int(num_docs),
                "num_clusters": int(num_clusters),
                "workset_name": str(workset_name),
                "selected_fixed_k": int(selection["fixed_k"]),
                "selected_routing_c": int(selection["routing_c"]),
                "selected_epsilon": float(selection["epsilon"]),
                "selected_candidate_output_cap": int(selection["candidate_output_cap"]),
                "selected_source": str(selection.get("selection_source", "")),
                "num_queries": int(summary.get("num_queries", 0)),
                "top_k": int(summary.get("top_k", 0)),
                "avg_exact_recall_at_k": float(summary.get("avg_exact_recall_at_k", 0.0)),
                "time_total_sec_avg": float(summary.get("time_total_sec_avg", 0.0)),
                "time_client_generate_query_sec_avg": float(
                    summary.get("time_client_generate_query_sec_avg", 0.0)
                ),
                "time_server_query_sec_avg": float(
                    summary.get("time_server_query_sec_avg", 0.0)
                ),
                "time_client_recover_docs_sec_avg": float(
                    summary.get("time_client_recover_docs_sec_avg", 0.0)
                ),
                "comm_two_stage_total_mb_avg": float(
                    summary.get("comm_two_stage_total_mb_avg", 0.0)
                ),
                "comm_client_generate_query_mb_avg": float(
                    summary.get("comm_client_generate_query_mb_avg", 0.0)
                ),
                "comm_server_query_mb_avg": float(
                    summary.get("comm_server_query_mb_avg", 0.0)
                ),
                "server_query_est_docs_touched_avg": float(
                    summary.get("server_query_est_docs_touched_avg", 0.0)
                ),
                "summary_json": str(summary_path),
            }
        )

    payload = {
        "baseline_slug": "plaintext_ann",
        "baseline_display_name": str(cfg.baseline_display_name),
        "selection_summary_json": str(selection_summary_path),
        "query_limit_requested": int(args.query_limit),
        "sizes": [int(row["num_docs"]) for row in rows],
        "definition": {
            "client_generate_query": "clean query embedding plus the same Top-c routing metadata as the mainline",
            "server_query": "same FAISS HNSW dense backend as the mainline, but with clean query embedding",
            "client_recover_docs": "same local exact rerank over the returned candidate payload",
            "gate": "disabled",
            "perturbation": "disabled",
            "routing_policy": "same soft_topc_fixed routing under the matched selected c",
        },
        "cost_stage_definition": {
            "comm_two_stage_total_mb_avg": "true per-query cross-boundary communication total = request bytes + response bytes",
            "comm_client_generate_query_mb_avg": "per-query request bytes in MB",
            "comm_server_query_mb_avg": "per-query response bytes in MB",
        },
        "artifacts": {
            "summary_json": str(json_path),
            "summary_csv": str(csv_path),
            "rows_jsonl": str(jsonl_path),
        },
        "rows": rows,
    }
    _save_json(json_path, payload)
    _write_csv(rows, csv_path)
    _write_jsonl(rows, jsonl_path)
    print(f"[saved] {json_path}")
    print(f"[saved] {csv_path}")
    print(f"[saved] {jsonl_path}")


if __name__ == "__main__":
    main()
