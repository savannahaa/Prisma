"""
Build local-shard-constrained nested Cohere MSMARCO v2.1 offset manifests.

This is the latency-only preparation path:
- the real 10M base is restricted to the contiguous locally available shard prefix
- 10K / 100K / 1M / 10M remain exact nested subsets
- we also emit an eligible query pool plus exact per-size top1k hit counts
"""

from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARD_META_PATH = (
    PROJECT_ROOT / "data" / "external" / "cohere_msmarco_v21" / "passages_shard_meta.json"
)
DEFAULT_LOCAL_SHARDS_ROOT = (
    PROJECT_ROOT / "data" / "external" / "cohere_msmarco_v21" / "passages_npy"
)

DEFAULT_SIZES = [10_000, 100_000, 1_000_000, 10_000_000]
DEFAULT_COVERAGE_RANKS = {
    10_000: 5,
    100_000: 20,
    1_000_000: 100,
    10_000_000: 100,
}
DEFAULT_SUPPORT_THRESHOLDS = [1, 5, 10, 20, 100]


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = str(line).strip()
            if text:
                rows.append(json.loads(text))
    return rows


def save_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_sizes(text: str) -> list[int]:
    vals = []
    for part in str(text).replace(";", ",").split(","):
        item = str(part).strip()
        if not item:
            continue
        vals.append(int(float(item)))
    out = sorted(set(vals))
    if not out:
        raise ValueError("sizes must not be empty")
    return out


def _parse_coverage_map(text: str, sizes: list[int]) -> dict[int, int]:
    mapping = {int(k): int(v) for k, v in DEFAULT_COVERAGE_RANKS.items()}
    raw = str(text).strip()
    if raw:
        mapping = {}
        for part in raw.replace(";", ",").split(","):
            item = str(part).strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(
                    f"invalid coverage map item '{item}', expected format like 10000:5"
                )
            left, right = item.split(":", 1)
            mapping[int(float(left))] = int(float(right))
    last = 5
    out: dict[int, int] = {}
    for size in sorted(sizes):
        if int(size) in mapping:
            last = int(mapping[int(size)])
        out[int(size)] = int(max(1, last))
    return out


def _parse_thresholds(text: str) -> list[int]:
    vals = []
    for part in str(text).replace(";", ",").split(","):
        item = str(part).strip()
        if not item:
            continue
        vals.append(int(float(item)))
    out = sorted(set(int(x) for x in vals if int(x) > 0))
    if not out:
        raise ValueError("support thresholds must not be empty")
    return out


def _query_id_from_row(row: dict) -> str:
    return str(
        row.get("query_id", row.get("raw_query_id", row.get("id", row.get("_id", ""))))
    ).strip()


def _resolve_local_path(shard_row: dict, local_root: Path) -> Path:
    path_text = str(shard_row.get("local_path", "")).strip()
    if path_text:
        return Path(path_text)
    file_name = str(shard_row.get("file_name", "")).strip()
    name = Path(file_name).name if file_name else f"msmarco_v2.1_doc_segmented_{int(shard_row['shard_idx']):02d}.npy"
    return (local_root / name).resolve()


def _detect_local_prefix(
    *,
    shard_meta_path: Path,
    local_shards_root: Path,
    max_local_shard_idx: int | None,
) -> dict:
    payload = json.loads(shard_meta_path.read_text(encoding="utf-8"))
    shard_rows = sorted(payload.get("shards", []), key=lambda row: int(row["shard_idx"]))
    if not shard_rows:
        raise RuntimeError(f"no shard rows found in {shard_meta_path}")

    contiguous_rows = []
    expected_idx = 0
    local_offset_limit = 0
    for row in shard_rows:
        shard_idx = int(row["shard_idx"])
        if shard_idx != int(expected_idx):
            break
        if max_local_shard_idx is not None and shard_idx > int(max_local_shard_idx):
            break
        local_path = _resolve_local_path(row, local_root=local_shards_root)
        if not local_path.exists():
            if max_local_shard_idx is not None and shard_idx <= int(max_local_shard_idx):
                raise FileNotFoundError(
                    f"requested local shard is missing: shard={shard_idx:02d} path={local_path}"
                )
            break
        row_copy = dict(row)
        row_copy["resolved_local_path"] = str(local_path)
        contiguous_rows.append(row_copy)
        local_offset_limit = int(row["global_end_exclusive"])
        expected_idx += 1

    if not contiguous_rows:
        raise RuntimeError(
            "failed to detect any contiguous local shard prefix starting from shard 00"
        )

    return {
        "repo_id": str(payload.get("repo_id", "")),
        "repo_type": str(payload.get("repo_type", "")),
        "local_offset_limit": int(local_offset_limit),
        "local_shard_indices": [int(row["shard_idx"]) for row in contiguous_rows],
        "local_shard_rows": contiguous_rows,
    }


def _local_top1k_offsets(row: dict, *, local_offset_limit: int) -> list[int]:
    out: list[int] = []
    seen = set()
    raw_offsets = row.get("top1k_offsets", [])
    if not isinstance(raw_offsets, list):
        return out
    for val in raw_offsets:
        try:
            off = int(val)
        except Exception:
            continue
        if int(off) >= int(local_offset_limit):
            continue
        if int(off) in seen:
            continue
        seen.add(int(off))
        out.append(int(off))
    return out


def _iter_query_offsets_local(rows: list[dict], topk: int) -> list[int]:
    out: list[int] = []
    for row in rows:
        offsets = row.get("local_top1k_offsets", [])
        if not isinstance(offsets, list):
            continue
        picked = 0
        for val in offsets:
            out.append(int(val))
            picked += 1
            if picked >= int(topk):
                break
    return out


def _count_union_size_with_prefix(*, core_offsets: list[int], prefix_end_exclusive: int) -> int:
    num_core_ge_prefix = int(sum(1 for x in core_offsets if int(x) >= int(prefix_end_exclusive)))
    return int(prefix_end_exclusive) + int(num_core_ge_prefix)


def _solve_prefix_end_for_size(*, core_offsets: list[int], size: int, total_passages: int) -> int:
    if len(core_offsets) > int(size):
        raise RuntimeError(
            f"core offset set too large for requested size: core={len(core_offsets)} > size={int(size)}"
        )
    lo = 0
    hi = int(min(size, total_passages))
    while lo < hi:
        mid = (lo + hi) // 2
        union_size = _count_union_size_with_prefix(
            core_offsets=core_offsets,
            prefix_end_exclusive=int(mid),
        )
        if int(union_size) >= int(size):
            hi = int(mid)
        else:
            lo = int(mid + 1)
    prefix_end = int(lo)
    if _count_union_size_with_prefix(core_offsets=core_offsets, prefix_end_exclusive=prefix_end) < int(size):
        raise RuntimeError(
            f"failed to solve prefix_end for size={int(size)} with total_passages={int(total_passages)}"
        )
    return int(prefix_end)


def _support_stats_from_rows(
    *,
    rows: list[dict],
    core_offsets: set[int],
    prefix_end_exclusive: int,
    thresholds: list[int],
) -> dict:
    out = {}
    for threshold in thresholds:
        cnt = 0
        for row in rows:
            offsets = row.get("local_top1k_offsets", [])
            if not isinstance(offsets, list):
                continue
            hit = 0
            for off in offsets:
                if int(off) < int(prefix_end_exclusive) or int(off) in core_offsets:
                    hit += 1
                    if hit >= int(threshold):
                        cnt += 1
                        break
        out[f"queries_with_at_least_{int(threshold)}_hits"] = int(cnt)
    return out


def _summarize_int_list(vals: list[int]) -> dict:
    if not vals:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    arr = sorted(int(x) for x in vals)
    n = len(arr)

    def q(frac: float) -> int:
        idx = min(n - 1, int(frac * (n - 1)))
        return int(arr[idx])

    return {
        "count": int(n),
        "min": int(arr[0]),
        "max": int(arr[-1]),
        "p10": int(q(0.10)),
        "p25": int(q(0.25)),
        "p50": int(q(0.50)),
        "p75": int(q(0.75)),
        "p90": int(q(0.90)),
        "p95": int(q(0.95)),
        "p99": int(q(0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build local-shard-constrained nested Cohere MSMARCO v2.1 offset manifests."
    )
    parser.add_argument("--query-jsonl-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sizes", default=",".join(str(x) for x in DEFAULT_SIZES))
    parser.add_argument(
        "--coverage-ranks-by-size",
        default=",".join(f"{k}:{v}" for k, v in sorted(DEFAULT_COVERAGE_RANKS.items())),
        help="Comma-separated size:topk map, e.g. 10000:5,100000:20,1000000:100,10000000:100",
    )
    parser.add_argument(
        "--support-thresholds",
        default=",".join(str(x) for x in DEFAULT_SUPPORT_THRESHOLDS),
    )
    parser.add_argument("--shard-meta-path", default=str(DEFAULT_SHARD_META_PATH))
    parser.add_argument("--local-shards-root", default=str(DEFAULT_LOCAL_SHARDS_ROOT))
    parser.add_argument(
        "--max-local-shard-idx",
        type=int,
        default=-1,
        help="Optional inclusive cap on the contiguous local shard prefix; -1 means auto-detect.",
    )
    parser.add_argument(
        "--min-local-hits-for-largest-size",
        type=int,
        default=0,
        help="Optional eligible-query threshold. When 0, reuse coverage_topk of the largest size.",
    )
    args = parser.parse_args()

    sizes = _parse_sizes(args.sizes)
    coverage_map = _parse_coverage_map(args.coverage_ranks_by_size, sizes=sizes)
    support_thresholds = _parse_thresholds(args.support_thresholds)

    local_prefix = _detect_local_prefix(
        shard_meta_path=Path(args.shard_meta_path).resolve(),
        local_shards_root=Path(args.local_shards_root).resolve(),
        max_local_shard_idx=(None if int(args.max_local_shard_idx) < 0 else int(args.max_local_shard_idx)),
    )
    local_offset_limit = int(local_prefix["local_offset_limit"])
    eligible_min_hits = int(
        args.min_local_hits_for_largest_size
        if int(args.min_local_hits_for_largest_size) > 0
        else coverage_map[int(max(sizes))]
    )

    query_rows = load_jsonl(args.query_jsonl_path)
    if not query_rows:
        raise RuntimeError(f"no query rows found at {args.query_jsonl_path}")

    all_local_hit_counts: list[int] = []
    eligible_rows: list[dict] = []
    ineligible_preview: list[dict] = []
    for row in query_rows:
        qid = _query_id_from_row(row)
        local_offsets = _local_top1k_offsets(row, local_offset_limit=local_offset_limit)
        local_hit_count = int(len(local_offsets))
        all_local_hit_counts.append(int(local_hit_count))
        prepared_row = {
            "query_id": str(qid),
            "raw_query_id": str(row.get("raw_query_id", qid)),
            "text": str(row.get("text", "")),
            "trec_year": row.get("trec_year"),
            "qrels": row.get("qrels", {}),
            "local_top1k_offsets": [int(x) for x in local_offsets],
            "local_top1k_hit_count": int(local_hit_count),
        }
        if int(local_hit_count) < int(eligible_min_hits):
            if len(ineligible_preview) < 20:
                ineligible_preview.append(
                    {
                        "query_id": str(qid),
                        "local_top1k_hit_count": int(local_hit_count),
                    }
                )
            continue
        eligible_rows.append(prepared_row)

    if not eligible_rows:
        raise RuntimeError(
            "no eligible queries remain after applying the local-hit threshold; "
            f"threshold={int(eligible_min_hits)} local_offset_limit={int(local_offset_limit)}"
        )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(out_dir / "eligible_queries_local.jsonl", eligible_rows)

    core_offsets_prev: set[int] = set()
    meta_rows = []
    size_to_membership: dict[int, tuple[set[int], int]] = {}
    for size in sizes:
        size = int(size)
        topk = int(coverage_map[int(size)])
        required_core = set(_iter_query_offsets_local(eligible_rows, topk=topk))
        core_offsets = set(core_offsets_prev)
        core_offsets.update(required_core)
        core_offsets_sorted = sorted(int(x) for x in core_offsets)
        prefix_end_exclusive = _solve_prefix_end_for_size(
            core_offsets=core_offsets_sorted,
            size=int(size),
            total_passages=int(local_offset_limit),
        )
        union_size = _count_union_size_with_prefix(
            core_offsets=core_offsets_sorted,
            prefix_end_exclusive=int(prefix_end_exclusive),
        )
        if int(union_size) < int(size):
            raise RuntimeError(
                f"compressed manifest underfilled size={int(size)}: union_size={int(union_size)}"
            )

        core_offsets_path = out_dir / f"db_{int(size)}_core_offsets.txt"
        with open(core_offsets_path, "w", encoding="utf-8") as f:
            for off in core_offsets_sorted:
                f.write(f"{int(off)}\n")

        manifest_path = out_dir / f"db_{int(size)}_manifest.json"
        core_offsets_set = set(core_offsets_sorted)
        stats = _support_stats_from_rows(
            rows=eligible_rows,
            core_offsets=core_offsets_set,
            prefix_end_exclusive=int(prefix_end_exclusive),
            thresholds=support_thresholds,
        )
        manifest = {
            "format": "cohere_offset_union_prefix_v1",
            "manifest_scope": "local_real_shard_prefix_only",
            "size": int(size),
            "coverage_topk": int(topk),
            "core_offsets_path": str(core_offsets_path),
            "core_offsets_count": int(len(core_offsets_sorted)),
            "prefix_end_exclusive": int(prefix_end_exclusive),
            "selected_count_effective": int(union_size),
            "total_passages": int(local_offset_limit),
            "local_offset_limit": int(local_offset_limit),
            "local_shard_indices": [int(x) for x in local_prefix["local_shard_indices"]],
            "eligible_query_count": int(len(eligible_rows)),
            **stats,
        }
        save_json(manifest_path, manifest)
        meta_rows.append(
            {
                "size": int(size),
                "coverage_topk": int(topk),
                "manifest_path": str(manifest_path),
                "core_offsets_count": int(len(core_offsets_sorted)),
                "prefix_end_exclusive": int(prefix_end_exclusive),
                "selected_count_effective": int(union_size),
                **stats,
            }
        )
        size_to_membership[int(size)] = (core_offsets_set, int(prefix_end_exclusive))
        core_offsets_prev = set(core_offsets_sorted)

    query_support_rows = []
    size_keys = [int(x) for x in sizes]
    for row in eligible_rows:
        support_row = {
            "query_id": str(row["query_id"]),
            "local_top1k_hit_count": int(row["local_top1k_hit_count"]),
        }
        offsets = [int(x) for x in row.get("local_top1k_offsets", [])]
        for size in size_keys:
            core_set, prefix_end = size_to_membership[int(size)]
            hits = int(sum(1 for off in offsets if int(off) < int(prefix_end) or int(off) in core_set))
            support_row[f"db_{int(size)}_hits"] = int(hits)
        query_support_rows.append(support_row)
    save_jsonl(out_dir / "query_support_by_size.jsonl", query_support_rows)

    save_json(
        out_dir / "local_offset_manifest_meta.json",
        {
            "pipeline": "build_cohere_msmarco_v21_local_offset_manifests",
            "manifest_format": "cohere_offset_union_prefix_v1",
            "query_jsonl_path": str(Path(args.query_jsonl_path).resolve()),
            "sizes": [int(x) for x in sizes],
            "coverage_ranks_by_size": {str(k): int(v) for k, v in coverage_map.items()},
            "support_thresholds": [int(x) for x in support_thresholds],
            "local_offset_limit": int(local_offset_limit),
            "local_shard_indices": [int(x) for x in local_prefix["local_shard_indices"]],
            "num_input_queries": int(len(query_rows)),
            "eligible_query_count": int(len(eligible_rows)),
            "eligible_min_local_hits": int(eligible_min_hits),
            "all_query_local_hit_summary": _summarize_int_list(all_local_hit_counts),
            "eligible_query_local_hit_summary": _summarize_int_list(
                [int(row["local_top1k_hit_count"]) for row in eligible_rows]
            ),
            "ineligible_preview": ineligible_preview,
            "eligible_queries_jsonl_path": str(out_dir / "eligible_queries_local.jsonl"),
            "query_support_jsonl_path": str(out_dir / "query_support_by_size.jsonl"),
            "per_size": meta_rows,
        },
    )

    print(
        json.dumps(
            {
                "status": "completed",
                "out_dir": str(out_dir),
                "local_offset_limit": int(local_offset_limit),
                "local_shard_indices": [int(x) for x in local_prefix["local_shard_indices"]],
                "eligible_query_count": int(len(eligible_rows)),
                "eligible_min_local_hits": int(eligible_min_hits),
                "per_size": meta_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
