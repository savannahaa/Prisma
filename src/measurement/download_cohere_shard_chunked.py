"""
Chunked downloader for large Cohere passage shards using curl subprocesses.

Why curl:
- tiny test downloads via curl succeed on this server
- requests/urllib3 long HTTPS flows are unstable here

This downloader:
- reads shard size/url from local shard-meta cache
- downloads medium-sized byte ranges with curl
- retries each chunk independently via curl
- can resume from completed chunk files
- can merge chunks into a final `.npy`
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
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_CHUNK_MB = 16
DEFAULT_RETRIES = 20
DEFAULT_WORKERS = 2
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARD_META_CACHE_PATH = (
    PROJECT_ROOT / "data" / "external" / "cohere_msmarco_v21" / "passages_shard_meta.json"
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return int(default)


def _load_shard_meta(cache_path: Path, shard_idx: int) -> dict:
    with open(cache_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for row in payload.get("shards", []):
        if int(row.get("shard_idx", -1)) == int(shard_idx):
            return dict(row)
    raise RuntimeError(f"missing shard_idx={int(shard_idx)} in {cache_path}")


def _dtype_itemsize(dtype_text: str) -> int:
    text = str(dtype_text).strip().lower()
    if text == "float16":
        return 2
    if text == "float32":
        return 4
    raise RuntimeError(f"unsupported dtype for chunked download: {dtype_text}")


def _expected_total_size(meta: dict) -> int:
    return int(meta["data_offset"]) + int(meta["num_rows"]) * int(meta["dim"]) * _dtype_itemsize(meta["dtype"])


def _run_curl_range(
    *,
    url: str,
    out_path: Path,
    start: int,
    end: int,
    retries: int,
) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    cmd = [
        "curl",
        "-L",
        "-4",
        "--fail",
        "--retry",
        str(int(retries)),
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "--max-time",
        "0",
        "--range",
        f"{int(start)}-{int(end)}",
        "-o",
        str(tmp_path),
        str(url),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"curl range failed start={int(start)} end={int(end)} rc={int(proc.returncode)} output={proc.stdout[-1200:]}"
        )
    os.replace(tmp_path, out_path)


def _download_one_chunk(
    *,
    url: str,
    part_path: Path,
    start: int,
    end: int,
    retries: int,
) -> tuple[int, int]:
    expected = int(end - start + 1)
    if part_path.exists() and int(part_path.stat().st_size) == int(expected):
        return int(expected), 0
    _run_curl_range(
        url=str(url),
        out_path=part_path,
        start=int(start),
        end=int(end),
        retries=int(retries),
    )
    got = int(part_path.stat().st_size)
    if int(got) != int(expected):
        raise RuntimeError(
            f"chunk size mismatch start={int(start)} end={int(end)} expected={int(expected)} got={int(got)}"
        )
    return int(got), 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunked curl downloader for a Cohere shard.")
    parser.add_argument("--shard-idx", type=int, required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--chunk-mb", type=int, default=_env_int("COHERE_CHUNK_MB", DEFAULT_CHUNK_MB))
    parser.add_argument("--workers", type=int, default=_env_int("COHERE_CHUNK_WORKERS", DEFAULT_WORKERS))
    parser.add_argument("--retries", type=int, default=_env_int("COHERE_CHUNK_RETRIES", DEFAULT_RETRIES))
    parser.add_argument("--max-chunks", type=int, default=0, help="For testing; 0 means all chunks.")
    parser.add_argument("--cache-path", default=str(DEFAULT_SHARD_META_CACHE_PATH))
    args = parser.parse_args()

    cache_path = Path(args.cache_path).resolve()
    meta = _load_shard_meta(cache_path, shard_idx=int(args.shard_idx))
    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = out_path.parent / f".{out_path.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    total_size = _expected_total_size(meta)
    chunk_size = int(max(1, int(args.chunk_mb)) * 1024 * 1024)
    num_chunks = int(math.ceil(float(total_size) / float(chunk_size)))
    if int(args.max_chunks) > 0:
        num_chunks = int(min(num_chunks, int(args.max_chunks)))

    print(
        json.dumps(
            {
                "status": "plan",
                "shard_idx": int(args.shard_idx),
                "url": str(meta["url"]),
                "total_size": int(total_size),
                "chunk_size": int(chunk_size),
                "num_chunks": int(num_chunks),
                "workers": int(args.workers),
                "out_path": str(out_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    downloaded_bytes = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = []
        for chunk_idx in range(int(num_chunks)):
            start = int(chunk_idx * chunk_size)
            end = int(min(total_size - 1, start + chunk_size - 1))
            part_path = parts_dir / f"part_{int(chunk_idx):05d}.bin"
            futures.append(
                ex.submit(
                    _download_one_chunk,
                    url=str(meta["url"]),
                    part_path=part_path,
                    start=int(start),
                    end=int(end),
                    retries=int(args.retries),
                )
            )
        for fut in as_completed(futures):
            size, attempts_used = fut.result()
            downloaded_bytes += int(size)
            print(
                f"[chunk-progress] downloaded_mb={float(downloaded_bytes / (1024 * 1024)):.2f} "
                f"attempts_used={int(attempts_used)}",
                flush=True,
            )

    if int(args.max_chunks) > 0:
        print(f"[test-only] finished first {int(num_chunks)} chunks", flush=True)
        return

    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_out, "wb") as fout:
        for chunk_idx in range(int(num_chunks)):
            part_path = parts_dir / f"part_{int(chunk_idx):05d}.bin"
            if not part_path.exists():
                raise RuntimeError(f"missing chunk file: {part_path}")
            with open(part_path, "rb") as fin:
                shutil.copyfileobj(fin, fout, length=1024 * 1024 * 16)
    os.replace(tmp_out, out_path)
    got = int(out_path.stat().st_size)
    if int(got) != int(total_size):
        raise RuntimeError(f"final file size mismatch: expected={int(total_size)} got={int(got)}")
    print(
        json.dumps(
            {
                "status": "completed",
                "shard_idx": int(args.shard_idx),
                "out_path": str(out_path),
                "size": int(got),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
