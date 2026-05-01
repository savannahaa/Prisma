from __future__ import annotations

import os


DEFAULT_CPU_NUM_THREADS = 32
DEFAULT_TORCH_NUM_INTEROP_THREADS = 1

_THREADPOOL_LIMITER = None
_CPU_THREADS_CONFIGURED = False
_CONFIGURED_CPU_NUM_THREADS = DEFAULT_CPU_NUM_THREADS


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return int(default)


def configure_runtime_cpu_threads(default_threads: int = DEFAULT_CPU_NUM_THREADS) -> int:
    global _THREADPOOL_LIMITER, _CPU_THREADS_CONFIGURED, _CONFIGURED_CPU_NUM_THREADS

    if _CPU_THREADS_CONFIGURED:
        return int(_CONFIGURED_CPU_NUM_THREADS)

    cpu_threads = int(max(1, _env_int("CPU_NUM_THREADS", int(default_threads))))
    torch_interop_threads = int(
        max(
            1,
            _env_int("TORCH_NUM_INTEROP_THREADS", DEFAULT_TORCH_NUM_INTEROP_THREADS),
        )
    )

    for env_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ.setdefault(env_name, str(cpu_threads))
    os.environ.setdefault("TORCH_NUM_INTEROP_THREADS", str(torch_interop_threads))

    try:
        from threadpoolctl import threadpool_limits

        _THREADPOOL_LIMITER = threadpool_limits(limits=int(cpu_threads))
        _THREADPOOL_LIMITER.__enter__()
    except Exception:
        _THREADPOOL_LIMITER = None

    try:
        import torch

        torch.set_num_threads(int(cpu_threads))
        try:
            torch.set_num_interop_threads(int(torch_interop_threads))
        except RuntimeError:
            pass
    except Exception:
        pass

    try:
        import faiss  # type: ignore

        if hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(int(cpu_threads))
    except Exception:
        pass

    _CPU_THREADS_CONFIGURED = True
    _CONFIGURED_CPU_NUM_THREADS = int(cpu_threads)
    return int(cpu_threads)
