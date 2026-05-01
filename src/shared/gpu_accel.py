from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


_GPU_CACHE_MAX_ENTRIES = int(os.environ.get("GPU_ACCEL_CACHE_MAX_ENTRIES", "8"))
_GPU_CACHE_MIN_ROWS = int(os.environ.get("GPU_ACCEL_CACHE_MIN_ROWS", "4096"))


@dataclass
class _GPUAccelRoleState:
    tensor_cache: "OrderedDict[tuple[Any, ...], Any]" = field(default_factory=OrderedDict)
    disabled_reason: str | None = None


_GPU_ROLE_STATES: dict[str, _GPUAccelRoleState] = {}


def normalize_gpu_role(role: str | None = None) -> str:
    text = "default" if role is None else str(role).strip().lower()
    if text in {"", "none", "default"}:
        return "default"
    if text in {"client", "server"}:
        return text
    return text


def resolve_device_spec(role: str | None = None) -> str:
    norm_role = normalize_gpu_role(role)
    generic = str(
        os.environ.get(
            "GPU_ACCEL_DEVICE",
            os.environ.get("DEFAULT_GPU_DEVICE", "cuda"),
        )
    ).strip()
    if norm_role == "client":
        return str(
            os.environ.get(
                "CLIENT_GPU_DEVICE",
                os.environ.get("GPU_ACCEL_CLIENT_DEVICE", generic),
            )
        ).strip()
    if norm_role == "server":
        return str(
            os.environ.get(
                "SERVER_GPU_DEVICE",
                os.environ.get("GPU_ACCEL_SERVER_DEVICE", generic),
            )
        ).strip()
    return generic if generic else "cuda"


def _role_state(role: str | None = None) -> _GPUAccelRoleState:
    norm_role = normalize_gpu_role(role)
    state = _GPU_ROLE_STATES.get(norm_role)
    if state is None:
        state = _GPUAccelRoleState()
        _GPU_ROLE_STATES[norm_role] = state
    return state


def gpu_available(role: str | None = None) -> bool:
    state = _role_state(role)
    return bool(state.disabled_reason is None and torch is not None and torch.cuda.is_available())


def _device(role: str | None = None):
    if not gpu_available(role=role):
        return None
    return torch.device(resolve_device_spec(role))


def _cache_key(
    arr: np.ndarray,
    *,
    np_dtype: np.dtype,
    normalize_rows: bool,
) -> tuple[Any, ...]:
    base = np.asarray(arr)
    ptr = int(base.__array_interface__["data"][0])
    return (
        ptr,
        tuple(int(x) for x in base.shape),
        str(np_dtype),
        bool(normalize_rows),
    )


def _maybe_cache_tensor(role: str | None, key: tuple[Any, ...], value) -> None:
    state = _role_state(role)
    state.tensor_cache[key] = value
    state.tensor_cache.move_to_end(key)
    while len(state.tensor_cache) > int(_GPU_CACHE_MAX_ENTRIES):
        state.tensor_cache.popitem(last=False)


def _is_cuda_runtime_fallback_error(exc: Exception) -> bool:
    if torch is not None and hasattr(torch, "AcceleratorError"):
        if isinstance(exc, torch.AcceleratorError):
            return True
    text = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "cuda",
        "cublas",
        "cudnn",
        "device-side assert",
        "busy or unavailable",
        "out of memory",
        "driver",
    )
    return any(token in text for token in needles)


def _disable_gpu_accel(reason: Exception | str, *, role: str | None = None) -> None:
    state = _role_state(role)
    state.disabled_reason = str(reason)
    state.tensor_cache.clear()
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def clear_gpu_tensor_cache(role: str | None = None) -> None:
    if role is None:
        for state in _GPU_ROLE_STATES.values():
            state.tensor_cache.clear()
    else:
        _role_state(role).tensor_cache.clear()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _as_gpu_tensor(
    arr: np.ndarray,
    *,
    np_dtype: np.dtype,
    normalize_rows: bool = False,
    cache_if_large: bool = False,
    role: str | None = None,
) -> Any | None:
    if not gpu_available(role=role):
        return None

    base = np.ascontiguousarray(np.asarray(arr, dtype=np_dtype))
    use_cache = bool(cache_if_large and base.ndim == 2 and int(base.shape[0]) >= int(_GPU_CACHE_MIN_ROWS))
    key = _cache_key(base, np_dtype=np.dtype(np_dtype), normalize_rows=bool(normalize_rows)) if use_cache else None
    if key is not None:
        state = _role_state(role)
        cached = state.tensor_cache.get(key)
        if cached is not None:
            state.tensor_cache.move_to_end(key)
            return cached

    try:
        tensor = torch.from_numpy(base).to(_device(role), non_blocking=True)
        if normalize_rows:
            norms = torch.linalg.vector_norm(tensor, dim=1, keepdim=True)
            tiny = torch.tensor(1e-12, device=tensor.device, dtype=tensor.dtype)
            tensor = tensor / torch.clamp(norms, min=tiny)
    except Exception as exc:
        if _is_cuda_runtime_fallback_error(exc):
            _disable_gpu_accel(exc, role=role)
            return None
        raise

    if key is not None:
        _maybe_cache_tensor(role, key, tensor)
    return tensor


def cosine_scores_1d(
    query: np.ndarray,
    docs: np.ndarray,
    *,
    assume_unit_norm: bool = True,
    prefer_float64: bool = False,
    cache_docs_if_large: bool = False,
    role: str | None = None,
) -> np.ndarray | None:
    if not gpu_available(role=role):
        return None

    np_dtype = np.float64 if prefer_float64 else np.float32
    docs_t = _as_gpu_tensor(
        docs,
        np_dtype=np_dtype,
        normalize_rows=False,
        cache_if_large=bool(cache_docs_if_large),
        role=role,
    )
    if docs_t is None:
        return None

    try:
        query_np = np.ascontiguousarray(np.asarray(query, dtype=np_dtype).reshape(-1))
        query_t = torch.from_numpy(query_np).to(_device(role), non_blocking=True)

        if assume_unit_norm:
            scores_t = torch.matmul(docs_t, query_t)
        else:
            numer_t = torch.matmul(docs_t, query_t)
            q_norm_t = torch.linalg.vector_norm(query_t)
            d_norm_t = torch.linalg.vector_norm(docs_t, dim=1)
            tiny = torch.tensor(1e-12, device=docs_t.device, dtype=docs_t.dtype)
            denom_t = torch.clamp(d_norm_t * q_norm_t, min=tiny)
            scores_t = numer_t / denom_t
        return scores_t.detach().cpu().numpy()
    except Exception as exc:
        if _is_cuda_runtime_fallback_error(exc):
            _disable_gpu_accel(exc, role=role)
            return None
        raise


def angular_distance_to_rows(
    query: np.ndarray,
    rows: np.ndarray,
    *,
    assume_unit_norm: bool = True,
    cache_rows_if_large: bool = False,
    role: str | None = None,
) -> np.ndarray | None:
    scores = cosine_scores_1d(
        query=query,
        docs=rows,
        assume_unit_norm=bool(assume_unit_norm),
        prefer_float64=False,
        cache_docs_if_large=bool(cache_rows_if_large),
        role=role,
    )
    if scores is None:
        return None
    return np.arccos(np.clip(np.asarray(scores, dtype=np.float32), -1.0, 1.0)).astype(np.float32)


def squared_l2_distance_matrix(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cache_x_if_large: bool = False,
    cache_y_if_large: bool = False,
    role: str | None = None,
) -> np.ndarray | None:
    if not gpu_available(role=role):
        return None

    x_t = _as_gpu_tensor(
        x,
        np_dtype=np.float32,
        normalize_rows=False,
        cache_if_large=bool(cache_x_if_large),
        role=role,
    )
    y_t = _as_gpu_tensor(
        y,
        np_dtype=np.float32,
        normalize_rows=False,
        cache_if_large=bool(cache_y_if_large),
        role=role,
    )
    if x_t is None or y_t is None:
        return None

    try:
        x_sq_t = torch.sum(x_t * x_t, dim=1, keepdim=True)
        y_sq_t = torch.sum(y_t * y_t, dim=1, keepdim=True).transpose(0, 1)
        cross_t = torch.matmul(x_t, y_t.transpose(0, 1))
        d2_t = torch.clamp(x_sq_t + y_sq_t - (2.0 * cross_t), min=0.0)
        return d2_t.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception as exc:
        if _is_cuda_runtime_fallback_error(exc):
            _disable_gpu_accel(exc, role=role)
            return None
        raise


def cosine_similarity_matrix(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cache_x_if_large: bool = False,
    cache_y_if_large: bool = False,
    role: str | None = None,
) -> np.ndarray | None:
    if not gpu_available(role=role):
        return None

    x_t = _as_gpu_tensor(
        x,
        np_dtype=np.float32,
        normalize_rows=False,
        cache_if_large=bool(cache_x_if_large),
        role=role,
    )
    y_t = _as_gpu_tensor(
        y,
        np_dtype=np.float32,
        normalize_rows=False,
        cache_if_large=bool(cache_y_if_large),
        role=role,
    )
    if x_t is None or y_t is None:
        return None
    try:
        sims_t = torch.matmul(x_t, y_t.transpose(0, 1))
        sims_t = torch.clamp(sims_t, min=-1.0, max=1.0)
        return sims_t.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception as exc:
        if _is_cuda_runtime_fallback_error(exc):
            _disable_gpu_accel(exc, role=role)
            return None
        raise


def topk_cosine_similarity_matrix(
    x: np.ndarray,
    y: np.ndarray,
    *,
    top_k: int,
    cache_x_if_large: bool = False,
    cache_y_if_large: bool = False,
    y_chunk_rows: int = 8192,
    role: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if not gpu_available(role=role):
        return None, None

    x_np = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    y_np = np.asarray(y, dtype=np.float32)
    if x_np.ndim != 2 or y_np.ndim != 2:
        raise ValueError("topk_cosine_similarity_matrix expects 2D inputs.")
    if x_np.shape[1] != y_np.shape[1]:
        raise ValueError("topk_cosine_similarity_matrix dim mismatch.")

    top_k = int(max(1, min(int(top_k), int(y_np.shape[0]))))
    chunk_rows = int(max(1, int(y_chunk_rows)))

    x_t = _as_gpu_tensor(
        x_np,
        np_dtype=np.float32,
        normalize_rows=False,
        cache_if_large=bool(cache_x_if_large),
        role=role,
    )
    if x_t is None:
        return None, None

    try:
        top_scores_t = None
        top_indices_t = None
        for start in range(0, int(y_np.shape[0]), int(chunk_rows)):
            end = min(int(y_np.shape[0]), int(start + chunk_rows))
            y_chunk_t = _as_gpu_tensor(
                np.ascontiguousarray(y_np[start:end], dtype=np.float32),
                np_dtype=np.float32,
                normalize_rows=False,
                cache_if_large=bool(cache_y_if_large),
                role=role,
            )
            if y_chunk_t is None:
                return None, None

            sims_t = torch.matmul(x_t, y_chunk_t.transpose(0, 1))
            sims_t = torch.clamp(sims_t, min=-1.0, max=1.0)
            chunk_idx_t = torch.arange(
                int(start),
                int(end),
                device=sims_t.device,
                dtype=torch.int64,
            ).view(1, -1).expand(int(x_np.shape[0]), -1)

            if top_scores_t is None:
                cand_scores_t = sims_t
                cand_indices_t = chunk_idx_t
            else:
                cand_scores_t = torch.cat([top_scores_t, sims_t], dim=1)
                cand_indices_t = torch.cat([top_indices_t, chunk_idx_t], dim=1)

            k_now = int(min(int(top_k), int(cand_scores_t.shape[1])))
            top_scores_t, top_pos_t = torch.topk(
                cand_scores_t,
                k=k_now,
                dim=1,
                largest=True,
                sorted=True,
            )
            top_indices_t = torch.gather(cand_indices_t, 1, top_pos_t)

        if top_scores_t is None or top_indices_t is None:
            return None, None
        return (
            top_scores_t.detach().cpu().numpy().astype(np.float32, copy=False),
            top_indices_t.detach().cpu().numpy().astype(np.int32, copy=False),
        )
    except Exception as exc:
        if _is_cuda_runtime_fallback_error(exc):
            _disable_gpu_accel(exc, role=role)
            return None, None
        raise
