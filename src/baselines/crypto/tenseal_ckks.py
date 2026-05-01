from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

from dataclasses import dataclass
import os
import time
from typing import Any

import numpy as np

try:
    import tenseal as ts
except Exception:  # pragma: no cover - optional at import time
    ts = None


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return str(raw).strip() if raw is not None else str(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _env_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    if raw is None:
        return [int(x) for x in default]
    try:
        vals = [int(float(part.strip())) for part in str(raw).split(",") if part.strip()]
    except Exception:
        vals = []
    return vals if vals else [int(x) for x in default]


BASELINE_CKKS_LIBRARY = _env_str("BASELINE_CKKS_LIBRARY", "tenseal").strip().lower()
BASELINE_CKKS_SCHEME = _env_str("BASELINE_CKKS_SCHEME", "ckks_selector_row").strip().lower()
BASELINE_CKKS_POLY_MODULUS_DEGREE = int(
    max(4096, _env_int("BASELINE_CKKS_POLY_MODULUS_DEGREE", 8192))
)
BASELINE_CKKS_COEFF_MOD_BIT_SIZES = _env_int_list(
    "BASELINE_CKKS_COEFF_MOD_BIT_SIZES",
    [60, 40, 40, 60],
)
BASELINE_CKKS_GLOBAL_SCALE_BITS = int(
    max(20, _env_int("BASELINE_CKKS_GLOBAL_SCALE_BITS", 40))
)
BASELINE_CKKS_DECRYPT_ROUND_TOLERANCE = float(
    max(1e-6, _env_float("BASELINE_CKKS_DECRYPT_ROUND_TOLERANCE", 0.25))
)


@dataclass
class CKKSRuntime:
    private_context: Any
    public_context: Any
    public_context_bytes: int
    private_context_bytes: int
    setup_time_sec: float
    slot_capacity: int


_RUNTIME_CACHE: dict[tuple[int, tuple[int, ...], int], CKKSRuntime] = {}


def _require_tenseal() -> None:
    if str(BASELINE_CKKS_LIBRARY) != "tenseal":
        raise RuntimeError(
            f"Expected BASELINE_CKKS_LIBRARY=tenseal, got {BASELINE_CKKS_LIBRARY}"
        )
    if str(BASELINE_CKKS_SCHEME) != "ckks_selector_row":
        raise RuntimeError(
            f"Expected BASELINE_CKKS_SCHEME=ckks_selector_row, got {BASELINE_CKKS_SCHEME}"
        )
    if ts is None:
        raise ImportError("TenSEAL is not available in the current environment.")


def _context_key() -> tuple[int, tuple[int, ...], int]:
    return (
        int(BASELINE_CKKS_POLY_MODULUS_DEGREE),
        tuple(int(x) for x in BASELINE_CKKS_COEFF_MOD_BIT_SIZES),
        int(BASELINE_CKKS_GLOBAL_SCALE_BITS),
    )


def build_runtime(*, required_slots: int, force_fresh: bool = False) -> CKKSRuntime:
    _require_tenseal()
    slot_capacity = int(BASELINE_CKKS_POLY_MODULUS_DEGREE) // 2
    if int(required_slots) > int(slot_capacity):
        raise RuntimeError(
            f"CKKS slot capacity exceeded: required_slots={int(required_slots)}, slot_capacity={int(slot_capacity)}"
        )
    key = _context_key()
    if (not bool(force_fresh)) and key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[key]

    t0 = time.perf_counter()
    private_context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=int(BASELINE_CKKS_POLY_MODULUS_DEGREE),
        coeff_mod_bit_sizes=[int(x) for x in BASELINE_CKKS_COEFF_MOD_BIT_SIZES],
    )
    private_context.generate_galois_keys()
    private_context.global_scale = float(2 ** int(BASELINE_CKKS_GLOBAL_SCALE_BITS))

    public_blob = private_context.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_galois_keys=True,
        save_relin_keys=False,
    )
    private_blob = private_context.serialize(
        save_public_key=True,
        save_secret_key=True,
        save_galois_keys=True,
        save_relin_keys=False,
    )
    runtime = CKKSRuntime(
        private_context=ts.context_from(private_blob),
        public_context=ts.context_from(public_blob),
        public_context_bytes=int(len(public_blob)),
        private_context_bytes=int(len(private_blob)),
        setup_time_sec=float(time.perf_counter() - t0),
        slot_capacity=int(slot_capacity),
    )
    if not bool(force_fresh):
        _RUNTIME_CACHE[key] = runtime
    return runtime


def encrypt_vector_request(
    *,
    runtime: CKKSRuntime,
    vector: np.ndarray,
) -> dict:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.size <= 0:
        raise ValueError("cannot encrypt an empty vector")
    t0 = time.perf_counter()
    encrypted_vector = ts.ckks_vector(runtime.private_context, values.tolist())
    request_blob = encrypted_vector.serialize()
    return {
        "request_blob": request_blob,
        "request_bytes": int(len(request_blob)),
        "client_encrypt_sec": float(time.perf_counter() - t0),
        "public_context_bytes_once": int(runtime.public_context_bytes),
        "private_context_bytes_once": int(runtime.private_context_bytes),
        "setup_time_sec_once": float(runtime.setup_time_sec),
        "vector_length": int(values.size),
    }


def server_matmul_response(
    *,
    runtime: CKKSRuntime,
    request_blob: bytes,
    plaintext_matrix: np.ndarray,
) -> dict:
    matrix = np.asarray(plaintext_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"plaintext_matrix must be 2D, got shape={matrix.shape}")
    t0 = time.perf_counter()
    encrypted_vector = ts.ckks_vector_from(runtime.public_context, request_blob)
    encrypted_result = encrypted_vector.matmul(matrix.tolist())
    response_blob = encrypted_result.serialize()
    return {
        "response_blob": response_blob,
        "response_bytes": int(len(response_blob)),
        "server_compute_sec": float(time.perf_counter() - t0),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "matrix_entries_touched": int(matrix.shape[0] * matrix.shape[1]),
    }


def decrypt_vector_response(
    *,
    runtime: CKKSRuntime,
    response_blob: bytes,
    expected_length: int,
) -> dict:
    t0 = time.perf_counter()
    encrypted_result = ts.ckks_vector_from(runtime.private_context, response_blob)
    recovered = np.asarray(encrypted_result.decrypt(), dtype=np.float64).reshape(-1)
    if int(recovered.size) != int(expected_length):
        raise RuntimeError(
            f"decrypted vector length mismatch: expected={int(expected_length)} got={int(recovered.size)}"
        )
    return {
        "values": recovered,
        "client_decrypt_sec": float(time.perf_counter() - t0),
    }


def decrypt_rounded_int_response(
    *,
    runtime: CKKSRuntime,
    response_blob: bytes,
    expected_length: int,
) -> dict:
    decrypted = decrypt_vector_response(
        runtime=runtime,
        response_blob=response_blob,
        expected_length=int(expected_length),
    )
    values = np.asarray(decrypted["values"], dtype=np.float64).reshape(-1)
    rounded = np.rint(values).astype(np.int32)
    max_abs_error = float(np.max(np.abs(values - rounded.astype(np.float64)))) if values.size else 0.0
    if float(max_abs_error) > float(BASELINE_CKKS_DECRYPT_ROUND_TOLERANCE):
        raise RuntimeError(
            "CKKS rounded integer recovery exceeded tolerance: "
            f"max_abs_error={max_abs_error:.6f} tolerance={float(BASELINE_CKKS_DECRYPT_ROUND_TOLERANCE):.6f}"
        )
    return {
        "values": values,
        "rounded": rounded,
        "client_decrypt_sec": float(decrypted["client_decrypt_sec"]),
        "max_abs_error": float(max_abs_error),
    }
