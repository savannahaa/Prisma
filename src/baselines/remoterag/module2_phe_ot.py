"""
RemoteRAG Module 2 prototype.

这版实现把论文里的 Module 2 落成一个可运行的本地协议原型：
1) 用户端对原始 query embedding 做 PHE（这里用纯 Python Paillier 风格加法同态）。
2) 云端仅拿到密文 query，在密文下对 Module 1 给出的 top-k' 候选计算 cosine distance。
3) 用户端解密距离并完成最终 top-k 排序。
4) 按论文 Theorem 3 的条件选择：
   - 满足条件：直接请求 top-k 文档。
   - 不满足条件：走 k-out-of-k' OT 取回文档。

说明：
- 这里的 PHE/OT 都是“本地可运行原型”，接口与消息流对齐论文，但不宣称达到生产级安全实现。
- PHE 使用整数定点量化来近似 cosine distance；结果会显式输出量化误差与 top-k 一致性。
- OT 采用 Chou-Orlandi 风格的 1-out-of-n 扩展到 k-out-of-k'，每个候选位置独立生成一条盲化选择消息。
"""

from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import argparse
import csv
import concurrent.futures
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from baselines.remoterag.config import RemoteRAGConfig
from shared.synthetic_doc_access import LazySyntheticTexts, load_doc_ids_or_synthetic


_MB = 1024.0 * 1024.0


@dataclass(frozen=True)
class PaillierPublicKey:
    n: int
    g: int
    n_sq: int


@dataclass(frozen=True)
class PaillierPrivateKey:
    lam: int
    mu: int


@dataclass(frozen=True)
class PaillierKeypair:
    public: PaillierPublicKey
    private: PaillierPrivateKey


@dataclass(frozen=True)
class OTRuntime:
    prime: int
    generator: int


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32)


def _normalize_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(x))
    if norm <= 1e-12:
        raise ValueError("vector norm too small")
    return (x / norm).astype(np.float32)


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


def _save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _int_size_bytes(value: int) -> int:
    value = int(abs(int(value)))
    if value <= 0:
        return 1
    return int((value.bit_length() + 7) // 8)


def _lcm(a: int, b: int) -> int:
    return int(abs(a * b) // math.gcd(a, b))


def _small_prime_sieve() -> tuple[int, ...]:
    return (
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
    )


def _is_probable_prime(n: int, rng: random.Random, rounds: int = 12) -> bool:
    n = int(n)
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if (n % 2) == 0:
        return False
    for p in _small_prime_sieve():
        if n == p:
            return True
        if (n % p) == 0:
            return False

    d = n - 1
    s = 0
    while (d % 2) == 0:
        d //= 2
        s += 1

    for _ in range(int(rounds)):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        witness = True
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                witness = False
                break
        if witness:
            return False
    return True


def _generate_prime(bits: int, rng: random.Random) -> int:
    bits = int(bits)
    if bits < 32:
        raise ValueError(f"prime bits too small: {bits}")
    while True:
        candidate = rng.getrandbits(bits)
        candidate |= (1 << (bits - 1))
        candidate |= 1
        if _is_probable_prime(candidate, rng):
            return int(candidate)


def _generate_paillier_keypair(bits: int, rng: random.Random) -> PaillierKeypair:
    half_bits = max(32, int(bits) // 2)
    p = _generate_prime(half_bits, rng)
    q = _generate_prime(half_bits, rng)
    while q == p:
        q = _generate_prime(half_bits, rng)
    n = int(p * q)
    lam = int(_lcm(p - 1, q - 1))
    g = int(n + 1)
    n_sq = int(n * n)
    u = pow(g, lam, n_sq)
    l_of_u = (u - 1) // n
    mu = int(pow(l_of_u, -1, n))
    return PaillierKeypair(
        public=PaillierPublicKey(n=n, g=g, n_sq=n_sq),
        private=PaillierPrivateKey(lam=lam, mu=mu),
    )


def _encode_signed_int(value: int, modulus: int) -> int:
    return int(value) % int(modulus)


def _decode_signed_int(value: int, modulus: int) -> int:
    value = int(value) % int(modulus)
    if value > (int(modulus) // 2):
        return int(value - int(modulus))
    return int(value)


def _paillier_encrypt(pub: PaillierPublicKey, message: int, rng: random.Random) -> int:
    message_mod = _encode_signed_int(int(message), int(pub.n))
    while True:
        r = rng.randrange(1, int(pub.n))
        if math.gcd(r, int(pub.n)) == 1:
            break
    # Fast path for the standard Paillier choice g = n + 1:
    # g^m mod n^2 == 1 + m*n mod n^2.
    if int(pub.g) == int(pub.n) + 1:
        g_pow_m = int((1 + (int(message_mod) * int(pub.n))) % int(pub.n_sq))
    else:
        g_pow_m = int(pow(int(pub.g), int(message_mod), int(pub.n_sq)))
    return int((g_pow_m * pow(r, int(pub.n), int(pub.n_sq))) % int(pub.n_sq))


def _paillier_decrypt(keypair: PaillierKeypair, ciphertext: int) -> int:
    pub = keypair.public
    priv = keypair.private
    u = pow(int(ciphertext), int(priv.lam), int(pub.n_sq))
    l_of_u = (u - 1) // int(pub.n)
    message = int((int(l_of_u) * int(priv.mu)) % int(pub.n))
    return _decode_signed_int(message, int(pub.n))


def _paillier_add(pub: PaillierPublicKey, left: int, right: int) -> int:
    return int((int(left) * int(right)) % int(pub.n_sq))


def _paillier_add_plain(pub: PaillierPublicKey, ciphertext: int, plain: int) -> int:
    plain_mod = _encode_signed_int(int(plain), int(pub.n))
    if int(pub.g) == int(pub.n) + 1:
        lift = int((1 + (int(plain_mod) * int(pub.n))) % int(pub.n_sq))
    else:
        lift = pow(int(pub.g), int(plain_mod), int(pub.n_sq))
    return int((int(ciphertext) * int(lift)) % int(pub.n_sq))


def _paillier_mul_plain(pub: PaillierPublicKey, ciphertext: int, scalar: int) -> int:
    scalar = int(scalar)
    if scalar == 0:
        return _paillier_encrypt(pub, 0, random.Random(0))
    if scalar == 1:
        return int(ciphertext)
    if scalar == -1:
        return int(pow(int(ciphertext), -1, int(pub.n_sq)))
    if scalar > 0:
        return int(pow(int(ciphertext), int(scalar), int(pub.n_sq)))
    inverse = pow(int(ciphertext), -1, int(pub.n_sq))
    return int(pow(int(inverse), int(-scalar), int(pub.n_sq)))


def _paillier_mul_plain_cached(
    pub: PaillierPublicKey,
    ciphertext: int,
    scalar: int,
    cache: dict[int, int],
) -> int:
    scalar = int(scalar)
    cached = cache.get(int(scalar))
    if cached is not None:
        return int(cached)
    out = _paillier_mul_plain(pub, int(ciphertext), int(scalar))
    cache[int(scalar)] = int(out)
    return int(out)


def _encrypt_query(pub: PaillierPublicKey, q_int: np.ndarray, rng: random.Random) -> tuple[list[int], int]:
    encrypted: list[int] = []
    total_bytes = 0
    for value in np.asarray(q_int, dtype=np.int64).tolist():
        c = _paillier_encrypt(pub, int(value), rng)
        encrypted.append(int(c))
        total_bytes += int(_int_size_bytes(c))
    return encrypted, int(total_bytes)


def _active_query_ciphertexts(
    encrypted_query: list[int],
    q_int: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    q_int = np.asarray(q_int, dtype=np.int64).reshape(-1)
    active_dims = np.flatnonzero(q_int != 0).astype(np.int32, copy=False)
    if int(active_dims.size) <= 0:
        return [], active_dims
    active_ciphertexts = [int(encrypted_query[int(idx)]) for idx in active_dims.tolist()]
    return active_ciphertexts, active_dims


def _encrypted_candidate_dot_products_serial(
    pub: PaillierPublicKey,
    encrypted_query_active: list[int],
    candidate_doc_ints_active: np.ndarray,
    scale_sq: int,
) -> tuple[list[int], int]:
    candidate_doc_ints_active = np.asarray(candidate_doc_ints_active, dtype=np.int64)
    encrypted_scores: list[int] = []
    total_bytes = 0
    enc_zero = _paillier_encrypt(pub, 0, random.Random(0))
    n_sq = int(pub.n_sq)
    encrypted_query_int = [int(x) for x in encrypted_query_active]
    per_dim_scalar_cache: list[dict[int, int]] = [{1: int(c_q)} for c_q in encrypted_query_int]
    for row in candidate_doc_ints_active:
        acc = int(enc_zero)
        for dim_idx, d_i in enumerate(row):
            scalar = int(d_i)
            if scalar == 0:
                continue
            lifted = _paillier_mul_plain_cached(
                pub,
                encrypted_query_int[dim_idx],
                scalar,
                per_dim_scalar_cache[dim_idx],
            )
            acc = int((int(acc) * int(lifted)) % n_sq)
        encrypted_distance = _paillier_add_plain(pub, acc, -int(scale_sq))
        encrypted_distance = _paillier_mul_plain(pub, encrypted_distance, -1)
        encrypted_scores.append(int(encrypted_distance))
        total_bytes += int(_int_size_bytes(encrypted_distance))
    return encrypted_scores, int(total_bytes)


def _encrypted_candidate_dot_products_worker(
    pub_payload: tuple[int, int, int],
    encrypted_query_active: list[int],
    candidate_doc_ints_active: np.ndarray,
    scale_sq: int,
) -> tuple[list[int], int]:
    pub = PaillierPublicKey(
        n=int(pub_payload[0]),
        g=int(pub_payload[1]),
        n_sq=int(pub_payload[2]),
    )
    return _encrypted_candidate_dot_products_serial(
        pub=pub,
        encrypted_query_active=encrypted_query_active,
        candidate_doc_ints_active=candidate_doc_ints_active,
        scale_sq=int(scale_sq),
    )


def _encrypted_candidate_dot_products(
    pub: PaillierPublicKey,
    encrypted_query: list[int],
    q_int: np.ndarray,
    candidate_doc_ints: np.ndarray,
    scale_sq: int,
    phe_workers: int = 1,
) -> tuple[list[int], int]:
    candidate_doc_ints = np.asarray(candidate_doc_ints, dtype=np.int64)
    encrypted_query_active, active_dims = _active_query_ciphertexts(
        encrypted_query=encrypted_query,
        q_int=q_int,
    )
    if int(active_dims.size) > 0:
        candidate_doc_ints_active = np.ascontiguousarray(candidate_doc_ints[:, active_dims], dtype=np.int64)
    else:
        candidate_doc_ints_active = np.zeros((int(candidate_doc_ints.shape[0]), 0), dtype=np.int64)

    workers = int(max(1, phe_workers))
    num_rows = int(candidate_doc_ints_active.shape[0])
    if workers <= 1 or num_rows <= 1:
        return _encrypted_candidate_dot_products_serial(
            pub=pub,
            encrypted_query_active=encrypted_query_active,
            candidate_doc_ints_active=candidate_doc_ints_active,
            scale_sq=int(scale_sq),
        )

    max_workers = int(min(workers, num_rows))
    chunk_rows = int(max(1, math.ceil(float(num_rows) / float(max_workers))))
    chunks = [
        np.ascontiguousarray(candidate_doc_ints_active[start : start + chunk_rows], dtype=np.int64)
        for start in range(0, num_rows, chunk_rows)
    ]
    pub_payload = (int(pub.n), int(pub.g), int(pub.n_sq))
    encrypted_scores: list[int] = []
    total_bytes = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                _encrypted_candidate_dot_products_worker,
                pub_payload,
                encrypted_query_active,
                chunk,
                int(scale_sq),
            )
            for chunk in chunks
        ]
        for fut in futures:
            chunk_scores, chunk_bytes = fut.result()
            encrypted_scores.extend(int(x) for x in chunk_scores)
            total_bytes += int(chunk_bytes)
    return encrypted_scores, int(total_bytes)


def _decrypt_distances(
    keypair: PaillierKeypair,
    encrypted_scores: list[int],
    scale_sq: int,
) -> np.ndarray:
    values = [_paillier_decrypt(keypair, int(c)) / float(scale_sq) for c in encrypted_scores]
    return np.asarray(values, dtype=np.float64)


def _generate_ot_runtime(bits: int, rng: random.Random) -> OTRuntime:
    prime = _generate_prime(int(bits), rng)
    return OTRuntime(prime=int(prime), generator=2)


def _hash_bytes(label: bytes, size: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < int(size):
        block = hashlib.sha256(label + counter.to_bytes(4, "big")).digest()
        out.extend(block)
        counter += 1
    return bytes(out[: int(size)])


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(left, right))


def _run_ot_transfer(
    *,
    runtime: OTRuntime,
    payloads: list[bytes],
    selected_positions: list[int],
    rng: random.Random,
) -> tuple[list[bytes], dict]:
    prime = int(runtime.prime)
    generator = int(runtime.generator)
    server_secret = rng.randrange(2, prime - 2)
    A = pow(generator, server_secret, prime)

    client_request_entries: list[dict] = []
    requested_lookup = set(int(x) for x in selected_positions)
    selected_b_values: dict[int, int] = {}
    B_values: list[int] = []
    for index in range(len(payloads)):
        b_i = rng.randrange(2, prime - 2)
        if int(index) in requested_lookup:
            B_i = pow(generator, b_i, prime)
            selected_b_values[int(index)] = int(b_i)
            bit = 0
        else:
            B_i = (pow(A, 1, prime) * pow(generator, b_i, prime)) % prime
            bit = 1
        B_values.append(int(B_i))
        client_request_entries.append(
            {
                "candidate_position": int(index),
                "masked_selection_bit": int(bit),
                "B_i_bytes": int(_int_size_bytes(B_i)),
            }
        )

    encrypted_messages: list[dict] = []
    server_response_bytes = 0
    for index, payload in enumerate(payloads):
        shared_value = pow(int(B_values[index]), int(server_secret), prime)
        key_stream = _hash_bytes(
            shared_value.to_bytes(_int_size_bytes(shared_value), "big"),
            len(payload),
        )
        cipher = _xor_bytes(payload, key_stream)
        encrypted_messages.append(
            {
                "candidate_position": int(index),
                "ciphertext_size_bytes": int(len(cipher)),
                "ciphertext": cipher,
            }
        )
        server_response_bytes += int(len(cipher))

    recovered_payloads: list[bytes] = []
    for index in selected_positions:
        b_i = int(selected_b_values[int(index)])
        shared_value = pow(A, b_i, prime)
        key_stream = _hash_bytes(
            shared_value.to_bytes(_int_size_bytes(shared_value), "big"),
            len(payloads[int(index)]),
        )
        cipher = encrypted_messages[int(index)]["ciphertext"]
        recovered_payloads.append(_xor_bytes(cipher, key_stream))

    trace = {
        "protocol": "chou_orlandi_style_k_out_of_kprime",
        "prime_bits": int(prime.bit_length()),
        "generator": int(generator),
        "server_init_A_bytes": int(_int_size_bytes(A)),
        "client_request_entries": client_request_entries,
        "server_encrypted_message_count": int(len(encrypted_messages)),
        "server_response_total_bytes": int(server_response_bytes),
        "selected_positions": [int(x) for x in selected_positions],
    }
    return recovered_payloads, trace


def _serialize_doc_payload(doc: dict) -> bytes:
    return json.dumps(doc, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _load_corpus_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            texts.append(str(row.get("text", "")))
    return texts


def _derive_module1_epsilon(
    *,
    cfg: RemoteRAGConfig,
    num_docs: int,
    epsilon: float,
    target_radius: float,
    prepare_assets: bool,
    reuse_existing: bool,
) -> float:
    if float(epsilon) > 0.0:
        return float(epsilon)
    _ = (cfg, num_docs, target_radius, prepare_assets, reuse_existing)
    return 0.0


def _module1_stem(num_docs: int, epsilon: float) -> str:
    return f"remoterag_module1_distancedp_n{int(num_docs)}_eps{int(round(float(epsilon)))}"


def _module1_target_radius_stem(num_docs: int, target_radius: float) -> str:
    text = str(f"{float(target_radius):g}").replace(".", "p")
    return f"remoterag_module1_distancedp_n{int(num_docs)}_targetr{text}"


def _safe_slug(text: str) -> str:
    chars: list[str] = []
    for ch in str(text).strip():
        if ch.isalnum() or ch in {"_", "-"}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_")


def _ensure_module1_outputs(
    *,
    project_root: Path,
    cfg: RemoteRAGConfig,
    num_docs: int,
    epsilon: float,
    target_radius: float,
    prepare_assets: bool,
    reuse_existing: bool,
    workset_name: str,
    module1_output_prefix: str,
    module1_query_limit: int,
    module1_k_prime_cap: int,
) -> tuple[Path, Path]:
    result_root = project_root / "results" / "repro_workflows" / "remoterag"
    if str(module1_output_prefix).strip():
        stem = str(module1_output_prefix).strip()
    elif float(epsilon) > 0.0:
        stem = _module1_stem(int(num_docs), float(epsilon))
    else:
        radius = float(target_radius) if float(target_radius) > 0.0 else float(cfg.default_module1_target_radius)
        stem = _module1_target_radius_stem(int(num_docs), float(radius))
    if str(workset_name).strip() and not str(module1_output_prefix).strip():
        stem = f"{stem}_{_safe_slug(str(workset_name))}"
    summary_path = result_root / f"{stem}.json"
    rows_path = result_root / f"{stem}.jsonl"
    if summary_path.exists() and rows_path.exists():
        return summary_path, rows_path
    script = project_root / "src" / "baselines" / "remoterag" / "module1_distancedp.py"
    cmd = [
        sys.executable,
        str(script),
        "--size",
        str(int(num_docs)),
        "--output-prefix",
        str(stem),
    ]
    if str(workset_name).strip():
        cmd.extend(["--workset-name", str(workset_name).strip()])
    if float(epsilon) > 0.0:
        cmd.extend(["--epsilon", str(float(epsilon))])
    elif float(target_radius) > 0.0:
        cmd.extend(["--target-radius", str(float(target_radius))])
    if int(module1_query_limit) > 0:
        cmd.extend(["--query-limit", str(int(module1_query_limit))])
    if int(module1_k_prime_cap) > 0:
        cmd.extend(["--k-prime-cap", str(int(module1_k_prime_cap))])
    if not bool(reuse_existing):
        cmd.append("--no-reuse-existing")
    subprocess.run(cmd, cwd=str(project_root), check=True)
    if not summary_path.exists() or not rows_path.exists():
        raise FileNotFoundError(
            f"RemoteRAG Module 1 outputs are still missing after generation attempt: {summary_path}, {rows_path}"
        )
    return summary_path, rows_path


def _quantize_embeddings(x: np.ndarray, scale: int) -> np.ndarray:
    return np.rint(np.asarray(x, dtype=np.float64) * float(scale)).astype(np.int64)


def _rank_smallest(values: np.ndarray, top_k: int) -> list[int]:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    return [int(x) for x in order[: int(top_k)].tolist()]


def _selected_payloads_from_positions(candidate_payloads: list[dict], positions: list[int]) -> list[dict]:
    return [candidate_payloads[int(pos)] for pos in positions]


def _direct_transfer_bytes(selected_payloads: list[dict], selected_positions: list[int]) -> tuple[int, int]:
    request_bytes = len(json.dumps({"positions": [int(x) for x in selected_positions]}, ensure_ascii=False).encode("utf-8"))
    response_bytes = sum(len(_serialize_doc_payload(doc)) for doc in selected_payloads)
    return int(request_bytes), int(response_bytes)


def _run_module2(
    *,
    project_root: Path,
    cfg: RemoteRAGConfig,
    summary_path: Path,
    rows_path: Path,
    query_limit: int,
    quantization_scale: int,
    paillier_bits: int,
    ot_prime_bits: int,
    seed: int,
    phe_workers: int,
    force_retrieve_mode: str,
) -> tuple[list[dict], dict, dict]:
    module1_summary = _load_json(summary_path)
    module1_rows = _load_jsonl(rows_path)
    if int(query_limit) > 0:
        module1_rows = module1_rows[: int(query_limit)]
    if not module1_rows:
        raise RuntimeError("module1 rows are empty; cannot run Module 2")

    input_paths = {key: Path(value) for key, value in module1_summary["input_paths"].items()}
    docs = _normalize_rows(np.load(input_paths["docs"]).astype(np.float32))
    queries = _normalize_rows(np.load(input_paths["queries"]).astype(np.float32))
    allow_synthetic = str(os.environ.get("ALLOW_SYNTHETIC_DOCID_TEXT_FALLBACK", "")).strip().lower() in {"1", "true", "yes", "on"}
    doc_ids, _doc_ids_are_synthetic = load_doc_ids_or_synthetic(
        input_paths["doc_ids"],
        num_docs=int(docs.shape[0]),
        allow_synthetic=bool(allow_synthetic),
        default_prefix=f"{input_paths['docs'].stem}_doc",
    )
    if input_paths["corpus"].exists():
        corpus_texts = _load_corpus_texts(input_paths["corpus"])
    elif bool(allow_synthetic):
        corpus_texts = LazySyntheticTexts(doc_ids=doc_ids)
    else:
        raise FileNotFoundError(f"missing required file: {input_paths['corpus']}")
    if len(corpus_texts) != len(doc_ids):
        raise RuntimeError(f"corpus/doc_ids size mismatch: {len(corpus_texts)} vs {len(doc_ids)}")

    key_rng = random.Random(int(seed))
    phe_t0 = time.perf_counter()
    keypair = _generate_paillier_keypair(int(paillier_bits), key_rng)
    phe_setup_sec = float(time.perf_counter() - phe_t0)
    ot_runtime = _generate_ot_runtime(int(ot_prime_bits), random.Random(int(seed) + 99173))

    rows_out: list[dict] = []
    ot_trace_rows: list[dict] = []
    retrieve_mode_counts = {"direct_indices": 0, "ot_k_out_of_kprime": 0}
    exact_order_matches: list[float] = []
    exact_overlap_counts: list[int] = []
    module1_cover_flags: list[float] = []
    order_matches_when_module1_covers: list[float] = []
    mean_abs_distance_errors: list[float] = []
    request_query_bytes: list[int] = []
    response_score_bytes: list[int] = []
    direct_transfer_request_bytes: list[int] = []
    direct_transfer_response_bytes: list[int] = []
    ot_total_client_bytes: list[int] = []
    ot_total_server_bytes: list[int] = []
    server_compute_sec_list: list[float] = []
    client_decrypt_sort_sec_list: list[float] = []
    client_encrypt_sec_list: list[float] = []
    retrieve_protocol_sec_list: list[float] = []
    total_request_bytes_list: list[int] = []
    total_response_bytes_list: list[int] = []
    total_latency_sec_list: list[float] = []

    for row_offset, module1_row in enumerate(module1_rows):
        query_index = int(module1_row["query_index"])
        query = _normalize_vec(queries[query_index])
        candidate_indices = [int(x) for x in module1_row["module1_candidate_indices"]]
        candidate_docs = docs[np.asarray(candidate_indices, dtype=np.int32)]
        candidate_doc_ids = [str(doc_ids[idx]) for idx in candidate_indices]
        candidate_texts = [str(corpus_texts[idx]) for idx in candidate_indices]
        candidate_payloads = [
            {
                "candidate_position": int(pos),
                "global_doc_index": int(global_idx),
                "doc_id": str(doc_id),
                "text": str(text),
            }
            for pos, (global_idx, doc_id, text) in enumerate(zip(candidate_indices, candidate_doc_ids, candidate_texts))
        ]

        q_int = _quantize_embeddings(query.reshape(1, -1), int(quantization_scale)).reshape(-1)
        d_int = _quantize_embeddings(candidate_docs, int(quantization_scale))
        scale_sq = int(int(quantization_scale) * int(quantization_scale))

        client_encrypt_t0 = time.perf_counter()
        enc_query, enc_query_bytes = _encrypt_query(
            keypair.public,
            q_int,
            random.Random(int(seed) + 1009 * int(row_offset) + 17),
        )
        client_encrypt_sec = float(time.perf_counter() - client_encrypt_t0)

        server_compute_t0 = time.perf_counter()
        enc_scores, enc_score_bytes = _encrypted_candidate_dot_products(
            keypair.public,
            enc_query,
            q_int,
            d_int,
            scale_sq,
            phe_workers=int(phe_workers),
        )
        server_compute_sec = float(time.perf_counter() - server_compute_t0)

        client_decrypt_t0 = time.perf_counter()
        decrypted_distances = _decrypt_distances(keypair, enc_scores, scale_sq)
        client_decrypt_sort_sec = float(time.perf_counter() - client_decrypt_t0)

        exact_distances = 1.0 - np.sum(candidate_docs * query.reshape(1, -1), axis=1, dtype=np.float64)
        abs_error = np.abs(np.asarray(decrypted_distances) - np.asarray(exact_distances))
        top_k = int(module1_row["k"])
        selected_positions = _rank_smallest(np.asarray(decrypted_distances), int(top_k))
        selected_payloads = _selected_payloads_from_positions(candidate_payloads, selected_positions)

        alpha_k = float(module1_row["original_alpha_k_rad"])
        omega_threshold = float(math.atan(math.tan(alpha_k) / math.sqrt(float(top_k))))
        distance_dp_bound = float(int(query.shape[0]) / float(module1_row["epsilon"]))
        if str(force_retrieve_mode) == "direct_indices":
            retrieve_mode = "direct_indices"
        elif str(force_retrieve_mode) == "ot_k_out_of_kprime":
            retrieve_mode = "ot_k_out_of_kprime"
        else:
            retrieve_mode = (
                "direct_indices"
                if float(omega_threshold) >= float(distance_dp_bound)
                else "ot_k_out_of_kprime"
            )

        recovered_payloads: list[dict]
        transfer_request_bytes = 0
        transfer_response_bytes = 0
        ot_trace = None
        transfer_t0 = time.perf_counter()
        if str(retrieve_mode) == "direct_indices":
            transfer_request_bytes, transfer_response_bytes = _direct_transfer_bytes(
                selected_payloads,
                selected_positions,
            )
            recovered_payloads = list(selected_payloads)
        else:
            ot_selected_bytes, ot_trace = _run_ot_transfer(
                runtime=ot_runtime,
                payloads=[_serialize_doc_payload(doc) for doc in candidate_payloads],
                selected_positions=selected_positions,
                rng=random.Random(int(seed) + 1000003 + int(row_offset)),
            )
            recovered_payloads = [json.loads(x.decode("utf-8")) for x in ot_selected_bytes]
            transfer_request_bytes = int(ot_trace["server_init_A_bytes"]) + sum(
                int(entry["B_i_bytes"]) for entry in ot_trace["client_request_entries"]
            )
            transfer_response_bytes = int(ot_trace["server_response_total_bytes"])
            ot_trace_rows.append(
                {
                    "query_id": str(module1_row["query_id"]),
                    "query_index": int(query_index),
                    "k": int(top_k),
                    "k_prime": int(len(candidate_payloads)),
                    "trace": ot_trace,
                }
            )
        retrieve_protocol_sec = float(time.perf_counter() - transfer_t0)

        selected_doc_ids = [str(doc["doc_id"]) for doc in recovered_payloads]
        selected_indices = [int(doc["global_doc_index"]) for doc in recovered_payloads]
        gt_doc_ids = [str(x) for x in module1_row["original_exact_topk_doc_ids"][:top_k]]
        module1_covers_exact_topk = bool(module1_row.get("include_original_exact_topk", False))
        overlap_count = int(len(set(selected_doc_ids).intersection(set(gt_doc_ids))))
        exact_order_match = bool(selected_doc_ids == gt_doc_ids)

        retrieve_mode_counts[str(retrieve_mode)] = int(retrieve_mode_counts.get(str(retrieve_mode), 0) + 1)
        exact_order_matches.append(1.0 if exact_order_match else 0.0)
        exact_overlap_counts.append(int(overlap_count))
        module1_cover_flags.append(1.0 if module1_covers_exact_topk else 0.0)
        if module1_covers_exact_topk:
            order_matches_when_module1_covers.append(1.0 if exact_order_match else 0.0)
        mean_abs_distance_errors.append(float(np.mean(abs_error)))
        request_query_bytes.append(int(enc_query_bytes))
        response_score_bytes.append(int(enc_score_bytes))
        direct_transfer_request_bytes.append(int(transfer_request_bytes))
        direct_transfer_response_bytes.append(int(transfer_response_bytes))
        if str(retrieve_mode) == "ot_k_out_of_kprime":
            ot_total_client_bytes.append(int(transfer_request_bytes))
            ot_total_server_bytes.append(int(transfer_response_bytes))
        server_compute_sec_list.append(float(server_compute_sec))
        client_decrypt_sort_sec_list.append(float(client_decrypt_sort_sec))
        client_encrypt_sec_list.append(float(client_encrypt_sec))
        retrieve_protocol_sec_list.append(float(retrieve_protocol_sec))
        total_request_bytes = int(enc_query_bytes) + int(transfer_request_bytes)
        total_response_bytes = int(enc_score_bytes) + int(transfer_response_bytes)
        total_latency_sec = float(
            client_encrypt_sec + server_compute_sec + client_decrypt_sort_sec + retrieve_protocol_sec
        )
        total_request_bytes_list.append(int(total_request_bytes))
        total_response_bytes_list.append(int(total_response_bytes))
        total_latency_sec_list.append(float(total_latency_sec))

        rows_out.append(
            {
                "query_index": int(query_index),
                "query_id": str(module1_row["query_id"]),
                "k": int(top_k),
                "k_prime": int(len(candidate_payloads)),
                "num_docs": int(module1_row["num_docs"]),
                "epsilon": float(module1_row["epsilon"]),
                "embedding_dim": int(query.shape[0]),
                "omega_threshold_rad": float(omega_threshold),
                "distance_dp_bound_n_over_epsilon": float(distance_dp_bound),
                "retrieve_mode": str(retrieve_mode),
                "time_client_encrypt_query_sec": float(client_encrypt_sec),
                "time_server_phe_score_sec": float(server_compute_sec),
                "time_client_decrypt_sort_sec": float(client_decrypt_sort_sec),
                "time_retrieve_protocol_sec": float(retrieve_protocol_sec),
                "latency_total_sec": float(total_latency_sec),
                "comm_client_encrypt_query_bytes": int(enc_query_bytes),
                "comm_server_score_response_bytes": int(enc_score_bytes),
                "comm_retrieve_request_bytes": int(transfer_request_bytes),
                "comm_retrieve_response_bytes": int(transfer_response_bytes),
                "comm_request_bytes_total": int(total_request_bytes),
                "comm_response_bytes_total": int(total_response_bytes),
                "comm_total_bytes": int(total_request_bytes + total_response_bytes),
                "module1_candidate_indices": [int(x) for x in candidate_indices],
                "module2_selected_positions": [int(x) for x in selected_positions],
                "module2_selected_indices": [int(x) for x in selected_indices],
                "module2_selected_doc_ids": [str(x) for x in selected_doc_ids],
                "module2_decrypted_distances": [float(x) for x in np.asarray(decrypted_distances).tolist()],
                "module2_exact_distances": [float(x) for x in np.asarray(exact_distances).tolist()],
                "module2_mean_abs_distance_error": float(np.mean(abs_error)),
                "module2_max_abs_distance_error": float(np.max(abs_error)),
                "module2_selected_text_lengths": [int(len(str(doc["text"]))) for doc in recovered_payloads],
                "module1_include_original_exact_topk": bool(module1_covers_exact_topk),
                "original_exact_topk_doc_ids": gt_doc_ids,
                "original_exact_topk_overlap_count": int(overlap_count),
                "original_exact_topk_order_match": bool(exact_order_match),
                "paper_condition": "atan(tan(alpha_k)/sqrt(k)) >= n/epsilon => direct retrieve else OT",
                "module2_payload_mode": (
                    "cloud_direct_doc_return"
                    if str(retrieve_mode) == "direct_indices"
                    else "cloud_ot_encrypted_doc_return"
                ),
            }
        )

    summary = {
        "module": "RemoteRAG Module 2",
        "paper_url": str(cfg.paper_url),
        "implementation_note": (
            "Implements a runnable prototype of PHE cosine-distance scoring and "
            "Theorem-3-based direct-vs-OT retrieval. PHE uses a pure-Python Paillier-style "
            "additively homomorphic scheme with fixed-point quantization; OT uses a "
            "Chou-Orlandi-style k-out-of-k' flow."
        ),
        "module1_summary_json": str(summary_path),
        "module1_rows_jsonl": str(rows_path),
        "num_queries": int(len(rows_out)),
        "top_k": int(rows_out[0]["k"]) if rows_out else 0,
        "paillier_modulus_bits": int(keypair.public.n.bit_length()),
        "paillier_setup_time_sec": float(phe_setup_sec),
        "ot_prime_bits": int(ot_runtime.prime.bit_length()),
        "quantization_scale": int(quantization_scale),
        "phe_workers": int(max(1, phe_workers)),
        "force_retrieve_mode": str(force_retrieve_mode),
        "retrieve_mode_counts": {
            str(key): int(value) for key, value in retrieve_mode_counts.items()
        },
        "cost_reporting_mode": "per_query_avg",
        "mean_abs_distance_error": float(np.mean(mean_abs_distance_errors)) if mean_abs_distance_errors else 0.0,
        "module1_candidate_cover_rate": float(np.mean(module1_cover_flags)) if module1_cover_flags else 0.0,
        "mean_exact_topk_overlap_count": float(np.mean(exact_overlap_counts)) if exact_overlap_counts else 0.0,
        "exact_topk_order_match_rate": float(np.mean(exact_order_matches)) if exact_order_matches else 0.0,
        "exact_topk_order_match_rate_when_module1_covers": (
            float(np.mean(order_matches_when_module1_covers))
            if order_matches_when_module1_covers
            else 0.0
        ),
        "latency_total_sec_avg": float(np.mean(total_latency_sec_list)) if total_latency_sec_list else 0.0,
        "latency_client_generate_sec_avg": float(np.mean(client_encrypt_sec_list)) if client_encrypt_sec_list else 0.0,
        "latency_server_query_sec_avg": float(np.mean(server_compute_sec_list)) if server_compute_sec_list else 0.0,
        "latency_client_recover_sec_avg": float(np.mean(client_decrypt_sort_sec_list)) if client_decrypt_sort_sec_list else 0.0,
        "time_retrieve_protocol_sec_avg": float(np.mean(retrieve_protocol_sec_list)) if retrieve_protocol_sec_list else 0.0,
        "comm_request_bytes_avg": float(np.mean(total_request_bytes_list)) if total_request_bytes_list else 0.0,
        "comm_response_bytes_avg": float(np.mean(total_response_bytes_list)) if total_response_bytes_list else 0.0,
        "comm_client_generate_query_mb": (float(np.mean(request_query_bytes)) / _MB) if request_query_bytes else 0.0,
        "comm_server_query_mb": (float(np.mean(response_score_bytes)) / _MB) if response_score_bytes else 0.0,
        "comm_two_stage_total_mb": (
            float(np.mean(total_request_bytes_list) + np.mean(total_response_bytes_list)) / _MB
            if total_request_bytes_list and total_response_bytes_list
            else 0.0
        ),
        "comm_client_encrypt_query_bytes_avg": float(np.mean(request_query_bytes)) if request_query_bytes else 0.0,
        "comm_server_score_response_bytes_avg": float(np.mean(response_score_bytes)) if response_score_bytes else 0.0,
        "comm_retrieve_request_bytes_avg": float(np.mean(direct_transfer_request_bytes)) if direct_transfer_request_bytes else 0.0,
        "comm_retrieve_response_bytes_avg": float(np.mean(direct_transfer_response_bytes)) if direct_transfer_response_bytes else 0.0,
        "comm_ot_client_bytes_avg": float(np.mean(ot_total_client_bytes)) if ot_total_client_bytes else 0.0,
        "comm_ot_server_bytes_avg": float(np.mean(ot_total_server_bytes)) if ot_total_server_bytes else 0.0,
        "time_client_encrypt_query_sec_avg": float(np.mean(client_encrypt_sec_list)) if client_encrypt_sec_list else 0.0,
        "time_server_phe_score_sec_avg": float(np.mean(server_compute_sec_list)) if server_compute_sec_list else 0.0,
        "time_client_decrypt_sort_sec_avg": float(np.mean(client_decrypt_sort_sec_list)) if client_decrypt_sort_sec_list else 0.0,
        "cost_stage_definition": {
            "latency_total_sec_avg": "per-query end-to-end Module 2 online latency = client PHE query encrypt + server PHE scoring + client decrypt/sort + retrieve protocol execution (direct return or OT payload transfer).",
            "comm_request_bytes_avg": "per-query total client->server online bytes across both online rounds: encrypted query request plus direct/OT retrieve request.",
            "comm_response_bytes_avg": "per-query total server->client online bytes across both online rounds: encrypted score response plus direct/OT retrieve response.",
            "comm_two_stage_total_mb": "per-query cross-boundary communication total over the full Module 2 online protocol.",
        },
        "workset_name": str(module1_summary["workset_name"]),
        "num_docs": int(module1_summary["num_docs"]),
        "epsilon": float(module1_summary["epsilon"]),
    }
    ot_trace_summary = {
        "module": "RemoteRAG Module 2 OT trace",
        "num_queries_with_ot": int(len(ot_trace_rows)),
        "rows": ot_trace_rows,
    }
    return rows_out, summary, ot_trace_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RemoteRAG Module 2 PHE + OT prototype on isolated assets.")
    parser.add_argument("--size", type=int, default=0, help="Database size. Default uses config.default_module1_size.")
    parser.add_argument("--workset-name", type=str, default="", help="Optional explicit PRISMA source workset name. Recommended when multiple datasets share the same size.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="DistanceDP epsilon used by Module 1. If <=0, derive from --target-radius.")
    parser.add_argument("--target-radius", type=float, default=0.0, help="Module 1 target expected radius. Used when --epsilon<=0.")
    parser.add_argument(
        "--prepare-assets",
        action="store_true",
        help="Build a public RemoteRAG asset manifest from ready PRISMA worksets before running.",
    )
    parser.set_defaults(reuse_existing=True)
    parser.add_argument("--reuse-existing", dest="reuse_existing", action="store_true")
    parser.add_argument("--no-reuse-existing", dest="reuse_existing", action="store_false")
    parser.add_argument("--module1-output-prefix", type=str, default="", help="Optional Module 1 output stem to consume.")
    parser.add_argument("--module1-query-limit", type=int, default=-1, help="Optional query limit forwarded when Module 1 must be generated.")
    parser.add_argument("--module1-k-prime-cap", type=int, default=0, help="Optional k' cap forwarded when Module 1 must be generated.")
    parser.add_argument("--query-limit", type=int, default=-1, help="Optional Module 2 query limit; <=0 uses config.default_module2_query_limit.")
    parser.add_argument("--seed", type=int, default=-1, help="RNG seed. <0 uses config.default_module2_seed.")
    parser.add_argument("--quantization-scale", type=int, default=0, help="Fixed-point quantization scale for PHE. <=0 uses config default.")
    parser.add_argument("--paillier-bits", type=int, default=0, help="Paillier modulus bits. <=0 uses config default.")
    parser.add_argument("--ot-prime-bits", type=int, default=0, help="OT group prime bits. <=0 uses config default.")
    parser.add_argument("--phe-workers", type=int, default=1, help="Number of worker processes used by the server-side PHE scoring loop. <=1 disables parallelism.")
    parser.add_argument(
        "--force-retrieve-mode",
        type=str,
        default="auto",
        choices=("auto", "direct_indices", "ot_k_out_of_kprime"),
        help="Override the paper condition and force Module 2 retrieval mode.",
    )
    parser.add_argument("--output-prefix", type=str, default="", help="Optional output stem override under results/repro_workflows/remoterag/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cfg = RemoteRAGConfig(project_root=project_root)
    if bool(args.prepare_assets):
        prep_script = project_root / "src" / "baselines" / "remoterag" / "prepare_assets.py"
        cmd = [sys.executable, str(prep_script)]
        if str(args.workset_name).strip():
            cmd.extend(["--workset-name", str(args.workset_name).strip()])
        subprocess.run(cmd, cwd=str(project_root), check=True)

    num_docs = int(args.size) if int(args.size) > 0 else int(cfg.default_module1_size)
    epsilon = _derive_module1_epsilon(
        cfg=cfg,
        num_docs=int(num_docs),
        epsilon=float(args.epsilon),
        target_radius=float(args.target_radius),
        prepare_assets=bool(args.prepare_assets),
        reuse_existing=bool(args.reuse_existing),
    )
    module1_query_limit = (
        int(args.module1_query_limit)
        if int(args.module1_query_limit) > 0
        else int(cfg.default_module1_query_limit)
    )
    summary_path, rows_path = _ensure_module1_outputs(
        project_root=project_root,
        cfg=cfg,
        num_docs=int(num_docs),
        epsilon=float(epsilon),
        target_radius=float(args.target_radius),
        prepare_assets=bool(args.prepare_assets),
        reuse_existing=bool(args.reuse_existing),
        workset_name=str(args.workset_name),
        module1_output_prefix=str(args.module1_output_prefix),
        module1_query_limit=int(module1_query_limit),
        module1_k_prime_cap=int(args.module1_k_prime_cap),
    )

    module1_summary = _load_json(summary_path)
    resolved_epsilon = float(module1_summary.get("epsilon", epsilon))

    query_limit = int(args.query_limit) if int(args.query_limit) > 0 else int(cfg.default_module2_query_limit)
    quantization_scale = (
        int(args.quantization_scale)
        if int(args.quantization_scale) > 0
        else int(cfg.default_module2_quantization_scale)
    )
    paillier_bits = int(args.paillier_bits) if int(args.paillier_bits) > 0 else int(cfg.default_module2_paillier_bits)
    ot_prime_bits = int(args.ot_prime_bits) if int(args.ot_prime_bits) > 0 else int(cfg.default_module2_ot_prime_bits)
    seed = int(args.seed) if int(args.seed) >= 0 else int(cfg.default_module2_seed)

    rows_out, summary, ot_trace = _run_module2(
        project_root=project_root,
        cfg=cfg,
        summary_path=summary_path,
        rows_path=rows_path,
        query_limit=int(query_limit),
        quantization_scale=int(quantization_scale),
        paillier_bits=int(paillier_bits),
        ot_prime_bits=int(ot_prime_bits),
        seed=int(seed),
        phe_workers=int(args.phe_workers),
        force_retrieve_mode=str(args.force_retrieve_mode),
    )

    if str(args.output_prefix).strip():
        stem = str(args.output_prefix).strip()
    else:
        base_stem = f"remoterag_module2_phe_ot_n{int(num_docs)}_eps{int(round(float(resolved_epsilon)))}"
        if str(args.workset_name).strip():
            stem = f"{base_stem}_{_safe_slug(str(args.workset_name))}"
        else:
            stem = base_stem
    result_root = project_root / "results" / "repro_workflows" / "remoterag"
    rows_jsonl = result_root / f"{stem}.jsonl"
    summary_json = result_root / f"{stem}.json"
    ot_trace_json = result_root / f"{stem}_ot_trace.json"
    compact_csv = result_root / f"{stem}_compact.csv"
    summary["rows_jsonl"] = str(rows_jsonl)
    summary["summary_json"] = str(summary_json)
    summary["ot_trace_json"] = str(ot_trace_json)
    summary["compact_csv"] = str(compact_csv)

    compact_rows = [
        {
            "query_index": int(row["query_index"]),
            "query_id": str(row["query_id"]),
            "retrieve_mode": str(row["retrieve_mode"]),
            "k": int(row["k"]),
            "k_prime": int(row["k_prime"]),
            "overlap_count": int(row["original_exact_topk_overlap_count"]),
            "order_match": int(bool(row["original_exact_topk_order_match"])),
            "mean_abs_distance_error": float(row["module2_mean_abs_distance_error"]),
            "time_server_phe_score_sec": float(row["time_server_phe_score_sec"]),
            "time_client_decrypt_sort_sec": float(row["time_client_decrypt_sort_sec"]),
        }
        for row in rows_out
    ]

    _write_jsonl(rows_jsonl, rows_out)
    _save_json(summary_json, summary)
    _save_json(ot_trace_json, ot_trace)
    _write_csv(
        compact_csv,
        compact_rows,
        fieldnames=[
            "query_index",
            "query_id",
            "retrieve_mode",
            "k",
            "k_prime",
            "overlap_count",
            "order_match",
            "mean_abs_distance_error",
            "time_server_phe_score_sec",
            "time_client_decrypt_sort_sec",
        ],
    )
    print(f"[saved] {rows_jsonl}")
    print(f"[saved] {summary_json}")
    print(f"[saved] {ot_trace_json}")
    print(f"[saved] {compact_csv}")


if __name__ == "__main__":
    main()
