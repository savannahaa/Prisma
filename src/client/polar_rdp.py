"""
Track-1（dense 分支）的 query 扰动实现。

流程：
1) 用 alpha/epsilon 计算 sigma 与 r_rdp_bar。
2) 在 [r_rdp_bar, r_max] 上做截断采样半径 r（sigma * chi_n）。
3) 在单位球面采样方向 v，得到扰动 query：e' = e + r v。
4) 输出几何诊断量（例如 Delta_alpha）供日志与分析使用。

说明：
- 这里只负责扰动，不负责候选检索与最终 rerank。
"""

from __future__ import annotations

# Allow running this file directly: `python src/client/polar_rdp.py`
if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import numpy as np

from shared.config import EPS


def compute_sigma(alpha: float, epsilon: float) -> float:
    # sigma 公式：sqrt(alpha / (2 * epsilon))
    return float(np.sqrt(alpha / (2.0 * epsilon)))


def compute_r_rdp_bar(alpha: float, epsilon: float, dim: int) -> float:
    # r_bar_RDP 公式：sqrt(n * alpha / (2 * epsilon)) = sigma * sqrt(dim)
    sigma = compute_sigma(alpha, epsilon)
    return float(sigma * np.sqrt(dim))


def sample_uniform_sphere(dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    在单位球面 S^{n-1} 上均匀采样方向向量 v。
    做法：
      t ~ N(0, I_n)
      v = t / ||t||_2
    """
    while True:
        t = rng.standard_normal(dim).astype(np.float32)
        norm = float(np.linalg.norm(t))
        if norm > EPS:
            return (t / norm).astype(np.float32)


def sample_sigma_chi_radius(
    sigma: float,
    dim: int,
    rng: np.random.Generator,
) -> float:
    """
    r ~ sigma * chi_n
    等价于：
      z ~ N(0, I_n)
      r = sigma * ||z||_2
    """
    z = rng.standard_normal(dim).astype(np.float32)
    return float(sigma * np.linalg.norm(z))


def sample_truncated_sigma_chi_radius(
    sigma: float,
    dim: int,
    low: float,
    high: float,
    rng: np.random.Generator,
    max_trials: int = 300000,
) -> float:
    """
    在 [low, high] 上做截断 sigma*chi_n 采样
    """
    if high < low:
        raise ValueError(
            f"非法截断区间：high={high:.10f} < low={low:.10f}"
        )

    if abs(high - low) <= 1e-12:
        return float(low)

    for _ in range(max_trials):
        r = sample_sigma_chi_radius(sigma, dim, rng)
        if low <= r <= high:
            return float(r)

    raise RuntimeError(
        "截断 χ 半径采样失败：请检查 [r_bar_RDP, r_max] 是否过窄，或增大 max_trials。"
    )


def compute_delta_alpha(query: np.ndarray, sampled_r: float) -> tuple[float, float]:
    """
    按论文 Algorithm 1 显式输出角偏移量：
        Delta_alpha = arctan(r / ||e_k||_2)
    """
    query = np.asarray(query, dtype=np.float32)
    query_l2 = float(np.linalg.norm(query))

    if query_l2 <= EPS:
        raise ValueError(
            f"query 的 L2 范数过小，无法计算 Delta_alpha：||query||={query_l2:.10f}"
        )

    delta_alpha_rad = float(np.arctan(sampled_r / query_l2))
    delta_alpha_deg = float(np.degrees(delta_alpha_rad))
    return delta_alpha_rad, delta_alpha_deg


def perturb_query_track1(
    query: np.ndarray,
    alpha: float,
    epsilon: float,
    r_max: float,
    rng: np.random.Generator,
    forced_radius: float | None = None,
    force_clip_to_r_max: bool = True,
):
    """
    Track 1 / Dense-region RDP retrieval

    贴合论文的步骤：
      1) 计算 sigma 和 r_bar_RDP
      2) 采样 r ~ sigma * chi_n，且截断到 [r_bar_RDP, r_max]
         - 若 forced_radius 不为 None，则改为使用固定扰动半径；
         - 此时默认只保留上界约束：r = min(forced_radius, r_max)。
      3) 在单位球面上采样方向 v
      4) 构造 e' = e + r v
      5) 显式输出 Delta_alpha = arctan(r / ||e||_2)

    注意：
      - e' 不做归一化；当前论文公式写的是 e' = e + r v
      - Delta_alpha 用原始 query 的 L2 范数来算
    """
    query = np.asarray(query, dtype=np.float32)
    dim = int(query.shape[0])

    sigma = compute_sigma(alpha, epsilon)
    r_rdp_bar = compute_r_rdp_bar(alpha, epsilon, dim)

    if r_max < r_rdp_bar:
        raise ValueError(
            f"当前 query 不满足 dense 条件：r_max={r_max:.10f} < r_rdp_bar={r_rdp_bar:.10f}"
        )

    radius_mode = "truncated_sigma_chi"
    requested_r = None
    sampled_r_raw = None
    sampled_r_clipped_to_r_max = False
    if forced_radius is not None:
        requested_r = float(forced_radius)
        if requested_r < 0.0:
            raise ValueError(f"forced_radius must be >= 0, got {requested_r:.10f}")
        radius_mode = "fixed_override"
        sampled_r_raw = float(requested_r)
        r = float(requested_r)
        if bool(force_clip_to_r_max) and r > float(r_max):
            r = float(r_max)
            sampled_r_clipped_to_r_max = True
    else:
        r = sample_truncated_sigma_chi_radius(
            sigma=sigma,
            dim=dim,
            low=r_rdp_bar,
            high=r_max,
            rng=rng,
        )
        sampled_r_raw = float(r)

    v = sample_uniform_sphere(dim, rng)

    delta = (r * v).astype(np.float32)
    perturbed_query = (query + delta).astype(np.float32)

    delta_l2 = float(np.linalg.norm(delta))
    if abs(delta_l2 - r) > 1e-5:
        raise RuntimeError(
            f"内部检查失败：||delta||={delta_l2:.10f} 与 sampled_r={r:.10f} 不一致。"
        )

    query_l2 = float(np.linalg.norm(query))
    delta_alpha_rad, delta_alpha_deg = compute_delta_alpha(query, r)

    return {
        "sigma": float(sigma),
        "r_rdp_bar": float(r_rdp_bar),
        "radius_mode": str(radius_mode),
        "requested_r": float(requested_r) if requested_r is not None else None,
        "sampled_r": float(r),
        "sampled_r_raw": float(sampled_r_raw),
        "sampled_r_clipped_to_r_max": bool(sampled_r_clipped_to_r_max),

        "query_l2": float(query_l2),
        "direction": v,
        "delta": delta,
        "delta_l2": float(delta_l2),

        "delta_alpha_rad": float(delta_alpha_rad),
        "delta_alpha_deg": float(delta_alpha_deg),

        "perturbed_query": perturbed_query,
        "perturbed_query_for_server": perturbed_query.copy(),
    }
