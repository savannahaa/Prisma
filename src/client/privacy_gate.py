"""
在线 gate 核心逻辑（论文公式的直接实现）。

新版口径（2026-04-08 论文修订）：
- 先找最近簇中心；
- 直接查离线 cluster-level 容量 R_max^(i)（即 cluster_r_max[i]）；
- 再根据 alpha、epsilon 和维度算 r_rdp_bar；
- 最后比较 cluster_r_max[i] 和 r_rdp_bar，决定该 query 是否仍在当前 track1/dense 主线公开范围内。

这个文件现在也负责 epsilon 区间裁剪逻辑：
如果给了 epsilon_min / epsilon_max，并且 RRDP_ENFORCE_EPSILON_INTERVAL=True，就把输入 epsilon 裁到可行区间内；
如果区间不可行，就只做 warning，不直接强行改 track。
"""

import numpy as np

EPS = 1e-12


def cosine_similarity_to_centers(query: np.ndarray, centers: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    centers = np.asarray(centers, dtype=np.float32)
    sims = centers @ query
    return np.clip(sims, -1.0, 1.0).astype(np.float32)


def angular_distance_to_centers(query: np.ndarray, centers: np.ndarray) -> np.ndarray:
    sims = cosine_similarity_to_centers(query=query, centers=centers)
    return np.arccos(sims).astype(np.float32)


def compute_sigma(alpha: float, epsilon: float) -> float:
    return float(np.sqrt(alpha / (2.0 * epsilon)))


def compute_r_rdp_bar(alpha: float, epsilon: float, dim: int) -> float:
    sigma = compute_sigma(alpha, epsilon)
    return float(sigma * np.sqrt(dim))


def choose_track(r_rdp_bar: float, r_max: float) -> str:
    return "dense_rdp" if r_max >= r_rdp_bar else "out_of_scope"


def resolve_epsilon_used(
    epsilon: float,
    epsilon_min: float = None,
    epsilon_max: float = None,
    enforce_epsilon_interval: bool = False,
):
    epsilon_input = float(epsilon)
    eps_min = float(epsilon_min) if epsilon_min is not None else None
    eps_max = float(epsilon_max) if epsilon_max is not None else None

    interval_defined = eps_min is not None and eps_max is not None
    interval_feasible = bool(
        interval_defined
        and np.isfinite(eps_min)
        and np.isfinite(eps_max)
        and eps_min > 0.0
        and eps_max > 0.0
        and eps_min <= eps_max
    )

    if bool(enforce_epsilon_interval) and interval_feasible:
        epsilon_used = float(np.clip(epsilon_input, eps_min, eps_max))
    else:
        epsilon_used = float(epsilon_input)

    return {
        "epsilon_input": float(epsilon_input),
        "epsilon_min": float(eps_min) if eps_min is not None else None,
        "epsilon_max": float(eps_max) if eps_max is not None else None,
        "epsilon_interval_defined": bool(interval_defined),
        "epsilon_interval_feasible": bool(interval_feasible),
        "epsilon_clipped": bool(abs(epsilon_used - epsilon_input) > 1e-12),
        "epsilon_used": float(epsilon_used),
    }


# 论文核心（新版）：对每个查询，找到最近簇中心后查离线 cluster_r_max，
# 再与 r_rdp_bar 比较选择 track。
# d_to_center / (d_k, d_fixed) 仅保留为诊断字段，不参与新版 gate 判定。
def gate_one_query(
    query: np.ndarray,
    centers: np.ndarray,
    cluster_r_max: np.ndarray,
    alpha: float,
    epsilon: float,
    cluster_r_k: np.ndarray = None,
    cluster_r_fixed: np.ndarray = None,
    epsilon_min: float = None,
    epsilon_max: float = None,
    enforce_epsilon_interval: bool = False,
):
    query = np.asarray(query, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    cluster_r_max = np.asarray(cluster_r_max, dtype=np.float32)
    cluster_r_k = np.asarray(cluster_r_k, dtype=np.float32) if cluster_r_k is not None else None
    cluster_r_fixed = np.asarray(cluster_r_fixed, dtype=np.float32) if cluster_r_fixed is not None else None

    if len(centers) == 0:
        raise ValueError("centers 为空。")
    if len(cluster_r_max) != len(centers):
        raise ValueError(
            f"cluster_r_max 与 centers 长度不一致：{len(cluster_r_max)} vs {len(centers)}"
        )
    if (cluster_r_k is not None) and (len(cluster_r_k) != len(centers)):
        raise ValueError(f"cluster_r_k 与 centers 长度不一致：{len(cluster_r_k)} vs {len(centers)}")
    if (cluster_r_fixed is not None) and (len(cluster_r_fixed) != len(centers)):
        raise ValueError(
            f"cluster_r_fixed 与 centers 长度不一致：{len(cluster_r_fixed)} vs {len(centers)}"
        )

    dim = int(query.shape[0])
    eps_info = resolve_epsilon_used(
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_max=epsilon_max,
        enforce_epsilon_interval=bool(enforce_epsilon_interval),
    )

    sigma = compute_sigma(alpha, float(eps_info["epsilon_used"]))
    r_rdp_bar = compute_r_rdp_bar(alpha, float(eps_info["epsilon_used"]), dim)

    # 在单位球上按最小角距离寻找最近父簇中心。
    theta = angular_distance_to_centers(query=query, centers=centers)
    cluster_id = int(np.argmin(theta))
    d_to_center = float(theta[cluster_id])
    d_k = float(cluster_r_k[cluster_id]) if cluster_r_k is not None else float("nan")
    d_fixed = float(cluster_r_fixed[cluster_id]) if cluster_r_fixed is not None else float("nan")
    r_max = float(cluster_r_max[cluster_id])

    # 新版论文口径：只按“离线 cluster-level 容量 vs r_rdp_bar”判定 track。
    track = choose_track(r_rdp_bar=r_rdp_bar, r_max=r_max)
    track_reason = "cluster_r_max_vs_r_rdp_bar"
    routing_cluster_source = "nearest_center_angular"
    routing_num_clusters = 1
    interval_infeasible_warning = bool(
        bool(eps_info["epsilon_interval_defined"])
        and (not bool(eps_info["epsilon_interval_feasible"]))
    )

    return {
        "cluster_id": cluster_id,
        "d_to_center": d_to_center,
        "d_k": d_k,
        "d_fixed": d_fixed,
        "r_max": float(r_max),
        "r_max_cluster": float(r_max),
        "r_max_mode": "cluster_level_quantile_offline",
        "sigma": float(sigma),
        "r_rdp_bar": float(r_rdp_bar),
        "track": track,
        "track_reason": str(track_reason),
        "routing_cluster_source": str(routing_cluster_source),
        "routing_num_clusters": int(routing_num_clusters),
        "epsilon_input": float(eps_info["epsilon_input"]),
        "epsilon_used": float(eps_info["epsilon_used"]),
        "epsilon_min": eps_info["epsilon_min"],
        "epsilon_max": eps_info["epsilon_max"],
        "epsilon_interval_defined": bool(eps_info["epsilon_interval_defined"]),
        "epsilon_interval_feasible": bool(eps_info["epsilon_interval_feasible"]),
        "epsilon_clipped": bool(eps_info["epsilon_clipped"]),
        "interval_infeasible_warning": bool(interval_infeasible_warning),
    }
