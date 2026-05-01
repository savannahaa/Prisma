"""
离线聚类统一入口

当前仅保留 method4：论文 Step1 prototype + 固定容量修正（最小增量搬运）+ 单次几何冻结。
- 聚类阶段严格只使用 workset 文档 embedding（不引入 query）。
"""

# Allow running this file directly: `python src/offline/cluster_offline.py`
if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

import os
import pickle
import shutil

import numpy as np

from shared.config import (
    NUM_CLUSTERS,
    TARGET_CLUSTER_SIZE,
    EVAL_K,
    FIXED_K,
    WORKSET_CLUSTER_INFO_PATH,
    WORKSET_CLUSTER_SUMMARY_JSON,
)
from shared.cluster_info_contract import assert_cluster_info_contract
from offline.cluster_offline_method4_balanced_spherical import (
    OUT_JSON as METHOD4_OUT_JSON,
    OUT_PKL as METHOD4_OUT_PKL,
    main as run_method4_main,
)


def _copy_if_needed(src_path: str, dst_path: str):
    src_abs = os.path.abspath(src_path)
    dst_abs = os.path.abspath(dst_path)
    if src_abs == dst_abs:
        return
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    shutil.copyfile(src_abs, dst_abs)


def _validate_cluster_info(path: str):
    with open(path, "rb") as f:
        cluster_info = pickle.load(f)

    chunks = cluster_info.get("chunks", [])
    centers = np.asarray(cluster_info.get("centers", []), dtype=np.float32)
    cluster_r_k = np.asarray(cluster_info.get("cluster_r_k", []), dtype=np.float32)
    cluster_r_fixed = np.asarray(cluster_info.get("cluster_r_fixed", []), dtype=np.float32)
    cluster_r_max = np.asarray(cluster_info.get("cluster_r_max", []), dtype=np.float32)

    if len(chunks) != int(NUM_CLUSTERS):
        raise RuntimeError(
            f"method4 输出簇数错误：expected {NUM_CLUSTERS}, got {len(chunks)}"
        )

    for cid, chunk in enumerate(chunks):
        if len(chunk) != int(TARGET_CLUSTER_SIZE):
            raise RuntimeError(
                f"method4 输出簇容量错误：cluster {cid}, expected {TARGET_CLUSTER_SIZE}, got {len(chunk)}"
            )

    if centers.shape[0] != int(NUM_CLUSTERS):
        raise RuntimeError(
            f"method4 centers 数量错误：expected {NUM_CLUSTERS}, got {centers.shape[0]}"
        )

    if len(cluster_r_k) != int(NUM_CLUSTERS) or len(cluster_r_fixed) != int(NUM_CLUSTERS):
        raise RuntimeError(
            "method4 半径向量长度错误："
            f"len(cluster_r_k)={len(cluster_r_k)}, len(cluster_r_fixed)={len(cluster_r_fixed)}"
        )
    if len(cluster_r_max) != int(NUM_CLUSTERS):
        raise RuntimeError(
            "method4 cluster_r_max 长度错误："
            f"expected {NUM_CLUSTERS}, got {len(cluster_r_max)}"
        )

    assert_cluster_info_contract(
        cluster_info,
        expected_eval_k=int(EVAL_K),
        expected_fixed_k=int(FIXED_K),
        expected_num_clusters=int(NUM_CLUSTERS),
        expected_target_cluster_size=int(TARGET_CLUSTER_SIZE),
    )


# 统一入口：固定执行 method4，然后同步到默认路径。
def main():
    print("=" * 90)
    print("Offline clustering entry (method4-only, step1+fixed-capacity-repair+freeze)")
    print("=" * 90)

    run_method4_main()
    _validate_cluster_info(METHOD4_OUT_PKL)

    _copy_if_needed(METHOD4_OUT_PKL, WORKSET_CLUSTER_INFO_PATH)
    _copy_if_needed(METHOD4_OUT_JSON, WORKSET_CLUSTER_SUMMARY_JSON)

    print("=" * 90)
    print("Method4 artifacts synchronized")
    print(f"METHOD4_OUT_PKL               : {METHOD4_OUT_PKL}")
    print(f"METHOD4_OUT_JSON              : {METHOD4_OUT_JSON}")
    print(f"WORKSET_CLUSTER_INFO_PATH     : {WORKSET_CLUSTER_INFO_PATH}")
    print(f"WORKSET_CLUSTER_SUMMARY_JSON  : {WORKSET_CLUSTER_SUMMARY_JSON}")
    print("=" * 90)


if __name__ == "__main__":
    main()
