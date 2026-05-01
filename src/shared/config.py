"""
全流程统一配置中心。

职责：
1) 统一声明数据/结果/模型的路径，避免在各脚本里硬编码路径。
2) 统一声明论文关键参数（EVAL_K、FIXED_K、ALPHA、EPSILON）。
3) 作为脚本间契约：prepare/cluster/query/online 都从这里拿同一套路径和参数。

你可以把这个文件看成“流水线 wiring 图”：
- raw 输入：`data/raw/*.jsonl|*.tsv`
- 中间产物：`data/*.npy|*.json`
- 在线输出：`results/*`
- 训练权重：`artifacts/*`
"""

import os

from shared.cpu_threads import configure_runtime_cpu_threads


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CPU_NUM_THREADS = int(configure_runtime_cpu_threads(default_threads=32))


def _with_suffix(path: str, suffix: str) -> str:
    if not suffix:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return int(default)


def _env_optional_float(name: str):
    raw = os.environ.get(name)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return str(default)
    text = str(raw).strip()
    return text if text else str(default)


def _env_int_list(name: str, default):
    raw = os.environ.get(name)
    if raw is None:
        return [int(x) for x in default]
    vals = []
    for part in str(raw).replace(";", ",").split(","):
        text = str(part).strip()
        if not text:
            continue
        try:
            vals.append(int(float(text)))
        except Exception:
            continue
    return [int(x) for x in vals] if vals else [int(x) for x in default]


PIPELINE_VARIANT = str(os.environ.get("PIPELINE_VARIANT", "")).strip().lower()
PIPELINE_IS_PAPERFAITHFUL_MAINLINE = PIPELINE_VARIANT == "paperfaithful_mainline"
PIPELINE_OUTPUT_TAG = _env_str("PIPELINE_OUTPUT_TAG", "")
PIPELINE_OUTPUT_SUFFIX = "_paperfaithful_mainline" if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else ""
if PIPELINE_OUTPUT_TAG:
    PIPELINE_OUTPUT_SUFFIX = f"{PIPELINE_OUTPUT_SUFFIX}_{PIPELINE_OUTPUT_TAG}"

# ===== paperfaithful_mainline 最优参数（用户锁定） =====
# 后续凡是口头说“用最优参数”，默认且仅指这一组。
# 如需改动，必须由用户显式重新指定。
PAPERFAITHFUL_MAINLINE_OPTIMAL_ALPHA = 2.0
PAPERFAITHFUL_MAINLINE_OPTIMAL_C = 7
PAPERFAITHFUL_MAINLINE_OPTIMAL_FIXED_K = 300
PAPERFAITHFUL_MAINLINE_OPTIMAL_EPSILON = 175000.0
PAPERFAITHFUL_MAINLINE_OPTIMAL_PRESET_NAME = "paperfaithful_mainline_locked_optimal_v1"
PAPERFAITHFUL_MAINLINE_OPTIMAL_PRESET = {
    "name": PAPERFAITHFUL_MAINLINE_OPTIMAL_PRESET_NAME,
    "alpha": float(PAPERFAITHFUL_MAINLINE_OPTIMAL_ALPHA),
    "routing_c": int(PAPERFAITHFUL_MAINLINE_OPTIMAL_C),
    "fixed_k": int(PAPERFAITHFUL_MAINLINE_OPTIMAL_FIXED_K),
    "epsilon": float(PAPERFAITHFUL_MAINLINE_OPTIMAL_EPSILON),
}
PAPERFAITHFUL_MAINLINE_GATE_MODE = _env_str(
    "PAPERFAITHFUL_MAINLINE_GATE_MODE",
    "auto",
).strip().lower()
if PAPERFAITHFUL_MAINLINE_GATE_MODE not in {"auto", "force_dense"}:
    raise ValueError(
        "upload-ready release keeps only the dense mainline path; "
        "PAPERFAITHFUL_MAINLINE_GATE_MODE must be one of: auto, force_dense"
    )
_track1_only_override = os.environ.get("PAPERFAITHFUL_MAINLINE_TRACK1_ONLY")
if _track1_only_override is not None and str(_track1_only_override).strip().lower() in {
    "0",
    "false",
    "no",
    "off",
}:
    raise ValueError(
        "upload-ready release keeps only the track1/dense mainline path; "
        "PAPERFAITHFUL_MAINLINE_TRACK1_ONLY cannot be disabled here"
    )
PAPERFAITHFUL_MAINLINE_TRACK1_ONLY = True

# ===== 工作集参数 =====
NUM_WORKSET_DOCS = int(max(1, _env_int("NUM_WORKSET_DOCS", 10000)))
NUM_CLUSTERS = int(max(1, _env_int("NUM_CLUSTERS", 20)))
_default_target_cluster_size = int(max(1, NUM_WORKSET_DOCS // max(1, NUM_CLUSTERS)))
TARGET_CLUSTER_SIZE = int(
    max(
        1,
        _env_int("TARGET_CLUSTER_SIZE", _default_target_cluster_size),
    )
)
if int(NUM_CLUSTERS) * int(TARGET_CLUSTER_SIZE) != int(NUM_WORKSET_DOCS):
    raise ValueError(
        "invalid fixed cluster setup: "
        f"NUM_CLUSTERS({NUM_CLUSTERS}) * TARGET_CLUSTER_SIZE({TARGET_CLUSTER_SIZE}) "
        f"!= NUM_WORKSET_DOCS({NUM_WORKSET_DOCS})"
    )

WORKSET_NAMESPACE = _env_str("WORKSET_NAMESPACE", "e5")
WORKSET_NAME_OVERRIDE = _env_str("WORKSET_NAME_OVERRIDE", "")
WORKSET_NAME = (
    str(WORKSET_NAME_OVERRIDE)
    if WORKSET_NAME_OVERRIDE
    else f"{WORKSET_NAMESPACE}_workset_{int(NUM_WORKSET_DOCS)}"
)
WORKSET_DOCS_STEM = f"docs_{WORKSET_NAME}"
WORKSET_DOC_IDS_STEM = f"doc_ids_{WORKSET_NAME}"
WORKSET_META_STEM = f"meta_{WORKSET_NAME}"
WORKSET_CORPUS_STEM = f"corpus_{WORKSET_NAME}"
WORKSET_QUERIES_STEM = f"queries_{WORKSET_NAME}"
WORKSET_QUERY_IDS_STEM = f"query_ids_{WORKSET_NAME}"
WORKSET_GT_TOPK_STEM = f"gt_topk_{WORKSET_NAME}"
WORKSET_CLUSTER_INFO_STEM = f"cluster_info_{WORKSET_NAME}_balanced_spherical"
WORKSET_QUERY_SPLIT_STEM = f"queries_{WORKSET_NAME}_split"
WORKSET_QUERY_CALIBRATION_STEM = f"queries_{WORKSET_NAME}_calibration"
WORKSET_QUERY_IDS_CALIBRATION_STEM = f"query_ids_{WORKSET_NAME}_calibration"
WORKSET_QRELS_STEM = f"qrels_{WORKSET_NAME}"
ONLINE_RESULTS_STEM = f"online_results_{WORKSET_NAME}"
ONLINE_SUMMARY_STEM = f"online_summary_{WORKSET_NAME}"
RRDP_PROFILE_STEM = f"rrdp_profile_{WORKSET_NAME}"

# ===== 原始全量语料 / query-free passage cache =====
CORPUS_JSONL_PATH = _env_str("CORPUS_JSONL_PATH", os.path.join(RAW_DIR, "corpus.jsonl"))

# 当前语义：这是 query-free passage cache，不再使用“targeted_pool”命名。
PASSAGE_CACHE_DOCS_PATH = _with_suffix(
    os.path.join(DATA_DIR, "docs_e5_passage_cache.npy"),
    PIPELINE_OUTPUT_SUFFIX,
)
PASSAGE_CACHE_DOC_IDS_PATH = _with_suffix(
    os.path.join(DATA_DIR, "doc_ids_e5_passage_cache.npy"),
    PIPELINE_OUTPUT_SUFFIX,
)
PASSAGE_CACHE_META_PATH = _with_suffix(
    os.path.join(DATA_DIR, "meta_e5_passage_cache.json"),
    PIPELINE_OUTPUT_SUFFIX,
)
PASSAGE_CACHE_PARTIAL_PATH = _with_suffix(
    os.path.join(DATA_DIR, "docs_build_partial.npy"),
    PIPELINE_OUTPUT_SUFFIX,
)
PASSAGE_CACHE_PROGRESS_PATH = _with_suffix(
    os.path.join(DATA_DIR, "docs_build_progress.json"),
    PIPELINE_OUTPUT_SUFFIX,
)

# ===== 原始真实 query / qrels =====
FULL_QUERIES_JSONL_PATH = _env_str("FULL_QUERIES_JSONL_PATH", os.path.join(RAW_DIR, "queries.jsonl"))
FULL_QRELS_TSV_PATH = _env_str("FULL_QRELS_TSV_PATH", os.path.join(RAW_DIR, "qrels.tsv"))

# ===== 当前 E5 工作集 =====
WORKSET_DOCS_PATH = _with_suffix(
    os.path.join(DATA_DIR, f"{WORKSET_DOCS_STEM}.npy"),
    PIPELINE_OUTPUT_SUFFIX,
)
WORKSET_DOC_IDS_PATH = _with_suffix(
    os.path.join(DATA_DIR, f"{WORKSET_DOC_IDS_STEM}.npy"),
    PIPELINE_OUTPUT_SUFFIX,
)
WORKSET_CORPUS_JSONL_PATH = _with_suffix(
    os.path.join(RAW_DIR, f"{WORKSET_CORPUS_STEM}.jsonl"),
    PIPELINE_OUTPUT_SUFFIX,
)
WORKSET_META_PATH = _with_suffix(
    os.path.join(DATA_DIR, f"{WORKSET_META_STEM}.json"),
    PIPELINE_OUTPUT_SUFFIX,
)

# ===== 离线 proxy map（论文口径：每个 parent cluster 只有一个 centroid） =====
# method4（唯一保留）路径：纯离线、无 query 参与的 balanced spherical 聚类。
WORKSET_CLUSTER_INFO_METHOD4_PATH = os.path.join(
    RESULTS_DIR, f"{WORKSET_CLUSTER_INFO_STEM}{PIPELINE_OUTPUT_SUFFIX}.pkl"
)
WORKSET_CLUSTER_SUMMARY_METHOD4_JSON = os.path.join(
    RESULTS_DIR, f"{WORKSET_CLUSTER_INFO_STEM}{PIPELINE_OUTPUT_SUFFIX}.json"
)

# 全流程默认读取的 cluster_info（run_online_pipeline / cluster_retrieval 等都会使用）。
WORKSET_CLUSTER_INFO_PATH = WORKSET_CLUSTER_INFO_METHOD4_PATH
WORKSET_CLUSTER_SUMMARY_JSON = WORKSET_CLUSTER_SUMMARY_METHOD4_JSON

# ===== 真实 query 工作集 =====
WORKSET_QUERIES_PATH = _with_suffix(
    os.path.join(DATA_DIR, f"{WORKSET_QUERIES_STEM}.npy"),
    PIPELINE_OUTPUT_SUFFIX,
)
WORKSET_QUERY_IDS_PATH = _with_suffix(
    os.path.join(DATA_DIR, f"{WORKSET_QUERY_IDS_STEM}.npy"),
    PIPELINE_OUTPUT_SUFFIX,
)
WORKSET_GT_TOPK_PATH = _with_suffix(
    os.path.join(DATA_DIR, f"{WORKSET_GT_TOPK_STEM}.npy"),
    PIPELINE_OUTPUT_SUFFIX,
)
WORKSET_CALIBRATION_QUERIES_PATH = _with_suffix(
    os.path.join(DATA_DIR, f"{WORKSET_QUERY_CALIBRATION_STEM}.npy"),
    PIPELINE_OUTPUT_SUFFIX,
)
WORKSET_CALIBRATION_QUERY_IDS_PATH = _with_suffix(
    os.path.join(DATA_DIR, f"{WORKSET_QUERY_IDS_CALIBRATION_STEM}.npy"),
    PIPELINE_OUTPUT_SUFFIX,
)
WORKSET_QRELS_PATH = _with_suffix(
    os.path.join(RAW_DIR, f"{WORKSET_QRELS_STEM}.tsv"),
    PIPELINE_OUTPUT_SUFFIX,
)
WORKSET_RELAXED_QRELS_PATH = _with_suffix(
    os.path.join(RAW_DIR, f"{WORKSET_QRELS_STEM}_relaxed.tsv"),
    PIPELINE_OUTPUT_SUFFIX,
)
WORKSET_QUERIES_JSONL_PATH = _with_suffix(
    os.path.join(RAW_DIR, f"{WORKSET_QUERIES_STEM}.jsonl"),
    PIPELINE_OUTPUT_SUFFIX,
)
WORKSET_CALIBRATION_QUERIES_JSONL_PATH = os.path.join(
    RAW_DIR, f"{WORKSET_QUERY_CALIBRATION_STEM}{PIPELINE_OUTPUT_SUFFIX}.jsonl"
)
WORKSET_QUERY_SPLIT_META_PATH = os.path.join(
    DATA_DIR, f"{WORKSET_QUERY_SPLIT_STEM}{PIPELINE_OUTPUT_SUFFIX}.meta.json"
)

# ===== query 池切分协议（防止 calibration/evaluation 污染） =====
# 采用稳定哈希切分：同一 query_id 在多次运行中总落在同一池。
# 为了让 r_max 下分位更稳定，默认提高 calibration 池占比。
# 在 query pool 规模足够时（建议 >=1200），calibration 约占 60%。
QUERY_CALIBRATION_RATIO = 0.65
QUERY_CALIBRATION_MIN_COUNT = 80
QUERY_EVALUATION_MIN_COUNT = 80
QUERY_SPLIT_SEED = 20260409

# ===== 在线结果 =====
ONLINE_RESULTS_JSONL = os.path.join(
    RESULTS_DIR, f"{ONLINE_RESULTS_STEM}{PIPELINE_OUTPUT_SUFFIX}.jsonl"
)
ONLINE_SUMMARY_JSON = os.path.join(
    RESULTS_DIR, f"{ONLINE_SUMMARY_STEM}{PIPELINE_OUTPUT_SUFFIX}.json"
)
RRDP_PROFILE_JSON = os.path.join(
    RESULTS_DIR, f"{RRDP_PROFILE_STEM}{PIPELINE_OUTPUT_SUFFIX}.json"
)
PAPERFAITHFUL_MAINLINE_AUDIT_JSON = os.path.join(
    RESULTS_DIR, f"paperfaithful_mainline_audit{PIPELINE_OUTPUT_SUFFIX}.json"
)
PAPERFAITHFUL_MAINLINE_AUDIT_CSV = os.path.join(
    RESULTS_DIR, f"paperfaithful_mainline_audit{PIPELINE_OUTPUT_SUFFIX}.csv"
)
PAPERFAITHFUL_MAINLINE_AUDIT_PKL = os.path.join(
    RESULTS_DIR, f"paperfaithful_mainline_audit{PIPELINE_OUTPUT_SUFFIX}.pkl"
)

# ===== 模型 =====
NEW_MODEL_NAME = "intfloat/e5-large-v2"
BATCH_SIZE = 8
MAX_LENGTH = 512

# ===== 工作集构造策略参数 =====
# 论文对齐：工作集严格 docs-only，不依赖 query/qrels。
WORKSET_BUILD_MODE = (
    "paperfaithful_docs_only_mainline"
    if PIPELINE_IS_PAPERFAITHFUL_MAINLINE
    else "query_free_long_passage_geometry_preserving_sampling"
)

# ===== 文档切分参数（query-free passage chunking） =====
# 收敛到更稳妥的中等单元：避免过碎也避免过胖。
# 推荐口径：2-5 句，64-160 token，target 约 96-128 token。
PASSAGE_MIN_TOKENS = 96 if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else 64
PASSAGE_TARGET_TOKENS = 192 if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else 112
PASSAGE_MAX_TOKENS = 256 if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else 160
PASSAGE_MIN_SENTENCES = 3 if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else 2
PASSAGE_MAX_SENTENCES = 6 if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else 5
PASSAGE_SENTENCE_STRIDE = 2 if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else 1
PASSAGE_OVERLAP_TOKENS = 40 if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else 0

# docs-only 主结果线：去重与 source 贡献约束（均不依赖 query/qrels）。
WORKSET_SOURCE_DOC_MAX_PASSAGES = 2 if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else 0
WORKSET_NEAR_DUP_HAMMING_MAX_WITHIN_SOURCE = 3
WORKSET_NEAR_DUP_HAMMING_MAX_GLOBAL = 2
WORKSET_COARSE_GROUPS = 64 if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else 32
WORKSET_PREENCODE_MULTIPLIER = 10

# ===== method4 离线聚类参数 =====
METHOD4_GAP_WEIGHT = 0.45
METHOD4_ROBUST_CORE_RATIO = 0.65
METHOD4_RESTART_PRIMARY_PERCENTILE = 60.0
# 论文主结果固定 seed（可用环境变量 METHOD4_RNG_SEED_OFFSET 临时覆盖）。
# 注意 method4 实际 seed = RANDOM_STATE + METHOD4_RNG_SEED_OFFSET。
METHOD4_RNG_SEED_OFFSET = 8135
METHOD4_SEED_SWEEP_ENABLED = _env_flag(
    "METHOD4_SEED_SWEEP_ENABLED",
    bool(PIPELINE_IS_PAPERFAITHFUL_MAINLINE),
)
METHOD4_SEED_SWEEP_START = _env_int("METHOD4_SEED_SWEEP_START", 0)
METHOD4_SEED_SWEEP_END = _env_int("METHOD4_SEED_SWEEP_END", 256)
METHOD4_DOC_REFINE_ENABLED = _env_flag(
    "METHOD4_DOC_REFINE_ENABLED",
    bool(PIPELINE_IS_PAPERFAITHFUL_MAINLINE) and int(NUM_CLUSTERS) <= 8,
)
METHOD4_DOC_REFINE_MAX_SWAPS = int(max(0, _env_int("METHOD4_DOC_REFINE_MAX_SWAPS", 64)))

# ===== 论文参数 =====
EVAL_K = int(max(1, _env_float("EVAL_K", 5)))
# 论文修正口径：
# - Track1 使用全局 HNSW fixed-k 返回预算；fixed_k 允许大于单簇容量，
#   不再被 TARGET_CLUSTER_SIZE 截断。
FIXED_K = int(
    max(
        int(EVAL_K),
        int(
            _env_float(
                "FIXED_K",
                PAPERFAITHFUL_MAINLINE_OPTIMAL_FIXED_K
                if PIPELINE_IS_PAPERFAITHFUL_MAINLINE
                else 300,
            )
        ),
    )
)
ALPHA = float(
    _env_float(
        "ALPHA",
        PAPERFAITHFUL_MAINLINE_OPTIMAL_ALPHA if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else 2.0,
    )
)
# 保持原始 RRDP 设定（r_rdp_bar≈0.03125，属于 0.03x）。
EPSILON = float(
    _env_float(
        "EPSILON",
        PAPERFAITHFUL_MAINLINE_OPTIMAL_EPSILON if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else 1048576.0,
    )
)
# 实验开关：若设置 TRACK1_FORCE_PERTURB_R，则 dense 分支不再截断采样，
# 而是对每个 Track1 query 使用固定扰动半径 r；可选地把 r 裁到当前 query 的 r_max。
TRACK1_FORCE_PERTURB_R = _env_optional_float("TRACK1_FORCE_PERTURB_R")
TRACK1_FORCE_PERTURB_R_CLIP_TO_RMAX = _env_flag("TRACK1_FORCE_PERTURB_R_CLIP_TO_RMAX", True)

# ===== Track1 ANN 参数（dense 分支 fixed-budget 检索） =====
# 论文主线当前实现：
# - Top-c 只用于本地 gate / 选 parent cluster；
# - dense 真执行时，服务器对扰动 query 做一次全局 HNSW fixed-k；
# - 因而 Track1 返回预算就是全局 fixed_k，而不是 c * fixed_k。
TRACK1_DENSE_RETRIEVAL_BACKEND = "faiss_hnsw_ann"
TRACK1_HNSW_SPACE = _env_str("TRACK1_HNSW_SPACE", "cosine").strip().lower() or "cosine"
TRACK1_HNSW_M = int(max(1, _env_float("TRACK1_HNSW_M", 32)))
TRACK1_HNSW_EF_CONSTRUCTION = int(max(1, _env_float("TRACK1_HNSW_EF_CONSTRUCTION", 200)))
# ef_search 实际生效值为 max(FIXED_K, TRACK1_HNSW_EF_SEARCH_BASE)。
TRACK1_HNSW_EF_SEARCH_BASE = int(max(1, _env_float("TRACK1_HNSW_EF_SEARCH_BASE", 640)))
# 在线固定 Top-c 软归属：
# - soft_topc_fixed: 每个 query 固定取 Top-c 最近质心，比较 max_j R_max^(j)
# - boundary_gap_ladder: 边界阈值动态多簇（保留作补充实验）
ROUTING_CLUSTER_SELECTION_POLICY = _env_str(
    "ROUTING_CLUSTER_SELECTION_POLICY",
    "soft_topc_fixed" if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else "boundary_gap_ladder",
).strip().lower()
ROUTING_FIXED_TOP_C = int(
    max(
        1,
        min(
            int(NUM_CLUSTERS),
            int(
                _env_float(
                    "ROUTING_FIXED_TOP_C",
                    float(
                        min(int(NUM_CLUSTERS), PAPERFAITHFUL_MAINLINE_OPTIMAL_C)
                        if PIPELINE_IS_PAPERFAITHFUL_MAINLINE
                        else 1
                    ),
                )
            ),
        ),
    )
)
# 边界 query 多簇路由（默认只扩到 top-2，且仅限最近中心距离差小的 query）。
ROUTING_ENABLE_BOUNDARY_MULTICLUSTER = _env_flag("ROUTING_ENABLE_BOUNDARY_MULTICLUSTER", False)
ROUTING_BOUNDARY_TOP_M = int(max(1, _env_float("ROUTING_BOUNDARY_TOP_M", 2)))
ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD = float(
    max(0.0, _env_float("ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD", 0.010))
)
ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_2 = float(
    max(
        0.0,
        _env_float(
            "ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_2",
            ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD,
        ),
    )
)
_routing_gap_threshold_3 = _env_optional_float("ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_3")
ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_3 = (
    float(max(0.0, _routing_gap_threshold_3))
    if _routing_gap_threshold_3 is not None
    else None
)
_routing_gap_threshold_4 = _env_optional_float("ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_4")
ROUTING_BOUNDARY_DISTANCE_GAP_THRESHOLD_4 = (
    float(max(0.0, _routing_gap_threshold_4))
    if _routing_gap_threshold_4 is not None
    else None
)
# 若开启，则边界多簇只保留逐簇满足 r_max >= r_rdp_bar 的簇；若全部不满足则报错并提示
# 当前 query 已超出 upload-ready track1-only 主线的公开范围。
ROUTING_MULTICLUSTER_RMAX_FILTER_ENABLE = _env_flag(
    "ROUTING_MULTICLUSTER_RMAX_FILTER_ENABLE",
    True,
)
# ===== RRDP 新版约束参数 =====
# R_{1-eta}: admissible radius 的分位参数（例如 eta=0.1 => 取 90% 分位）
RRDP_ETA = 0.10
# D_{K_safe}: 第 K_safe 近邻距离的保守分位参数（beta=0.1 => 取 90% 分位）
RRDP_BETA = 0.10
# 离线 cluster-level r_max surrogate 的保守下分位参数（论文中的 gamma）。
# 例如 gamma=0.10 表示采用每簇 r_max^ideal 的 10% 分位，给出 90% 置信覆盖。
RMAX_CLUSTER_QUANTILE_GAMMA = 0.10
# cluster_r_max 的 anchor 来源策略（论文主结果推荐 calibration_query_only）：
# - calibration_query_only: 只用独立 calibration queries（主结果）
# - calibration_query_plus_docs: calibration queries + docs 回填（补充实验）
# - docs_only: 只用 docs（补充实验）
RMAX_ANCHOR_POLICY = _env_str("RMAX_ANCHOR_POLICY", "calibration_query_only").strip().lower()
# 每簇 anchor 规模控制。
# 说明：在当前 2000-doc/4x500 几何与现有 calibration query 池下，
# min=120~150 不可行；这里采用经离线 seed 扫描验证可达的严格下限。
RMAX_TARGET_ANCHORS_PER_CLUSTER = _env_int("RMAX_TARGET_ANCHORS_PER_CLUSTER", 180)
# core-300 center 后弱簇 routed calibration 支持度下降；
# 主流程保持 enforce_min=True，同时把最小阈值收敛到当前可达下限。
RMAX_MIN_ANCHORS_PER_CLUSTER = _env_int(
    "RMAX_MIN_ANCHORS_PER_CLUSTER",
    8 if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else 16,
)
# 主结果版 anchor 选择不做 membership/core 过滤，保留该字段仅为向后兼容元数据。
RMAX_ANCHOR_MEMBERSHIP_MIN_RATIO = 0.0
# 若某簇达不到最小 anchor 数，是否直接报错。
# 对 calibration-query 口径，默认继续运行；若少量弱簇 anchor 稍少，
# 允许用 quantile + shrinkage 继续估计，而不是直接失败。
RMAX_ANCHOR_ENFORCE_MIN_PER_CLUSTER = _env_flag(
    "RMAX_ANCHOR_ENFORCE_MIN_PER_CLUSTER",
    False,
)
# 按到簇中心距离做分层均匀抽样（代表性样本，不偏 core）。
RMAX_ANCHOR_NUM_DISTANCE_STRATA = 3
# 低样本簇保守处理（论文口径：仍为 cluster-level offline constant）。
# - n_i >= RMAX_SUPPORT_NORMAL_MIN：按原 quantile 直接使用；
# - RMAX_SUPPORT_SOFT_MIN <= n_i < RMAX_SUPPORT_NORMAL_MIN：r_max 乘 soft scale；
# - n_i < RMAX_SUPPORT_SOFT_MIN：r_max 设为 hard value（从而尽早暴露超出当前 track1-only 范围的 query）。
RMAX_SUPPORT_AWARE_CONSERVATIVE_ENABLE = _env_flag(
    "RMAX_SUPPORT_AWARE_CONSERVATIVE_ENABLE",
    False,
)
RMAX_SUPPORT_NORMAL_MIN = _env_int(
    "RMAX_SUPPORT_NORMAL_MIN",
    8 if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else 40,
)
RMAX_SUPPORT_SOFT_MIN = _env_int(
    "RMAX_SUPPORT_SOFT_MIN",
    4 if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else 20,
)
RMAX_SUPPORT_SOFT_SCALE = _env_float(
    "RMAX_SUPPORT_SOFT_SCALE",
    0.75 if PIPELINE_IS_PAPERFAITHFUL_MAINLINE else 0.90,
)
RMAX_SUPPORT_HARD_RMAX_VALUE = 0.0

# paper-faithful 主结果线：低样本簇 quantile 向全局保守收缩（cluster-level offline constant）。
RMAX_SHRINKAGE_ENABLE = bool(PIPELINE_IS_PAPERFAITHFUL_MAINLINE)
RMAX_SHRINKAGE_TAU = 24.0
RMAX_SHRINKAGE_MIN_BLEND = 0.15
# K_safe: 语义不可区分所需的最小有效扰动计数（不等同于 FIXED_K）。
# 经验上取远大于 EVAL_K 但显著小于 FIXED_K 的规模，避免过度保守导致区间退化。
RRDP_K_SAFE = 32
# 是否把输入 epsilon 限制在 [epsilon_min, epsilon_max]
RRDP_ENFORCE_EPSILON_INTERVAL = True
# 全局 epsilon 区间不可行时在线策略：warn_only
RRDP_GLOBAL_INTERVAL_POLICY = "warn_only"
# 离线 profiling query 的条数上限；<=0 表示不截断
RRDP_PROFILE_QUERY_LIMIT = 0

RANDOM_STATE = 42
RNG_SEED = 20260328
EPS = 1e-12
