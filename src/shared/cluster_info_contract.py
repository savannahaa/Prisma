"""
cluster_info 契约校验工具。

目的：
1) 防止脚本误读不兼容的 cluster_info 产物。
2) 统一检查论文主线口径是否对齐：
   - soft_topc_fixed 主线允许 c-aware 的 route-union r_max surrogate；
   - 单簇 / 非 soft_topc_fixed 仍要求 within_cluster 版本；
   - anchor 应由最近中心分簇后的代表性样本构成（主结果不做 membership/core 过滤）
"""

from __future__ import annotations

# Allow running this file directly: `python src/shared/cluster_info_contract.py`
if __package__ in (None, ""):
    import os
    import sys

    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

from typing import Any, Dict, Optional

import numpy as np

from shared.config import ROUTING_CLUSTER_SELECTION_POLICY


EXPECTED_RMAX_SCOPE = (
    "within_topc_overlap_route_union_docs"
    if str(ROUTING_CLUSTER_SELECTION_POLICY).strip().lower() == "soft_topc_fixed"
    else "within_owner_cluster_docs"
)
LEGACY_SOFT_TOPC_SELECTED_SCOPE = "within_topc_selected_cluster_global_hnsw"
EXPECTED_MEMBERSHIP_SCOPE = "not_applicable_representative_sampling"
NEAREST_CLUSTER_LOCAL_MEMBERSHIP_SCOPE = "nearest_cluster_top_fixed_k_within_cluster_docs"
LEGACY_MEMBERSHIP_SCOPE = "global_top_fixed_k_over_all_workset_docs"
EXPECTED_CENTER_SOURCE = "full_500_centroid_then_normalize"


def _to_int_or_none(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def validate_cluster_info_contract(
    cluster_info: dict,
    *,
    expected_eval_k: int | None = None,
    expected_fixed_k: int | None = None,
    expected_num_clusters: int | None = None,
    expected_target_cluster_size: int | None = None,
) -> Dict[str, Any]:
    errors = []
    warnings = []

    if not isinstance(cluster_info, dict):
        return {
            "ok": False,
            "errors": ["cluster_info is not a dict"],
            "warnings": [],
            "signature": {},
        }

    centers = np.asarray(cluster_info.get("centers", []), dtype=np.float32)
    chunks = list(cluster_info.get("chunks", []))
    cluster_r_max = np.asarray(cluster_info.get("cluster_r_max", []), dtype=np.float32)
    eval_k = _to_int_or_none(cluster_info.get("eval_k"))
    fixed_k = _to_int_or_none(cluster_info.get("fixed_k"))
    num_clusters_field = _to_int_or_none(cluster_info.get("num_clusters"))

    if centers.ndim != 2 or centers.shape[0] <= 0:
        errors.append("centers is empty or not 2D")

    if len(chunks) <= 0:
        errors.append("chunks is empty")

    inferred_clusters = int(centers.shape[0]) if centers.ndim == 2 else int(len(chunks))
    if len(chunks) > 0 and centers.ndim == 2 and int(len(chunks)) != int(centers.shape[0]):
        errors.append(
            f"chunks/centers mismatch: len(chunks)={len(chunks)} vs centers={centers.shape[0]}"
        )

    if len(cluster_r_max) != inferred_clusters:
        errors.append(
            f"cluster_r_max size mismatch: len(cluster_r_max)={len(cluster_r_max)} vs num_clusters={inferred_clusters}"
        )

    if num_clusters_field is not None and int(num_clusters_field) != int(inferred_clusters):
        errors.append(
            f"num_clusters field mismatch: num_clusters={num_clusters_field} vs inferred={inferred_clusters}"
        )

    if expected_num_clusters is not None and int(expected_num_clusters) != int(inferred_clusters):
        errors.append(
            f"expected_num_clusters mismatch: expected={expected_num_clusters} vs inferred={inferred_clusters}"
        )

    if expected_eval_k is not None and eval_k is not None and int(expected_eval_k) != int(eval_k):
        errors.append(f"eval_k mismatch: expected={expected_eval_k} vs got={eval_k}")
    if expected_fixed_k is not None and fixed_k is not None and int(expected_fixed_k) != int(fixed_k):
        errors.append(f"fixed_k mismatch: expected={expected_fixed_k} vs got={fixed_k}")

    if expected_target_cluster_size is not None and len(chunks) > 0:
        bad = []
        for cid, chunk in enumerate(chunks):
            size = int(len(np.asarray(chunk, dtype=np.int32)))
            if size != int(expected_target_cluster_size):
                bad.append((int(cid), int(size)))
        if bad:
            preview = ", ".join([f"{cid}:{sz}" for cid, sz in bad[:5]])
            errors.append(
                "cluster size mismatch with expected_target_cluster_size="
                f"{expected_target_cluster_size}; bad_clusters={preview}"
            )

    rmax = cluster_info.get("rmax_surrogate", {})
    if not isinstance(rmax, dict):
        errors.append("rmax_surrogate missing or not dict")
        rmax = {}

    rmax_scope = _as_str(rmax.get("rmax_scope"))
    formula = _as_str(rmax.get("formula"))
    anchor_policy = _as_str(rmax.get("anchor_policy"))
    anchor_source = _as_str(rmax.get("anchor_source"))
    anchor_cluster_assign_source = _as_str(rmax.get("anchor_cluster_assign_source"))
    selector_meta = rmax.get("anchor_selector_meta", {})
    if not isinstance(selector_meta, dict):
        selector_meta = {}

    selector = _as_str(selector_meta.get("selector"))
    fixed_k_for_membership = _to_int_or_none(selector_meta.get("fixed_k_for_membership"))
    membership_scope = _as_str(selector_meta.get("membership_scope"))
    membership_min_ratio = selector_meta.get("membership_min_ratio")
    routing_fixed_top_c = _to_int_or_none(rmax.get("routing_fixed_top_c"))
    routing_policy = str(ROUTING_CLUSTER_SELECTION_POLICY).strip().lower()
    soft_topc_single_cluster = routing_policy == "soft_topc_fixed" and int(
        routing_fixed_top_c or 0
    ) <= 1
    expected_rmax_scope = "within_owner_cluster_docs" if soft_topc_single_cluster else EXPECTED_RMAX_SCOPE

    if routing_policy == "soft_topc_fixed" and not soft_topc_single_cluster:
        accepted_rmax_scopes = {EXPECTED_RMAX_SCOPE, LEGACY_SOFT_TOPC_SELECTED_SCOPE}
        if rmax_scope not in accepted_rmax_scopes:
            errors.append(
                "rmax_scope mismatch for soft_topc_fixed: "
                f"expected one of {sorted(accepted_rmax_scopes)} vs got={rmax_scope}"
            )
        elif rmax_scope == LEGACY_SOFT_TOPC_SELECTED_SCOPE:
            warnings.append(
                "deprecated soft_topc_fixed rmax_scope detected: "
                "within_topc_selected_cluster_global_hnsw; "
                "route-union surrogate is preferred so c participates in the gate proxy."
            )
    elif rmax_scope != expected_rmax_scope:
        errors.append(f"rmax_scope mismatch: expected={expected_rmax_scope} vs got={rmax_scope}")

    if routing_policy == "soft_topc_fixed":
        if soft_topc_single_cluster:
            if formula and "within_cluster" not in formula:
                errors.append(
                    f"rmax formula missing within_cluster marker for soft_topc_fixed,c=1: {formula}"
                )
        elif formula and all(
            marker not in formula
            for marker in ("topc_overlap_route_union", "topc_selected_cluster_global_hnsw")
        ):
            errors.append(
                "rmax formula missing accepted soft_topc_fixed marker "
                f"(route_union or selected_cluster): {formula}"
            )
    else:
        if formula and "within_cluster" not in formula:
            errors.append(f"rmax formula missing within_cluster marker: {formula}")
    if formula and "theta_fixed-theta_k" not in formula:
        warnings.append(f"rmax formula does not contain theta_fixed-theta_k marker: {formula}")

    fixed_k_required_scopes = {NEAREST_CLUSTER_LOCAL_MEMBERSHIP_SCOPE, LEGACY_MEMBERSHIP_SCOPE}
    if membership_scope in fixed_k_required_scopes:
        if fixed_k is not None and fixed_k_for_membership is not None:
            if int(fixed_k_for_membership) != int(fixed_k):
                errors.append(
                    "membership fixed_k mismatch: "
                    f"fixed_k_for_membership={fixed_k_for_membership} vs fixed_k={fixed_k}"
                )
        elif anchor_policy != "docs_only":
            warnings.append("fixed_k_for_membership missing in anchor_selector_meta")

    if membership_scope:
        accepted_scopes = {
            EXPECTED_MEMBERSHIP_SCOPE,
            NEAREST_CLUSTER_LOCAL_MEMBERSHIP_SCOPE,
            LEGACY_MEMBERSHIP_SCOPE,
        }
        if membership_scope not in accepted_scopes:
            errors.append(
                "membership_scope mismatch: "
                "expected one of "
                "["
                f"{EXPECTED_MEMBERSHIP_SCOPE}, "
                f"{NEAREST_CLUSTER_LOCAL_MEMBERSHIP_SCOPE}, "
                f"{LEGACY_MEMBERSHIP_SCOPE}"
                f"] vs got={membership_scope}"
            )
        elif membership_scope == LEGACY_MEMBERSHIP_SCOPE:
            msg = (
                "legacy membership_scope detected: global_top_fixed_k_over_all_workset_docs; "
                "main-result contract requires representative sampling scope."
            )
            if anchor_policy == "calibration_query_only":
                errors.append(msg)
            else:
                warnings.append(msg)
        elif membership_scope == NEAREST_CLUSTER_LOCAL_MEMBERSHIP_SCOPE:
            msg = (
                "membership_scope is nearest-cluster local top-fixed_k; "
                "main-result contract requires representative sampling without membership filter."
            )
            if anchor_policy == "calibration_query_only":
                errors.append(msg)
            else:
                warnings.append(msg)
    elif anchor_policy != "docs_only":
        warnings.append(
            "membership_scope missing; expected representative sampling scope marker in anchor_selector_meta"
        )

    if anchor_policy == "calibration_query_only":
        if routing_policy == "soft_topc_fixed":
            if soft_topc_single_cluster:
                if anchor_cluster_assign_source != "provided_owner_cluster_ids":
                    errors.append(
                        "anchor_cluster_assign_source mismatch for soft_topc_fixed,c=1 calibration_query_only: "
                        f"expected=provided_owner_cluster_ids vs got={anchor_cluster_assign_source}"
                    )
            elif not anchor_cluster_assign_source.startswith("topc_nearest_centroids_by_angular_distance"):
                errors.append(
                    "anchor_cluster_assign_source mismatch for soft_topc_fixed calibration_query_only: "
                    "expected topc_nearest_centroids_by_angular_distance:* "
                    f"vs got={anchor_cluster_assign_source}"
                )
        else:
            if anchor_cluster_assign_source != "provided_owner_cluster_ids":
                errors.append(
                    "anchor_cluster_assign_source mismatch for calibration_query_only: "
                    f"expected=provided_owner_cluster_ids vs got={anchor_cluster_assign_source}"
                )

    if anchor_policy == "calibration_query_only":
        if not selector or ("representative" not in selector):
            errors.append(
                "selector mismatch for calibration_query_only main-result contract: "
                f"expected representative selector marker, got={selector}"
            )
    elif anchor_policy != "docs_only" and selector:
        if ("representative" not in selector) and ("membership_filter" not in selector):
            warnings.append(f"selector does not contain representative/membership marker: {selector}")

    signature = {
        "rmax_scope": rmax_scope,
        "formula": formula,
        "anchor_policy": anchor_policy,
        "anchor_source": anchor_source,
        "anchor_cluster_assign_source": anchor_cluster_assign_source,
        "routing_cluster_selection_policy": _as_str(rmax.get("routing_cluster_selection_policy")),
        "routing_fixed_top_c": routing_fixed_top_c,
        "selector": selector,
        "fixed_k_for_membership": fixed_k_for_membership,
        "membership_scope": membership_scope,
        "membership_min_ratio": membership_min_ratio,
        "num_clusters": inferred_clusters,
        "eval_k": eval_k,
        "fixed_k": fixed_k,
    }

    clustering_method = cluster_info.get("clustering_method", {})
    if isinstance(clustering_method, dict):
        final_geometry = clustering_method.get("final_geometry", {})
        if isinstance(final_geometry, dict):
            center_source = _as_str(final_geometry.get("center_source"))
            if center_source and center_source != EXPECTED_CENTER_SOURCE:
                errors.append(
                    "final_geometry.center_source mismatch: "
                    f"expected={EXPECTED_CENTER_SOURCE} vs got={center_source}"
                )

            public_center_member_count = _to_int_or_none(
                final_geometry.get("public_center_member_count")
            )
            legacy_public_center_core_size = _to_int_or_none(
                final_geometry.get("public_center_core_size")
            )
            if public_center_member_count is not None and expected_target_cluster_size is not None:
                if int(public_center_member_count) != int(expected_target_cluster_size):
                    errors.append(
                        "final_geometry.public_center_member_count mismatch: "
                        f"expected={expected_target_cluster_size} vs got={public_center_member_count}"
                    )
            if legacy_public_center_core_size is not None:
                warnings.append(
                    "legacy final_geometry.public_center_core_size detected; "
                    "new mainline artifacts should use public_center_member_count"
                )
                if expected_target_cluster_size is not None and int(legacy_public_center_core_size) != int(
                    expected_target_cluster_size
                ):
                    errors.append(
                        "legacy final_geometry.public_center_core_size mismatch: "
                        f"expected={expected_target_cluster_size} vs got={legacy_public_center_core_size}"
                    )
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "signature": signature,
    }


def assert_cluster_info_contract(
    cluster_info: dict,
    *,
    expected_eval_k: int | None = None,
    expected_fixed_k: int | None = None,
    expected_num_clusters: int | None = None,
    expected_target_cluster_size: int | None = None,
    raise_on_warning: bool = False,
) -> Dict[str, Any]:
    report = validate_cluster_info_contract(
        cluster_info,
        expected_eval_k=expected_eval_k,
        expected_fixed_k=expected_fixed_k,
        expected_num_clusters=expected_num_clusters,
        expected_target_cluster_size=expected_target_cluster_size,
    )
    if not bool(report["ok"]):
        msg = "cluster_info contract check failed:\n- " + "\n- ".join(report["errors"])
        if report["warnings"]:
            msg += "\n(warnings)\n- " + "\n- ".join(report["warnings"])
        raise RuntimeError(msg)
    if bool(raise_on_warning) and report["warnings"]:
        msg = "cluster_info contract warning treated as error:\n- " + "\n- ".join(
            report["warnings"]
        )
        raise RuntimeError(msg)
    return report
