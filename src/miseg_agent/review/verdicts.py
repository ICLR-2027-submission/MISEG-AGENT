"""Deterministic accept/reject/review outcomes from independent model probes."""
from __future__ import annotations
from typing import Any
from copy import deepcopy
import numpy as np

def _expert_review_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve only the controls used by executable 3D point review."""
    defaults = {'automatic_negative_prompts_enabled': True,
     'prompt_only_response_radius_px': 5,
     'prompt_only_ed_support_threshold': 0.5,
     'prompt_only_correct_threshold': 0.5,
     'metric_min_local_response_fraction': 0.55,
     'metric_min_directional_change_px': 1,
     'metric_min_dice': 0.35,
     'metric_min_rough_overlap': 0.3,
     'metric_min_area_ratio': 0.35,
     'metric_max_area_ratio': 2.5,
     'metric_max_change_ratio_vs_rough': 1.5,
     'metric_max_component_count_increase': 3,
     'metric_remote_positive_distance_mm': 20.0,
     'metric_max_incremental_correct_loss': 0.005,
     'metric_min_positive_ed_probability_at_point': 0.85,
     'metric_min_positive_directional_support_fraction': 0.9,
     'metric_min_negative_ed_probability_at_point': 0.95,
     'metric_min_negative_directional_support_fraction': 0.98,
     'metric_min_negative_local_response_fraction': 0.8,
     'metric_min_negative_correct_foreground_retention': 0.995,
     'metric_max_negative_removed_fraction': 0.1,
     'metric_max_negative_change_ratio_vs_rough': 0.15,
     'manual_review_min_component_voxels': 64,
     'manual_review_min_error_mass': 48.0,
     'manual_review_error_mass_multiplier': 2.0}
    configured = dict(config.get("expert_prompt_review") or {})
    unknown = set(configured) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown review settings: {', '.join(sorted(unknown))}")
    result = {**defaults, **configured}
    for key, default in defaults.items():
        if isinstance(default, bool):
            if not isinstance(result[key], bool):
                raise ValueError(f"{key} must be a boolean.")
            continue
        value = float(result[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
        result[key] = int(value) if isinstance(default, int) else value
    if result["prompt_only_response_radius_px"] < 1:
        raise ValueError("The response neighborhood radius must be positive.")
    if result["metric_min_area_ratio"] > result["metric_max_area_ratio"]:
        raise ValueError("Minimum volume ratio cannot exceed the maximum.")
    return result


def _metric_slice_reviews(
    candidates: list[dict[str, Any]],
    *,
    slice_evidence: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate_metrics = dict(slice_evidence.get("candidate_metrics") or {})
    reviews: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        original = [int(round(float(value))) for value in candidate["point_xyz"][:3]]
        original_label = 0 if int(candidate.get("label", 1)) == 0 else 1
        prompt_kind = (
            "scribble" if str(candidate.get("prompt_kind") or "point") == "scribble" else "point"
        )
        original_scribble = (
            deepcopy(candidate.get("recommended_scribble"))
            if prompt_kind == "scribble" and isinstance(candidate.get("recommended_scribble"), dict)
            else None
        )
        local = dict(candidate_metrics.get(candidate_id) or {})
        comparison = dict(local.get("comparison_metrics") or {})
        comparison_available = bool(comparison.get("available"))
        available = bool(comparison_available and local.get("available"))
        failures: list[str] = []
        direction = "added" if original_label == 1 else "removed"
        direction_px = int(comparison.get(f"{direction}_px") or 0)
        support = comparison.get(
            "added_fn_supported_fraction" if original_label == 1 else "removed_fp_supported_fraction"
        )
        if available:
            if local.get("singleton_passed") is not True:
                failures.extend(str(item) for item in local.get("singleton_failures") or [])
        else:
            failures.extend(str(item) for item in local.get("singleton_failures") or [])

        failures = list(dict.fromkeys(failures))
        large_ed_region, ed_region_metrics = _large_ed_region(candidate, config=config)
        decisive_reject = _has_decisive_reject_failure(failures)
        manual_review = bool(
            not decisive_reject
            and (local.get("requires_manual_review") or large_ed_region)
            and (
                not available
                or (failures and not decisive_reject)
            )
        )
        verdict = (
            "accept"
            if available and not failures
            else "review"
            if manual_review
            else "reject"
            if available
            else "unavailable"
        )
        rationale = (
            "Accepted from an independent nnInteractive 3D intervention: the point response was "
            "consistent; independent spatial and incremental-change checks passed."
            if verdict == "accept"
            else "Needs manual adjustment: ED marks a large error region, but the independent "
            "nnInteractive 3D response was structurally ambiguous. Open this slice and place or edit "
            "the prompt manually."
            if verdict == "review"
            else "Rejected from the independent nnInteractive 3D intervention: " + "; ".join(failures)
            if verdict == "reject"
            else f"Metric review unavailable: {slice_evidence.get('reason') or 'nnInteractive comparison failed'}."
        )
        suggested_scribble = deepcopy(original_scribble)
        if suggested_scribble is not None:
            suggested_scribble.update(
                {
                    "label": original_label,
                    "slice_index": original[2],
                    "mode": "fast_3d",
                    "candidate_id": candidate_id,
                }
            )
        reviews.append(
            {
                "candidate_id": candidate_id,
                "verdict": verdict,
                "response": {"accept": "accepted", "reject": "rejected", "review": "needs_review"}.get(
                    verdict, "unavailable"
                ),
                "original_point": original,
                "original_label": original_label,
                "suggested_point": list(original),
                "suggested_label": original_label,
                "prompt_kind": prompt_kind,
                "original_scribble": original_scribble,
                "suggested_scribble": suggested_scribble,
                "mode": "fast_3d",
                "parent_candidate_id": str(candidate.get("parent_candidate_id") or candidate_id),
                "candidate_scope": str(
                    candidate.get("candidate_scope") or "policy_region_representative"
                ),
                "confidence": (
                    0.90 if verdict == "accept" else 0.85 if verdict == "reject" else 0.50 if verdict == "review" else 0.0
                ),
                "rationale": rationale[:600],
                "manual_review_recommended": manual_review,
                "large_ed_region": large_ed_region,
                "ed_region_metrics": ed_region_metrics,
                "relocation_validated": False,
                "relocation_reason": "prompt modification is disabled",
                "suggested_probability": round(float(candidate.get("probability") or 0.0), 4),
                "relocation_distance_px": 0.0,
                "metric_reviewed": available,
                "model_reviewed": False,
                "decision_source": "nninteractive_3d_metrics",
                "decision_metrics": {
                    "spatial_evidence": deepcopy(local.get("spatial_evidence") or {}),
                    "check_failures": list(failures),
                    "spatial_dimension": 3,
                    "count_unit": "voxels",
                    "rough_overlap": comparison.get("rough_overlap"),
                    "volume_ratio": comparison.get("area_ratio"),
                    "directional_change_voxels": direction_px,
                    "dice": comparison.get("dice"),
                    "iou": comparison.get("iou"),
                    "area_ratio": comparison.get("area_ratio"),
                    "change_ratio_vs_rough": comparison.get("change_ratio_vs_rough"),
                    "directional_change_px": direction_px,
                    "directional_support_fraction": support,
                    "local_expected_response_fraction": local.get(
                        "local_expected_response_fraction"
                    ),
                    "correct_foreground_retention": comparison.get(
                        "correct_foreground_retention"
                    ),
                    "component_count_delta": comparison.get("component_count_delta"),
                    "singleton_passed": local.get("singleton_passed"),
                },
                **(
                    {"review_error": str(slice_evidence.get("reason") or "metrics unavailable")[:800]}
                    if not available
                    else {}
                ),
            }
        )
    return reviews


def _large_ed_region(
    candidate: dict[str, Any],
    *,
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Return whether ED burden is large enough to warrant human adjustment."""

    features = dict(candidate.get("features") or {})
    component_voxels = int(
        _finite_float(
            candidate.get(
                "slice_component_voxels",
                candidate.get(
                    "component_voxels",
                    features.get("slice_component_voxels", 0),
                ),
            ),
            0,
        )
    )
    error_mass = _finite_float(
        features.get("slice_error_mass", features.get("component_error_mass", 0.0)),
        0.0,
    )
    recommended_mass = max(
        0.0,
        _finite_float(features.get("recommendation_min_error_mass"), 0.0),
    )
    minimum_voxels = int(config["manual_review_min_component_voxels"])
    minimum_mass = max(
        float(config["manual_review_min_error_mass"]),
        recommended_mass * float(config["manual_review_error_mass_multiplier"]),
    )
    # This is intentionally a threshold decision rather than a per-volume
    # top-k. Difficult volumes may expose more than twenty review slices, while
    # low-burden volumes expose fewer or none.
    large = bool(component_voxels >= minimum_voxels and error_mass >= minimum_mass)
    return large, {
        "component_voxels": component_voxels,
        "error_mass": round(float(error_mass), 4),
        "minimum_component_voxels": minimum_voxels,
        "minimum_error_mass": round(float(minimum_mass), 4),
        "relative_error_burden": (
            None
            if recommended_mass <= 0.0
            else round(float(error_mass / recommended_mass), 4)
        ),
        "minimum_relative_error_burden": float(
            config["manual_review_error_mass_multiplier"]
        ),
        "selection_policy": "extent_and_case_normalized_burden_thresholds",
    }


def _has_decisive_reject_failure(failures: list[str]) -> bool:
    decisive = (
        "spatial rejection:",
        "automatic negative prompts are disabled by validated policy",
        "contradicted the prompt polarity",
        "no meaningful added region",
        "no meaningful removed region",
    )
    return any(any(marker in failure for marker in decisive) for failure in failures)


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default
