"""Full-mask and local-response checks used in executable 3D review."""
from __future__ import annotations
from typing import Any

def response_failures(
    candidate: dict[str, Any],
    *,
    local: dict[str, Any],
    comparison: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    """Judge independent prompt utility with stricter removal safeguards."""

    failures: list[str] = []
    label = 0 if int(candidate.get("label", 1)) == 0 else 1
    if label == 0 and not bool(config.get("automatic_negative_prompts_enabled", True)):
        # Keep an explicit opt-out in addition to the per-negative safeguards.
        failures.append("automatic negative prompts are disabled by validated policy")
    direction = "removed" if label == 0 else "added"
    if local.get("prompt_response_consistent") is not True:
        failures.append("prompt response contradicted the prompt polarity")
    local_fraction = local.get("local_expected_response_fraction")
    minimum_local_response = float(config.get("metric_min_local_response_fraction", 0.55))
    if label == 0:
        minimum_local_response = max(
            minimum_local_response,
            float(config.get("metric_min_negative_local_response_fraction", 0.80)),
        )
    if local_fraction is None or float(local_fraction) < minimum_local_response:
        failures.append("local expected-response fraction was too low")
    if int(comparison.get(f"{direction}_px") or 0) < int(config.get("metric_min_directional_change_px", 1)):
        failures.append(f"no meaningful {direction} region was produced")
    rough_area = int(comparison.get("rough_area_px") or 0)
    if rough_area > 0:
        dice = comparison.get("dice")
        if dice is None or float(dice) < float(config.get("metric_min_dice", 0.35)):
            failures.append("refined-vs-rough Dice was below the safety floor")
        area_ratio = comparison.get("area_ratio")
        if area_ratio is None or not (
            float(config.get("metric_min_area_ratio", 0.35))
            <= float(area_ratio)
            <= float(config.get("metric_max_area_ratio", 2.50))
        ):
            failures.append("refined-mask area ratio was outside the safe range")
        rough_overlap = comparison.get("rough_overlap")
        if rough_overlap is None or float(rough_overlap) < float(
            config.get("metric_min_rough_overlap", 0.30)
        ):
            failures.append("nnInteractive retained too little overlap with the rough mask")
        change_ratio = comparison.get("change_ratio_vs_rough")
        if change_ratio is None or float(change_ratio) > float(
            config.get("metric_max_change_ratio_vs_rough", 1.50)
        ):
            failures.append("nnInteractive changed too much of the rough-mask shape")
    if int(comparison.get("component_count_delta") or 0) > int(
        config.get("metric_max_component_count_increase", 3)
    ):
        failures.append("nnInteractive introduced too many new components")
    if label == 1:
        minimum_ed_probability = float(
            config.get("metric_min_positive_ed_probability_at_point", 0.85)
        )
        primary_probability = local.get("ed_primary_probability_at_point")
        if primary_probability is None or float(primary_probability) < minimum_ed_probability:
            failures.append("positive point ED probability was below the strict utility floor")
        minimum_support = float(
            config.get("metric_min_positive_directional_support_fraction", 0.90)
        )
        support = comparison.get("added_fn_supported_fraction")
        if support is None or float(support) < minimum_support:
            failures.append("positive additions had insufficient ED false-negative support")
    else:
        if local.get("rough_contains_point") is not True:
            failures.append("negative point was outside the rough foreground")
        primary_probability = local.get("ed_primary_probability_at_point")
        if primary_probability is None or float(primary_probability) < float(
            config.get("metric_min_negative_ed_probability_at_point", 0.95)
        ):
            failures.append("negative point ED probability was below the strict utility floor")
        support = comparison.get("removed_fp_supported_fraction")
        if support is None or float(support) < float(
            config.get("metric_min_negative_directional_support_fraction", 0.98)
        ):
            failures.append("negative removals had insufficient ED false-positive support")
        retained = comparison.get("correct_foreground_retention")
        if retained is not None and float(retained) < float(
            config.get("metric_min_negative_correct_foreground_retention", 0.995)
        ):
            failures.append("negative prompt removed too much ED-supported correct foreground")
        if rough_area > 0:
            if int(comparison.get("removed_px") or 0) / rough_area > float(
                config.get("metric_max_negative_removed_fraction", 0.10)
            ):
                failures.append("negative prompt removed too large a fraction of the rough mask")
            change_ratio = comparison.get("change_ratio_vs_rough")
            if change_ratio is None or float(change_ratio) > float(
                config.get("metric_max_negative_change_ratio_vs_rough", 0.15)
            ):
                failures.append("negative prompt changed too much of the rough-mask shape")
    return failures
