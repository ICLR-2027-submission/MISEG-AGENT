"""Independent nnInteractive interventions measured on complete 3D volumes."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .checks import response_failures


def volume_prompt_evidence(state: dict[str, Any], candidates: list[dict[str, Any]], *, registry, config, baseline_cache=None) -> dict[str, Any]:
    """Preserve the existing gates; legacy *_px/area keys now count voxels.

    Only scalar metrics leave this function. Each call to the tool resets all
    interactions to the supplied rough prior, exactly as a submit request does.
    """
    metrics = {}
    image = np.asarray(state["image"])
    rough = np.asarray(state["current_mask"], dtype=bool)
    probabilities = state.get("voxel_error_probabilities") or {}
    try:
        if image.ndim != 3 or rough.shape != image.shape:
            raise ValueError("Expert nnInteractive review requires matching 3D image and rough mask.")
        tool = registry.get("nninteractive")
        if not tool.available:
            raise RuntimeError("nnInteractive is unavailable.")
        maps = {name: np.asarray(probabilities[name]) for name in ("false_negative", "false_positive", "correct")}
        if any(p.shape != rough.shape or not np.isfinite(p).all() for p in maps.values()):
            raise ValueError("ED probabilities must be finite and match the complete volume.")
        structure = ndi.generate_binary_structure(3, 2)
        rough_count = int(rough.sum())
        rough_components = ndi.label(rough, structure)[1]
        correct = rough & (maps["correct"] >= config["prompt_only_correct_threshold"])
        threshold = config["prompt_only_ed_support_threshold"]
        cache = baseline_cache if baseline_cache is not None else {}
        if "mask" not in cache:
            cache["mask"] = _predict(tool, image, rough, {})
            metadata = state.get("metadata") or {}
            image_metadata = metadata.get("image") or {}
            spacing = np.asarray(image_metadata.get("spacing_xyz", metadata.get("spacing_xyz", [1, 1, 1])), dtype=float)
            if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
                raise ValueError("Invalid voxel spacing for spatial review")
            cache["distance"] = ndi.distance_transform_edt(~rough, sampling=spacing[::-1]) if rough.any() else None
        baseline = cache["mask"]
        for candidate in candidates:
            cid = str(candidate["candidate_id"])
            try:
                point = [int(round(v)) for v in candidate["point_xyz"][:3]]
                zyx = tuple(point[::-1])
                if len(zyx) != 3 or any(v < 0 or v >= size for v, size in zip(zyx, rough.shape)):
                    raise ValueError("Candidate point is outside the volume.")
                label = int(candidate.get("label", 1))
                result = tool.run(image=image, current_mask=rough, conditioning_mode="rough_refine",
                                  prompt={"points": [point], "labels": [label], "mode": "fast_3d"})
                if not result.get("success"):
                    raise RuntimeError(str(result.get("error") or "nnInteractive prediction failed"))
                pred = np.asarray(result.get("raw_model_mask", result.get("mask")))
                if pred.shape != rough.shape or not np.isfinite(pred).all() or not np.isin(pred, (0, 1)).all():
                    raise ValueError("nnInteractive returned an invalid 3D binary mask.")
                pred = pred.astype(bool)
                added, removed = pred & ~rough, rough & ~pred
                n, intersection = int(pred.sum()), int((pred & rough).sum())
                union = n + rough_count - intersection
                radius = max(1, int(config["prompt_only_response_radius_px"]))
                bounds = tuple(slice(max(0, v-radius), min(size, v+radius+1)) for v, size in zip(zyx, rough.shape))
                grids = np.ogrid[tuple(slice(b.start-v, b.stop-v) for b, v in zip(bounds, zyx))]
                sphere = sum(g*g for g in grids) <= radius**2
                response = pred[bounds] if label else ~pred[bounds]
                comparison = {
                    "available": True, "spatial_dimension": 3, "count_unit": "voxels",
                    "rough_area_px": rough_count, "added_px": int(added.sum()), "removed_px": int(removed.sum()),
                    "dice": 2*intersection/max(1, n+rough_count), "iou": intersection/max(1, union),
                    "rough_overlap": intersection/max(1, rough_count), "area_ratio": n/max(1, rough_count),
                    "change_ratio_vs_rough": int((added | removed).sum())/max(1, rough_count),
                    "component_count_delta": int(ndi.label(pred, structure)[1])-rough_components,
                    "added_fn_supported_fraction": int((added & (maps['false_negative'] >= threshold)).sum())/max(1, int(added.sum())),
                    "removed_fp_supported_fraction": int((removed & (maps['false_positive'] >= threshold)).sum())/max(1, int(removed.sum())),
                    "correct_foreground_retention": int((correct & pred).sum())/int(correct.sum()) if correct.any() else None,
                }
                local = {
                    "available": True, "comparison_metrics": comparison,
                    "initial_refinement_scope": (result.get("metadata") or {}).get("initial_refinement_scope"),
                    "prompt_response_consistent": bool(pred[zyx]) == bool(label),
                    "rough_contains_point": bool(rough[zyx]),
                    "local_expected_response_fraction": float(response[sphere].mean()),
                    "ed_primary_probability_at_point": float(maps['false_negative' if label else 'false_positive'][zyx]),
                }
                spatial, spatial_failures, uncertain = _spatial_evidence(
                    tool, image, rough, baseline, pred, point, label, maps, correct,
                    structure, cache["distance"], config)
                local["spatial_evidence"] = spatial
                local["requires_manual_review"] = uncertain
                failures = response_failures(candidate, local=local, comparison=comparison, config=config)
                # New uncertainty can withhold an otherwise accepted prompt,
                # but must not relax an existing rejection.
                local["requires_manual_review"] = uncertain and not failures
                failures.extend(spatial_failures)
                local.update(singleton_passed=not failures, singleton_failures=failures)
                metrics[cid] = local
            except Exception as exc:
                metrics[cid] = {"available": False, "comparison_metrics": {"available": False}, "singleton_failures": [str(exc)]}
    except Exception as exc:
        metrics = {str(c['candidate_id']): {"available": False, "comparison_metrics": {"available": False}, "singleton_failures": [str(exc)]} for c in candidates}
    return {"status": "available" if all(m['available'] for m in metrics.values()) else "unavailable",
            "reason": "nnInteractive 3D metrics unavailable" if any(not m['available'] for m in metrics.values()) else None,
            "candidate_metrics": metrics, "count_unit": "voxels", "spatial_dimension": 3}


def _predict(tool, image, rough, prompt):
    args = {"image": image, "prompt": prompt, "conditioning_mode": "prompt_only" if rough is None else "rough_refine"}
    if rough is not None:
        args["current_mask"] = rough
    result = tool.run(**args)
    if not result.get("success"):
        raise RuntimeError(str(result.get("error") or "Independent prediction failed"))
    mask = np.asarray(result.get("raw_model_mask", result.get("mask")))
    if mask.shape != image.shape or not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError("Invalid independent prediction mask")
    return mask.astype(bool)


def _point_component(mask, point_zyx, structure):
    labels, _ = ndi.label(mask, structure)
    label = int(labels[point_zyx])
    return labels == label if label else np.zeros_like(mask, dtype=bool)


def _spatial_evidence(tool, image, rough, baseline, pred, point, label, maps, correct, structure, distance, config):
    """Attribute changes to the point, not to mask-only refinement or global overlap."""
    zyx = tuple(point[::-1])
    added, removed = pred & ~baseline, baseline & ~pred
    threshold = float(config["prompt_only_ed_support_threshold"])
    evidence = {"incremental_added_voxels": int(added.sum()), "incremental_removed_voxels": int(removed.sum()),
                "point_distance_to_rough_mm": float(distance[zyx]) if distance is not None else None}
    if label == 1:
        independent = _predict(tool, image, None, {"points": [point], "labels": [1]})
        component = _point_component(independent, zyx, structure)
        coupled = _point_component(pred, zyx, structure)
        local_added = added & coupled
        evidence.update(prompt_only_component_voxels=int(component.sum()),
                        prompt_only_component_overlap_voxels=int((component & rough).sum()),
                        conditioned_component_overlap_voxels=int((coupled & rough).sum()),
                        point_component_added_voxels=int(local_added.sum()),
                        incremental_fn_support=float((local_added & (maps["false_negative"] >= threshold)).sum()) / max(1, int(local_added.sum())))
    else:
        evidence["incremental_fp_support"] = float((removed & (maps["false_positive"] >= threshold)).sum()) / max(1, int(removed.sum()))
    correct_baseline = correct & baseline
    evidence["incremental_correct_foreground_lost_fraction"] = int((correct_baseline & ~pred).sum()) / max(1, int(correct_baseline.sum()))
    failures, uncertain = spatial_decision(evidence, label, config)
    return evidence, failures, uncertain


def spatial_decision(evidence, label, config):
    """Keep deterministic decision rules replayable from recorded scalar evidence."""
    failures, uncertain = [], False
    if label == 1:
        if not evidence["prompt_only_component_voxels"]:
            failures.append("spatial rejection: independent prediction did not contain the positive point")
        elif evidence["point_distance_to_rough_mm"] is None:
            uncertain = True
            failures.append("independent spatial validation has no rough foreground reference")
        elif evidence["prompt_only_component_overlap_voxels"] == 0:
            remote = evidence["point_distance_to_rough_mm"] > float(config.get("metric_remote_positive_distance_mm", 20.0))
            if remote and evidence["conditioned_component_overlap_voxels"] == 0:
                failures.append("spatial rejection: remote positive component is disconnected from rough in both predictions")
            else:
                uncertain = True
                failures.append("independent positive component has no rough overlap; possible missed structure needs review")
        if not evidence["point_component_added_voxels"]:
            failures.append("spatial rejection: positive point caused no additional foreground in its component")
        elif evidence["incremental_fn_support"] < float(config["metric_min_positive_directional_support_fraction"]):
            uncertain = True
            failures.append("incremental ED uncertainty: point-attributable additions lack ED false-negative support")
    else:
        if not evidence["incremental_removed_voxels"]:
            failures.append("spatial rejection: negative point caused no additional removal")
        elif evidence["incremental_fp_support"] < float(config["metric_min_negative_directional_support_fraction"]):
            uncertain = True
            failures.append("incremental ED uncertainty: point-attributable removals lack ED false-positive support")
    if evidence["incremental_correct_foreground_lost_fraction"] > float(config.get("metric_max_incremental_correct_loss", 0.005)):
        uncertain = True
        failures.append("incremental ED uncertainty: point destroyed supported foreground beyond mask-only refinement")
    return failures, uncertain
