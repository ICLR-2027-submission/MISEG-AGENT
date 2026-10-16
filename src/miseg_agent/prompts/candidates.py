"""Convert dense FN/FP diagnosis maps into a small, safe candidate set."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .contracts import MODE_FAST_3D, MODE_FINE_2D, PromptCandidate


class PromptCandidateGenerator:
    """Generate balanced component-internal point candidates from ED outputs."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})

    def generate(self, state: dict[str, Any]) -> list[PromptCandidate]:
        mask_value = state.get("current_mask")
        if mask_value is None:
            return []
        mask = _as_volume(mask_value).astype(bool)
        fn_map, fp_map = _error_maps(state, mask.shape)
        if fn_map is None or fp_map is None:
            return []
        correct_map = _matching_probability(
            (state.get("voxel_error_probabilities") or {}).get("correct"),
            mask.shape,
        )
        threshold = float(self.config.get("probability_threshold", 0.50))
        dominance_margin = max(0.0, float(self.config.get("candidate_dominance_margin", 0.05)))
        min_component = max(1, int(self.config.get("min_component_voxels", 4)))
        maximum = max(1, int(self.config.get("max_candidates_per_label", 8)))
        maximum_total = max(1, int(self.config.get("max_prompt_candidates", 12)))
        broad_slices = max(2, int(self.config.get("min_3d_component_slices", 3)))
        broad_voxels = max(
            min_component,
            int(self.config.get("min_3d_component_voxels", 32)),
        )
        broad_mean_slice_voxels = max(
            1.0,
            float(self.config.get("min_3d_mean_slice_voxels", 32.0)),
        )
        connectivity = int(np.clip(int(self.config.get("candidate_connectivity", 2)), 1, 3))
        spacing_xyz = _spacing_xyz(state)
        history_points = _history_points(state.get("refinement_prompt_history") or [])
        history_points.extend(_feedback_points(state.get("policy_feedback_history") or []))
        min_distance = max(
            0.0,
            float(
                self.config.get(
                    "min_prompt_distance_mm",
                    self.config.get("min_prompt_distance", 12.0),
                )
            ),
        )
        candidates: list[PromptCandidate] = []
        correct = np.zeros(mask.shape, dtype=np.float32) if correct_map is None else correct_map
        specs = (
            (
                "false_negative",
                1,
                fn_map,
                (~mask)
                & np.isfinite(fn_map)
                & (fn_map >= threshold)
                & (fn_map >= correct + dominance_margin),
            ),
            (
                "false_positive",
                0,
                fp_map,
                mask
                & np.isfinite(fp_map)
                & (fp_map >= threshold)
                & (fp_map >= correct + dominance_margin),
            ),
        )
        for source, label_value, probabilities, binary in specs:
            generated = _component_candidates(
                binary=binary,
                probabilities=probabilities,
                source=source,
                label_value=label_value,
                volume_shape=mask.shape,
                min_component=min_component,
                broad_slices=broad_slices,
                broad_voxels=broad_voxels,
                broad_mean_slice_voxels=broad_mean_slice_voxels,
                connectivity=connectivity,
                spacing_xyz=spacing_xyz,
                correct_probabilities=correct_map,
            )
            kept: list[PromptCandidate] = []
            for candidate in generated:
                if _near_any(
                    candidate.point_xyz,
                    [point for point, label in history_points if label is None or label == label_value],
                    min_distance,
                    spacing_xyz,
                ):
                    continue
                if _near_any(
                    candidate.point_xyz,
                    [item.point_xyz for item in kept],
                    min_distance,
                    spacing_xyz,
                ):
                    continue
                kept.append(candidate)
                if len(kept) >= maximum:
                    break
            candidates.extend(kept)
        ordered = sorted(candidates, key=_candidate_sort_key, reverse=True)
        if len(ordered) <= maximum_total:
            return ordered
        selected: list[PromptCandidate] = []
        for label in (1, 0):
            candidate = next((item for item in ordered if item.label == label), None)
            if candidate is not None:
                selected.append(candidate)
        for candidate in ordered:
            if candidate in selected:
                continue
            selected.append(candidate)
            if len(selected) >= maximum_total:
                break
        return sorted(selected[:maximum_total], key=_candidate_sort_key, reverse=True)


def _component_candidates(
    *,
    binary: np.ndarray,
    probabilities: np.ndarray,
    source: str,
    label_value: int,
    volume_shape: tuple[int, int, int],
    min_component: int,
    broad_slices: int,
    broad_voxels: int,
    broad_mean_slice_voxels: float,
    connectivity: int,
    spacing_xyz: tuple[float, float, float],
    correct_probabilities: np.ndarray | None,
) -> list[PromptCandidate]:
    labels, _count = ndi.label(
        binary,
        structure=ndi.generate_binary_structure(3, connectivity),
    )
    result: list[PromptCandidate] = []
    volume_voxels = float(max(1, np.prod(volume_shape)))
    component_boxes = ndi.find_objects(labels)
    for component_id, box in enumerate(component_boxes, 1):
        if box is None:
            continue
        local_labels = labels[box]
        component = local_labels == component_id
        voxels = int(component.sum())
        if voxels < min_component:
            continue
        z_indices = np.flatnonzero(component.any(axis=(1, 2)))
        slices = int(z_indices.size)
        z_min = int((box[0].start or 0) + int(z_indices.min()))
        z_max = int((box[0].start or 0) + int(z_indices.max()))
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        distance_3d, distance_2d = _component_distance_maps(component, spacing_zyx)
        local_probabilities = probabilities[box]
        probability_values = np.asarray(local_probabilities[component], dtype=np.float32)
        probability_mean = float(np.mean(probability_values))
        probability_p95 = float(np.percentile(probability_values, 95.0))
        error_mass = float(np.sum(probability_values))
        component_probs = np.where(component, local_probabilities, 0.0)
        distance_3d_scale = distance_3d / max(float(distance_3d.max()), 1e-6)
        distance_2d_scale = distance_2d / max(float(distance_2d.max()), 1e-6)
        utility = np.where(
            component,
            0.55 * component_probs + 0.30 * distance_2d_scale + 0.15 * distance_3d_scale,
            -np.inf,
        )
        local_z, local_y, local_x = np.unravel_index(int(np.argmax(utility)), utility.shape)
        z = int((box[0].start or 0) + local_z)
        y = int((box[1].start or 0) + local_y)
        x = int((box[2].start or 0) + local_x)
        probability = float(probabilities[z, y, x])
        correct_probability = 0.0 if correct_probabilities is None else float(correct_probabilities[z, y, x])
        confidence_margin = probability - correct_probability
        size_score = min(
            1.0,
            float(np.log1p(voxels) / np.log1p(max(volume_voxels * 0.01, 2.0))),
        )
        component_priority = float(
            np.clip(0.45 * probability_p95 + 0.30 * probability_mean + 0.25 * size_score, 0.0, 1.0)
        )
        physical_x = max(float(volume_shape[2] - 1) * spacing_xyz[0], spacing_xyz[0])
        physical_y = max(float(volume_shape[1] - 1) * spacing_xyz[1], spacing_xyz[1])
        physical_z = max(float(volume_shape[0] - 1) * spacing_xyz[2], spacing_xyz[2])
        inplane_diagonal = max(float(np.hypot(physical_x, physical_y)), 1e-6)
        region_diagonal = max(
            float(np.sqrt(physical_x**2 + physical_y**2 + physical_z**2)),
            1e-6,
        )
        allowed = (
            (MODE_FINE_2D, MODE_FAST_3D)
            if slices >= broad_slices and voxels >= broad_voxels
            else (MODE_FINE_2D,)
        )
        mean_slice_voxels = float(voxels / max(slices, 1))
        recommended_mode = (
            MODE_FAST_3D
            if MODE_FAST_3D in allowed and mean_slice_voxels >= broad_mean_slice_voxels
            else MODE_FINE_2D
        )
        region_name = "FN" if label_value == 1 else "FP"
        mode_reason = (
            "3D correction recommended"
            if recommended_mode == MODE_FAST_3D
            else (
                "3D correction available; 2D preferred for the small in-slice footprint"
                if MODE_FAST_3D in allowed
                else "localized 2D correction"
            )
        )
        explanation = (
            f"Representative {region_name} point from one 3D connected region: "
            f"{voxels} voxels across {slices} slices (z={z_min + 1}–{z_max + 1}); "
            f"mean={probability_mean:.2f}, p95={probability_p95:.2f}, "
            f"margin over correct={confidence_margin:.2f}; {mode_reason}."
        )
        result.append(
            PromptCandidate(
                candidate_id=f"{source}:{component_id}:{x}-{y}-{z}",
                point_xyz=(int(x), int(y), int(z)),
                label=label_value,
                source=source,
                probability=probability,
                component_id=component_id,
                component_voxels=voxels,
                component_slices=slices,
                component_z_range=(z_min, z_max),
                explanation=explanation,
                features={
                    "label_positive": float(label_value == 1),
                    "probability": probability,
                    "correct_probability": correct_probability,
                    "component_volume_fraction": float(voxels / volume_voxels),
                    "component_span_fraction": float(slices / max(volume_shape[0], 1)),
                    "component_mean_slice_voxels": mean_slice_voxels,
                    "interior_distance": float(distance_2d[local_z, local_y, local_x]),
                    "inplane_interior_distance_mm": float(distance_2d[local_z, local_y, local_x]),
                    "region_interior_distance_mm": float(distance_3d[local_z, local_y, local_x]),
                    "component_mean_probability": probability_mean,
                    "component_p95_probability": probability_p95,
                    "component_error_mass": error_mass,
                    "component_error_mass_fraction": float(error_mass / volume_voxels),
                    "component_priority": component_priority,
                    "confidence_margin": confidence_margin,
                    "point_x_fraction": float(x / max(volume_shape[2] - 1, 1)),
                    "point_y_fraction": float(y / max(volume_shape[1] - 1, 1)),
                    "point_z_fraction": float(z / max(volume_shape[0] - 1, 1)),
                    "inplane_interior_distance_fraction": float(
                        distance_2d[local_z, local_y, local_x] / inplane_diagonal
                    ),
                    "region_interior_distance_fraction": float(
                        distance_3d[local_z, local_y, local_x] / region_diagonal
                    ),
                },
                allowed_modes=allowed,
                recommended_mode=recommended_mode,
            )
        )
    return sorted(result, key=_candidate_sort_key, reverse=True)


def _component_distance_maps(
    component: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Measure interior depth with explicit background outside a tight component box."""

    padded_3d = np.pad(component, 1, mode="constant", constant_values=False)
    distance_3d = ndi.distance_transform_edt(padded_3d, sampling=spacing_zyx)[1:-1, 1:-1, 1:-1]
    distance_2d = np.zeros(component.shape, dtype=np.float32)
    for z in range(component.shape[0]):
        plane = component[z]
        if not plane.any():
            continue
        padded_2d = np.pad(plane, 1, mode="constant", constant_values=False)
        distance_2d[z] = ndi.distance_transform_edt(
            padded_2d,
            sampling=spacing_zyx[1:],
        )[1:-1, 1:-1]
    return np.asarray(distance_3d, dtype=np.float32), distance_2d


def _error_maps(state: dict[str, Any], shape: tuple[int, int, int]) -> tuple[np.ndarray | None, np.ndarray | None]:
    probabilities = state.get("voxel_error_probabilities") or {}
    fn_map = _matching_probability(probabilities.get("false_negative"), shape)
    fp_map = _matching_probability(probabilities.get("false_positive"), shape)
    if fn_map is not None and fp_map is not None:
        return fn_map, fp_map
    heatmap = _matching_probability(state.get("error_heatmap"), shape)
    if heatmap is None:
        return None, None
    return heatmap, heatmap


def _matching_probability(value: Any, shape: tuple[int, int, int]) -> np.ndarray | None:
    if value is None:
        return None
    array = _as_volume(value).astype(np.float32, copy=False)
    if array.shape != shape:
        return None
    if np.isfinite(array).all():
        return array
    return np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)


def _as_volume(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2:
        return array[None]
    if array.ndim != 3:
        raise ValueError(f"Prompt policy expects a 2D/3D array, got {array.shape}.")
    return array


def _history_points(
    history: list[dict[str, Any]],
) -> list[tuple[tuple[int, int, int], int | None]]:
    result: list[tuple[tuple[int, int, int], int | None]] = []
    for prompt in history:
        fallback = prompt.get("slice_index")
        labels = list(prompt.get("labels") or [])
        for index, point in enumerate(prompt.get("points") or []):
            values = list(point)
            label = None if index >= len(labels) else int(labels[index])
            if len(values) >= 3:
                result.append(((int(values[0]), int(values[1]), int(values[2])), label))
            elif len(values) >= 2 and fallback is not None:
                result.append(((int(values[0]), int(values[1]), int(fallback)), label))
    return result


def _feedback_points(
    history: list[dict[str, Any]],
) -> list[tuple[tuple[int, int, int], int | None]]:
    result: list[tuple[tuple[int, int, int], int | None]] = []
    for item in history:
        label = item.get("label")
        normalized_label = None if label is None else int(label)
        for point in (item.get("point"), item.get("original_point")):
            if point is None:
                continue
            values = list(point)
            if len(values) >= 3:
                normalized = (int(values[0]), int(values[1]), int(values[2]))
                pair = (normalized, normalized_label)
                if pair not in result:
                    result.append(pair)
    return result


def _near_any(
    point: tuple[int, int, int],
    others: list[tuple[int, int, int]],
    distance: float,
    spacing_xyz: tuple[float, float, float],
) -> bool:
    if distance <= 0:
        return False
    spacing = np.asarray(spacing_xyz, dtype=np.float32)
    p = np.asarray(point, dtype=np.float32) * spacing
    return any(
        float(np.linalg.norm(p - np.asarray(other, dtype=np.float32) * spacing)) < distance
        for other in others
    )


def _spacing_xyz(state: dict[str, Any]) -> tuple[float, float, float]:
    image_metadata = dict((state.get("metadata") or {}).get("image") or {})
    raw = list(image_metadata.get("spacing_xyz") or [1.0, 1.0, 1.0])
    raw.extend([1.0] * (3 - len(raw)))
    return tuple(max(float(value), 1e-6) for value in raw[:3])


def _candidate_sort_key(candidate: PromptCandidate) -> tuple[float, float, int]:
    return (
        float(candidate.features.get("component_priority", candidate.probability)),
        float(candidate.features.get("component_error_mass", 0.0)),
        int(candidate.component_voxels),
    )
