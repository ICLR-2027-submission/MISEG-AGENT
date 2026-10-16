"""Suppress previously reviewed or submitted points."""
from __future__ import annotations
from typing import Any

def _prior_slice_points(state: dict[str, Any]) -> tuple[set[str], list[tuple[tuple[int, int, int], int]]]:
    ids: set[str] = set()
    points: list[tuple[tuple[int, int, int], int]] = []
    feedback = list(state.get("policy_feedback_history") or []) + list(
        state.get("slice_prompt_feedback_history") or []
    )
    for item in feedback:
        candidate_id = str(item.get("candidate_id") or "")
        if candidate_id:
            ids.add(candidate_id)
        point = list(item.get("point") or item.get("original_point") or [])
        if len(point) >= 3:
            points.append(((int(point[0]), int(point[1]), int(point[2])), 0 if int(item.get("label", 1)) == 0 else 1))
    for prompt in state.get("refinement_prompt_history") or []:
        labels = list(prompt.get("labels") or [])
        fallback_z = prompt.get("slice_index")
        for index, point in enumerate(prompt.get("points") or []):
            values = list(point)
            if len(values) < 3 and fallback_z is not None:
                values.append(fallback_z)
            if len(values) >= 3:
                label = 0 if index < len(labels) and int(labels[index]) == 0 else 1
                points.append(((int(values[0]), int(values[1]), int(values[2])), label))
    return ids, points


def _spacing_xyz(state: dict[str, Any]) -> tuple[float, float, float]:
    raw = ((state.get("metadata") or {}).get("image") or {}).get("spacing_xyz") or [1.0, 1.0, 1.0]
    values = [max(float(item), 1e-6) for item in list(raw)[:3]]
    return tuple((values + [1.0, 1.0, 1.0])[:3])
