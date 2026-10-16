"""One interactive point per eligible 3D ED connected component."""

from __future__ import annotations

from typing import Any

import numpy as np

from .history import _prior_slice_points, _spacing_xyz
from .candidates import PromptCandidateGenerator


def derive_recommended_region_prompt_candidates(
    state: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Enumerate all 3D region representatives, suppressing handled points.

    Each point determines its own axial review slice. Distinct components
    are retained without a per-volume top-k limit.
    """
    mask = state.get("current_mask")
    if mask is None:
        return []
    settings = dict(config or {})
    minimum_distance = max(0.0, float(settings.get("min_prompt_distance_mm", 12.0)))
    settings.update(
        max_candidates_per_label=max(1, int(np.size(mask))),
        max_prompt_candidates=max(1, int(np.size(mask))),
        min_prompt_distance_mm=0.0,
    )
    handled_ids, handled_points = _prior_slice_points(state)
    spacing = np.asarray(_spacing_xyz(state))
    result = []
    for candidate in PromptCandidateGenerator(settings).generate(state):
        item = candidate.as_dict()
        point = np.asarray(candidate.point_xyz)
        if candidate.candidate_id in handled_ids or any(
            label == candidate.label
            and (
                np.array_equal(point, other)
                or np.linalg.norm((point - np.asarray(other)) * spacing) < minimum_distance
            )
            for other, label in handled_points
        ):
            continue
        item.update(
            candidate_scope="region_3d",
            prompt_kind="point",
            recommended_scribble=None,
        )
        result.append(item)
    return result
