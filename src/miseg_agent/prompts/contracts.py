"""A connected-region point recommendation."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any
MODE_FAST_3D = "fast_3d"
MODE_FINE_2D = "fine_2d"

@dataclass(frozen=True)
class PromptCandidate:
    """One finite spatial action proposed from an ED probability component."""

    candidate_id: str
    point_xyz: tuple[int, int, int]
    label: int
    source: str
    probability: float
    component_id: int
    component_voxels: int
    component_slices: int
    component_z_range: tuple[int, int] = (0, 0)
    explanation: str = ""
    features: dict[str, float] = field(default_factory=dict)
    allowed_modes: tuple[str, ...] = (MODE_FINE_2D,)
    recommended_mode: str = MODE_FINE_2D

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["point_xyz"] = list(self.point_xyz)
        result["component_z_range"] = list(self.component_z_range)
        result["allowed_modes"] = list(self.allowed_modes)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PromptCandidate":
        return cls(
            candidate_id=str(value["candidate_id"]),
            point_xyz=tuple(int(item) for item in value["point_xyz"][:3]),
            label=int(value["label"]),
            source=str(value["source"]),
            probability=float(value["probability"]),
            component_id=int(value["component_id"]),
            component_voxels=int(value["component_voxels"]),
            component_slices=int(value["component_slices"]),
            component_z_range=tuple(
                int(item) for item in value.get("component_z_range") or (0, 0)
            )[:2],
            explanation=str(value.get("explanation") or ""),
            features={str(key): float(item) for key, item in dict(value.get("features") or {}).items()},
            allowed_modes=tuple(str(item) for item in value.get("allowed_modes") or (MODE_FINE_2D,)),
            recommended_mode=str(value.get("recommended_mode") or MODE_FINE_2D),
        )

    def to_prompt(self, *, mode: str | None = None) -> dict[str, Any]:
        x, y, z = self.point_xyz
        selected_mode = str(mode or self.recommended_mode or MODE_FINE_2D)
        return {
            "prompt_type": "positive_click" if self.label == 1 else "negative_click",
            "points": [[x, y, z]],
            "labels": [self.label],
            "positive_points": [[x, y, z]] if self.label == 1 else [],
            "negative_points": [[x, y, z]] if self.label == 0 else [],
            "slice_index": z,
            "source": "prompt_policy",
            "candidate_id": self.candidate_id,
            "candidate_ids": [self.candidate_id],
            "candidate_source": self.source,
            "component_id": self.component_id,
            "component_voxels": self.component_voxels,
            "component_slices": self.component_slices,
            "component_z_range": list(self.component_z_range),
            "candidate_explanation": self.explanation,
            "recommended_mode": self.recommended_mode,
            "point_modes": [selected_mode],
            "refinement_mode": selected_mode,
        }
