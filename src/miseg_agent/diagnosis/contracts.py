"""Stable data and deployment contracts for ED V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


CHECKPOINT_FORMAT = "miseg_agent_error_diagnosis"
MODEL_VERSION = 2
CHECKPOINT_ARCHITECTURE = "factorized_resunet3d"
MODEL_INPUT_CONTRACT = "ct_and_rough_mask_only"
VOXEL_CLASSES = ("correct", "false_negative", "false_positive")


@dataclass(frozen=True)
class PreprocessingConfig:
    soft_tissue_window: tuple[float, float] = (-200.0, 300.0)
    wide_window: tuple[float, float] = (-1000.0, 1000.0)
    distance_clip_mm: float = 20.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "PreprocessingConfig":
        value = dict(value or {})
        return cls(
            soft_tissue_window=tuple(value.get("soft_tissue_window", (-200.0, 300.0))),
            wide_window=tuple(value.get("wide_window", (-1000.0, 1000.0))),
            distance_clip_mm=float(value.get("distance_clip_mm", 20.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def checkpoint_metadata_template() -> dict[str, Any]:
    """Return the immutable part of every ED V2 checkpoint."""

    return {
        "format": CHECKPOINT_FORMAT,
        "model_version": MODEL_VERSION,
        "architecture": CHECKPOINT_ARCHITECTURE,
        "input_contract": MODEL_INPUT_CONTRACT,
    }
