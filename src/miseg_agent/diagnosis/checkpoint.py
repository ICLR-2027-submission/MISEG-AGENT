"""Strict, independently versioned ED V2 checkpoint protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .contracts import (
    CHECKPOINT_ARCHITECTURE,
    CHECKPOINT_FORMAT,
    MODEL_INPUT_CONTRACT,
    MODEL_VERSION,
    checkpoint_metadata_template,
)
from .model import FactorizedResUNet3D


REQUIRED_FIELDS = {
    "format",
    "model_version",
    "architecture",
    "input_contract",
    "state_dict",
    "model_config",
    "preprocessing_config",
    "calibration",
    "deployment_thresholds",
    "training_manifest_digest",
    "validation_metrics",
}


def validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_FIELDS - set(checkpoint))
    if missing:
        raise ValueError(f"Not an ED V2 checkpoint; missing fields: {', '.join(missing)}")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format')!r}")
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(f"ED V2 loader requires model_version=2, got {checkpoint.get('model_version')!r}.")
    if checkpoint.get("architecture") != CHECKPOINT_ARCHITECTURE:
        raise ValueError(f"Unsupported ED V2 architecture: {checkpoint.get('architecture')!r}")
    if checkpoint.get("input_contract") != MODEL_INPUT_CONTRACT:
        raise ValueError(
            f"ED V2 requires input_contract={MODEL_INPUT_CONTRACT!r}, "
            f"got {checkpoint.get('input_contract')!r}."
        )


def build_checkpoint(
    model: FactorizedResUNet3D,
    *,
    preprocessing_config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    deployment_thresholds: Mapping[str, Any],
    training_manifest_digest: str,
    validation_metrics: Mapping[str, Any],
    optimizer_state_dict: Mapping[str, Any] | None = None,
    epoch: int | None = None,
) -> dict[str, Any]:
    checkpoint = {
        **checkpoint_metadata_template(),
        "state_dict": model.state_dict(),
        "model_config": dict(model.model_config),
        "preprocessing_config": dict(preprocessing_config),
        "calibration": dict(calibration),
        "deployment_thresholds": dict(deployment_thresholds),
        "training_manifest_digest": str(training_manifest_digest),
        "validation_metrics": dict(validation_metrics),
    }
    if optimizer_state_dict is not None:
        checkpoint["optimizer_state_dict"] = dict(optimizer_state_dict)
    if epoch is not None:
        checkpoint["epoch"] = int(epoch)
    validate_checkpoint(checkpoint)
    return checkpoint


def save_checkpoint(path: str | Path, checkpoint: Mapping[str, Any]) -> None:
    validate_checkpoint(checkpoint)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(checkpoint), path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ED V2 checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("ED V2 checkpoint must be a mapping.")
    validate_checkpoint(checkpoint)
    return checkpoint


def load_model(path: str | Path, *, device: str | torch.device = "cpu") -> tuple[FactorizedResUNet3D, dict[str, Any]]:
    checkpoint = load_checkpoint(path, map_location=device)
    model = FactorizedResUNet3D(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def validate_resume_checkpoint(path: str | Path) -> dict[str, Any]:
    """Training resume is deliberately strict and therefore rejects every V1 file."""

    return load_checkpoint(path, map_location="cpu")
