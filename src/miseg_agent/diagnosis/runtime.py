"""Public ED V2 inference entry point."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .calibration import apply_calibration
from .checkpoint import load_model
from .preprocessing import build_input_channels, read_volume


def _as_volume(value: Any, *, is_label: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(value, (str, Path)):
        return read_volume(value, is_label=is_label)
    array = np.asarray(value)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"ED V2 expects a 3D volume, got shape {array.shape}.")
    return array, {}


def _starts(size: int, patch: int, overlap: float) -> list[int]:
    if size <= patch:
        return [0]
    stride = max(int(round(patch * (1.0 - overlap))), 1)
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _pad_channels(channels: np.ndarray, patch_size: Sequence[int]) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    pads = []
    slices = []
    for size, patch in zip(channels.shape[-3:], patch_size):
        missing = max(int(patch) - int(size), 0)
        before = missing // 2
        after = missing - before
        pads.append((before, after))
        slices.append(slice(before, before + size))
    padded = np.pad(channels, ((0, 0), *pads), mode="constant")
    return padded, tuple(slices)


def sliding_window_logits(
    model: torch.nn.Module,
    channels: np.ndarray,
    *,
    patch_size: Sequence[int],
    overlap: float,
    device: torch.device,
    amp: bool = True,
) -> dict[str, np.ndarray]:
    if not 0.0 <= overlap < 1.0:
        raise ValueError("Patch overlap must be in [0, 1).")
    patch_size = tuple(int(item) for item in patch_size)
    padded, original_slices = _pad_channels(np.asarray(channels, dtype=np.float32), patch_size)
    shape = padded.shape[-3:]
    accumulators = {name: np.zeros(shape, dtype=np.float32) for name in ("error_logit", "polarity_logit", "gt_foreground_logit")}
    weights = np.zeros(shape, dtype=np.float32)
    # A strictly positive center-weighting suppresses seams without losing outer voxels.
    axes = [np.hanning(size + 2)[1:-1] if size > 2 else np.ones(size) for size in patch_size]
    window = np.maximum(axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :], 1e-3).astype(np.float32)
    amp_enabled = bool(amp and device.type == "cuda")
    with torch.inference_mode():
        for z in _starts(shape[0], patch_size[0], overlap):
            for y in _starts(shape[1], patch_size[1], overlap):
                for x in _starts(shape[2], patch_size[2], overlap):
                    spatial = (slice(z, z + patch_size[0]), slice(y, y + patch_size[1]), slice(x, x + patch_size[2]))
                    patch = torch.from_numpy(padded[(slice(None), *spatial)][None]).to(device=device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                        outputs = model(patch)
                    for name in accumulators:
                        accumulators[name][spatial] += outputs[name][0, 0].float().cpu().numpy() * window
                    weights[spatial] += window
    weights = np.maximum(weights, 1e-6)
    return {name: (value / weights)[original_slices].astype(np.float32) for name, value in accumulators.items()}


class ErrorDiagnosisRuntime:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
        patch_size: Sequence[int] | None = None,
        patch_overlap: float = 0.5,
        amp: bool = True,
    ):
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.checkpoint = load_model(self.checkpoint_path, device=self.device)
        configured_patch = self.checkpoint.get("model_config", {}).get("inference_patch_size", (64, 96, 96))
        self.patch_size = tuple(int(item) for item in (patch_size or configured_patch))
        self.patch_overlap = float(patch_overlap)
        self.amp = bool(amp)

    def predict(
        self,
        image: Any,
        current_mask: Any,
        image_meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        image_array, path_meta = _as_volume(image)
        mask_array, _ = _as_volume(current_mask, is_label=True)
        if image_array.shape != mask_array.shape:
            raise ValueError(f"Image/current-mask shape mismatch: {image_array.shape} vs {mask_array.shape}.")
        metadata = {**path_meta, **dict(image_meta or {})}
        channels = build_input_channels(
            image_array,
            mask_array,
            image_meta=metadata,
            config=self.checkpoint["preprocessing_config"],
        )
        logits = sliding_window_logits(
            self.model,
            channels,
            patch_size=self.patch_size,
            overlap=self.patch_overlap,
            device=self.device,
            amp=self.amp,
        )
        probabilities = apply_calibration(logits["error_logit"], logits["polarity_logit"], self.checkpoint["calibration"])
        voxel_probabilities = {
            name: probabilities[name] for name in ("correct", "false_negative", "false_positive")
        }
        thresholds = dict(self.checkpoint.get("deployment_thresholds") or {})
        error_threshold = float(thresholds.get("error", 0.5))
        error_region = probabilities["error"] >= error_threshold
        fn_mass = float(probabilities["false_negative"][error_region].mean()) if np.any(error_region) else 0.0
        fp_mass = float(probabilities["false_positive"][error_region].mean()) if np.any(error_region) else 0.0
        if not np.any(error_region):
            error_type = "acceptable"
        elif min(fn_mass, fp_mass) >= 0.25 * max(fn_mass, fp_mass):
            error_type = "mixed_error"
        elif fn_mass >= fp_mass:
            error_type = "under_segmentation"
        else:
            error_type = "over_segmentation"
        error_heatmap = probabilities["error"].astype(np.float32)
        severity = float(np.clip(error_heatmap.mean() + 0.5 * error_heatmap.max(), 0.0, 1.0))
        confidence = float(np.max(np.stack(tuple(voxel_probabilities.values()), axis=0), axis=0).mean())
        return {
            "voxel_error_probabilities": voxel_probabilities,
            "error_heatmap": error_heatmap,
            "error_region": error_region.astype(np.uint8),
            "error_type": error_type,
            "severity": severity,
            "confidence": confidence,
            "metadata": {
                "method": "factorized_resunet3d",
                "format": self.checkpoint["format"],
                "model_version": 2,
                "architecture": self.checkpoint["architecture"],
                "input_contract": self.checkpoint["input_contract"],
                "checkpoint_path": self.checkpoint_path,
                "model_inputs": ["ct_soft_window", "ct_wide_window", "rough_mask", "rough_signed_distance"],
                "patch_size": list(self.patch_size),
                "patch_overlap": self.patch_overlap,
                "calibration": dict(self.checkpoint["calibration"]),
                "deployment_thresholds": thresholds,
                "probability_factorization": "P(error) * P(polarity|error)",
            },
        }


@lru_cache(maxsize=4)
def _cached_runtime(
    checkpoint_path: str,
    device: str | None,
    patch_size: tuple[int, int, int] | None,
    patch_overlap: float,
    amp: bool,
) -> ErrorDiagnosisRuntime:
    return ErrorDiagnosisRuntime(
        checkpoint_path,
        device=device,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        amp=amp,
    )


def predict(
    *,
    checkpoint_path: str | Path,
    image: Any,
    current_mask: Any,
    image_meta: Mapping[str, Any] | None = None,
    device: str | None = None,
    patch_size: Sequence[int] | None = None,
    patch_overlap: float = 0.5,
    amp: bool = True,
) -> dict[str, Any]:
    """Stateless public API backed by a small checkpoint/runtime cache."""

    runtime = _cached_runtime(
        str(Path(checkpoint_path).expanduser().resolve()),
        device,
        None if patch_size is None else tuple(int(item) for item in patch_size),
        float(patch_overlap),
        bool(amp),
    )
    return runtime.predict(image, current_mask, image_meta)
