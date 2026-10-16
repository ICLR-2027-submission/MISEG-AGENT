"""Standalone ED V2 preprocessing in NumPy array order ``(z, y, x)``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi

from .contracts import PreprocessingConfig

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover
    sitk = None


def read_volume(path: str | Path, *, is_label: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Read NPY/NPZ or a SimpleITK-supported volume without pipeline imports."""

    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        array = np.load(path)
        return np.asarray(array), {"spacing": [1.0, 1.0, 1.0], "path": str(path)}
    if suffixes.endswith(".npz"):
        with np.load(path) as data:
            key = "array" if "array" in data else data.files[0]
            array = np.asarray(data[key])
        return array, {"spacing": [1.0, 1.0, 1.0], "path": str(path)}
    if sitk is None:
        raise ImportError("SimpleITK is required to read medical image volumes.")
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    metadata = {
        "spacing": list(reversed(image.GetSpacing())),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "path": str(path),
    }
    dtype = np.uint8 if is_label else np.float32
    return np.asarray(array, dtype=dtype), metadata


def write_volume(path: str | Path, array: np.ndarray, *, reference_path: str | Path | None = None) -> None:
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffixes.endswith(".npy"):
        np.save(path, np.asarray(array))
        return
    if sitk is None:
        raise ImportError("SimpleITK is required to write medical image volumes.")
    image = sitk.GetImageFromArray(np.asarray(array))
    if reference_path:
        reference = sitk.ReadImage(str(reference_path))
        if reference.GetSize() == image.GetSize():
            image.CopyInformation(reference)
    sitk.WriteImage(image, str(path))


def window_channel(image: np.ndarray, window: Sequence[float]) -> np.ndarray:
    low, high = (float(window[0]), float(window[1]))
    if high <= low:
        raise ValueError(f"Invalid CT window: {window}")
    return np.clip((np.asarray(image, dtype=np.float32) - low) / (high - low), 0.0, 1.0)


def spacing_zyx(image_meta: Mapping[str, Any] | None) -> tuple[float, float, float]:
    meta = dict(image_meta or {})
    if "spacing_zyx" in meta:
        value = meta["spacing_zyx"]
        reverse = False
    elif "spacing_xyz" in meta:
        value = meta["spacing_xyz"]
        reverse = True
    else:
        value = meta.get("spacing", (1.0, 1.0, 1.0))
        reverse = str(meta.get("spacing_order", "zyx")).lower() == "xyz"
    if len(value) != 3:
        return (1.0, 1.0, 1.0)
    spacing = tuple(max(float(item), 1e-6) for item in value)
    return tuple(reversed(spacing)) if reverse else spacing


def signed_distance(mask: np.ndarray, spacing: Sequence[float], clip_mm: float = 20.0) -> np.ndarray:
    """Signed distance: negative inside the rough mask and positive outside."""

    binary = np.asarray(mask) > 0
    outside = ndi.distance_transform_edt(~binary, sampling=spacing)
    inside = ndi.distance_transform_edt(binary, sampling=spacing)
    distance = outside - inside
    return np.clip(distance, -clip_mm, clip_mm).astype(np.float32)


def build_input_channels(
    image: np.ndarray,
    rough_mask: np.ndarray,
    *,
    image_meta: Mapping[str, Any] | None = None,
    config: PreprocessingConfig | Mapping[str, Any] | None = None,
) -> np.ndarray:
    config = config if isinstance(config, PreprocessingConfig) else PreprocessingConfig.from_mapping(config)
    image = np.asarray(image, dtype=np.float32)
    rough = (np.asarray(rough_mask) > 0).astype(np.float32)
    if image.shape != rough.shape or image.ndim != 3:
        raise ValueError(f"Image and rough mask must be same-shape 3D arrays, got {image.shape} and {rough.shape}.")
    distance = signed_distance(rough, spacing_zyx(image_meta), config.distance_clip_mm)
    distance /= max(config.distance_clip_mm, 1e-6)
    return np.stack(
        (
            window_channel(image, config.soft_tissue_window),
            window_channel(image, config.wide_window),
            rough,
            distance,
        ),
        axis=0,
    ).astype(np.float32)


def resize_array(array: np.ndarray, shape: Sequence[int], *, is_label: bool = False) -> np.ndarray:
    target = tuple(int(item) for item in shape)
    source = np.asarray(array)
    if source.shape == target:
        return source.copy()
    factors = tuple(out / inp for out, inp in zip(target, source.shape))
    resized = ndi.zoom(source, factors, order=0 if is_label else 1, mode="nearest", prefilter=not is_label)
    if resized.shape != target:
        fixed = np.zeros(target, dtype=resized.dtype)
        overlap = tuple(slice(0, min(a, b)) for a, b in zip(target, resized.shape))
        fixed[overlap] = resized[overlap]
        resized = fixed
    return resized


def resize_probabilities(probabilities: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    restored = np.stack([resize_array(channel, shape, is_label=False) for channel in probabilities], axis=0)
    return np.clip(restored, 0.0, 1.0).astype(np.float32)


def target_maps(gt: np.ndarray, rough: np.ndarray) -> dict[str, np.ndarray]:
    gt_binary = np.asarray(gt) > 0
    rough_binary = np.asarray(rough) > 0
    if gt_binary.shape != rough_binary.shape:
        raise ValueError("GT and rough mask shapes differ.")
    false_negative = gt_binary & ~rough_binary
    false_positive = ~gt_binary & rough_binary
    error = false_negative | false_positive
    return {
        "error": error.astype(np.float32),
        "polarity": false_negative.astype(np.float32),
        "gt_foreground": gt_binary.astype(np.float32),
        "false_negative": false_negative,
        "false_positive": false_positive,
        "correct": ~error,
    }
