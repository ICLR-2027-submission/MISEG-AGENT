"""Physical-grid NIfTI loading, shared workspace creation, and mask export."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass
class VolumeCase:
    image: np.ndarray
    mask: np.ndarray
    spacing_xyz: tuple[float, float, float]
    reference: sitk.Image
    native_reference: sitk.Image

    def save(self, path, array, *, native=False, is_mask=True):
        array = np.asarray(array)
        if array.shape != self.image.shape:
            raise ValueError("Output array does not match the workspace.")
        image = sitk.GetImageFromArray(array.astype(np.uint8 if is_mask else np.float32))
        image.CopyInformation(self.reference)
        if native:
            image = sitk.Resample(
                image, self.native_reference, sitk.Transform(),
                sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear, 0.0, image.GetPixelID()
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(path))


def load_case(image_path, mask_path, *, shape_zyx=(256, 256, 256), mask_label=None):
    """Use linear CT and nearest-neighbor mask interpolation on one physical grid.

    Masks must already be binary, or a single integer foreground label must be
    supplied explicitly. This prevents combining unrelated organs silently.
    """
    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    mask = sitk.ReadImage(str(mask_path))
    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError("Image and mask must both be three-dimensional.")
    raw_mask = sitk.GetArrayFromImage(mask)
    if not np.isfinite(raw_mask).all():
        raise ValueError("Mask contains non-finite values.")
    if mask_label is None:
        if not np.isin(raw_mask, (0, 1)).all():
            raise ValueError("Provide a binary mask or specify mask_label for a multiclass mask.")
        selected = raw_mask.astype(np.uint8)
    else:
        selected = (raw_mask == mask_label).astype(np.uint8)
    binary = sitk.GetImageFromArray(selected)
    binary.CopyInformation(mask)
    shape = tuple(int(size) for size in shape_zyx)
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError("Workspace dimensions must each be at least two voxels.")
    target_size = tuple(reversed(shape))
    reference = sitk.Image(target_size, sitk.sitkFloat32)
    # Preserve first and last voxel-center positions on each physical axis.
    spacing = tuple(old_spacing * max(old_size - 1, 1) / (new_size - 1)
                    for old_size, new_size, old_spacing in zip(image.GetSize(), target_size, image.GetSpacing()))
    reference.SetSpacing(spacing)
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    image_out = sitk.Resample(image, reference, sitk.Transform(), sitk.sitkLinear, -1000.0, sitk.sitkFloat32)
    mask_out = sitk.Resample(binary, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return VolumeCase(sitk.GetArrayFromImage(image_out), sitk.GetArrayFromImage(mask_out), spacing, reference, image)
