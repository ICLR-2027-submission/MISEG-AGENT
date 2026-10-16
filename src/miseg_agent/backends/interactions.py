"""Shared parsing for voxel-coordinate interactions."""
from __future__ import annotations
import math
from typing import Any, Mapping, Sequence
import numpy as np

def _parse_point(point: Any, *, default_slice: Any, depth: int) -> tuple[float, float, int]:
    if isinstance(point, Mapping):
        if "x" not in point or "y" not in point:
            raise ValueError(f"Point mapping requires x and y: {point}")
        x, y = float(point["x"]), float(point["y"])
        slice_value = point.get("slice_index", point.get("z", default_slice))
    else:
        values = list(point)
        if len(values) < 2:
            raise ValueError(f"Point requires at least [x, y]: {point}")
        x, y = float(values[0]), float(values[1])
        slice_value = values[2] if len(values) >= 3 else default_slice
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"Point coordinates must be finite: {point}")
    if slice_value is None:
        if depth != 1:
            raise ValueError("2D point coordinates on a 3D volume require slice_index or a z coordinate.")
        slice_value = 0
    slice_float = float(slice_value)
    if not math.isfinite(slice_float):
        raise ValueError(f"Point slice coordinate must be finite: {point}")
    slice_index = int(round(slice_float))
    if not 0 <= slice_index < depth:
        raise ValueError(
            f"Point slice {slice_index} is outside input depth {depth}; z was not clipped or moved."
        )
    return x, y, slice_index


def _normalize_scribble_entries(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping) or _looks_like_scribble_tuple(value):
        return [value]
    return list(value)


def _looks_like_scribble_tuple(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 4
        and not isinstance(value[0], Mapping)
        and isinstance(value[3], Sequence)
        and not isinstance(value[3], (str, bytes))
    )


def _normalize_point_entries(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return [value]
    entries = list(value)
    if len(entries) >= 2 and all(np.isscalar(item) for item in entries):
        return [entries]
    return entries


def _parse_scribble(entry: Any, *, default_slice: Any, depth: int) -> dict[str, Any]:
    if isinstance(entry, Mapping):
        label = entry.get("label", 1)
        slice_value = entry.get("slice_index", default_slice)
        brush_size = entry.get("brush_size", 1)
        points = entry.get("points")
    else:
        label, slice_value, brush_size, points = entry
    if points is None or not _has_entries(points):
        raise ValueError("Each scribble requires at least one point.")
    if slice_value is None:
        if depth != 1:
            raise ValueError("A scribble on a 3D volume requires slice_index.")
        slice_value = 0
    slice_float = float(slice_value)
    if not math.isfinite(slice_float):
        raise ValueError(f"Scribble slice coordinate must be finite: {entry}")
    slice_index = int(round(slice_float))
    if not 0 <= slice_index < depth:
        raise ValueError(
            f"Scribble slice {slice_index} is outside input depth {depth}; z was not clipped or moved."
        )
    brush = max(float(brush_size), 1.0)
    parsed_points: list[list[float]] = []
    for point in points:
        if isinstance(point, Mapping):
            x, y = float(point["x"]), float(point["y"])
        else:
            values = list(point)
            if len(values) < 2:
                raise ValueError(f"Scribble point requires [x, y]: {point}")
            x, y = float(values[0]), float(values[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Scribble coordinates must be finite: {point}")
        parsed_points.append([x, y])
    return {
        "label": _parse_label(label),
        "slice_index": slice_index,
        "brush_size": brush,
        "points": parsed_points,
    }


def _parse_label(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "positive", "pos", "foreground", "fg", "include", "true"}:
            return 1
        if normalized in {"0", "negative", "neg", "background", "bg", "exclude", "false"}:
            return 0
        raise ValueError(f"Unknown prompt label: {value!r}")
    label = int(value)
    if label not in (0, 1):
        raise ValueError(f"Prompt label must be 0 or 1, got {value!r}.")
    return label


def _has_entries(value: Any) -> bool:
    return value is not None and len(value) > 0


def _rasterize_scribbles(strokes: Sequence[Mapping[str, Any]], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    canvas = np.zeros((2, height, width), dtype=np.float32)
    for stroke in strokes:
        label = int(stroke["label"])
        channel = 0 if label == 1 else 1
        points = stroke["points"]
        radius = max(0, int(math.floor(float(stroke["brush_size"]) / 2.0)))
        samples: list[tuple[float, float]] = []
        if len(points) == 1:
            samples.append((points[0][0], points[0][1]))
        else:
            for start, end in zip(points[:-1], points[1:]):
                dx, dy = end[0] - start[0], end[1] - start[1]
                steps = max(1, int(math.ceil(max(abs(dx), abs(dy)))))
                for fraction in np.linspace(0.0, 1.0, steps + 1):
                    samples.append((start[0] + fraction * dx, start[1] + fraction * dy))
        for x, y in samples:
            _paint_disk(canvas, channel, x=x, y=y, radius=radius)
    return canvas


def _paint_disk(canvas: np.ndarray, channel: int, *, x: float, y: float, radius: int) -> None:
    center_x, center_y = int(round(x)), int(round(y))
    height, width = canvas.shape[-2:]
    x_min, x_max = max(0, center_x - radius), min(width - 1, center_x + radius)
    y_min, y_max = max(0, center_y - radius), min(height - 1, center_y + radius)
    if x_min > x_max or y_min > y_max:
        return
    yy, xx = np.ogrid[y_min : y_max + 1, x_min : x_max + 1]
    disk = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    view = canvas[channel, y_min : y_max + 1, x_min : x_max + 1]
    other = canvas[1 - channel, y_min : y_max + 1, x_min : x_max + 1]
    view[disk] = 1.0
    other[disk] = 0.0
