"""Persistent official nnInteractive inference on complete [Z, Y, X] volumes."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .base import BaseTool
from .interactions import (
    _normalize_point_entries,
    _normalize_scribble_entries,
    _parse_label,
    _parse_point,
    _parse_scribble,
    _rasterize_scribbles,
)


DEFAULT_REPO_PATH = "foundation_models/nnInteractive"
DEFAULT_MODEL_DIR = "checkpoints/models/nnInteractive_v1.0"


class NNInteractiveTool(BaseTool):
    """Segment from prompts, or refine a rough volume with cumulative interactions.

    ``point(s)`` use project coordinates [x,y,z]. Scribbles use the same
    ``label/slice_index/brush_size/points`` contract as the axial UI. Boxes use
    [x1,y1,z1,x2,y2,z2], with exclusive upper bounds; a four-coordinate box
    requires ``slice_index``. Boxes run before points, followed by scribbles.
    No conservative union or substitute inference backend is applied.
    """

    name = "nninteractive"
    description = "Official nnInteractive 3D prompt segmentation and rough-mask refinement."

    def __init__(
        self,
        *,
        checkpoint_path: str | None = None,
        model_dir: str | None = None,
        repo_path: str | None = None,
        device: str | None = None,
        session: Any | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        merged = {**dict(config or {}), **kwargs}
        for key, value in (("model_dir", model_dir), ("repo_path", repo_path), ("device", device)):
            if value is not None:
                merged[key] = value
        super().__init__(checkpoint_path=checkpoint_path, config=merged)
        self._session = session
        self._lock = threading.RLock()
        self._image_key: tuple[tuple[int, ...], str] | None = None

    @property
    def availability_error(self) -> str | None:
        if self._session is not None:
            return None
        model_dir, fold, checkpoint_name = self._model_location()
        required = [model_dir / name for name in ("plans.json", "dataset.json")]
        required.append(model_dir / f"fold_{fold}" / checkpoint_name)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            return "nnInteractive model files missing: " + ", ".join(missing)
        repo_package = self._repo_path() / "nnInteractive" / "inference" / "inference_session.py"
        try:
            installed = importlib.util.find_spec("nnInteractive") is not None
            torch_installed = importlib.util.find_spec("torch") is not None
        except (ImportError, ValueError):
            installed = torch_installed = False
        if not installed and not repo_package.is_file():
            return "nnInteractive package is unavailable; install the configured official repository."
        if not torch_installed:
            return "PyTorch is required by nnInteractive."
        return None

    @property
    def available(self) -> bool:
        return self.availability_error is None

    def _run(
        self,
        *,
        image,
        current_mask=None,
        prompt: Mapping[str, Any] | None = None,
        conditioning_mode: str = "rough_refine",
        previous_slice_logits=None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if conditioning_mode not in {"rough_refine", "prompt_only"}:
            raise ValueError("nnInteractive supports conditioning_mode='rough_refine' or 'prompt_only'.")
        if previous_slice_logits:
            raise ValueError("nnInteractive does not consume ScribblePrompt previous_slice_logits.")
        image_array = np.asarray(image)
        if image_array.ndim != 3 or any(size <= 0 for size in image_array.shape):
            raise ValueError(f"nnInteractive requires a complete 3D [Z,Y,X] image; got {image_array.shape}.")
        volume = np.array(image_array, dtype=np.float32, order="C", copy=True)
        if not np.isfinite(volume).all():
            raise ValueError("nnInteractive image must contain only finite intensities.")
        if current_mask is None and conditioning_mode == "rough_refine":
            raise ValueError("nnInteractive refinement requires current_mask as a rough 3D prior.")
        if current_mask is not None and conditioning_mode == "prompt_only":
            raise ValueError("prompt_only mode does not accept a current_mask prior.")
        rough = np.zeros(volume.shape, dtype=np.uint8) if current_mask is None else np.asarray(current_mask)
        if rough.shape != volume.shape:
            raise ValueError(f"current_mask shape {rough.shape} does not match image shape {volume.shape}.")
        if not np.isfinite(rough).all() or not np.isin(rough, (0, 1)).all():
            raise ValueError("nnInteractive current_mask must be binary (0 or 1) and finite.")
        rough = rough.astype(np.uint8, copy=True)
        if prompt is not None and not isinstance(prompt, Mapping):
            raise TypeError("prompt must be a mapping.")
        prompt_data = dict(prompt or {})
        for key in ("point", "points", "labels", "positive_points", "negative_points", "scribbles",
                    "slice_index", "prompt_type", "box", "boxes", "box_label"):
            if key not in prompt_data and key in kwargs:
                prompt_data[key] = kwargs[key]
        interactions = _prepare_interactions(prompt_data, volume.shape)
        if conditioning_mode == "prompt_only" and not interactions:
            raise ValueError("Initial segmentation requires at least one explicit prompt.")
        refine_initial = conditioning_mode == "rough_refine" and bool(self.config.get("refine_initial_mask", True))
        if not refine_initial and not interactions:
            raise ValueError("refine_initial_mask=False requires at least one explicit interaction.")
        image_key = (volume.shape, hashlib.sha256(memoryview(volume).cast("B")).hexdigest())
        prepared_at = time.perf_counter()

        # A session holds both image preprocessing and accumulated prompts. Serialize
        # the entire transaction, including the output copy, across web requests.
        with self._lock:
            session = self._load_session()
            loaded_at = time.perf_counter()
            image_cached = self._image_key == image_key
            try:
                if not image_cached:
                    session.set_image(volume[None])
                    self._image_key = image_key
                target = np.zeros(volume.shape, dtype=np.uint8)
                session.set_target_buffer(target)
                # The official initial-seg API clears EVERY previous interaction.
                if conditioning_mode == "rough_refine":
                    session.add_initial_seg_interaction(rough.copy(), run_prediction=refine_initial)
                else:
                    # Public official API: no initial-mask interaction, and no
                    # prompts left over from a previous request on the same CT.
                    session.reset_interactions()
                for kind, payload, label in interactions:
                    if kind == "point":
                        session.add_point_interaction(payload, include_interaction=bool(label), run_prediction=True)
                    elif kind == "box":
                        session.add_bbox_interaction(payload, include_interaction=bool(label), run_prediction=True)
                    else:
                        stroke, bbox = payload
                        session.add_scribble_interaction(
                            stroke, include_interaction=bool(label), run_prediction=True, interaction_bbox=bbox,
                        )
                result_mask = target.copy()
            except Exception:
                # Incomplete asynchronous preprocessing or prediction must not be
                # considered reusable on the next call, even for the same CT.
                self._image_key = None
                raise
            predicted_at = time.perf_counter()

        changed_slices = np.flatnonzero(np.any(result_mask != rough, axis=(1, 2))).tolist()
        model_dir, fold, checkpoint_name = self._model_location()
        return {
            "mask": result_mask,
            "raw_model_mask": result_mask.copy(),
            "slice_logits": {},
            "metadata": {
                "method": self.name,
                "backend": "official_nninteractive",
                "official_model": "nnInteractive",
                "is_3d": True,
                "input_shape": list(volume.shape),
                "output_shape": list(result_mask.shape),
                "conditioning_mode": conditioning_mode,
                "rough_mask_used_as_model_input": current_mask is not None,
                "previous_mask_conditioned": current_mask is not None,
                "refine_initial_mask": refine_initial,
                "initial_refinement_scope": "full_volume" if refine_initial else "none",
                "prompt_execution": "sequential_cumulative_after_initial_mask" if current_mask is not None else "sequential_cumulative_prompt_only",
                "interaction_count": len(interactions),
                "interaction_counts": {kind: sum(item[0] == kind for item in interactions) for kind in ("point", "scribble", "box")},
                "predictor_calls": int(refine_initial) + len(interactions),
                "predictor_calls_semantics": "official interaction prediction requests; each may run multiple 3D patches",
                "image_preprocessing_cached": image_cached,
                "changed_slices": changed_slices,
                "edited_slices": changed_slices,
                "positive_only_conservative_union_enabled": False,
                "model_dir": str(model_dir),
                "checkpoint_path": str(model_dir / f"fold_{fold}" / checkpoint_name),
                "use_fold": fold,
                "stage_timings_sec": {
                    "prepare": prepared_at - started,
                    "load_and_lock_wait": loaded_at - prepared_at,
                    "predict": predicted_at - loaded_at,
                    "total_measured": time.perf_counter() - started,
                },
                "prompt_coordinate_contract": "project [x,y,z] -> nnInteractive [z,y,x]; boxes have exclusive upper bounds",
            },
        }

    def _repo_path(self) -> Path:
        return self.resolve_path(self.config.get("repo_path") or DEFAULT_REPO_PATH).resolve()

    def _model_location(self) -> tuple[Path, int | str, str]:
        fold = self.config.get("use_fold", 0)
        checkpoint_name = str(self.config.get("checkpoint_name") or "checkpoint_final.pth")
        if self.checkpoint_path:
            configured = self.resolve_path(self.checkpoint_path).resolve()
            if configured.suffix.lower() in {".pth", ".pt"}:
                if not configured.parent.name.startswith("fold_"):
                    raise ValueError("nnInteractive checkpoint_path must be inside a fold_N model directory.")
                fold_value = configured.parent.name.removeprefix("fold_")
                fold = int(fold_value) if fold_value.isdigit() else fold_value
                return configured.parent.parent, fold, configured.name
            return configured, fold, checkpoint_name
        return self.resolve_path(self.config.get("model_dir") or DEFAULT_MODEL_DIR).resolve(), fold, checkpoint_name

    def _load_session(self):
        if self._session is not None:
            return self._session
        error = self.availability_error
        if error:
            raise RuntimeError(error)
        import torch

        requested_device = str(self.config.get("device") or "auto").lower()
        device = ("cuda:0" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested nnInteractive device {device!r}, but CUDA is unavailable.")
        repo_path = str(self._repo_path())
        try:
            module = importlib.import_module("nnInteractive.inference.inference_session")
        except ModuleNotFoundError as exc:
            if exc.name != "nnInteractive":
                raise ImportError(f"nnInteractive dependency missing: {exc.name}") from exc
            sys.path.insert(0, repo_path)
            try:
                module = importlib.import_module("nnInteractive.inference.inference_session")
            finally:
                sys.path.remove(repo_path)
        model_dir, fold, checkpoint_name = self._model_location()
        session = module.nnInteractiveInferenceSession(
            device=torch.device(device),
            use_torch_compile=bool(self.config.get("use_torch_compile", False)),
            verbose=bool(self.config.get("verbose", False)),
            torch_n_threads=int(self.config.get("torch_n_threads", 8)),
            do_autozoom=bool(self.config.get("do_autozoom", True)),
            enable_undo=False,
        )
        session.initialize_from_trained_model_folder(str(model_dir), use_fold=fold, checkpoint_name=checkpoint_name)
        self._session = session
        return session


def _prepare_interactions(prompt: Mapping[str, Any], shape: tuple[int, ...]) -> list[tuple[str, Any, int]]:
    depth, height, width = shape
    default_slice = prompt.get("slice_index")
    interactions: list[tuple[str, Any, int]] = []
    boxes = list(prompt.get("boxes") or [])
    if prompt.get("box") is not None:
        boxes.insert(0, {"box": prompt["box"], "label": prompt.get("box_label", 1)})
    for entry in boxes:
        data = entry if isinstance(entry, Mapping) else {"box": entry}
        values = np.asarray(data.get("box"), dtype=float)
        if values.ndim != 1 or len(values) not in (4, 6) or not np.isfinite(values).all():
            raise ValueError("A box requires finite [x1,y1,x2,y2] or [x1,y1,z1,x2,y2,z2] coordinates.")
        if len(values) == 4:
            z_value = data.get("slice_index", default_slice)
            _, _, z = _parse_point([0, 0], default_slice=z_value, depth=depth)
            values = np.array([values[0], values[1], z, values[2], values[3], z + 1])
        lower, upper = values[:3], values[3:]
        bounds = np.array([width, height, depth])
        if np.any(lower < 0) or np.any(upper > bounds) or np.any(upper <= lower):
            raise ValueError(f"Box {values.tolist()} is empty or outside image bounds {bounds.tolist()}.")
        lower, upper = np.rint(lower).astype(int), np.rint(upper).astype(int)
        if np.any(upper <= lower):
            raise ValueError("Box becomes empty after conversion to voxel indices.")
        bbox = [[int(lower[i]), int(upper[i])] for i in (2, 1, 0)]
        interactions.append(("box", bbox, _strict_label(data.get("label", 1))))

    points = prompt.get("points")
    entries: list[tuple[Any, Any]] = []
    if points is not None and len(points):
        point_list = _normalize_point_entries(points)
        labels = prompt.get("labels")
        if labels is not None and len(labels) != len(point_list):
            raise ValueError("prompt labels must have the same length as prompt points.")
        for index, point in enumerate(point_list):
            label = labels[index] if labels is not None else point.get("label", 1) if isinstance(point, Mapping) else 1
            entries.append((point, label))
    else:
        if prompt.get("labels") is not None and len(prompt["labels"]):
            raise ValueError("prompt labels were provided without points.")
        for key, label in (("positive_points", 1), ("negative_points", 0)):
            if prompt.get(key) is not None:
                entries.extend((point, label) for point in _normalize_point_entries(prompt[key]))
        if prompt.get("point") is not None:
            entries.append((prompt["point"], 0 if "negative" in str(prompt.get("prompt_type", "")) else 1))
    for point, label in entries:
        if not isinstance(point, Mapping) and len(point) not in (2, 3):
            raise ValueError("A point requires [x,y,z], or [x,y] with slice_index.")
        x, y, z = _parse_point(point, default_slice=default_slice, depth=depth)
        coordinates = [z, int(round(y)), int(round(x))]
        if not (0 <= x < width and 0 <= y < height) or any(not 0 <= v < limit for v, limit in zip(coordinates, shape)):
            raise ValueError(f"Prompt point {[x,y,z]} is outside the input volume; coordinates were not clipped.")
        interactions.append(("point", coordinates, _strict_label(label)))

    for entry in _normalize_scribble_entries(prompt.get("scribbles")):
        raw_brush = entry.get("brush_size", 1) if isinstance(entry, Mapping) else entry[2]
        if not math.isfinite(float(raw_brush)) or float(raw_brush) <= 0:
            raise ValueError("Scribble brush_size must be finite and positive.")
        stroke = _parse_scribble(entry, default_slice=default_slice, depth=depth)
        raw_label = entry.get("label", 1) if isinstance(entry, Mapping) else entry[0]
        label = _strict_label(raw_label)
        for x, y in stroke["points"]:
            if not (0 <= x < width and 0 <= y < height and 0 <= round(x) < width and 0 <= round(y) < height):
                raise ValueError("Scribble point is outside the input slice; coordinates were not clipped.")
        canvas = _rasterize_scribbles([stroke], (height, width))[0 if label else 1].astype(np.uint8)[None]
        z = stroke["slice_index"]
        interactions.append(("scribble", (canvas, [[z, z + 1], [0, height], [0, width]]), label))
    return interactions


def _strict_label(value: Any) -> int:
    if not isinstance(value, str) and (not np.isscalar(value) or value not in (0, 1)):
        raise ValueError(f"Prompt label must be 0 or 1, got {value!r}.")
    return _parse_label(value)
