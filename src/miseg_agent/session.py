"""Versioned diagnosis, independent review, joint preview, and explicit commit."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np

from .prompts.regions import derive_recommended_region_prompt_candidates
from .review.api import review_candidates


@dataclass(frozen=True)
class MaskPreview:
    preview_id: str
    mask_version: int
    mask: np.ndarray
    points: tuple[tuple[int, int, int], ...]
    labels: tuple[int, ...]


class MISegAgentSession:
    """Core workflow for a single image and binary target; no GUI or GT input.

    Call ``diagnose`` before proposing or reviewing points. User-selected points
    need current, executable probe evidence before ``preview``. Only ``commit``
    with explicit approval changes the mask. A commit invalidates all evidence.
    Runtime and refinement tools are injected to keep deployment independent.
    """

    def __init__(self, image, mask, *, diagnosis, refinement, spacing_xyz=(1.0, 1.0, 1.0), config=None):
        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask)
        spacing = np.asarray(spacing_xyz, dtype=float)
        if image.ndim != 3 or min(image.shape) < 1 or not np.isfinite(image).all():
            raise ValueError("image must be a non-empty, finite [Z,Y,X] volume.")
        if mask.shape != image.shape or not np.isin(mask, (0, 1)).all():
            raise ValueError("mask must be binary and have the image shape.")
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("spacing_xyz must contain three positive finite values.")
        self._state: dict[str, Any] = {
            "image": image.copy(), "current_mask": mask.astype(np.uint8, copy=True),
            "metadata": {"image": {"spacing_xyz": spacing.tolist()}},
            "refinement_prompt_history": [], "policy_feedback_history": [],
        }
        self._diagnosis = diagnosis
        self._refinement = refinement
        self._config = deepcopy(config or {})
        self._version = 0
        self._diagnosis_version = None
        self._reviewed = {}
        self._pending = {}

    @property
    def mask(self):
        return self._state["current_mask"].copy()

    @property
    def mask_version(self):
        return self._version

    @property
    def history(self):
        return deepcopy(self._state["refinement_prompt_history"])

    def diagnose(self):
        # Invalidate earlier probe evidence even if a repeated diagnosis fails.
        self._diagnosis_version = None
        self._reviewed.clear()
        self._pending.clear()
        self._state.pop("voxel_error_probabilities", None)
        result = self._diagnosis.predict(
            self._state["image"], self._state["current_mask"], self._state["metadata"]["image"]
        )
        maps = result["voxel_error_probabilities"]
        checked = {}
        for name in ("correct", "false_negative", "false_positive"):
            value = np.asarray(maps[name], dtype=np.float32)
            if value.shape != self._state["image"].shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name} diagnosis map.")
            if np.any(value < 0) or np.any(value > 1):
                raise ValueError(f"{name} scores must be in [0,1].")
            checked[name] = value.copy()
        if not np.allclose(sum(checked.values()), 1.0, atol=1e-4):
            raise ValueError("The three diagnosis probabilities must sum to one.")
        self._state["voxel_error_probabilities"] = checked
        self._diagnosis_version = self._version
        return deepcopy(checked)

    def _require_diagnosis(self):
        if self._diagnosis_version != self._version:
            raise RuntimeError("Run diagnose() for the current mask before requesting guidance.")

    def propose(self):
        self._require_diagnosis()
        candidates = derive_recommended_region_prompt_candidates(
            self._state, config=self._config.get("prompt_policy", {})
        )
        for item in candidates:
            item["mask_version"] = self._version
        return candidates

    def point(self, xyz, label, *, candidate_id=None):
        """Construct a user-added or edited point for a fresh executable probe."""
        self._require_diagnosis()
        raw = np.asarray(xyz, dtype=float)
        if raw.shape != (3,) or not np.isfinite(raw).all() or not np.equal(raw, np.rint(raw)).all():
            raise ValueError("Point coordinates must be three integer voxel indices [x,y,z].")
        coordinates = tuple(int(v) for v in raw)
        if any(v < 0 or v >= size for v, size in zip(coordinates[::-1], self._state["image"].shape)):
            raise ValueError("Point is outside the workspace.")
        if label not in (0, 1):
            raise ValueError("Point label must be 1 (foreground) or 0 (background).")
        source = "false_negative" if label else "false_positive"
        probability = self._state["voxel_error_probabilities"][source][coordinates[::-1]]
        return {"candidate_id": candidate_id or f"user:{uuid4().hex}", "point_xyz": list(coordinates),
                "label": int(label), "probability": float(probability), "source": source,
                "mask_version": self._version, "prompt_kind": "point"}

    def _signature(self, candidate):
        if candidate.get("mask_version") != self._version:
            raise ValueError("Candidate belongs to a different mask version; propose or create it again.")
        item = self.point(candidate["point_xyz"], candidate["label"], candidate_id=candidate["candidate_id"])
        return (self._version, str(item["candidate_id"]), tuple(item["point_xyz"]), item["label"])

    def review(self, candidates=None):
        self._require_diagnosis()
        candidates = self.propose() if candidates is None else deepcopy(candidates)
        signatures = [self._signature(item) for item in candidates]
        for signature in signatures:
            self._reviewed.pop(signature, None)
        # Re-review changes the evidence supporting a pending preview.
        self._pending.clear()
        results = review_candidates(self._state, candidates, self._refinement, config=self._config)
        for signature, result in zip(signatures, results):
            result["mask_version"] = self._version
            self._reviewed[signature] = deepcopy(result)
        return results

    def preview(self, selected_candidates):
        """Apply explicitly user-selected points jointly, without committing.

        A reviewed reject/review verdict can be overridden by user selection;
        unavailable evidence cannot. Moving or relabeling a point needs review.
        """
        self._require_diagnosis()
        selected = list(selected_candidates)
        if not selected:
            raise ValueError("Select at least one reviewed point for a preview.")
        signatures = [self._signature(item) for item in selected]
        if len(set(signatures)) != len(signatures):
            raise ValueError("The same point cannot be submitted twice.")
        for signature in signatures:
            record = self._reviewed.get(signature)
            if not record or not record.get("metric_reviewed"):
                raise RuntimeError("Every selected point needs a successful current probe; review edited points again.")
        points = tuple(item[2] for item in signatures)
        labels = tuple(item[3] for item in signatures)
        result = self._refinement.run(
            image=self._state["image"], current_mask=self._state["current_mask"],
            prompt={"points": [list(p) for p in points], "labels": list(labels)}, conditioning_mode="rough_refine"
        )
        if not result.get("success"):
            raise RuntimeError(result.get("error", "Joint refinement failed."))
        mask = np.asarray(result.get("raw_model_mask", result.get("mask")))
        if mask.shape != self._state["image"].shape or not np.isin(mask, (0, 1)).all():
            raise ValueError("Joint refinement returned an invalid mask.")
        token = uuid4().hex
        self._pending[token] = (self._version, mask.astype(np.uint8, copy=True), points, labels)
        return MaskPreview(token, self._version, mask.astype(np.uint8, copy=True), points, labels)

    def commit(self, preview_id: str, *, approved: bool):
        """Commit a preview only after the caller has obtained human approval."""
        if approved is not True:
            raise ValueError("Mask commitment requires explicit human approval.")
        if preview_id not in self._pending:
            raise ValueError("Unknown, rejected, or stale preview.")
        version, mask, points, labels = self._pending[preview_id]
        if version != self._version:
            raise ValueError("Preview does not belong to the current mask.")
        self._state["current_mask"] = mask.copy()
        self._state["refinement_prompt_history"].append({
            "mask_version": version, "points": [list(p) for p in points], "labels": list(labels),
            "preview_id": preview_id,
        })
        self._version += 1
        self._diagnosis_version = None
        self._state.pop("voxel_error_probabilities", None)
        self._reviewed.clear()
        self._pending.clear()
        return self.mask

    def reject(self, preview_id: str):
        """Discard a preview while retaining the current mask and diagnosis."""
        if preview_id not in self._pending:
            raise ValueError("Unknown preview.")
        del self._pending[preview_id]

    def dismiss(self, candidate):
        """Record a user-dismissed recommendation for history suppression."""
        signature = self._signature(candidate)
        self._state["policy_feedback_history"].append({
            "candidate_id": signature[1], "point": list(signature[2]), "label": signature[3], "response": "rejected"
        })
