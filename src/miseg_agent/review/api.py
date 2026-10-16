"""Independent candidate probes; no probe is committed to the current mask."""
from __future__ import annotations

from typing import Any

from .evidence import volume_prompt_evidence
from .verdicts import _expert_review_config, _metric_slice_reviews


def review_candidates(state: dict[str, Any], candidates: list[dict[str, Any]], tool, *, config=None):
    """Return executable review records using one shared mask-only control.

    ``tool`` must implement the NNInteractiveTool interface. Each candidate is
    reset to the same mask. A positive candidate also receives a point-only probe.
    All measurements are relative to the mask and ED maps, never ground truth.
    """
    if not candidates:
        return []
    ids = [str(item["candidate_id"]) for item in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate identifiers must be unique within one review.")
    settings = _expert_review_config(dict(config or {}))
    evidence = volume_prompt_evidence(
        state, candidates, registry={"nninteractive": tool}, config=settings, baseline_cache={}
    )
    return _metric_slice_reviews(candidates, slice_evidence=evidence, config=settings)
