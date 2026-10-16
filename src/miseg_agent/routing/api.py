"""Filter capabilities before applying the benchmark routing policy."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any

from .catalog import foundation_model_catalog
from .policy import benchmark_policy_choice, _normalize_objective
from .targeting import normalize_target_name, tool_supports_target


def select_initial_model(
    target: str,
    available_models: Iterable[str],
    *,
    case_id: str,
    objective: str = "accuracy",
    dataset_hint: str = "",
    explicit_model: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Route a CT request among models the caller has actually installed.

    Availability is supplied by the deployment, never inferred from the catalog.
    This function selects a backend; it does not download weights or run models.
    """
    settings = dict(config or {})
    available_models = set(available_models)
    normalized = normalize_target_name(target)
    if not normalized or normalized == "unknown" or not str(case_id).strip():
        raise ValueError("A specific target and stable case_id are required.")
    if _normalize_objective(objective) is None:
        raise ValueError(f"Unknown routing objective: {objective}")
    models = foundation_model_catalog(settings).get("models", {})
    candidates = {
        name for name in available_models
        if (entry := models.get(name))
        and entry.get("initial_rough_eligible") is True
        and 3 in entry.get("dimensions", [])
        and any(normalize_target_name(item) in {"ct", "medical"} for item in entry.get("modalities", []))
        and entry.get("prompt_mode") in {"automatic", "class_prompt", "text"}
        and tool_supports_target(settings, name, normalized)
    }
    # The catalog expresses kidney label mappings separately from routing aliases.
    if normalized == "bilateral kidney" and "totalsegmentator" in available_models:
        if tool_supports_target(settings, "totalsegmentator", "kidney"):
            candidates.add("totalsegmentator")
    if explicit_model is not None:
        if explicit_model not in candidates:
            raise ValueError(f"Requested model is unavailable or incompatible: {explicit_model}")
        return {"tool": explicit_model, "reason": "Explicit compatible model choice.", "objective": "explicit"}
    if not candidates:
        raise ValueError(f"No available automatic 3D model supports {target!r}.")
    state = {
        "image": SimpleNamespace(shape=(1, 1, 1)),
        "target_description": {"target_class": normalized, "modality_hint": "ct"},
        "metadata": {"case_id": case_id, "dataset_hint": dataset_hint, "foundation_model_objective": objective},
    }
    choice = benchmark_policy_choice(state, candidates, settings)
    if choice is None:
        return {"tool": None, "reason": "No matching benchmark evidence; choose a compatible model explicitly.",
                "compatible_models": sorted(candidates)}
    return choice
