"""Shared target-capability helpers for segmentation model routing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .catalog import foundation_model_entry


def normalize_target_name(value: Any) -> str:
    """Normalize user/config target names without changing their semantics."""

    return " ".join(
        str(value or "").replace("_", " ").replace("-", " ").strip().lower().split()
    )


def configured_targets_for_tool(config: Mapping[str, Any], name: str) -> set[str] | None:
    """Return a tool's explicit semantic coverage, or ``None`` for unrestricted.

    ``target_rois`` is the preferred schema for tools whose user-facing target
    expands to one or more backend labels.  The older fields remain supported
    so existing local configurations keep their routing behavior.
    """

    tool_config = dict((config.get("tools", {}) or {}).get(name, {}) or {})
    catalog_entry = foundation_model_entry(config, name)
    target_rois = tool_config.get("target_rois") or catalog_entry.get("target_rois")
    if isinstance(target_rois, Mapping) and target_rois:
        return _normalized_set(target_rois.keys())

    supported_targets = tool_config.get("supported_targets") or catalog_entry.get("supported_targets")
    if supported_targets:
        return _normalized_set(_as_items(supported_targets))

    roi_subset = tool_config.get("roi_subset")
    if roi_subset:
        return _normalized_set(_as_items(roi_subset))

    label_prompts = tool_config.get("label_prompts")
    if isinstance(label_prompts, Mapping) and label_prompts:
        return _normalized_set(label_prompts.keys())

    return None


def tool_supports_target(config: Mapping[str, Any], name: str, target: Any) -> bool:
    normalized_target = normalize_target_name(target)
    if not normalized_target or normalized_target == "unknown":
        return True
    configured = configured_targets_for_tool(config, name)
    return configured is None or normalized_target in configured


def _as_items(value: Any) -> Iterable[Any]:
    if isinstance(value, str):
        return (value,)
    return value or ()


def _normalized_set(values: Iterable[Any]) -> set[str]:
    return {normalized for value in values if (normalized := normalize_target_name(value))}
