"""Load the auditable foundation-model capability catalog."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "resources" / "model_catalog.yaml"


def foundation_model_catalog(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    inline = config.get("foundation_model_catalog")
    if isinstance(inline, Mapping):
        return dict(inline)

    configured_path = config.get("foundation_model_catalog_path")
    path = Path(str(configured_path)) if configured_path else DEFAULT_CATALOG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return {"models": {}}
    return _read_catalog(path.resolve())


def foundation_model_entry(config: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    catalog = foundation_model_catalog(config)
    return dict((catalog.get("models", {}) or {}).get(name, {}) or {})


@lru_cache(maxsize=8)
def _read_catalog(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Foundation model catalog must be a mapping: {path}")
    return loaded
