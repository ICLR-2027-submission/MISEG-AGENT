"""Base tool interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path.cwd()


class BaseTool(ABC):
    """Small tool base class with normalized result envelopes."""

    name: str = "base_tool"
    description: str = "Base tool"

    def __init__(self, *, checkpoint_path: str | None = None, config: dict[str, Any] | None = None):
        self.checkpoint_path = checkpoint_path
        self.config = config or {}

    def resolve_path(self, path: str | Path | None) -> Path | None:
        """Resolve user/config paths against cwd first, then the code directory."""

        if path in (None, ""):
            return None
        candidate = Path(str(path)).expanduser()
        if candidate.is_absolute():
            return candidate
        if candidate.exists():
            return candidate.resolve()
        project_candidate = PROJECT_ROOT / candidate
        if project_candidate.exists():
            return project_candidate.resolve()
        return candidate

    @property
    def available(self) -> bool:
        """Whether this tool has a usable implementation for selection."""

        return True

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool and wrap output in a consistent dictionary."""

        try:
            result = self._run(**kwargs)
            if not isinstance(result, dict):
                result = {"output": result}
            result.setdefault("success", True)
            result.setdefault("tool_name", self.name)
            return result
        except Exception as exc:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(exc),
            }

    @abstractmethod
    def _run(self, **kwargs: Any) -> dict[str, Any]:
        """Tool-specific implementation."""
