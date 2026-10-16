"""Load and apply the benchmark-backed foundation-model routing policy."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from .targeting import normalize_target_name


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY_PATH = PROJECT_ROOT / "resources" / "policy.txt"


def foundation_selection_policy(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the routing policy instructions and audited benchmark evidence."""

    config = config or {}
    selection_config = dict(config.get("foundation_selection") or {})
    policy_path = _resolve_path(selection_config.get("policy_path") or DEFAULT_POLICY_PATH)
    evidence_path = _resolve_path(
        selection_config.get("evidence_path")
        or policy_path.parent / "benchmark_evidence.yaml"
    )
    return _read_policy_bundle(policy_path.resolve(), evidence_path.resolve())


def selection_objective(state: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> str:
    """Resolve an explicit routing objective; accuracy remains the default."""

    config = config or {}
    target = dict(state.get("target_description") or {})
    metadata = dict(state.get("metadata") or {})
    configured = dict(config.get("foundation_selection") or {})
    raw = (
        metadata.get("foundation_model_objective")
        or target.get("selection_objective")
    )
    normalized = _normalize_objective(raw)
    if normalized:
        return normalized

    query = str(state.get("user_query") or "").strip().lower()
    disregard_quality = any(
        token in query
        for token in (
            "regardless of quality",
            "regardless quality",
            "不考虑质量",
            "不管质量",
            "无论质量",
        )
    )
    if any(token in query for token in ("low gpu", "low vram", "gpu memory", "显存", "gpu占用")):
        return "unrestricted_gpu_memory" if disregard_quality else "gpu_memory"
    if any(token in query for token in ("fastest", "low latency", "speed priority", "最快", "低延迟", "速度优先")):
        return "unrestricted_latency" if disregard_quality else "latency"
    if any(token in query for token in ("balanced", "balance quality", "综合平衡", "平衡模式")):
        return "balanced"
    return _normalize_objective(configured.get("default_objective")) or "accuracy"


def benchmark_policy_choice(
    state: Mapping[str, Any],
    candidates: set[str],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Rank compatible candidates using only evidence declared by the policy.

    This function has no diversity state and applies no repetition penalty. The
    same candidate is returned repeatedly when it remains best for the supplied
    target and objective.
    """

    if not candidates:
        return None
    config = config or {}
    bundle = foundation_selection_policy(config)
    evidence = dict(bundle.get("evidence") or {})
    target = dict(state.get("target_description") or {})
    metadata = dict(state.get("metadata") or {})
    image = state.get("image")
    if len(getattr(image, "shape", ()) or ()) != 3:
        return None
    modality = normalize_target_name(target.get("modality_hint"))
    if modality not in {"ct", "cta", "computed tomography"}:
        return None
    target_name = normalize_target_name(target.get("target_class"))
    aliases = {
        normalize_target_name(key): normalize_target_name(value)
        for key, value in dict(evidence.get("target_aliases") or {}).items()
    }
    evidence_target = aliases.get(target_name, target_name)
    objective = selection_objective(state, config)
    dataset_hint = str(
        metadata.get("dataset_hint")
        or metadata.get("dataset")
        or target.get("dataset_hint")
        or ""
    ).strip()
    scope_name, scope = _evidence_scope(evidence, evidence_target, dataset_hint)
    quality_grades = dict(scope.get("quality_grade") or {}) if scope else {}
    reliability_grades = dict(scope.get("reliability_grade") or {}) if scope else {}
    benchmarked_candidates = sorted(set(candidates).intersection(quality_grades))
    if not benchmarked_candidates:
        return None

    grade_order = {
        normalize_target_name(name): int(rank)
        for name, rank in dict((evidence.get("quality_grades") or {}).get("rank") or {}).items()
    }
    resources = dict(evidence.get("model_resources") or {})
    ranked_rows = []
    for name in benchmarked_candidates:
        quality = normalize_target_name(quality_grades.get(name))
        reliability = normalize_target_name(reliability_grades.get(name) or quality)
        resource = dict(resources.get(name) or {})
        if quality not in grade_order or reliability not in grade_order:
            continue
        if "mean_cold_start_wall_time_sec" not in resource or "peak_gpu_memory_delta_mb" not in resource:
            continue
        ranked_rows.append(
            {
                "tool": name,
                "quality_grade": quality,
                "quality_rank": grade_order[quality],
                "reliability_grade": reliability,
                "reliability_rank": grade_order[reliability],
                "latency": float(resource["mean_cold_start_wall_time_sec"]),
                "gpu_memory": float(resource["peak_gpu_memory_delta_mb"]),
            }
        )
    if not ranked_rows:
        return None

    best_quality_rank = max(row["quality_rank"] for row in ranked_rows)
    top_quality = [row for row in ranked_rows if row["quality_rank"] == best_quality_rank]
    selected_row: dict[str, Any]
    equivalent_rows: list[dict[str, Any]]
    case_key_source: str | None = None

    if objective == "accuracy":
        best_reliability_rank = max(row["reliability_rank"] for row in top_quality)
        equivalent_rows = [
            row for row in top_quality if row["reliability_rank"] == best_reliability_rank
        ]
        selected_row, case_key_source = _stable_case_choice(state, equivalent_rows, objective)
        metric_name = "quality_grade"
        metric_value: str | float = str(selected_row["quality_grade"])
        tie_breaker = "stable_case_key" if len(equivalent_rows) > 1 else "quality_and_reliability_grade"
        detail = (
            f"quality_grade={selected_row['quality_grade']} and "
            f"reliability_grade={selected_row['reliability_grade']}"
        )
        if len(equivalent_rows) > 1:
            detail += (
                f"; {len(equivalent_rows)} candidates remain evidence-equivalent, so a stable "
                f"case-bound choice ({case_key_source}) selects {selected_row['tool']}"
            )
        ranking = [
            {
                "tool": row["tool"],
                "quality_grade": row["quality_grade"],
                "reliability_grade": row["reliability_grade"],
            }
            for row in sorted(
                ranked_rows,
                key=lambda row: (-row["quality_rank"], -row["reliability_rank"], row["tool"]),
            )
        ]
    elif objective == "balanced":
        equivalent_rows = _pareto_front(top_quality)
        selected_row, case_key_source = _stable_case_choice(state, equivalent_rows, objective)
        metric_name = "quality_grade"
        metric_value = str(selected_row["quality_grade"])
        tie_breaker = "balanced_pareto_then_stable_case_key"
        detail = (
            f"quality_grade={selected_row['quality_grade']}; selected from a {len(equivalent_rows)}-model "
            "Pareto frontier over reliability grade, latency, and GPU memory"
        )
        ranking = [
            {
                "tool": row["tool"],
                "quality_grade": row["quality_grade"],
                "reliability_grade": row["reliability_grade"],
                "latency_sec": row["latency"],
                "gpu_memory_mb": row["gpu_memory"],
                "on_pareto_front": row in equivalent_rows,
            }
            for row in ranked_rows
        ]
    else:
        unrestricted = objective in {"unrestricted_latency", "unrestricted_gpu_memory"}
        resource_name = "gpu_memory" if "gpu_memory" in objective else "latency"
        resource_field = (
            "peak_gpu_memory_delta_mb"
            if resource_name == "gpu_memory"
            else "mean_cold_start_wall_time_sec"
        )
        pool = ranked_rows if unrestricted else top_quality
        equivalent_rows = sorted(pool, key=lambda row: (row[resource_name], row["tool"]))
        selected_row = equivalent_rows[0]
        metric_name = resource_field
        metric_value = float(selected_row[resource_name])
        tie_breaker = "resource_across_all_quality_grades" if unrestricted else "resource_within_top_quality_grade"
        quality_note = "all quality grades" if unrestricted else f"top quality grade={selected_row['quality_grade']}"
        detail = f"{quality_note}; minimum {resource_field}={metric_value:.4f}"
        ranking = [
            {
                "tool": row["tool"],
                "quality_grade": row["quality_grade"],
                "reliability_grade": row["reliability_grade"],
                "value": row[resource_name],
            }
            for row in equivalent_rows
        ]

    selected = str(selected_row["tool"])

    cases = int(scope.get("cases") or 0)
    reason = (
        f"Benchmark-informed routing for target '{target_name}' and objective '{objective}': "
        f"using {scope_name}, {detail} (n={cases})."
    )
    return {
        "tool": selected,
        "reason": reason,
        "confidence": 0.95 if objective == "accuracy" and cases >= 10 else 0.85,
        "objective": objective,
        "evidence_scope": scope_name,
        "evidence_cases": cases,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "quality_grade": selected_row["quality_grade"],
        "reliability_grade": selected_row["reliability_grade"],
        "tie_breaker": tie_breaker,
        "equivalent_candidates": sorted(row["tool"] for row in equivalent_rows),
        "case_key_source": case_key_source,
        "ranking": ranking,
        "policy_path": str(bundle["policy_path"]),
        "evidence_path": str(bundle["evidence_path"]),
    }


def _evidence_scope(
    evidence: Mapping[str, Any],
    target: str,
    dataset_hint: str,
) -> tuple[str, dict[str, Any]]:
    datasets = dict(evidence.get("datasets") or {})
    if dataset_hint:
        matched_name = next(
            (name for name in datasets if name.casefold() == dataset_hint.casefold()),
            None,
        )
        if matched_name:
            row = dict(datasets[matched_name] or {})
            minimum_cases = int(
                (evidence.get("routing_policy") or {}).get("minimum_dataset_cases") or 1
            )
            if normalize_target_name(row.get("target")) == target and int(row.get("cases") or 0) >= minimum_cases:
                return f"dataset:{matched_name}", row
    return f"target:{target}", dict((evidence.get("targets") or {}).get(target) or {})


def _normalize_objective(value: Any) -> str | None:
    normalized = normalize_target_name(value)
    aliases = {
        "accuracy": "accuracy",
        "dice": "accuracy",
        "quality": "accuracy",
        "balanced": "balanced",
        "balance": "balanced",
        "latency": "latency",
        "speed": "latency",
        "runtime": "latency",
        "gpu memory": "gpu_memory",
        "vram": "gpu_memory",
        "memory": "gpu_memory",
        "unrestricted latency": "unrestricted_latency",
        "fastest regardless quality": "unrestricted_latency",
        "unrestricted gpu memory": "unrestricted_gpu_memory",
        "lowest gpu regardless quality": "unrestricted_gpu_memory",
    }
    return aliases.get(normalized)


def _stable_case_choice(
    state: Mapping[str, Any],
    rows: list[dict[str, Any]],
    objective: str,
) -> tuple[dict[str, Any], str | None]:
    ordered = sorted(rows, key=lambda row: row["tool"])
    if len(ordered) == 1:
        return ordered[0], None
    case_key, source = _case_routing_key(state, objective)
    digest = hashlib.sha256(case_key.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], byteorder="big", signed=False) % len(ordered)
    return ordered[index], source


def _case_routing_key(state: Mapping[str, Any], objective: str) -> tuple[str, str]:
    metadata = dict(state.get("metadata") or {})
    image_metadata = dict(metadata.get("image") or {})
    identity_fields = (
        "study_uid",
        "series_uid",
        "case_id",
        "source_image_path",
        "workspace_image_path",
        "source_path",
    )
    identity = ""
    source = "request_fallback"
    for container_name, container in (("metadata", metadata), ("metadata.image", image_metadata)):
        for field in identity_fields:
            value = str(container.get(field) or "").strip()
            if value:
                identity = value
                source = f"{container_name}.{field}"
                break
        if identity:
            break
    target = dict(state.get("target_description") or {})
    image_shape = tuple(int(value) for value in getattr(state.get("image"), "shape", ()) or ())
    if not identity:
        identity = "|".join(
            (
                str(state.get("user_query") or ""),
                str(metadata.get("dataset_hint") or metadata.get("dataset") or ""),
                repr(image_shape),
            )
        )
    key = "|".join(
        (
            identity,
            normalize_target_name(target.get("target_class")),
            normalize_target_name(target.get("modality_hint")),
            objective,
        )
    )
    return key, source


def _pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for row in rows:
        dominated = any(
            other is not row
            and other["reliability_rank"] >= row["reliability_rank"]
            and other["latency"] <= row["latency"]
            and other["gpu_memory"] <= row["gpu_memory"]
            and (
                other["reliability_rank"] > row["reliability_rank"]
                or other["latency"] < row["latency"]
                or other["gpu_memory"] < row["gpu_memory"]
            )
            for other in rows
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: row["tool"])


def _resolve_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=8)
def _read_policy_bundle(policy_path: Path, evidence_path: Path) -> dict[str, Any]:
    if not policy_path.is_file():
        raise FileNotFoundError(f"Foundation routing policy does not exist: {policy_path}")
    if not evidence_path.is_file():
        raise FileNotFoundError(f"Foundation routing evidence does not exist: {evidence_path}")
    instructions = policy_path.read_text(encoding="utf-8")
    with evidence_path.open("r", encoding="utf-8") as handle:
        evidence = yaml.safe_load(handle) or {}
    if not isinstance(evidence, dict):
        raise ValueError(f"Foundation routing evidence must be a mapping: {evidence_path}")
    return {
        "policy_path": policy_path,
        "evidence_path": evidence_path,
        "instructions": instructions,
        "evidence": evidence,
    }
