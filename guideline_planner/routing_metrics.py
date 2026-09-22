"""Trajectory quality and current-state routing evaluation metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


try:
    from medclaw.trajectory.action_set import dynamic_rubric_scores_from_action_set
except ImportError:  # pragma: no cover - planner-only environments
    dynamic_rubric_scores_from_action_set = None  # type: ignore[assignment]


DEFAULT_QUALITY_WEIGHTS = {
    "required": 0.30,
    "acceptable": 0.15,
    "defer": 0.10,
    "progress": 0.30,
    "unsafe": 0.15,
}


def compute_trajectory_quality(
    planner_output: Mapping[str, Any],
    dynamic_rubric: Mapping[str, Any] | None,
    future_evidence: Mapping[str, Any] | None,
    patient_state_before: Mapping[str, Any],
    patient_state_after: Mapping[str, Any],
) -> float:
    """Combine evaluator-provided quality components without clinical rules."""

    details = trajectory_quality_details(
        planner_output,
        dynamic_rubric,
        future_evidence,
        patient_state_before,
        patient_state_after,
    )
    return float(details["quality"])


def trajectory_quality_details(
    planner_output: Mapping[str, Any],
    dynamic_rubric: Mapping[str, Any] | None,
    future_evidence: Mapping[str, Any] | None,
    patient_state_before: Mapping[str, Any],
    patient_state_after: Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    rubric = dict(dynamic_rubric or {})
    evidence = dict(future_evidence or {})
    action_set = rubric.get("action_set")
    if isinstance(action_set, Mapping) and dynamic_rubric_scores_from_action_set is not None:
        derived = dynamic_rubric_scores_from_action_set(action_set)
        for key, value in derived.items():
            if key not in rubric and value is not None:
                rubric[key] = value
    components: dict[str, float | None] = {
        "required": _component("required_score", rubric, evidence),
        "acceptable": _component("acceptable_score", rubric, evidence),
        "defer": _component("defer_score", rubric, evidence),
        "progress": _component("state_progress_score", rubric, evidence),
        "unsafe": _component("unsafe_score", rubric, evidence),
    }
    if components["progress"] is None:
        components["progress"] = _state_progress(patient_state_before, patient_state_after)
    selected_weights = dict(DEFAULT_QUALITY_WEIGHTS)
    if weights:
        selected_weights.update({str(key): float(value) for key, value in weights.items()})
    positive_keys = [
        key for key in ("required", "acceptable", "defer", "progress")
        if components[key] is not None
    ]
    positive_weight = sum(selected_weights[key] for key in positive_keys)
    positive = (
        sum(selected_weights[key] * float(components[key]) for key in positive_keys)
        / positive_weight
        if positive_weight > 0
        else 0.0
    )
    unsafe = float(components["unsafe"] or 0.0)
    quality = min(max(positive - selected_weights["unsafe"] * unsafe, 0.0), 1.0)
    return {
        "quality": quality,
        "components": components,
        "weights": selected_weights,
        "unsafe_known": components["unsafe"] is not None,
        "planner_action_present": bool(planner_output),
    }


def evaluate_routing_runs(run_dirs: Iterable[str | Path]) -> dict[str, Any]:
    per_run = [_evaluate_run(Path(item)) for item in run_dirs]
    per_run = [item for item in per_run if item is not None]
    metric_names = (
        "action_score",
        "unsafe_hit_rate",
        "state_progress_rate",
        "redundant_action_rate",
        "premature_action_rate",
        "average_active_memory_count",
        "gate_entropy",
        "memory_counterfactual_utility",
    )
    aggregate: dict[str, Any] = {}
    for name in metric_names:
        values = [item[name] for item in per_run if isinstance(item.get(name), (int, float))]
        aggregate[name] = mean(values) if values else None
        aggregate[f"{name}_coverage"] = len(values) / len(per_run) if per_run else 0.0
    return {
        "run_count": len(per_run),
        "aggregate": aggregate,
        "runs": per_run,
    }


def write_routing_evaluation(run_dir: str | Path) -> dict[str, Any]:
    """Evaluate one run and persist metrics with explicit coverage fields."""

    path = Path(run_dir)
    payload = evaluate_routing_runs([path])
    output_path = path / "routing_metrics.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _evaluate_run(run_dir: Path) -> dict[str, Any] | None:
    routing = _read_json(run_dir / "memory_routing.json")
    if not isinstance(routing, Mapping):
        return None
    steps = [item for item in routing.get("steps", []) if isinstance(item, Mapping)]
    judge = _read_json(run_dir / "judge_scores.json")
    action_score = None
    if isinstance(judge, Mapping) and isinstance(judge.get("final_total"), (int, float)):
        maximum = sum(float(value) for value in (judge.get("weights") or {}).values())
        action_score = float(judge["final_total"]) / maximum if maximum > 0 else None
    unsafe_values = [item.get("unsafe_score") for item in steps]
    unsafe_values = [float(item) for item in unsafe_values if isinstance(item, (int, float))]
    active_counts = [len(item.get("active_memories", [])) for item in steps]
    entropies = [
        float(item["gate_entropy"])
        for item in steps
        if isinstance(item.get("gate_entropy"), (int, float))
    ]
    utility_values = [
        float(item["counterfactual_utility"])
        for item in steps
        if isinstance(item.get("counterfactual_utility"), (int, float))
    ]
    state_progress, redundant = _state_and_redundancy(run_dir)
    premature_values = [
        bool(item.get("premature_action"))
        for item in steps
        if isinstance(item.get("premature_action"), bool)
    ]
    return {
        "run_dir": str(run_dir),
        "case_id": routing.get("case_id"),
        "run_id": routing.get("run_id"),
        "action_score": action_score,
        "unsafe_hit_rate": (
            sum(value > 0 for value in unsafe_values) / len(unsafe_values)
            if unsafe_values
            else None
        ),
        "state_progress_rate": state_progress,
        "redundant_action_rate": redundant,
        "premature_action_rate": (
            sum(premature_values) / len(premature_values) if premature_values else None
        ),
        "average_active_memory_count": mean(active_counts) if active_counts else None,
        "gate_entropy": mean(entropies) if entropies else None,
        "memory_counterfactual_utility": (
            mean(utility_values) if utility_values else None
        ),
    }


def _judge_quality(run_dir: Path) -> tuple[float | None, float | None]:
    judge = _read_json(run_dir / "judge_scores.json")
    if not isinstance(judge, Mapping):
        return None, None
    weights = judge.get("weights", {})
    total = judge.get("final_total")
    maximum = sum(float(value) for value in weights.values()) if isinstance(weights, Mapping) else 0
    quality = None
    dual_layer = judge.get("dual_layer_scores")
    if isinstance(dual_layer, Mapping) and isinstance(
        dual_layer.get("combined_total"),
        (int, float),
    ):
        quality = min(max(float(dual_layer["combined_total"]) / 100.0, 0.0), 1.0)
    if quality is None:
        trajectory = _read_json(run_dir / "trajectory_scores.json")
        trajectory_total = trajectory.get("macro_trajectory_total")
        if isinstance(trajectory_total, (int, float)):
            quality = min(max(float(trajectory_total) / 100.0, 0.0), 1.0)
    if quality is None and isinstance(total, (int, float)) and maximum:
        quality = float(total) / maximum
    scores = judge.get("llm_scores", {})
    critical = scores.get("CR", {}) if isinstance(scores, Mapping) else {}
    unsafe = None
    if isinstance(critical, Mapping):
        score = critical.get("score")
        max_score = critical.get("max_score")
        if isinstance(score, (int, float)) and isinstance(max_score, (int, float)) and max_score:
            unsafe = 0.0 if float(score) >= float(max_score) else 1.0
    return quality, unsafe


def _state_progress(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    ignored = {
        "current_time",
        "state_update_summary",
        "warnings",
        "previous_actions",
        "pending_actions",
    }
    keys = (set(before) | set(after)) - ignored
    return 1.0 if any(before.get(key) != after.get(key) for key in keys) else 0.0


def _state_and_redundancy(run_dir: Path) -> tuple[float | None, float | None]:
    records = _read_jsonl(run_dir / "dual_agent_rounds.jsonl")
    if not records:
        return None, None
    progressed = 0
    redundant = 0
    seen_without_progress: set[str] = set()
    action_count = 0
    for record in records:
        delta = record.get("patient_state_delta", {})
        has_progress = bool(delta)
        progressed += int(has_progress)
        for skill in record.get("tool_skills", []):
            action_count += 1
            skill_name = str(skill)
            if not has_progress and skill_name in seen_without_progress:
                redundant += 1
            if not has_progress:
                seen_without_progress.add(skill_name)
    return progressed / len(records), (redundant / action_count if action_count else None)


def _component(name: str, *sources: Mapping[str, Any]) -> float | None:
    for source in sources:
        value = source.get(name)
        if isinstance(value, (int, float)):
            return min(max(float(value), 0.0), 1.0)
        feedback = source.get("routing_feedback")
        if isinstance(feedback, Mapping) and isinstance(feedback.get(name), (int, float)):
            return min(max(float(feedback[name]), 0.0), 1.0)
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            result.append(dict(value))
    return result
