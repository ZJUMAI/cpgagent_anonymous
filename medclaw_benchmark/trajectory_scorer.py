"""Deterministic process-level scoring from guideline trajectories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from medclaw.trajectory.action_set import resolve_action_set, resolve_verifications
from medclaw.utils import read_json, write_json


DEFAULT_MICRO_WEIGHT = 0.5
DEFAULT_MACRO_WEIGHT = 0.5
DEFAULT_UNSAFE_PENALTY = 0.25


def score_trajectory_run(run_dir: str | Path) -> dict[str, Any]:
    """Score process-level guideline adherence via dynamic rubric step alignment."""

    from medclaw_benchmark.dyn_traj_scorer import score_dyn_trajectory_run

    return score_dyn_trajectory_run(run_dir)


def score_trajectory_run_legacy(run_dir: str | Path) -> dict[str, Any]:
    """Legacy final-answer keyword coverage scorer kept for comparison."""

    run_path = Path(run_dir).resolve()
    trajectory_path = run_path / "guideline_trajectory.json"
    final_answer_path = run_path / "final_answers.json"
    if not trajectory_path.is_file():
        return _missing_legacy_result(run_path, "guideline_trajectory.json is missing")
    if not final_answer_path.is_file():
        return _missing_legacy_result(run_path, "final_answers.json is missing")

    trajectory = read_json(trajectory_path)
    final_answers = read_json(final_answer_path)
    answer_text = str(final_answers.get("answer_text", ""))
    steps = trajectory.get("trajectory", [])
    step_scores = []
    expected_total = 0
    matched_total = 0
    supported_expected = 0
    supported_matched = 0
    guideline_unobserved_expected = 0
    guideline_unobserved_matched = 0
    unsafe_total = 0
    unsafe_hit_total = 0
    acceptable_matched_total = 0
    conditional_matched_total = 0

    if not isinstance(steps, list):
        steps = []

    for step in steps:
        if not isinstance(step, Mapping):
            continue
        action_set = resolve_action_set(step)
        verification = {
            str(item.get("action_id")): item
            for item in resolve_verifications(step)
            if item.get("action_id") is not None
        }

        required_items, required_matched, required_missed = _score_bucket(
            answer_text,
            action_set.get("required") or [],
            bucket="required",
            verification=verification,
        )
        for item in required_items:
            expected_total += 1
            classification = item["verification_classification"]
            if classification == "supported_by_report_and_guideline":
                supported_expected += 1
            elif classification == "guideline_supported_but_unobserved":
                guideline_unobserved_expected += 1
        for item in required_matched:
            matched_total += 1
            classification = item["verification_classification"]
            if classification == "supported_by_report_and_guideline":
                supported_matched += 1
            elif classification == "guideline_supported_but_unobserved":
                guideline_unobserved_matched += 1

        _, acceptable_matched, _ = _score_bucket(
            answer_text,
            action_set.get("acceptable") or [],
            bucket="acceptable",
            verification=verification,
        )
        _, conditional_matched, _ = _score_bucket(
            answer_text,
            action_set.get("conditional") or [],
            bucket="conditional",
            verification=verification,
        )
        acceptable_matched_total += len(acceptable_matched)
        conditional_matched_total += len(conditional_matched)

        unsafe_hits = []
        for action in action_set.get("unsafe") or []:
            if not isinstance(action, Mapping):
                continue
            unsafe_total += 1
            action_text = str(action.get("action", ""))
            if _matches_unsafe_action(answer_text, action_text):
                unsafe_hit_total += 1
                unsafe_hits.append(
                    {
                        "action_id": action.get("action_id"),
                        "action": action_text,
                        "bucket": "unsafe",
                    }
                )

        coverage = (
            len(required_matched) / len(required_items) if required_items else None
        )
        step_scores.append(
            {
                "step": step.get("step"),
                "phase": step.get("phase"),
                "expected_count": len(required_items),
                "matched_count": len(required_matched),
                "coverage": coverage,
                "matched_actions": required_matched,
                "missed_actions": required_missed,
                "soft_matches": {
                    "acceptable": acceptable_matched,
                    "conditional": conditional_matched,
                },
                "action_set_counts": {
                    "required": len(action_set.get("required") or []),
                    "acceptable": len(action_set.get("acceptable") or []),
                    "conditional": len(action_set.get("conditional") or []),
                    "unsafe": len(action_set.get("unsafe") or []),
                },
                "unsafe_hits": unsafe_hits,
            }
        )

    coverage_score = matched_total / expected_total if expected_total else 0.0
    unsafe_hit_rate = unsafe_hit_total / unsafe_total if unsafe_total else 0.0
    penalized = max(coverage_score - DEFAULT_UNSAFE_PENALTY * unsafe_hit_rate, 0.0)
    report_supported_score = (
        supported_matched / supported_expected if supported_expected else None
    )
    guideline_unobserved_score = (
        guideline_unobserved_matched / guideline_unobserved_expected
        if guideline_unobserved_expected
        else None
    )
    macro_total = round(penalized * 100, 3)

    return {
        "status": "success",
        "case_id": trajectory.get("case_id") or final_answers.get("case_id"),
        "run_id": final_answers.get("run_id"),
        "score_type": "process_level_guideline_trajectory",
        "macro_trajectory_total": macro_total,
        "coverage": round(coverage_score, 4),
        "coverage_after_unsafe_penalty": round(penalized, 4),
        "expected_action_count": expected_total,
        "matched_action_count": matched_total,
        "unsafe": {
            "total": unsafe_total,
            "hits": unsafe_hit_total,
            "hit_rate": round(unsafe_hit_rate, 4),
            "penalty_weight": DEFAULT_UNSAFE_PENALTY,
        },
        "soft_buckets": {
            "acceptable_matched": acceptable_matched_total,
            "conditional_matched": conditional_matched_total,
        },
        "retrospective_consistency": {
            "report_supported_expected": supported_expected,
            "report_supported_matched": supported_matched,
            "score": None
            if report_supported_score is None
            else round(report_supported_score, 4),
        },
        "guideline_optimality": {
            "guideline_unobserved_expected": guideline_unobserved_expected,
            "guideline_unobserved_matched": guideline_unobserved_matched,
            "score": None
            if guideline_unobserved_score is None
            else round(guideline_unobserved_score, 4),
        },
        "step_scores": step_scores,
        "limitations": [
            "Hard score uses action_set.required coverage minus unsafe hit penalty.",
            "action_set.acceptable and action_set.conditional are soft matches only.",
            "Gold steps are read from dual-LM planner.action_set and verifier.verifications.",
        ],
    }


def combine_micro_macro_scores(
    *,
    micro_total: float | int | None,
    macro_total: float | int | None,
    micro_weight: float = DEFAULT_MICRO_WEIGHT,
    macro_weight: float = DEFAULT_MACRO_WEIGHT,
) -> dict[str, Any]:
    if micro_total is None or macro_total is None:
        return {
            "micro_final_answer_total": micro_total,
            "macro_trajectory_total": macro_total,
            "combined_total": None,
            "weights": {"micro": micro_weight, "macro": macro_weight},
            "status": "incomplete",
        }
    combined = float(micro_total) * micro_weight + float(macro_total) * macro_weight
    return {
        "micro_final_answer_total": float(micro_total),
        "macro_trajectory_total": float(macro_total),
        "combined_total": round(combined, 3),
        "weights": {"micro": micro_weight, "macro": macro_weight},
        "status": "success",
    }


def _score_bucket(
    answer_text: str,
    actions: list[Any],
    *,
    bucket: str,
    verification: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    matched: list[dict[str, Any]] = []
    missed: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        action_id = str(action.get("action_id") or "")
        action_text = str(action.get("action", ""))
        action_type = str(action.get("action_type", ""))
        item = {
            "action_id": action_id,
            "action": action_text,
            "action_type": action_type,
            "bucket": bucket,
            "condition": action.get("condition"),
            "verification_classification": verification.get(action_id, {}).get(
                "classification"
            ),
        }
        scored.append(item)
        if _matches_action(answer_text, action_text, action_type):
            matched.append(item)
        else:
            missed.append(item)
    return scored, matched, missed


def _missing_legacy_result(run_dir: Path, reason: str) -> dict[str, Any]:
    return {
        "status": "missing",
        "score_type": "process_level_guideline_trajectory",
        "macro_trajectory_total": None,
        "reason": reason,
    }


def _missing_result(run_dir: Path, reason: str) -> dict[str, Any]:
    result = {
        "status": "missing",
        "score_type": "process_level_guideline_trajectory",
        "macro_trajectory_total": None,
        "reason": reason,
    }
    write_json(run_dir / "trajectory_scores.json", result)
    return result


def _matches_unsafe_action(answer_text: str, forbidden_text: str) -> bool:
    haystack = _normalized(answer_text)
    forbidden = _normalized(forbidden_text)
    if not forbidden:
        return False
    return forbidden in haystack or _matches_action(
        answer_text, forbidden_text, "treatment_planning"
    )


def _matches_action(answer_text: str, action_text: str, action_type: str) -> bool:
    haystack = _normalized(answer_text)
    action = action_text.lower()
    if action_type == "staging":
        return any(token in haystack for token in ["tnm", "分期", "stage", "ajcc"])
    if action_type == "molecular_testing":
        return any(
            token in haystack
            for token in [
                "molecular",
                "biomarker",
                "egfr",
                "alk",
                "ros1",
                "pd l1",
                "pd-l1",
                "分子",
                "驱动基因",
                "生物标志物",
            ]
        )
    if action_type == "pathology":
        return any(token in haystack for token in ["patholog", "histolog", "病理", "组织学"])
    if "surgery" in action or "surgical" in action or "resectable" in action:
        return any(token in haystack for token in ["surgery", "surgical", "resect", "手术", "切除"])
    if "adjuvant" in action:
        return any(token in haystack for token in ["adjuvant", "辅助"])
    if "performance status" in action:
        return any(token in haystack for token in ["performance status", "ecog", "体能"])
    keywords = _keywords(action_text)
    if not keywords:
        return False
    hits = sum(1 for keyword in keywords if keyword in haystack)
    return hits >= min(2, len(keywords))


def _keywords(text: str) -> list[str]:
    stop = {
        "the",
        "and",
        "or",
        "to",
        "of",
        "if",
        "is",
        "are",
        "from",
        "with",
        "because",
        "according",
        "recommend",
        "consider",
    }
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower())
        if len(token) > 1 and token not in stop
    ]


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower()))
