"""Fixed-set Planner/Router V2 acceptance metrics and release gates."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from guideline_planner.schemas_v2 import V2SchemaError, validate_planner_action_v2


PLANNER_THRESHOLDS = {
    "action_hit_at_1": (">=", 0.85),
    "unsafe_or_premature_rate": ("<=", 0.01),
    "completed_skill_repeat_rate": ("<=", 0.05),
    "confirmed_evidence_repeat_rate": ("<=", 0.05),
    "state_progress_rate": (">=", 0.80),
    "phase_transition_f1": (">=", 0.85),
    "provenance_legal_rate": (">=", 1.00),
    "supporting_rule_recall": (">=", 0.90),
    "schema_pass_rate": (">=", 1.00),
}

ROUTER_THRESHOLDS = {
    "positive_recall_at_4": (">=", 0.85),
    "positive_gate_mass": (">=", 0.60),
    "single_positive_normalized_entropy": ("<=", 0.90),
    "median_positive_top1_margin": (">=", 0.05),
}

COUNTERFACTUAL_THRESHOLDS = {
    "decision_relevant_change_rate": (">=", 0.75),
    "irrelevant_perturbation_retention_rate": (">=", 0.90),
}


def evaluate_planner_predictions(
    examples: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(examples)
    if not rows:
        raise ValueError("Planner V2 evaluation requires non-empty fixed test examples.")
    events = [_planner_event(row) for row in rows]
    micro = _mean_metrics(events)
    strata: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for row, event in zip(rows, events):
        state = row["trajectory"]["state_before"]
        strata[(state["cancer_family"], state["current_phase"])].append(event)
    per_stratum = {
        f"{family}/{phase}": _mean_metrics(values)
        for (family, phase), values in sorted(strata.items())
    }
    macro = {
        key: sum(value[key] for value in per_stratum.values()) / len(per_stratum)
        for key in next(iter(per_stratum.values()))
    }
    macro["phase_transition_f1"] = _phase_macro_f1(rows)
    counterfactual = _counterfactual_metrics(rows)
    gates = _gate_results(macro, PLANNER_THRESHOLDS)
    gates.update(_gate_results(counterfactual, COUNTERFACTUAL_THRESHOLDS))
    return {
        "schema_version": "planner_evaluation.v2",
        "count": len(rows),
        "micro": micro,
        "macro": macro,
        "per_cancer_phase": per_stratum,
        "counterfactual": counterfactual,
        "gates": gates,
        "accepted": all(gates.values()),
    }


def evaluate_router_predictions(
    examples: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(examples)
    if not rows:
        raise ValueError("Router V2 evaluation requires non-empty fixed test examples.")
    metrics = []
    entropy_values = []
    margin_values = []
    for row in rows:
        gold = row["routing_labels"]
        route = row["routing"]
        positive = set(gold["strong_positive_memory_ids"]) | set(
            gold["weak_positive_memory_ids"]
        )
        active = list(route.get("active_memories") or [])
        top4 = [str(item.get("memory_id")) for item in active[:4]]
        mass = sum(
            float(item.get("weight") or 0.0)
            for item in active
            if str(item.get("memory_id")) in positive
        )
        metrics.append(
            {
                "positive_recall_at_4": float(bool(positive.intersection(top4))),
                "positive_gate_mass": mass,
            }
        )
        if len(positive) == 1:
            diagnostics = route.get("gate_diagnostics") or {}
            entropy_values.append(
                float(
                    route.get("normalized_gate_entropy")
                    if route.get("normalized_gate_entropy") is not None
                    else diagnostics.get("normalized_entropy") or 0.0
                )
            )
        scores = {
            str(item.get("memory_id")): float(item.get("gate_score") or 0.0)
            for item in route.get("candidate_activations") or []
        }
        positive_scores = [score for memory, score in scores.items() if memory in positive]
        negative_scores = [score for memory, score in scores.items() if memory not in positive]
        if positive_scores and negative_scores:
            margin_values.append(max(positive_scores) - max(negative_scores))
    aggregate = _mean_metrics(metrics)
    aggregate["single_positive_normalized_entropy"] = (
        sum(entropy_values) / len(entropy_values) if entropy_values else 1.0
    )
    aggregate["median_positive_top1_margin"] = _median(margin_values)
    gates = _gate_results(aggregate, ROUTER_THRESHOLDS)
    return {
        "schema_version": "router_evaluation.v2",
        "count": len(rows),
        "metrics": aggregate,
        "gates": gates,
        "accepted": all(gates.values()),
    }


def paired_bootstrap_lower_bound(
    daa_scores: Sequence[float],
    baseline_scores: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 17,
) -> float:
    if len(daa_scores) != len(baseline_scores) or not daa_scores:
        raise ValueError("Paired bootstrap requires equal non-empty score vectors.")
    rng = random.Random(seed)
    count = len(daa_scores)
    differences = []
    for _ in range(max(int(samples), 100)):
        indices = [rng.randrange(count) for _ in range(count)]
        differences.append(
            sum(daa_scores[index] - baseline_scores[index] for index in indices) / count
        )
    differences.sort()
    return differences[max(math.floor(0.025 * len(differences)), 0)]


def evaluate_daa_comparison(
    daa_scores: Sequence[float],
    latent_topk_scores: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    lower = paired_bootstrap_lower_bound(
        daa_scores,
        latent_topk_scores,
        samples=samples,
        seed=seed,
    )
    return {
        "schema_version": "daa_comparison.v2",
        "paired_count": len(daa_scores),
        "mean_difference": sum(
            daa - baseline for daa, baseline in zip(daa_scores, latent_topk_scores)
        )
        / len(daa_scores),
        "bootstrap_95ci_lower": lower,
        "accepted_as_default": lower >= -0.02,
    }


def planner_action_macro_score(row: Mapping[str, Any]) -> float:
    """Return the per-example action score used for paired DAA comparison."""

    event = _planner_event(row)
    utilities = (
        event["action_hit_at_1"],
        1.0 - event["unsafe_or_premature_rate"],
        1.0 - event["completed_skill_repeat_rate"],
        1.0 - event["confirmed_evidence_repeat_rate"],
        event["state_progress_rate"],
        event["provenance_legal_rate"],
        event["supporting_rule_recall"],
        event["schema_pass_rate"],
    )
    return sum(utilities) / len(utilities)


def _planner_event(row: Mapping[str, Any]) -> dict[str, float]:
    trajectory = row["trajectory"]
    prediction = row["prediction"]
    state = trajectory["state_before"]
    positive_memories = set(trajectory["routing_labels"]["strong_positive_memory_ids"])
    positive_memories.update(trajectory["routing_labels"]["weak_positive_memory_ids"])
    schema_valid = True
    try:
        validate_planner_action_v2(
            prediction,
            patient_state=state,
            active_memories=list(row.get("active_memories") or []),
        )
    except V2SchemaError:
        schema_valid = False
    actions = list(prediction.get("actions") or [])
    first = actions[0] if actions else {}
    positive_actions = [
        *trajectory["action_set"]["required"],
        *trajectory["action_set"]["acceptable"],
        *[
            item
            for item in trajectory["action_set"]["conditional"]
            if item.get("condition_satisfied") is True
        ],
    ]
    negative_actions = [
        *trajectory["action_set"]["premature"],
        *trajectory["action_set"]["unsafe"],
    ]
    completed = set(state.get("completed_skills") or [])
    delta_fields = set(trajectory["state_delta"])
    # Keep the historical metric key for report compatibility.  Reusing a skill
    # is not itself an error: only a reused skill whose action cannot account for
    # any observed state delta is considered an unproductive repeat.
    unproductive_repeat = any(
        completed.intersection(action.get("required_skills") or [])
        and not delta_fields.intersection(action.get("expected_state_delta") or [])
        for action in actions
    )
    cited_rules = {
        str(rule)
        for action in actions
        for provenance in action.get("provenance") or []
        for rule in provenance.get("rule_ids") or []
    }
    gold_rules = {
        str(rule)
        for item in positive_actions
        for rule in item.get("guideline_rule_ids") or []
    }
    cited_memories = {
        str(provenance.get("memory_id"))
        for action in actions
        for provenance in action.get("provenance") or []
    }
    expected_delta = {
        field
        for action in actions
        for field in action.get("expected_state_delta") or []
    }
    phase_target = trajectory["state_after"]["current_phase"]
    phase_prediction = prediction.get("proposed_phase") or prediction.get("current_phase")
    return {
        "action_hit_at_1": float(any(_action_matches(first, item) for item in positive_actions)),
        "unsafe_or_premature_rate": float(
            any(
                _action_matches(action, item)
                for action in actions
                for item in negative_actions
            )
        ),
        "completed_skill_repeat_rate": float(unproductive_repeat),
        "confirmed_evidence_repeat_rate": float(
            _requests_confirmed_evidence(
                state,
                prediction.get("missing_information") or [],
            )
        ),
        "state_progress_rate": float(bool(delta_fields.intersection(expected_delta))),
        "phase_transition_f1": float(str(phase_prediction) == str(phase_target)),
        "provenance_legal_rate": float(schema_valid and cited_memories.issubset(positive_memories)),
        "supporting_rule_recall": len(cited_rules.intersection(gold_rules)) / max(len(gold_rules), 1),
        "schema_pass_rate": float(schema_valid),
    }


def _action_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if not left or str(left.get("action_type")) != str(right.get("action_type")):
        return False
    left_id = str(left.get("action_id") or "")
    right_id = str(right.get("action_id") or "")
    if left_id and right_id and left_id != right_id:
        return False
    return set(left.get("required_skills") or []).issubset(
        set(right.get("required_skills") or [])
    )


def _counterfactual_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    anchors = {
        (
            str(row["trajectory"]["trajectory_id"]),
            int(row["trajectory"]["turn_index"]),
        ): row
        for row in rows
        if row["trajectory"].get("case_source") == "real"
    }
    relevant: list[float] = []
    irrelevant: list[float] = []
    for row in rows:
        metadata = row["trajectory"].get("counterfactual")
        if not isinstance(metadata, Mapping):
            continue
        anchor = anchors.get(
            (
                str(metadata.get("anchor_trajectory_id") or ""),
                int(metadata.get("anchor_turn_index") or 0),
            )
        )
        if anchor is None:
            continue
        changed = _prediction_signature(row["prediction"]) != _prediction_signature(
            anchor["prediction"]
        )
        if metadata.get("perturbation_type") == "decision_relevant":
            relevant.append(float(changed))
        elif metadata.get("perturbation_type") == "irrelevant":
            irrelevant.append(float(not changed))
    return {
        "decision_relevant_change_rate": sum(relevant) / len(relevant) if relevant else 0.0,
        "irrelevant_perturbation_retention_rate": (
            sum(irrelevant) / len(irrelevant) if irrelevant else 0.0
        ),
    }


def _prediction_signature(prediction: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        (
            str(action.get("action_type") or ""),
            tuple(sorted(action.get("required_skills") or [])),
            tuple(
                sorted(
                    str(provenance.get("memory_id") or "")
                    for provenance in action.get("provenance") or []
                )
            ),
        )
        for action in prediction.get("actions") or []
    )


def _requests_confirmed_evidence(
    state: Mapping[str, Any],
    missing_information: Sequence[Any],
) -> bool:
    aliases = {
        "known_diagnosis": {"known_diagnosis", "diagnosis"},
        "known_stage": {"known_stage", "stage", "staging"},
        "known_biomarkers": {"known_biomarkers", "biomarker", "biomarkers", "molecular"},
        "risk_stratification": {"risk_stratification", "risk", "risk_group"},
    }
    requested = {
        str(item).strip().lower().replace(" ", "_") for item in missing_information
    }
    return any(
        state.get(field) not in (None, {}, []) and bool(requested.intersection(names))
        for field, names in aliases.items()
    )


def _phase_macro_f1(rows: Sequence[Mapping[str, Any]]) -> float:
    pairs = []
    for row in rows:
        trajectory = row["trajectory"]
        prediction = row["prediction"]
        gold = str(trajectory["state_after"]["current_phase"])
        predicted = str(
            prediction.get("proposed_phase") or prediction.get("current_phase") or ""
        )
        pairs.append((gold, predicted))
    labels = sorted({item for pair in pairs for item in pair if item})
    if not labels:
        return 0.0
    values = []
    for label in labels:
        true_positive = sum(gold == label and predicted == label for gold, predicted in pairs)
        false_positive = sum(gold != label and predicted == label for gold, predicted in pairs)
        false_negative = sum(gold == label and predicted != label for gold, predicted in pairs)
        denominator = 2 * true_positive + false_positive + false_negative
        values.append((2 * true_positive / denominator) if denominator else 0.0)
    return sum(values) / len(values)


def _mean_metrics(events: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not events:
        return {}
    return {
        key: sum(float(event[key]) for event in events) / len(events)
        for key in events[0]
    }


def _gate_results(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, tuple[str, float]],
) -> dict[str, bool]:
    return {
        key: float(metrics.get(key, float("nan"))) >= threshold
        if operator == ">="
        else float(metrics.get(key, float("nan"))) <= threshold
        for key, (operator, threshold) in thresholds.items()
    }


def _median(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
