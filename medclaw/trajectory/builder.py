"""Build step-level guideline-grounded trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from medclaw.trajectory.action_set import (
    DYNAMIC_RUBRIC_SCHEMA_VERSION,
    assemble_action_set,
)
from medclaw.trajectory.report import extract_patient_events
from medclaw.trajectory.rubric import nsclc_2010_rubric_rules
from medclaw.trajectory.schema import (
    STAGE_ORDER,
    STEP_BY_STAGE,
    ActionSet,
    ClinicalAction,
    PatientEvent,
    RubricRule,
    Stage,
    TrajectoryStep,
)
from medclaw.trajectory.verification import verify_action
from medclaw.utils import write_json


def build_guideline_trajectory(
    case_id: str,
    report_path: Path | str,
    *,
    guideline_id: str = "NSCLC_2010",
    guideline_version: str = "2010",
    decision_date: str = "2010-12-31",
    diagnosis_year: int | None = None,
    clinical_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build the report/rubric/trajectory artifact for one patient."""

    if guideline_id != "NSCLC_2010" or str(guideline_version) != "2010":
        raise ValueError(
            "This released trajectory rubric supports only NSCLC_2010@2010; "
            f"received {guideline_id}@{guideline_version}."
        )
    events = extract_patient_events(case_id, report_path, clinical_path=clinical_path)
    rules = nsclc_2010_rubric_rules(guideline_source=guideline_id)
    steps = build_steps(events, rules)
    canonical_state = canonical_state_from_events(events)
    data: dict[str, Any] = {
        "case_id": case_id,
        "schema_version": DYNAMIC_RUBRIC_SCHEMA_VERSION,
        "guideline_id": guideline_id,
        "guideline_version": str(guideline_version),
        "decision_date": decision_date,
        "evaluation_mode": "fixed_historical_guideline",
        "diagnosis_year": diagnosis_year,
        "patient_event_table": [event.to_dict() for event in events],
        "guideline_rubric_rules": [rule.to_dict() for rule in rules],
        "trajectory": [step.to_dict() for step in steps],
        "canonical_state": canonical_state,
    }
    if output_path is not None:
        write_json(Path(output_path), data)
    return data


def build_steps(
    events: Iterable[PatientEvent], rules: Iterable[RubricRule]
) -> list[TrajectoryStep]:
    event_list = sorted(events, key=lambda event: (event.available_at_step, event.event_id))
    rule_list = list(rules)
    max_step = max(
        [event.available_at_step for event in event_list]
        + [STEP_BY_STAGE[rule.applicable_stage] for rule in rule_list]
        + [0]
    )

    steps: list[TrajectoryStep] = []
    for step in range(max_step + 1):
        phase = STAGE_ORDER[min(step, len(STAGE_ORDER) - 1)]
        visible_events = [event for event in event_list if event.available_at_step <= step]
        future_events = [event for event in event_list if event.available_at_step > step]
        applicable_rules = [
            rule for rule in rule_list if STEP_BY_STAGE[rule.applicable_stage] == step
        ]
        visible_state = visible_state_from_events(visible_events, phase=phase)
        candidates = _candidate_actions_from_rules(
            applicable_rules, step=step, state=visible_state
        )
        forbidden = tuple(
            dict.fromkeys(
                forbidden
                for rule in applicable_rules
                for forbidden in rule.forbidden_actions
            )
        )
        action_set_dict = assemble_action_set(
            candidates,
            forbidden_actions=forbidden,
            state=visible_state,
            step=step,
        )
        action_set = _action_set_from_dict(action_set_dict)
        required_actions = _clinical_actions_from_dicts(action_set_dict.get("required") or [])
        verification = tuple(
            verify_action(
                action.to_dict(),
                future_events=future_events,
                rubric_rules=applicable_rules,
            ).to_dict()
            for action in required_actions
        )
        steps.append(
            TrajectoryStep(
                step=step,
                phase=phase,
                visible_state=visible_state,
                hidden_future=tuple(
                    dict.fromkeys(event.event_type for event in future_events)
                ),
                action_set=action_set,
                observed_events=tuple(event.event_id for event in visible_events),
                model_prediction=None,
                verification=verification,
                canonical_state_after_step=canonical_state_from_events(visible_events),
            )
        )
    return steps


def visible_state_from_events(
    events: Iterable[PatientEvent], *, phase: Stage | None = None
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    open_questions = set()
    completed_actions = []

    for event in events:
        attrs = event.attributes
        if event.stage == "baseline":
            for key in ["age", "sex", "race", "primary_site", "disease_type"]:
                if attrs.get(key) is not None:
                    state[key] = attrs[key]
        elif event.stage == "diagnosis_confirmation":
            state["diagnosis"] = attrs.get("diagnosis", event.content)
            completed_actions.append(
                {
                    "action": "pathologic confirmation",
                    "status": "completed",
                    "evidence": event.evidence_span,
                }
            )
        elif event.stage == "staging":
            for key in ["T", "N", "M", "stage_group"]:
                if attrs.get(key):
                    state[key] = attrs[key]
            completed_actions.append(
                {
                    "action": "TNM staging",
                    "status": "completed",
                    "evidence": event.evidence_span,
                }
            )
        elif event.stage == "biomarker_assessment":
            molecular = dict(state.get("molecular_status", {}))
            gene = attrs.get("gene")
            if gene:
                molecular[str(gene)] = attrs.get("result") or "reported"
                state["molecular_status"] = molecular
        elif event.stage == "treatment_observed":
            treatments = list(state.get("observed_treatments", []))
            treatment_record = {
                "treatment_type": attrs.get("treatment_type", event.content),
                "status": attrs.get("status", "recorded_unknown"),
            }
            if attrs.get("intent"):
                treatment_record["intent"] = attrs["intent"]
            if attrs.get("note"):
                treatment_record["note"] = attrs["note"]
            treatments.append(treatment_record)
            state["observed_treatments"] = treatments
        elif event.stage == "followup_or_outcome":
            state["outcome"] = event.content

    if "diagnosis" not in state:
        open_questions.add("pathology unknown")
    if not all(key in state for key in ["T", "N", "M"]):
        open_questions.add("complete TNM stage unknown")
    if "molecular_status" not in state:
        open_questions.add("molecular status unknown")
        if phase == "biomarker_assessment":
            state["molecular_status"] = "unknown"
    if "PD-L1" not in state.get("molecular_status", {}):
        open_questions.add("PD-L1 unknown")
        if phase == "biomarker_assessment":
            state["PD-L1"] = "unknown"
    if "stage_group" not in state:
        open_questions.add("AJCC stage group unknown")

    if completed_actions:
        state["completed_actions"] = completed_actions
    if open_questions:
        state["open_questions"] = sorted(open_questions)
    return state


def canonical_state_from_events(events: Iterable[PatientEvent]) -> dict[str, Any]:
    visible = visible_state_from_events(events)
    completed_actions = visible.pop("completed_actions", [])
    open_questions = visible.pop("open_questions", [])
    return {
        "known_facts": visible,
        "completed_actions": completed_actions,
        "open_questions": open_questions,
    }


def _candidate_actions_from_rules(
    rules: Iterable[RubricRule], *, step: int, state: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for rule in rules:
        bucket = "required" if rule.priority == "must" else "acceptable"
        priority = "high" if rule.priority == "must" else ("medium" if rule.priority == "should" else "low")
        for action_template in rule.recommended_actions:
            action = _render_action(action_template, state or {})
            if action in seen:
                existing = actions[seen[action]]
                existing["guideline_rule_ids"] = list(
                    dict.fromkeys(existing["guideline_rule_ids"] + [rule.rule_id])
                )
                existing["required_evidence"] = list(
                    dict.fromkeys(existing["required_evidence"] + list(rule.required_evidence))
                )
                continue
            if action.lower().startswith("if "):
                action_obj = {
                    "action_id": f"s{step}_a{len(actions)}",
                    "action_type": _action_type(action),
                    "action": action,
                    "priority": priority,
                    "bucket": "conditional",
                    "condition": action,
                    "guideline_rule_ids": [rule.rule_id],
                    "required_evidence": list(rule.required_evidence),
                    "rationale": f"Recommended by {rule.guideline_source} rule {rule.rule_id}.",
                }
            else:
                action_obj = {
                    "action_id": f"s{step}_a{len(actions)}",
                    "action_type": _action_type(action),
                    "action": action,
                    "priority": priority,
                    "bucket": bucket,
                    "guideline_rule_ids": [rule.rule_id],
                    "required_evidence": list(rule.required_evidence),
                    "rationale": f"Recommended by {rule.guideline_source} rule {rule.rule_id}.",
                }
            seen[action] = len(actions)
            actions.append(action_obj)
    return actions


def _clinical_actions_from_dicts(items: Iterable[Mapping[str, Any]]) -> list[ClinicalAction]:
    actions: list[ClinicalAction] = []
    for item in items:
        actions.append(
            ClinicalAction(
                action_id=str(item["action_id"]),
                action_type=str(item["action_type"]),
                action=str(item["action"]),
                priority=item.get("priority", "high"),  # type: ignore[arg-type]
                guideline_rule_ids=tuple(item.get("guideline_rule_ids") or ()),
                required_evidence=tuple(item.get("required_evidence") or ()),
                rationale=str(item.get("rationale") or ""),
                bucket=item.get("bucket", "required"),  # type: ignore[arg-type]
                condition=item.get("condition"),
            )
        )
    return actions


def _action_set_from_dict(payload: Mapping[str, Any]) -> ActionSet:
    return ActionSet(
        required=tuple(_clinical_actions_from_dicts(payload.get("required") or [])),
        acceptable=tuple(_clinical_actions_from_dicts(payload.get("acceptable") or [])),
        conditional=tuple(_clinical_actions_from_dicts(payload.get("conditional") or [])),
        unsafe=tuple(_clinical_actions_from_dicts(payload.get("unsafe") or [])),
    )


def _actions_from_rules(rules: Iterable[RubricRule], *, step: int) -> list[ClinicalAction]:
    return _clinical_actions_from_dicts(
        _candidate_actions_from_rules(rules, step=step, state={})
    )


def _render_action(action: str, state: Mapping[str, Any]) -> str:
    if "{tnm}" not in action:
        return action
    values = [str(state.get(key) or "").strip() for key in ("T", "N", "M")]
    tnm = "".join(values) if all(values) else "the documented TNM classification"
    if all(values) and any("X" in value.upper() for value in values):
        return (
            "complete unknown TNM components before deriving AJCC stage group "
            f"(currently {tnm})"
        )
    return action.replace("{tnm}", tnm)


def _action_type(action: str) -> str:
    lowered = action.lower()
    if "stage" in lowered or "tnm" in lowered:
        return "staging"
    if "molecular" in lowered or "biomarker" in lowered or "pd-l1" in lowered:
        return "molecular_testing"
    if "treatment" in lowered or "therapy" in lowered or "surgery" in lowered:
        return "treatment_planning"
    if "histological" in lowered or "patholog" in lowered:
        return "pathology"
    return "diagnostic_workup"
