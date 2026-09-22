"""Deterministic phase transition rules for patient_state.v2."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

from guideline_planner.schemas_v2 import V2SchemaError, validate_patient_state_v2


PHASE_ORDER = (
    "diagnostic_workup",
    "diagnosis_confirmation",
    "staging",
    "risk_or_biomarker_stratification",
    "treatment_selection",
    "treatment_monitoring",
    "followup",
)


def can_commit_phase(
    state: Mapping[str, Any],
    proposed_phase: str | None,
) -> tuple[bool, str]:
    current = str(state.get("current_phase") or "")
    proposed = str(proposed_phase or current)
    if proposed == current:
        return True, "phase unchanged"
    if current not in PHASE_ORDER or proposed not in PHASE_ORDER:
        return False, "phase is outside the V2 phase taxonomy"
    current_index = PHASE_ORDER.index(current)
    proposed_index = PHASE_ORDER.index(proposed)
    if proposed_index != current_index + 1:
        return False, "phase transitions must advance exactly one verified stage"
    predicate = _ENTRY_REQUIREMENTS[proposed]
    if not predicate(state):
        return False, f"entry requirements for {proposed!r} are not satisfied"
    return True, "entry requirements satisfied"


def commit_phase(
    state: Mapping[str, Any],
    proposed_phase: str | None,
) -> dict[str, Any]:
    normalized = validate_patient_state_v2(state)
    allowed, reason = can_commit_phase(normalized, proposed_phase)
    if not allowed:
        raise V2SchemaError(f"Cannot commit proposed phase: {reason}.")
    result = deepcopy(normalized)
    if proposed_phase:
        result["current_phase"] = str(proposed_phase)
    return result


def _has_diagnosis(state: Mapping[str, Any]) -> bool:
    return bool(state.get("known_diagnosis"))


def _has_stage(state: Mapping[str, Any]) -> bool:
    return bool(state.get("known_stage"))


def _has_risk_or_biomarkers(state: Mapping[str, Any]) -> bool:
    return bool(state.get("known_biomarkers") or state.get("risk_stratification"))


def _has_selected_treatment(state: Mapping[str, Any]) -> bool:
    return bool(state.get("treatment_history") or state.get("selected_treatment"))


def _has_treatment_observation(state: Mapping[str, Any]) -> bool:
    if not _has_selected_treatment(state):
        return False
    return any(
        str(item.get("status") or "").lower() in {"started", "completed", "observed"}
        for item in state.get("treatment_history", [])
        if isinstance(item, Mapping)
    )


def _has_followup_evidence(state: Mapping[str, Any]) -> bool:
    return any(
        str(item.get("field") or "") in {"response_assessment", "followup_status", "progression"}
        for item in state.get("evidence_ledger", [])
        if isinstance(item, Mapping)
    )


_ENTRY_REQUIREMENTS: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "diagnostic_workup": lambda state: True,
    "diagnosis_confirmation": _has_diagnosis,
    "staging": _has_diagnosis,
    "risk_or_biomarker_stratification": _has_stage,
    "treatment_selection": lambda state: _has_stage(state) and _has_risk_or_biomarkers(state),
    "treatment_monitoring": _has_selected_treatment,
    "followup": lambda state: _has_treatment_observation(state) and _has_followup_evidence(state),
}
