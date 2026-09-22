"""Typed records for guideline-grounded trajectory reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Stage = Literal[
    "baseline",
    "diagnosis_confirmation",
    "staging",
    "biomarker_assessment",
    "initial_treatment_decision",
    "treatment_observed",
    "followup_or_outcome",
]

STAGE_ORDER: tuple[Stage, ...] = (
    "baseline",
    "diagnosis_confirmation",
    "staging",
    "biomarker_assessment",
    "initial_treatment_decision",
    "treatment_observed",
    "followup_or_outcome",
)

STEP_BY_STAGE: dict[Stage, int] = {
    stage: index for index, stage in enumerate(STAGE_ORDER)
}


@dataclass(frozen=True)
class PatientEvent:
    event_id: str
    case_id: str
    stage: Stage
    event_type: str
    content: str
    source: str
    available_at_step: int
    evidence_span: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RubricRule:
    rule_id: str
    applicable_stage: Stage
    condition: dict[str, Any]
    recommended_actions: tuple[str, ...]
    required_evidence: tuple[str, ...]
    forbidden_actions: tuple[str, ...]
    priority: Literal["must", "should", "optional"]
    guideline_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClinicalAction:
    action_id: str
    action_type: str
    action: str
    priority: Literal["high", "medium", "low"]
    guideline_rule_ids: tuple[str, ...]
    required_evidence: tuple[str, ...]
    rationale: str
    bucket: Literal["required", "acceptable", "conditional", "unsafe"] = "required"
    condition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data.get("condition"):
            data.pop("condition", None)
        return data


@dataclass(frozen=True)
class ActionSet:
    required: tuple[ClinicalAction, ...] = ()
    acceptable: tuple[ClinicalAction, ...] = ()
    conditional: tuple[ClinicalAction, ...] = ()
    unsafe: tuple[ClinicalAction, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": [action.to_dict() for action in self.required],
            "acceptable": [action.to_dict() for action in self.acceptable],
            "conditional": [action.to_dict() for action in self.conditional],
            "unsafe": [action.to_dict() for action in self.unsafe],
        }


@dataclass(frozen=True)
class TrajectoryStep:
    step: int
    phase: Stage | str
    visible_state: dict[str, Any]
    hidden_future: tuple[str, ...]
    observed_events: tuple[str, ...]
    action_set: ActionSet
    model_prediction: dict[str, Any] | None = None
    verification: tuple[dict[str, Any], ...] = ()
    canonical_state_after_step: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action_set"] = self.action_set.to_dict()
        # Legacy mirrors are intentionally omitted; consumers should use action_set.
        return data


@dataclass(frozen=True)
class ActionVerification:
    action_id: str
    action: str
    report_support: Literal["supported", "not_observed", "contradicted"]
    guideline_support: Literal["supported", "uncertain", "contradicted"]
    classification: Literal[
        "supported_by_report_and_guideline",
        "guideline_supported_but_unobserved",
        "observed_but_guideline_uncertain",
        "contradicted_or_premature",
    ]
    evidence_event_ids: tuple[str, ...]
    evidence_text: tuple[str, ...]
    state_update: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["category"] = self.classification
        return data
