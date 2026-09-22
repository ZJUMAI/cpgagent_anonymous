"""Strict V2 data contracts for patient-state-conditioned guideline planning."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from guideline_planner.constants import (
    PATIENT_STATE_SCHEMA_VERSION,
    PLANNER_ACTION_SCHEMA_VERSION,
    PLANNER_TRAJECTORY_SCHEMA_VERSION,
    SUPPORTED_PLANNER_CANCER_FAMILIES,
)
from guideline_planner.release_scope import (
    action_target_allows,
    action_target_families,
    copy_release_scope,
    load_release_scope,
    release_scope_hash,
)


class V2SchemaError(ValueError):
    """Raised when a V2 planner artifact violates its contract."""


_NONEMPTY_STRING = {"type": "string", "minLength": 1}
_STRING_ARRAY = {"type": "array", "items": _NONEMPTY_STRING, "uniqueItems": True}

GUIDELINE_REF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["guideline_id", "version"],
    "properties": {
        "guideline_id": _NONEMPTY_STRING,
        "version": _NONEMPTY_STRING,
        "effective_from": {"type": ["string", "null"]},
        "effective_to": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

MEMORY_PROVENANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["memory_id", "rule_ids", "source_spans"],
    "properties": {
        "memory_id": _NONEMPTY_STRING,
        "rule_ids": {"type": "array", "items": _NONEMPTY_STRING, "minItems": 1},
        "source_spans": {"type": "array", "items": _NONEMPTY_STRING, "minItems": 1},
        "guideline_id": {"type": ["string", "null"]},
        "version": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

PLANNER_ACTION_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "objective",
        "action_type",
        "required_skills",
        "preconditions",
        "expected_state_delta",
        "provenance",
    ],
    "properties": {
        "action_id": {"type": ["string", "null"]},
        "objective": _NONEMPTY_STRING,
        "action_type": _NONEMPTY_STRING,
        "required_skills": _STRING_ARRAY,
        "preconditions": {"type": "array", "items": {"type": "object"}},
        "unmet_preconditions": _STRING_ARRAY,
        "expected_state_delta": _STRING_ARRAY,
        "provenance": {
            "type": "array",
            "items": MEMORY_PROVENANCE_SCHEMA,
            "minItems": 1,
        },
        "priority": {"type": ["integer", "string", "null"]},
        "repeat_justification": {"type": ["string", "null"]},
        "risk_level": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

BLOCKED_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["objective", "reason", "category"],
    "properties": {
        "objective": _NONEMPTY_STRING,
        "reason": _NONEMPTY_STRING,
        "category": {
            "type": "string",
            "enum": ["conditional", "premature", "unsafe", "unavailable"],
        },
        "until": _STRING_ARRAY,
    },
    "additionalProperties": True,
}

EVIDENCE_LEDGER_ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["field", "value", "skill_name", "time", "status", "source_ref"],
    "properties": {
        "field": _NONEMPTY_STRING,
        "value": {},
        "skill_name": _NONEMPTY_STRING,
        "time": {"type": ["string", "integer", "number", "null"]},
        "status": _NONEMPTY_STRING,
        "source_ref": {"type": ["string", "null"]},
        "call_id": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

PATIENT_STATE_V2_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "schema_version",
        "case_id",
        "cancer_family",
        "disease_subtype",
        "current_phase",
        "decision_date",
        "guideline_context",
        "known_diagnosis",
        "known_stage",
        "known_biomarkers",
        "risk_stratification",
        "available_modalities",
        "completed_skills",
        "completed_actions",
        "treatment_history",
        "evidence_ledger",
        "last_transition",
        "pending_actions",
        "blocked_actions",
        "unresolved_information",
    ],
    "properties": {
        "schema_version": {"const": PATIENT_STATE_SCHEMA_VERSION},
        "case_id": _NONEMPTY_STRING,
        "cancer_family": {
            "type": "string",
            "enum": list(SUPPORTED_PLANNER_CANCER_FAMILIES),
        },
        "disease_subtype": {"type": ["string", "null"]},
        "current_phase": _NONEMPTY_STRING,
        "decision_date": _NONEMPTY_STRING,
        "guideline_context": {
            "type": "object",
            "required": ["decision_date", "guidelines"],
            "properties": {
                "decision_date": _NONEMPTY_STRING,
                "guidelines": {
                    "type": "array",
                    "items": GUIDELINE_REF_SCHEMA,
                    "minItems": 1,
                },
            },
            "additionalProperties": True,
        },
        "known_diagnosis": {"type": ["string", "null"]},
        "known_stage": {"type": ["string", "null"]},
        "known_biomarkers": {"type": "object"},
        "risk_stratification": {"type": "object"},
        "available_modalities": {"type": ["object", "array"]},
        "completed_skills": _STRING_ARRAY,
        "completed_actions": {"type": "array", "items": {"type": "object"}},
        "treatment_history": {"type": "array", "items": {"type": "object"}},
        "current_treatment_line": {"type": ["integer", "null"], "minimum": 0},
        "evidence_ledger": {
            "type": "array",
            "items": EVIDENCE_LEDGER_ENTRY_SCHEMA,
        },
        "last_transition": {"type": ["object", "null"]},
        "pending_actions": {"type": "array"},
        "blocked_actions": {"type": "array"},
        "unresolved_information": _STRING_ARRAY,
    },
    "additionalProperties": True,
}

PLANNER_ACTION_V2_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "schema_version",
        "current_phase",
        "proposed_phase",
        "missing_information",
        "actions",
        "blocked_actions",
        "should_stop",
        "reason",
    ],
    "properties": {
        "schema_version": {"const": PLANNER_ACTION_SCHEMA_VERSION},
        "current_phase": _NONEMPTY_STRING,
        "proposed_phase": {"type": ["string", "null"]},
        "missing_information": _STRING_ARRAY,
        "actions": {
            "type": "array",
            "items": PLANNER_ACTION_ITEM_SCHEMA,
            "maxItems": 3,
        },
        "blocked_actions": {"type": "array", "items": BLOCKED_ACTION_SCHEMA},
        "should_stop": {"type": "boolean"},
        "reason": _NONEMPTY_STRING,
    },
    "additionalProperties": False,
}

TRAINING_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "action_id",
        "objective",
        "action_type",
        "required_skills",
        "preconditions",
        "expected_state_delta",
        "guideline_rule_ids",
        "supporting_memory_ids",
        "source_spans",
    ],
    "properties": {
        "action_id": _NONEMPTY_STRING,
        "objective": _NONEMPTY_STRING,
        "action_type": _NONEMPTY_STRING,
        "required_skills": _STRING_ARRAY,
        "preconditions": {"type": "array", "items": {"type": "object"}},
        "expected_state_delta": _STRING_ARRAY,
        "guideline_rule_ids": {
            "type": "array",
            "items": _NONEMPTY_STRING,
            "minItems": 1,
        },
        "supporting_memory_ids": {
            "type": "array",
            "items": _NONEMPTY_STRING,
            "minItems": 1,
        },
        "source_spans": {"type": "array", "items": _NONEMPTY_STRING, "minItems": 1},
        "condition_satisfied": {"type": ["boolean", "null"]},
        "risk_level": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

ACTION_SET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["required", "acceptable", "conditional", "premature", "unsafe"],
    "properties": {
        bucket: {"type": "array", "items": TRAINING_ACTION_SCHEMA}
        for bucket in ("required", "acceptable", "conditional", "premature", "unsafe")
    },
    "additionalProperties": False,
}

COUNTERFACTUAL_SCHEMA: dict[str, Any] = {
    "type": ["object", "null"],
    "required": [
        "anchor_trajectory_id",
        "anchor_turn_index",
        "perturbation_type",
        "changed_fields",
    ],
    "properties": {
        "anchor_trajectory_id": _NONEMPTY_STRING,
        "anchor_turn_index": {"type": "integer", "minimum": 0},
        "perturbation_type": {
            "type": "string",
            "enum": ["decision_relevant", "irrelevant"],
        },
        "changed_fields": {"type": "array", "items": _NONEMPTY_STRING, "minItems": 1},
    },
    "additionalProperties": True,
}

PLANNER_TRAJECTORY_V2_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "schema_version",
        "trajectory_id",
        "base_case_id",
        "case_id",
        "turn_index",
        "split",
        "case_source",
        "guideline_context",
        "state_before",
        "action_set",
        "accepted_plan_variants",
        "tool_executions",
        "state_after",
        "state_delta",
        "routing_labels",
        "provenance",
    ],
    "properties": {
        "schema_version": {"const": PLANNER_TRAJECTORY_SCHEMA_VERSION},
        "trajectory_id": _NONEMPTY_STRING,
        "base_case_id": _NONEMPTY_STRING,
        "case_id": _NONEMPTY_STRING,
        "turn_index": {"type": "integer", "minimum": 0},
        "split": {"type": "string", "enum": ["train", "validation", "test"]},
        "case_source": {"type": "string", "enum": ["real", "counterfactual"]},
        "supersedes_trajectory_ids": {
            "type": "array",
            "items": _NONEMPTY_STRING,
            "uniqueItems": True,
        },
        "counterfactual": COUNTERFACTUAL_SCHEMA,
        "guideline_context": {
            "type": "object",
            "required": ["decision_date", "guidelines"],
            "properties": {
                "decision_date": _NONEMPTY_STRING,
                "guidelines": {
                    "type": "array",
                    "items": GUIDELINE_REF_SCHEMA,
                    "minItems": 1,
                },
            },
            "additionalProperties": True,
        },
        "state_before": PATIENT_STATE_V2_SCHEMA,
        "action_set": ACTION_SET_SCHEMA,
        "accepted_plan_variants": {
            "type": "array",
            "items": PLANNER_ACTION_V2_SCHEMA,
            "minItems": 1,
        },
        "tool_executions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["skill_name", "status"],
                "properties": {
                    "skill_name": _NONEMPTY_STRING,
                    "status": {
                        "type": "string",
                        "enum": ["success", "failed", "skipped", "blocked"],
                    },
                    "result_summary": {"type": ["string", "null"]},
                    "evidence_refs": {"type": "array"},
                },
                "additionalProperties": True,
            },
        },
        "state_after": PATIENT_STATE_V2_SCHEMA,
        "state_delta": {"type": "object", "minProperties": 1},
        "routing_labels": {
            "type": "object",
            "required": [
                "strong_positive_memory_ids",
                "weak_positive_memory_ids",
                "hard_negative_memory_ids",
                "easy_negative_memory_ids",
            ],
            "properties": {
                key: _STRING_ARRAY
                for key in (
                    "strong_positive_memory_ids",
                    "weak_positive_memory_ids",
                    "hard_negative_memory_ids",
                    "easy_negative_memory_ids",
                )
            },
            "additionalProperties": False,
        },
        "provenance": {
            "type": "object",
            "required": [
                "rule_compiler_version",
                "case_source_ref",
                "teacher_model",
                "teacher_model_version",
                "teacher_prompt_hash",
                "teacher_output_hash",
                "review_status",
                "reviewer_id",
                "reviewed_at",
                "qc_flags",
            ],
            "properties": {
                "rule_compiler_version": _NONEMPTY_STRING,
                "case_source_ref": _NONEMPTY_STRING,
                "teacher_model": _NONEMPTY_STRING,
                "teacher_model_version": _NONEMPTY_STRING,
                "teacher_prompt_hash": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$",
                },
                "teacher_output_hash": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$",
                },
                "review_status": {
                    "type": "string",
                    "enum": ["pending", "approved", "rejected"],
                },
                "reviewer_id": {"type": ["string", "null"]},
                "reviewed_at": {"type": ["string", "null"]},
                "qc_flags": _STRING_ARRAY,
            },
            "additionalProperties": True,
        },
    },
    "additionalProperties": False,
}


_PATIENT_VALIDATOR = Draft202012Validator(PATIENT_STATE_V2_SCHEMA)
_ACTION_VALIDATOR = Draft202012Validator(PLANNER_ACTION_V2_SCHEMA)
_TRAJECTORY_VALIDATOR = Draft202012Validator(PLANNER_TRAJECTORY_V2_SCHEMA)


def validate_patient_state_v2(state: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(state))
    _raise_schema_errors(_PATIENT_VALIDATOR, result, "patient_state.v2")
    _validate_decision_date(result)
    if result["guideline_context"]["decision_date"] != result["decision_date"]:
        raise V2SchemaError(
            "patient_state.v2 decision_date must match guideline_context.decision_date."
        )
    _validate_guideline_effectivity(result)
    return result


def validate_planner_action_v2(
    payload: Mapping[str, Any],
    *,
    patient_state: Mapping[str, Any] | None = None,
    active_memories: Sequence[Mapping[str, Any] | str] | None = None,
) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    _raise_schema_errors(_ACTION_VALIDATOR, result, "planner_action.v2")
    if result["should_stop"] and result["actions"]:
        raise V2SchemaError(
            "planner_action.v2 cannot contain actions when should_stop=true."
        )
    if not result["should_stop"] and not result["actions"]:
        raise V2SchemaError(
            "planner_action.v2 requires at least one action unless should_stop=true."
        )

    state = (
        validate_patient_state_v2(patient_state) if patient_state is not None else None
    )
    if state is not None and result["current_phase"] != state["current_phase"]:
        raise V2SchemaError(
            "planner_action.v2 current_phase must echo patient_state.v2 current_phase."
        )
    active_index = _active_memory_index(active_memories or [])
    active_ids = set(active_index)
    target_guidelines = _target_guideline_pairs(state) if state else set()

    for action in result["actions"]:
        # Skills describe reusable capabilities, not one-shot tasks.  A report
        # reader may legitimately be called several times for different files,
        # sections, questions, or cross-checks.  Loop/no-progress detection
        # therefore belongs to trajectory execution and evaluation, where the
        # action target and resulting state delta are available; it must not be
        # inferred from membership in patient_state.completed_skills.
        if action.get("unmet_preconditions"):
            raise V2SchemaError(
                "planner_action.v2 selected an action with unmet_preconditions."
            )
        if state is not None and action["preconditions"]:
            from guideline_planner.rule_engine import evaluate_condition

            if not all(
                evaluate_condition(condition, state)
                for condition in action["preconditions"]
            ):
                raise V2SchemaError(
                    "planner_action.v2 selected an action whose preconditions are false."
                )
        for provenance in action["provenance"]:
            memory_id = provenance["memory_id"]
            if active_ids and memory_id not in active_ids:
                raise V2SchemaError(
                    f"planner_action.v2 provenance memory {memory_id!r} is not active."
                )
            metadata = active_index.get(memory_id, {})
            _validate_provenance_against_memory(provenance, metadata)
            pair = (
                str(
                    provenance.get("guideline_id") or metadata.get("guideline_id") or ""
                ),
                str(provenance.get("version") or metadata.get("version") or ""),
            )
            if target_guidelines and all(pair) and pair not in target_guidelines:
                raise V2SchemaError(
                    f"planner_action.v2 cites guideline/version {pair!r} outside the patient target context."
                )
    return result


def validate_trajectory_record_v2(record: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(record))
    _raise_schema_errors(_TRAJECTORY_VALIDATOR, result, "planner_trajectory.v2")
    before = validate_patient_state_v2(result["state_before"])
    after = validate_patient_state_v2(result["state_after"])
    if result["guideline_context"] != before["guideline_context"]:
        raise V2SchemaError(
            "planner_trajectory.v2 guideline_context must equal state_before.guideline_context."
        )
    if before["case_id"] != result["case_id"] or after["case_id"] != result["case_id"]:
        raise V2SchemaError("planner_trajectory.v2 case IDs do not match its states.")
    counterfactual = result.get("counterfactual")
    if result["case_source"] == "counterfactual" and not isinstance(
        counterfactual, Mapping
    ):
        raise V2SchemaError(
            "Counterfactual trajectories require anchor and perturbation metadata."
        )
    if result["case_source"] == "real" and counterfactual is not None:
        raise V2SchemaError("Real trajectories cannot carry counterfactual metadata.")
    if result["provenance"]["review_status"] == "approved" and not all(
        str(result["provenance"].get(field) or "").strip()
        for field in ("reviewer_id", "reviewed_at")
    ):
        raise V2SchemaError(
            "Approved trajectories require reviewer_id and reviewed_at."
        )
    if before == after:
        raise V2SchemaError(
            "planner_trajectory.v2 state_before and state_after must differ."
        )
    expected_delta = state_delta(before, after)
    if result["state_delta"] != expected_delta:
        raise V2SchemaError(
            "planner_trajectory.v2 state_delta does not match state_before/state_after."
        )
    positive_ids = set(result["routing_labels"]["strong_positive_memory_ids"])
    positive_ids.update(result["routing_labels"]["weak_positive_memory_ids"])
    if not positive_ids:
        raise V2SchemaError("planner_trajectory.v2 requires positive routing memories.")
    active = [{"memory_id": item} for item in sorted(positive_ids)]
    allowed_actions = [
        *result["action_set"]["required"],
        *result["action_set"]["acceptable"],
        *[
            item
            for item in result["action_set"]["conditional"]
            if item.get("condition_satisfied") is True
        ],
    ]
    for variant in result["accepted_plan_variants"]:
        validate_planner_action_v2(
            variant,
            patient_state=before,
            active_memories=active,
        )
        for action in variant["actions"]:
            if not any(
                _plan_action_is_allowed(action, item) for item in allowed_actions
            ):
                raise V2SchemaError(
                    "Teacher plan action is not grounded in a required/acceptable/"
                    "satisfied-conditional rule candidate."
                )
        expected_progress = {
            field
            for action in variant["actions"]
            for field in action.get("expected_state_delta") or []
        }
        if not expected_progress.intersection(expected_delta):
            raise V2SchemaError(
                "Accepted plan variants must predict at least one observed state_delta field."
            )
    positives = (
        len(result["action_set"]["required"])
        + len(result["action_set"]["acceptable"])
        + sum(
            item.get("condition_satisfied") is True
            for item in result["action_set"]["conditional"]
        )
    )
    negatives = sum(
        len(result["action_set"][bucket]) for bucket in ("premature", "unsafe")
    )
    if positives == 0 or negatives == 0:
        raise V2SchemaError(
            "planner_trajectory.v2 requires at least one positive and one negative action."
        )
    negative_actions = [
        *result["action_set"]["premature"],
        *result["action_set"]["unsafe"],
        *[
            item
            for item in result["action_set"]["conditional"]
            if item.get("condition_satisfied") is False
        ],
    ]
    ambiguous = {
        _training_action_signature(item) for item in allowed_actions
    }.intersection(_training_action_signature(item) for item in negative_actions)
    if ambiguous:
        raise V2SchemaError(
            "planner_trajectory.v2 has semantically identical positive and negative actions."
        )
    successful_skills = {
        str(item.get("skill_name") or "")
        for item in result["tool_executions"]
        if isinstance(item, Mapping)
        and str(item.get("status") or "").lower() == "success"
        and str(item.get("skill_name") or "")
    }
    newly_completed = set(after["completed_skills"]) - set(before["completed_skills"])
    if newly_completed != successful_skills:
        raise V2SchemaError(
            "planner_trajectory.v2 successful tool executions and newly completed skills differ."
        )
    if after.get("last_transition") is None:
        raise V2SchemaError(
            "planner_trajectory.v2 state_after must record last_transition."
        )
    return result


def _plan_action_is_allowed(
    plan_action: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    if str(plan_action.get("action_type")) != str(candidate.get("action_type")):
        return False
    if not set(plan_action.get("required_skills") or []).issubset(
        set(candidate.get("required_skills") or [])
    ):
        return False
    if {
        json.dumps(item, ensure_ascii=False, sort_keys=True)
        for item in plan_action.get("preconditions") or []
    } != {
        json.dumps(item, ensure_ascii=False, sort_keys=True)
        for item in candidate.get("preconditions") or []
    }:
        return False
    rule_ids = {
        str(item)
        for provenance in plan_action.get("provenance") or []
        for item in provenance.get("rule_ids") or []
        if isinstance(provenance, Mapping)
    }
    memory_ids = {
        str(provenance.get("memory_id") or "")
        for provenance in plan_action.get("provenance") or []
        if isinstance(provenance, Mapping)
    }
    source_spans = {
        str(item)
        for provenance in plan_action.get("provenance") or []
        for item in provenance.get("source_spans") or []
        if isinstance(provenance, Mapping)
    }
    return (
        bool(rule_ids)
        and rule_ids.issubset(set(candidate.get("guideline_rule_ids") or []))
        and memory_ids.issubset(set(candidate.get("supporting_memory_ids") or []))
        and source_spans.issubset(set(candidate.get("source_spans") or []))
    )


def _training_action_signature(action: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(action.get("action_type") or ""),
        tuple(sorted(action.get("required_skills") or [])),
        tuple(
            sorted(
                json.dumps(item, ensure_ascii=False, sort_keys=True)
                for item in action.get("preconditions") or []
            )
        ),
        tuple(sorted(action.get("expected_state_delta") or [])),
        tuple(sorted(action.get("guideline_rule_ids") or [])),
        tuple(sorted(action.get("supporting_memory_ids") or [])),
    )


def state_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: {"before": deepcopy(before.get(key)), "after": deepcopy(after.get(key))}
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    }


def stable_case_splits(
    case_families: Mapping[str, str],
    *,
    seed: int = 17,
) -> dict[str, str]:
    """Assign every base case to one stable group split, stratified by cancer family."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for case_id, family in case_families.items():
        if family not in SUPPORTED_PLANNER_CANCER_FAMILIES:
            raise V2SchemaError(f"Unsupported Planner V2 cancer family: {family!r}")
        grouped[family].append(str(case_id))
    result: dict[str, str] = {}
    for family, case_ids in grouped.items():
        ordered = sorted(
            set(case_ids),
            key=lambda item: hashlib.sha256(
                f"{seed}:{family}:{item}".encode()
            ).hexdigest(),
        )
        count = len(ordered)
        if count < 10:
            raise V2SchemaError(
                f"Cancer family {family!r} needs at least 10 cases to form grouped splits."
            )
        validation_count = max(5, round(count * 0.15))
        test_count = max(5, round(count * 0.15))
        if validation_count + test_count >= count:
            raise V2SchemaError(
                f"Cancer family {family!r} does not have enough cases after holdout allocation."
            )
        for case_id in ordered[:test_count]:
            result[case_id] = "test"
        for case_id in ordered[test_count : test_count + validation_count]:
            result[case_id] = "validation"
        for case_id in ordered[test_count + validation_count :]:
            result[case_id] = "train"
    return result


@dataclass(frozen=True)
class DatasetAdmissionReport:
    ready: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    statistics: dict[str, Any]
    release_scope: dict[str, Any] | None = None
    release_scope_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": "planner_dataset_admission.v2",
            "ready": self.ready,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "statistics": deepcopy(self.statistics),
        }
        if self.release_scope is not None:
            result["release_scope"] = deepcopy(self.release_scope)
            result["release_scope_hash"] = self.release_scope_hash
        return result


def audit_trajectory_dataset_v2(
    records: Iterable[Mapping[str, Any]],
    *,
    release_gates: bool = True,
    release_scope: str | Path | Mapping[str, Any] | None = None,
    require_test_approval: bool = True,
) -> DatasetAdmissionReport:
    try:
        resolved_scope = load_release_scope(release_scope)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise V2SchemaError(f"Invalid Planner release scope: {exc}") from exc
    scoped_families = set(action_target_families(resolved_scope))
    rows = [dict(record) for record in records]
    errors: list[str] = []
    warnings: list[str] = []
    admission_exceptions = resolved_scope.get("admission_exceptions") or []
    conditional_exceptions = {
        (str(item["cancer_family"]), bool(item["condition_satisfied"])): str(
            item["reason"]
        )
        for item in admission_exceptions
        if item.get("criterion") == "conditional_status"
    }
    phase_exceptions = {
        (str(item["cancer_family"]), str(item["phase"])): str(item["reason"])
        for item in admission_exceptions
        if item.get("criterion") == "phase_minimum"
    }
    valid: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            valid.append(validate_trajectory_record_v2(row))
        except V2SchemaError as exc:
            errors.append(f"record[{index}]: {exc}")

    case_split: dict[str, str] = {}
    case_family: dict[str, str] = {}
    real_cases: dict[str, set[str]] = defaultdict(set)
    split_cases: dict[tuple[str, str], set[str]] = defaultdict(set)
    phase_counts: Counter[tuple[str, str, str]] = Counter()
    bucket_counts: Counter[tuple[str, str]] = Counter()
    conditional_status_counts: Counter[tuple[str, bool]] = Counter()
    review_strata: Counter[tuple[str, str, str, str]] = Counter()
    approved_strata: Counter[tuple[str, str, str, str]] = Counter()
    counterfactual_coverage: dict[str, set[str]] = defaultdict(set)
    for row in valid:
        case_id = str(row["base_case_id"])
        family = str(row["state_before"]["cancer_family"])
        subtype = row["state_before"].get("disease_subtype")
        guideline_ids = {
            str(item["guideline_id"]) for item in row["guideline_context"]["guidelines"]
        }
        if not action_target_allows(
            resolved_scope,
            cancer_family=family,
            disease_subtype=str(subtype) if subtype is not None else None,
            guideline_ids=guideline_ids,
        ):
            errors.append(
                "Record "
                f"{row['trajectory_id']!r}/{row['turn_index']} target "
                f"{family}/{subtype or '<none>'} is outside release scope "
                f"{resolved_scope['scope_id']!r}."
            )
        split = str(row["split"])
        if case_id in case_split and case_split[case_id] != split:
            errors.append(f"base_case_id {case_id!r} leaks across splits.")
        if case_id in case_family and case_family[case_id] != family:
            errors.append(f"base_case_id {case_id!r} changes cancer family.")
        case_split[case_id] = split
        case_family[case_id] = family
        if row["case_source"] == "real":
            real_cases[family].add(case_id)
        else:
            counterfactual_coverage[case_id].add(
                str(row["counterfactual"]["perturbation_type"])
            )
        split_cases[(family, split)].add(case_id)
        phase = str(row["state_before"]["current_phase"])
        phase_counts[(family, phase, split)] += 1
        present_buckets = [
            bucket for bucket, items in row["action_set"].items() if items
        ]
        for bucket in present_buckets:
            bucket_counts[(family, bucket)] += len(row["action_set"][bucket])
        for action in row["action_set"]["conditional"]:
            status = action.get("condition_satisfied")
            if isinstance(status, bool):
                conditional_status_counts[(family, status)] += 1
        versions = "+".join(
            sorted(
                f"{item['guideline_id']}@{item['version']}"
                for item in row["guideline_context"]["guidelines"]
            )
        )
        risk_bucket = str(row.get("risk_bucket") or "default")
        stratum = (family, phase, versions, risk_bucket)
        if split in {"train", "validation"}:
            review_strata[stratum] += 1
            if row["provenance"]["review_status"] == "approved":
                approved_strata[stratum] += 1
        if (
            require_test_approval
            and split == "test"
            and row["provenance"]["review_status"] != "approved"
        ):
            errors.append(
                f"Test record {row['trajectory_id']!r} is not human-approved."
            )

    record_index = {
        (str(row["trajectory_id"]), int(row["turn_index"])): row for row in valid
    }
    for row in valid:
        metadata = row.get("counterfactual")
        if not isinstance(metadata, Mapping):
            continue
        anchor = record_index.get(
            (
                str(metadata["anchor_trajectory_id"]),
                int(metadata["anchor_turn_index"]),
            )
        )
        label = f"{row['trajectory_id']}/{row['turn_index']}"
        if anchor is None or anchor["case_source"] != "real":
            errors.append(f"Counterfactual {label} has no real anchor record.")
            continue
        if (
            anchor["base_case_id"] != row["base_case_id"]
            or anchor["split"] != row["split"]
        ):
            errors.append(
                f"Counterfactual {label} does not share anchor case and split."
            )
        changed = {
            key
            for key in set(anchor["state_before"]) | set(row["state_before"])
            if key != "case_id"
            and anchor["state_before"].get(key) != row["state_before"].get(key)
        }
        declared = set(metadata["changed_fields"])
        if not declared.issubset(changed):
            errors.append(
                f"Counterfactual {label} declares unchanged fields: {sorted(declared - changed)}."
            )
        action_changed = _accepted_plan_signature(anchor) != _accepted_plan_signature(
            row
        )
        memory_changed = _positive_memory_signature(
            anchor
        ) != _positive_memory_signature(row)
        if metadata["perturbation_type"] == "decision_relevant":
            if not action_changed and not memory_changed:
                errors.append(
                    f"Decision-relevant counterfactual {label} changes neither action nor memory labels."
                )
        elif action_changed or memory_changed:
            errors.append(
                f"Irrelevant counterfactual {label} changes action or memory labels."
            )

    _audit_trajectory_continuity(valid, errors)

    if release_gates:
        for family in sorted(scoped_families):
            if len(real_cases[family]) < 30:
                errors.append(
                    f"Cancer family {family!r} has {len(real_cases[family])} real cases; 30 required."
                )
            for split in ("validation", "test"):
                if len(split_cases[(family, split)]) < 5:
                    errors.append(
                        f"Cancer family {family!r} split {split!r} needs at least 5 base cases."
                    )
            for bucket in ("conditional", "premature", "unsafe"):
                if bucket_counts[(family, bucket)] == 0:
                    errors.append(
                        f"Cancer family {family!r} has no {bucket!r} supervision."
                    )
            for status in (True, False):
                if conditional_status_counts[(family, status)] == 0:
                    message = (
                        f"Cancer family {family!r} has no conditional actions with "
                        f"condition_satisfied={status}."
                    )
                    exception_reason = conditional_exceptions.get((family, status))
                    if exception_reason:
                        warnings.append(
                            f"Declared admission exception: {message} {exception_reason}"
                        )
                    else:
                        errors.append(message)
            missing_counterfactuals = sorted(
                case_id
                for case_id in real_cases[family]
                if counterfactual_coverage[case_id]
                != {"decision_relevant", "irrelevant"}
            )
            if missing_counterfactuals:
                errors.append(
                    f"Cancer family {family!r} has {len(missing_counterfactuals)} real cases "
                    "without both decision-relevant and irrelevant counterfactuals."
                )
        phases = {
            (family, phase)
            for family, phase, _ in phase_counts
            if family in scoped_families
        }
        for family, phase in phases:
            phase_messages = []
            if phase_counts[(family, phase, "train")] < 20:
                phase_messages.append(
                    f"Phase {family}/{phase} has fewer than 20 train transitions."
                )
            if phase_counts[(family, phase, "test")] < 5:
                phase_messages.append(
                    f"Phase {family}/{phase} has fewer than 5 test transitions."
                )
            exception_reason = phase_exceptions.get((family, phase))
            for message in phase_messages:
                if exception_reason:
                    warnings.append(
                        f"Declared admission exception: {message} {exception_reason}"
                    )
                else:
                    errors.append(message)
        for stratum, count in review_strata.items():
            required = min(count, max(5, math.ceil(count * 0.10)))
            if approved_strata[stratum] < required:
                errors.append(
                    "Review stratum "
                    + "/".join(stratum)
                    + f" has {approved_strata[stratum]}/{required} approved examples."
                )
    elif not rows:
        warnings.append("Trajectory dataset is empty.")

    statistics = {
        "record_count": len(rows),
        "valid_record_count": len(valid),
        "real_case_counts": {
            key: len(value) for key, value in sorted(real_cases.items())
        },
        "split_case_counts": {
            f"{family}/{split}": len(value)
            for (family, split), value in sorted(split_cases.items())
        },
        "phase_counts": {
            f"{family}/{phase}/{split}": count
            for (family, phase, split), count in sorted(phase_counts.items())
        },
        "bucket_counts": {
            f"{family}/{bucket}": count
            for (family, bucket), count in sorted(bucket_counts.items())
        },
        "counterfactual_case_counts": {
            kind: sum(kind in values for values in counterfactual_coverage.values())
            for kind in ("decision_relevant", "irrelevant")
        },
        "conditional_status_counts": {
            f"{family}/{status}": count
            for (family, status), count in sorted(conditional_status_counts.items())
        },
    }
    return DatasetAdmissionReport(
        ready=not errors,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
        statistics=statistics,
        release_scope=copy_release_scope(resolved_scope),
        release_scope_hash=release_scope_hash(resolved_scope),
    )


def _accepted_plan_signature(record: Mapping[str, Any]) -> tuple[Any, ...]:
    variants = []
    for plan in record["accepted_plan_variants"]:
        variants.append(
            tuple(
                (
                    str(action.get("action_type") or ""),
                    tuple(sorted(action.get("required_skills") or [])),
                    tuple(
                        sorted(
                            str(rule)
                            for provenance in action.get("provenance") or []
                            for rule in provenance.get("rule_ids") or []
                        )
                    ),
                )
                for action in plan.get("actions") or []
            )
        )
    return tuple(variants)


def _positive_memory_signature(record: Mapping[str, Any]) -> tuple[str, ...]:
    labels = record["routing_labels"]
    return tuple(
        sorted(
            {
                *labels["strong_positive_memory_ids"],
                *labels["weak_positive_memory_ids"],
            }
        )
    )


def _audit_trajectory_continuity(
    records: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> None:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["trajectory_id"])].append(record)
    for trajectory_id, turns in grouped.items():
        ordered = sorted(turns, key=lambda item: int(item["turn_index"]))
        indices = [int(item["turn_index"]) for item in ordered]
        if len(indices) != len(set(indices)):
            errors.append(f"Trajectory {trajectory_id!r} has duplicate turn indices.")
            continue
        if indices and indices[0] != 0:
            errors.append(f"Trajectory {trajectory_id!r} must start at turn_index=0.")
        for previous, current in pairwise(ordered):
            if int(current["turn_index"]) != int(previous["turn_index"]) + 1:
                errors.append(f"Trajectory {trajectory_id!r} has non-adjacent turns.")
            if previous["state_after"] != current["state_before"]:
                errors.append(
                    f"Trajectory {trajectory_id!r} has a discontinuous patient state."
                )
            if (
                previous["case_id"] != current["case_id"]
                or previous["base_case_id"] != current["base_case_id"]
                or previous["split"] != current["split"]
            ):
                errors.append(
                    f"Trajectory {trajectory_id!r} changes case identity or split."
                )


def teacher_cache_key(
    *,
    model: str,
    prompt: str,
    rule_compiler_version: str,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "rule_compiler_version": rule_compiler_version,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_v2_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise V2SchemaError(f"{path}:{line_number} must contain a JSON object.")
        rows.append(dict(value))
    return rows


def _raise_schema_errors(
    validator: Draft202012Validator,
    value: Mapping[str, Any],
    label: str,
) -> None:
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
    if not errors:
        return
    rendered = []
    for error in errors[:8]:
        location = ".".join(str(item) for item in error.absolute_path) or "<root>"
        rendered.append(f"{location}: {error.message}")
    raise V2SchemaError(f"Invalid {label}: " + "; ".join(rendered))


def _validate_decision_date(state: Mapping[str, Any]) -> None:
    try:
        date.fromisoformat(str(state["decision_date"]))
    except (TypeError, ValueError) as exc:
        raise V2SchemaError(
            "patient_state.v2 decision_date must use YYYY-MM-DD."
        ) from exc


def _validate_guideline_effectivity(state: Mapping[str, Any]) -> None:
    decision = date.fromisoformat(str(state["decision_date"]))
    seen: set[tuple[str, str]] = set()
    for reference in state["guideline_context"]["guidelines"]:
        pair = (str(reference["guideline_id"]), str(reference["version"]))
        if pair in seen:
            raise V2SchemaError(
                f"patient_state.v2 duplicates guideline target {pair!r}."
            )
        seen.add(pair)
        for field in ("effective_from", "effective_to"):
            value = reference.get(field)
            if value in (None, ""):
                continue
            try:
                date.fromisoformat(str(value))
            except (TypeError, ValueError) as exc:
                raise V2SchemaError(
                    f"patient_state.v2 {field} must use YYYY-MM-DD."
                ) from exc
        effective_from = reference.get("effective_from")
        effective_to = reference.get("effective_to")
        if effective_from and decision < date.fromisoformat(str(effective_from)):
            raise V2SchemaError(
                f"Decision date precedes target guideline {pair!r} effective range."
            )
        if effective_to and decision > date.fromisoformat(str(effective_to)):
            raise V2SchemaError(
                f"Decision date exceeds target guideline {pair!r} effective range."
            )


def _target_guideline_pairs(state: Mapping[str, Any] | None) -> set[tuple[str, str]]:
    if not state:
        return set()
    context = state.get("guideline_context")
    if not isinstance(context, Mapping):
        return set()
    return {
        (str(item.get("guideline_id") or ""), str(item.get("version") or ""))
        for item in context.get("guidelines", [])
        if isinstance(item, Mapping)
    }


def _active_memory_index(
    memories: Sequence[Mapping[str, Any] | str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in memories:
        if isinstance(item, str):
            result[item] = {}
            continue
        metadata = (
            item.get("metadata") if isinstance(item.get("metadata"), Mapping) else item
        )
        memory_id = str(
            item.get("memory_id")
            or item.get("guideline_memory_id")
            or metadata.get("guideline_memory_id")
            or ""
        )
        if memory_id:
            result[memory_id] = dict(metadata)
    return result


def _validate_provenance_against_memory(
    provenance: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    if not metadata:
        return
    available_rules = set(
        metadata.get("source_rule_ids") or metadata.get("rule_ids") or []
    )
    requested_rules = set(provenance.get("rule_ids") or [])
    if available_rules and not requested_rules.issubset(available_rules):
        raise V2SchemaError(
            "planner_action.v2 cites rule IDs absent from its active memory."
        )
    available_spans = set(
        metadata.get("source_span_ids") or metadata.get("source_spans") or []
    )
    requested_spans = set(provenance.get("source_spans") or [])
    if available_spans and not requested_spans.issubset(available_spans):
        raise V2SchemaError(
            "planner_action.v2 cites source spans absent from its active memory."
        )
