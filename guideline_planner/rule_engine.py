"""Deterministic condition-action compiler for versioned Planner V2 rubrics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

from guideline_planner.schemas_v2 import V2SchemaError, validate_patient_state_v2


ACTION_BUCKETS = ("required", "acceptable", "conditional", "premature", "unsafe")
_OPERATORS = {
    "eq",
    "ne",
    "in",
    "not_in",
    "exists",
    "missing",
    "contains",
    "gte",
    "lte",
}


def compile_action_set(
    patient_state: Mapping[str, Any],
    rules: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Compile grounded action buckets without model-generated facts."""

    state = validate_patient_state_v2(patient_state)
    family = str(state["cancer_family"])
    phase = str(state["current_phase"])
    targets = {
        (str(item["guideline_id"]), str(item["version"]))
        for item in state["guideline_context"]["guidelines"]
    }
    result: dict[str, list[dict[str, Any]]] = {
        bucket: [] for bucket in ACTION_BUCKETS
    }
    for rule in rules:
        validate_compilable_rule(rule)
        if str(rule["cancer_family"]) != family or str(rule["phase"]) != phase:
            continue
        if (str(rule["guideline_id"]), str(rule["version"])) not in targets:
            continue
        rule_satisfied = evaluate_condition(rule["condition"], state)
        for template in rule["action_templates"]:
            template_condition = template.get("condition", {"all": []})
            satisfied = rule_satisfied and evaluate_condition(template_condition, state)
            bucket = template.get("bucket_if_true" if satisfied else "bucket_if_false")
            if bucket in (None, ""):
                continue
            if bucket not in ACTION_BUCKETS:
                raise V2SchemaError(
                    f"Rule {rule['rule_id']!r} selects unknown action bucket {bucket!r}."
                )
            action = {
                "action_id": str(template["action_id"]),
                "objective": str(template["objective"]),
                "action_type": str(template["action_type"]),
                "required_skills": _unique_strings(template["required_skills"]),
                "preconditions": _condition_list(template["preconditions"]),
                "expected_state_delta": _unique_strings(
                    template["expected_state_delta"]
                ),
                "guideline_rule_ids": [str(rule["rule_id"])],
                "supporting_memory_ids": _unique_strings(
                    template.get("supporting_memory_ids") or rule["memory_ids"]
                ),
                "source_spans": _unique_strings(
                    template.get("source_spans") or rule["source_spans"]
                ),
                "condition_satisfied": satisfied if bucket == "conditional" else None,
                "risk_level": template.get("risk_level"),
            }
            result[bucket].append(action)
    for bucket in ACTION_BUCKETS:
        result[bucket].sort(key=lambda item: (str(item["action_id"]), str(item["objective"])))
    return result


def evaluate_condition(condition: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """Evaluate a small, auditable JSON condition DSL."""

    if not isinstance(condition, Mapping):
        raise V2SchemaError("Rule conditions must be JSON objects.")
    keys = {key for key in ("all", "any", "not") if key in condition}
    if keys:
        if len(keys) != 1 or len(condition) != 1:
            raise V2SchemaError("Composite rule conditions must contain exactly one operator.")
        operator = next(iter(keys))
        value = condition[operator]
        if operator == "not":
            if not isinstance(value, Mapping):
                raise V2SchemaError("Rule condition 'not' requires one object.")
            return not evaluate_condition(value, state)
        if not isinstance(value, list):
            raise V2SchemaError(f"Rule condition {operator!r} requires a list.")
        evaluated = [evaluate_condition(item, state) for item in value]
        return all(evaluated) if operator == "all" else any(evaluated)

    field = str(condition.get("field") or "")
    operator = str(condition.get("operator") or "")
    if not field or operator not in _OPERATORS:
        raise V2SchemaError(
            "Leaf rule conditions require field and a supported operator."
        )
    actual = _field_value(state, field)
    expected = condition.get("value")
    if operator == "exists":
        return actual not in (None, "", [], {})
    if operator == "missing":
        return actual in (None, "", [], {})
    if operator == "eq":
        return actual == expected
    if operator == "ne":
        return actual != expected
    if operator in {"in", "not_in"}:
        if not isinstance(expected, list):
            raise V2SchemaError(f"Rule operator {operator!r} requires a list value.")
        matched = actual in expected
        return matched if operator == "in" else not matched
    if operator == "contains":
        if isinstance(actual, Mapping):
            return expected in actual
        if isinstance(actual, (list, tuple, set, str)):
            return expected in actual
        return False
    if operator in {"gte", "lte"}:
        try:
            return actual >= expected if operator == "gte" else actual <= expected
        except TypeError as exc:
            raise V2SchemaError(
                f"Rule comparison {operator!r} received incompatible values."
            ) from exc
    raise AssertionError(operator)


def action_matches_template(
    action: Mapping[str, Any],
    template: Mapping[str, Any],
) -> bool:
    """Check teacher/compiler output against the non-generative rubric fields."""

    return all(
        (
            str(action.get("action_id") or "") == str(template.get("action_id") or ""),
            str(action.get("action_type") or "")
            == str(template.get("action_type") or ""),
            set(action.get("required_skills") or [])
            == set(template.get("required_skills") or []),
            {
                _canonical_condition(item)
                for item in action.get("preconditions") or []
            }
            == {
                _canonical_condition(item)
                for item in template.get("preconditions") or []
            },
            set(action.get("expected_state_delta") or [])
            == set(template.get("expected_state_delta") or []),
        )
    )


def validate_compilable_rule(rule: Mapping[str, Any]) -> None:
    required = {
        "rule_id",
        "cancer_family",
        "guideline_id",
        "version",
        "phase",
        "condition",
        "allowed_action_types",
        "memory_ids",
        "source_spans",
        "action_templates",
    }
    missing = sorted(required - set(rule))
    if rule.get("schema_version") != "guideline_rule.v2" or missing:
        raise V2SchemaError(
            f"Invalid guideline_rule.v2 {rule.get('rule_id')!r}; missing={missing}."
        )
    if not isinstance(rule["action_templates"], list) or not rule["action_templates"]:
        raise V2SchemaError(
            f"Rule {rule['rule_id']!r} requires non-empty action_templates."
        )
    allowed_action_types = set(_unique_strings(rule["allowed_action_types"]))
    if not allowed_action_types:
        raise V2SchemaError(
            f"Rule {rule['rule_id']!r} requires allowed_action_types."
        )
    if not _unique_strings(rule["memory_ids"]) or not _unique_strings(rule["source_spans"]):
        raise V2SchemaError(
            f"Rule {rule['rule_id']!r} requires memory_ids and source_spans."
        )
    required_template = {
        "action_id",
        "objective",
        "action_type",
        "required_skills",
        "preconditions",
        "expected_state_delta",
        "bucket_if_true",
        "bucket_if_false",
    }
    for index, template in enumerate(rule["action_templates"]):
        if not isinstance(template, Mapping):
            raise V2SchemaError(
                f"Rule {rule['rule_id']!r} action_templates[{index}] must be an object."
            )
        missing_template = sorted(required_template - set(template))
        if missing_template:
            raise V2SchemaError(
                f"Rule {rule['rule_id']!r} action_templates[{index}] "
                f"is missing {missing_template}."
            )
        for field in ("bucket_if_true", "bucket_if_false"):
            bucket = template[field]
            if bucket not in (*ACTION_BUCKETS, None):
                raise V2SchemaError(
                    f"Rule {rule['rule_id']!r} has invalid {field}={bucket!r}."
                )
        if str(template["action_type"]) not in allowed_action_types:
            raise V2SchemaError(
                f"Rule {rule['rule_id']!r} template action type is not allowed."
            )


def _field_value(state: Mapping[str, Any], field: str) -> Any:
    current: Any = state
    for part in field.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _unique_strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        raise V2SchemaError("Rule action list fields must be arrays.")
    return list(dict.fromkeys(str(item) for item in deepcopy(list(value)) if str(item)))


def _condition_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise V2SchemaError("Rule action preconditions must be arrays of condition objects.")
    return [deepcopy(dict(item)) for item in value]


def _canonical_condition(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
