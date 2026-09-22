"""Versioned rule and memory provenance admission checks for Planner V2."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from guideline_planner.io_utils import read_jsonl
from guideline_planner.retrieval import load_memory_metadata_records
from guideline_planner.rule_engine import (
    action_matches_template,
    compile_action_set,
    validate_compilable_rule,
)
from guideline_planner.schemas_v2 import V2SchemaError

GPT_GUIDELINE_CLAIM_SCHEMA_VERSION = "gpt_guideline_claim.v2"


def validate_grounding_assets(
    records: Iterable[Mapping[str, Any]],
    *,
    rule_registry_path: str | Path | None,
    memory_dir: str | Path | None,
) -> list[str]:
    """Resolve every action label to a version-matched rule, memory, and span."""

    if rule_registry_path is None:
        return ["A versioned condition-action rule registry is required."]
    if memory_dir is None:
        return ["A V2 Memory Store is required for provenance admission."]
    rules = read_jsonl(Path(rule_registry_path))
    memories = load_memory_metadata_records(memory_dir)
    rule_index: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    legacy_rules: list[dict[str, Any]] = []
    for index, rule in enumerate(rules):
        try:
            if rule.get("schema_version") == GPT_GUIDELINE_CLAIM_SCHEMA_VERSION:
                _validate_gpt_guideline_claim(rule)
            else:
                validate_compilable_rule(rule)
                legacy_rules.append(dict(rule))
        except V2SchemaError as exc:
            errors.append(f"rule[{index}] is invalid: {exc}")
            continue
        rule_id = str(rule["rule_id"])
        if rule_id in rule_index:
            errors.append(f"Duplicate rule_id {rule_id!r}.")
            continue
        rule_index[rule_id] = dict(rule)
    memory_index = {str(memory["guideline_memory_id"]): memory for memory in memories}
    for row in records:
        family = str(row["state_before"]["cancer_family"])
        try:
            compiled_action_set = compile_action_set(row["state_before"], legacy_rules)
        except V2SchemaError as exc:
            errors.append(
                f"{row['trajectory_id']}/{row['turn_index']}: rule compilation failed: {exc}"
            )
            continue
        context = {
            (str(item["guideline_id"]), str(item["version"]))
            for item in row["guideline_context"]["guidelines"]
        }
        for bucket, actions in row["action_set"].items():
            for action in actions:
                label = f"{row['trajectory_id']}/{row['turn_index']}/{bucket}/{action['action_id']}"
                compiler_matched = any(
                    action_matches_template(action, candidate)
                    and set(action["guideline_rule_ids"]).issubset(
                        set(candidate["guideline_rule_ids"])
                    )
                    and set(action["supporting_memory_ids"]).issubset(
                        set(candidate["supporting_memory_ids"])
                    )
                    and set(action["source_spans"]).issubset(
                        set(candidate["source_spans"])
                    )
                    for candidate in compiled_action_set[bucket]
                )
                direct_claim_matched = any(
                    _gpt_claim_supports_action(
                        action,
                        rule_index.get(str(rule_id)),
                        family=family,
                        guideline_context=context,
                    )
                    for rule_id in action["guideline_rule_ids"]
                )
                if not compiler_matched and not direct_claim_matched:
                    errors.append(
                        f"{label}: action is supported by neither compiled rule "
                        "conditions nor a provenance-only GPT guideline claim."
                    )
                matched_template = False
                for rule_id in action["guideline_rule_ids"]:
                    rule = rule_index.get(str(rule_id))
                    if rule is None:
                        errors.append(f"{label}: unknown rule {rule_id!r}.")
                        continue
                    if rule["cancer_family"] != family:
                        errors.append(
                            f"{label}: rule {rule_id!r} has the wrong cancer family."
                        )
                    if (str(rule["guideline_id"]), str(rule["version"])) not in context:
                        errors.append(
                            f"{label}: rule {rule_id!r} has the wrong guideline version."
                        )
                    if action["action_type"] not in rule["allowed_action_types"]:
                        errors.append(
                            f"{label}: action type is not allowed by rule {rule_id!r}."
                        )
                    if not set(action["source_spans"]).issubset(
                        set(rule["source_spans"])
                    ):
                        errors.append(
                            f"{label}: source spans are not grounded by rule {rule_id!r}."
                        )
                    if not set(action["supporting_memory_ids"]).issubset(
                        set(rule["memory_ids"])
                    ):
                        errors.append(
                            f"{label}: memories are not grounded by rule {rule_id!r}."
                        )
                    if rule.get("schema_version") == GPT_GUIDELINE_CLAIM_SCHEMA_VERSION:
                        if _gpt_claim_supports_action(
                            action,
                            rule,
                            family=family,
                            guideline_context=context,
                        ):
                            matched_template = True
                    elif any(
                        action_matches_template(action, template)
                        for template in rule["action_templates"]
                        if isinstance(template, Mapping)
                    ):
                        matched_template = True
                if not matched_template:
                    errors.append(
                        f"{label}: action is not declared by any cited rule action_template."
                    )
                for memory_id in action["supporting_memory_ids"]:
                    memory = memory_index.get(str(memory_id))
                    if memory is None:
                        errors.append(f"{label}: unknown memory {memory_id!r}.")
                        continue
                    memory_family = _canonical_family(memory.get("cancer_type"))
                    if memory_family != family:
                        errors.append(
                            f"{label}: memory {memory_id!r} is not independently tagged as {family}."
                        )
                    pair = (
                        str(memory.get("guideline_id") or ""),
                        str(memory.get("version") or ""),
                    )
                    if pair not in context:
                        errors.append(
                            f"{label}: memory {memory_id!r} has the wrong guideline version."
                        )
                    memory_spans = set(
                        memory.get("source_span_ids")
                        or memory.get("source_chunk_ids")
                        or []
                    )
                    if not set(action["source_spans"]).issubset(memory_spans):
                        errors.append(
                            f"{label}: source span is absent from memory {memory_id!r}."
                        )
        if family == "endometrial" and not any(
            rule.get("cancer_family") == "endometrial" for rule in rule_index.values()
        ):
            errors.append(
                "Endometrial training is blocked: no versioned authoritative rules."
            )
        if family == "nasopharyngeal" and any(
            _canonical_family(memory_index.get(item, {}).get("cancer_type"))
            != "nasopharyngeal"
            for item in row["routing_labels"]["strong_positive_memory_ids"]
        ):
            errors.append(
                "Nasopharyngeal training is blocked until its guideline subsection is independently split."
            )
    return list(dict.fromkeys(errors))


def _validate_gpt_guideline_claim(claim: Mapping[str, Any]) -> None:
    """Validate a provenance-only claim emitted by the GPT data workflow.

    These records deliberately contain no condition/action templates.  They
    prove that a generated action cites a real guideline span and compatible
    memory, but cannot be used by :func:`compile_action_set` to generate a
    training target.
    """

    required = {
        "rule_id",
        "cancer_family",
        "guideline_id",
        "version",
        "claim_text",
        "allowed_action_types",
        "memory_ids",
        "source_spans",
        "provenance_only",
    }
    missing = sorted(required - set(claim))
    if missing or claim.get("schema_version") != GPT_GUIDELINE_CLAIM_SCHEMA_VERSION:
        raise V2SchemaError(
            f"Invalid {GPT_GUIDELINE_CLAIM_SCHEMA_VERSION} "
            f"{claim.get('rule_id')!r}; missing={missing}."
        )
    if claim.get("provenance_only") is not True:
        raise V2SchemaError(
            f"GPT guideline claim {claim.get('rule_id')!r} must be provenance_only."
        )
    if "action_templates" in claim:
        raise V2SchemaError(
            f"GPT guideline claim {claim.get('rule_id')!r} cannot contain action_templates."
        )
    if claim.get("claim_id") not in (None, claim.get("rule_id")):
        raise V2SchemaError(
            f"GPT guideline claim {claim.get('rule_id')!r} has a conflicting claim_id."
        )
    for field in ("rule_id", "cancer_family", "guideline_id", "version", "claim_text"):
        if not str(claim.get(field) or "").strip():
            raise V2SchemaError(
                f"GPT guideline claim {claim.get('rule_id')!r} has empty {field}."
            )
    for field in ("allowed_action_types", "memory_ids", "source_spans"):
        value = claim.get(field)
        if (
            not isinstance(value, list)
            or not value
            or any(not str(item).strip() for item in value)
        ):
            raise V2SchemaError(
                f"GPT guideline claim {claim.get('rule_id')!r} requires non-empty {field}."
            )


def _gpt_claim_supports_action(
    action: Mapping[str, Any],
    claim: Mapping[str, Any] | None,
    *,
    family: str,
    guideline_context: set[tuple[str, str]],
) -> bool:
    if not claim or claim.get("schema_version") != GPT_GUIDELINE_CLAIM_SCHEMA_VERSION:
        return False
    return all(
        (
            str(claim.get("cancer_family")) == family,
            (str(claim.get("guideline_id")), str(claim.get("version")))
            in guideline_context,
            str(action.get("action_type"))
            in {str(item) for item in claim.get("allowed_action_types") or []},
            set(action.get("supporting_memory_ids") or []).issubset(
                {str(item) for item in claim.get("memory_ids") or []}
            ),
            set(action.get("source_spans") or []).issubset(
                {str(item) for item in claim.get("source_spans") or []}
            ),
        )
    )


def _canonical_family(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"luad", "lusc", "nsclc", "sclc", "lung", "lung_cancer"}:
        return "lung"
    if text in {"ucec", "endometrial", "endometrial_cancer"}:
        return "endometrial"
    if text in {"npc", "nasopharyngeal", "nasopharyngeal_carcinoma"}:
        return "nasopharyngeal"
    return text or None
