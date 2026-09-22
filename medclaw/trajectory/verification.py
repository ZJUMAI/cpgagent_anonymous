"""Verification and scoring helpers for generated clinical actions."""

from __future__ import annotations

from typing import Iterable, Mapping

from medclaw.trajectory.schema import ActionVerification, PatientEvent, RubricRule


def verify_action(
    action: Mapping[str, object],
    *,
    future_events: Iterable[PatientEvent],
    rubric_rules: Iterable[RubricRule],
) -> ActionVerification:
    """Verify one generated action against future report events and rubric rules."""

    action_id = str(action.get("action_id") or "generated_action")
    action_text = str(action.get("action") or "")
    report_hits = _matching_events(action_text, future_events)
    guideline_hits = _matching_rules(action_text, rubric_rules)
    premature = _is_premature_or_forbidden(action_text, rubric_rules)

    if premature:
        report_support = "contradicted"
        guideline_support = "contradicted"
        classification = "contradicted_or_premature"
    elif report_hits and guideline_hits:
        report_support = "supported"
        guideline_support = "supported"
        classification = "supported_by_report_and_guideline"
    elif guideline_hits:
        report_support = "not_observed"
        guideline_support = "supported"
        classification = "guideline_supported_but_unobserved"
    elif report_hits:
        report_support = "supported"
        guideline_support = "uncertain"
        classification = "observed_but_guideline_uncertain"
    else:
        report_support = "not_observed"
        guideline_support = "uncertain"
        classification = "observed_but_guideline_uncertain"

    return ActionVerification(
        action_id=action_id,
        action=action_text,
        report_support=report_support,
        guideline_support=guideline_support,
        classification=classification,
        evidence_event_ids=tuple(event.event_id for event in report_hits),
        evidence_text=tuple(event.evidence_span for event in report_hits),
        state_update=_state_update_from_events(report_hits),
    )


def score_action_set(
    generated_actions: Iterable[Mapping[str, object]],
    expected_actions: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Compute expected-action coverage for generated action sets."""

    generated_texts = [str(item.get("action") or item) for item in generated_actions]
    expected_texts = [str(item.get("action") or item) for item in expected_actions]
    matched = [
        expected
        for expected in expected_texts
        if any(_text_overlap(expected, generated) for generated in generated_texts)
    ]
    coverage = len(matched) / len(expected_texts) if expected_texts else 1.0
    return {
        "expected_count": len(expected_texts),
        "matched_count": len(matched),
        "coverage": coverage,
        "matched_expected_actions": matched,
    }


def _matching_events(
    action_text: str, events: Iterable[PatientEvent]
) -> list[PatientEvent]:
    return [
        event
        for event in events
        if _event_matches_action(event, action_text)
    ]


def _matching_rules(action_text: str, rules: Iterable[RubricRule]) -> list[RubricRule]:
    return [
        rule
        for rule in rules
        if any(_text_overlap(action_text, item) for item in rule.recommended_actions)
    ]


def _is_premature_or_forbidden(action_text: str, rules: Iterable[RubricRule]) -> bool:
    if action_text.lower().strip().startswith(("do not ", "do not recommend ")):
        return False
    return any(
        _strict_forbidden_match(action_text, forbidden)
        for rule in rules
        for forbidden in rule.forbidden_actions
    )


def _strict_forbidden_match(action_text: str, forbidden_text: str) -> bool:
    action = _normalized_text(action_text)
    forbidden = _normalized_text(forbidden_text)
    if not action or not forbidden:
        return False
    return forbidden in action or action in forbidden


def _text_overlap(left: str, right: str) -> bool:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return False
    overlap = left_tokens & right_tokens
    return len(overlap) >= min(2, len(left_tokens), len(right_tokens))


def _tokens(text: str) -> set[str]:
    stop = {
        "the",
        "and",
        "or",
        "to",
        "of",
        "before",
        "according",
        "with",
        "if",
        "no",
        "known",
    }
    return {
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        if len(token) > 1 and token not in stop
    }


def _normalized_text(text: str) -> str:
    return " ".join(
        "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
    )


def _event_matches_action(event: PatientEvent, action_text: str) -> bool:
    lowered = action_text.lower()
    if lowered.strip().startswith(("do not ", "do not recommend ")):
        return False
    if any(term in lowered for term in ["testing", "test ", "pd-l1", "molecular", "biomarker"]):
        return event.event_type == "molecular_test"
    if any(term in lowered for term in ["surgery", "surgical", "resectable"]):
        return event.event_type == "therapy" and "surgery" in event.content.lower()
    return (
        (event.event_type == "tnm" and ("stage" in lowered or "tnm" in lowered))
        or (event.event_type == "pathology" and ("patholog" in lowered or "histolog" in lowered))
        or (event.event_type == "therapy" and ("treatment" in lowered or "therapy" in lowered))
    )


def _state_update_from_events(events: Iterable[PatientEvent]) -> dict[str, object]:
    update: dict[str, object] = {}
    for event in events:
        if event.event_type == "tnm":
            for key in ["T", "N", "M", "stage_group"]:
                if event.attributes.get(key):
                    update[key] = event.attributes[key]
        elif event.event_type == "pathology":
            update["diagnosis"] = event.attributes.get("diagnosis", event.content)
        elif event.event_type == "molecular_test":
            molecular = dict(update.get("molecular_status", {}))
            gene = event.attributes.get("gene")
            if gene:
                molecular[str(gene)] = event.attributes.get("result") or "reported"
            update["molecular_status"] = molecular
        elif event.event_type == "therapy":
            observed = list(update.get("observed_treatments", []))
            record = {
                "treatment_type": event.attributes.get("treatment_type", event.content),
                "status": event.attributes.get("status", "recorded_unknown"),
            }
            if event.attributes.get("intent"):
                record["intent"] = event.attributes["intent"]
            if event.attributes.get("note"):
                record["note"] = event.attributes["note"]
            observed.append(record)
            update["observed_treatments"] = observed
    return update
