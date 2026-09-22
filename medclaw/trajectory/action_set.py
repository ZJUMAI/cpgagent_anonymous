"""Dynamic rubric action-set assembly and compatibility mirrors."""

from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping

ActionBucket = Literal["required", "acceptable", "conditional", "unsafe"]
Priority = Literal["high", "medium", "low"]

ACTION_SET_KEYS: tuple[ActionBucket, ...] = (
    "required",
    "acceptable",
    "conditional",
    "unsafe",
)

DYNAMIC_RUBRIC_SCHEMA_VERSION = "trajectory.dynamic_rubric.v1"
DUAL_LM_SCHEMA_VERSION = "trajectory.dual_lm.v1"


def empty_action_set() -> dict[str, list[dict[str, Any]]]:
    return {key: [] for key in ACTION_SET_KEYS}


def evidence_missing(state: Mapping[str, Any], evidence_keys: Iterable[str]) -> bool:
    """Return True when any required evidence key is absent or explicitly missing."""

    for key in evidence_keys:
        normalized = key.strip().lower()
        if normalized in {"ecog", "performance status"}:
            if str(state.get("ECOG", "missing")).lower() == "missing":
                return True
            continue
        if normalized in {"molecular report", "biomarkers", "biomarker status"}:
            if _biomarker_status_unknown(state):
                return True
            continue
        if normalized in {"pd-l1", "pd l1"}:
            if _pd_l1_unknown(state):
                return True
            continue
        if normalized in {"egfr", "egfr mutation", "egfr activating mutation"}:
            if _gene_status_unknown(state, "EGFR"):
                return True
            continue
        if normalized in {"alk", "alk fusion"}:
            if _gene_status_unknown(state, "ALK"):
                return True
            continue
        if normalized in {"margin status", "margin", "residual disease"}:
            if not state.get("margin_or_residual_status"):
                return True
            continue
        if normalized in {"stage", "pathologic stage", "tnm", "pT", "pN", "pM"}:
            if not state.get("pathologic_stage") and not state.get("tnm"):
                return True
            continue
        if normalized in {"histology"}:
            if not state.get("histology") and not state.get("diagnosis"):
                return True
            continue
        if normalized in {"recurrence timing", "metastasis status"}:
            if state.get("recurrence_or_metastasis") and "incomplete" in " ".join(
                state.get("open_questions", [])
            ).lower():
                return True
            continue
        value = state.get(key)
        if value in (None, "", "missing", "unknown"):
            return True
        if isinstance(value, str) and value.lower() in {"missing", "unknown"}:
            return True
    return False


def assemble_action_set(
    candidates: Iterable[Mapping[str, Any]],
    *,
    forbidden_actions: Iterable[str] | None = None,
    state: Mapping[str, Any] | None = None,
    step: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Route candidate actions into req/acc/cond/unsafe buckets."""

    action_set = empty_action_set()
    state_mapping = dict(state or {})
    counters = {key: 0 for key in ACTION_SET_KEYS}

    for candidate in candidates:
        bucket = _resolve_bucket(candidate, state_mapping)
        payload = _normalize_action(candidate, bucket=bucket, step=step, index=counters[bucket])
        counters[bucket] += 1
        action_set[bucket].append(payload)

    for forbidden in forbidden_actions or []:
        text = str(forbidden).strip()
        if not text:
            continue
        action_set["unsafe"].append(
            {
                "action_id": f"s{step}_u{counters['unsafe']}",
                "action_type": _action_type(text),
                "action": text,
                "priority": "high",
                "guideline_rule_ids": [],
                "required_evidence": [],
                "rationale": "Forbidden or condition-mismatched action at the current state.",
            }
        )
        counters["unsafe"] += 1

    return action_set


def resolve_action_set(step: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Normalize either supported trajectory step to the four action buckets.

    New artifacts use ``step.action_set``. Historical dual-LM inputs may store
    the same value at ``step.planner.action_set``; conflicting duplicates are
    rejected rather than silently choosing one.
    """

    direct = step.get("action_set")
    planner = step.get("planner")
    nested = planner.get("action_set") if isinstance(planner, Mapping) else None
    direct_resolved = _normalize_action_set(direct)
    nested_resolved = _normalize_action_set(nested)
    if isinstance(direct, Mapping) and isinstance(nested, Mapping):
        if direct_resolved != nested_resolved:
            raise ValueError(
                "Trajectory step contains conflicting step.action_set and "
                "step.planner.action_set values."
            )
    if isinstance(direct, Mapping):
        return direct_resolved
    if isinstance(nested, Mapping):
        return nested_resolved
    return empty_action_set()


def _normalize_action_set(raw: Any) -> dict[str, list[dict[str, Any]]]:
    resolved = empty_action_set()
    if not isinstance(raw, Mapping):
        return resolved
    for key in ACTION_SET_KEYS:
        items = raw.get(key) or []
        resolved[key] = [
            dict(item) if isinstance(item, Mapping) else {"action": str(item)}
            for item in items
            if item is not None
        ]
    return resolved


def resolve_verifications(step: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return verification entries from either supported trajectory schema."""

    direct = step.get("verification")
    verifier = step.get("verifier")
    nested = verifier.get("verifications") if isinstance(verifier, Mapping) else None
    direct_resolved = _normalize_verifications(direct)
    nested_resolved = _normalize_verifications(nested)
    if isinstance(direct, list) and isinstance(nested, list):
        if direct_resolved != nested_resolved:
            raise ValueError(
                "Trajectory step contains conflicting step.verification and "
                "step.verifier.verifications values."
            )
    if isinstance(direct, list):
        return direct_resolved
    if isinstance(nested, list):
        return nested_resolved
    return []


def _normalize_verifications(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items if isinstance(item, Mapping)]


def canonicalize_trajectory_step(step: Mapping[str, Any]) -> dict[str, Any]:
    """Move compatibility fields to the canonical root-level step layout."""

    payload = dict(step)
    payload["action_set"] = resolve_action_set(step)
    payload["verification"] = resolve_verifications(step)

    planner = payload.get("planner")
    if isinstance(planner, Mapping) and "action_set" in planner:
        planner_payload = dict(planner)
        planner_payload.pop("action_set", None)
        payload["planner"] = planner_payload

    verifier = payload.get("verifier")
    if isinstance(verifier, Mapping) and "verifications" in verifier:
        verifier_payload = dict(verifier)
        verifier_payload.pop("verifications", None)
        payload["verifier"] = verifier_payload
    return payload


def canonicalize_guideline_trajectory(
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a trajectory whose steps use one canonical action-set layout."""

    payload = dict(trajectory)
    steps = trajectory.get("trajectory")
    if isinstance(steps, list):
        payload["trajectory"] = [
            canonicalize_trajectory_step(step) if isinstance(step, Mapping) else step
            for step in steps
        ]
    return payload


def flatten_visible_state(visible: Mapping[str, Any] | None) -> dict[str, Any]:
    """Flatten dual-LM ``visible_state.known_facts`` into a scorer-friendly dict."""

    if not isinstance(visible, Mapping):
        return {}
    known_facts = visible.get("known_facts")
    flat: dict[str, Any] = {}
    if isinstance(known_facts, Mapping):
        flat.update(dict(known_facts))
    for key, value in visible.items():
        if key == "known_facts":
            continue
        # Prefer known_facts values when both present.
        flat.setdefault(key, value)
    return flat


def action_set_to_step_fields(
    action_set: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Deprecated mirror helper kept for transitional tooling."""

    required = list(action_set.get("required") or [])
    forbidden = [
        str(item.get("action", ""))
        for item in action_set.get("unsafe") or []
        if isinstance(item, Mapping) and item.get("action")
    ]
    return required, forbidden


def flatten_action_set(action_set: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return all bucketed actions in evaluation order."""

    ordered: list[dict[str, Any]] = []
    for key in ACTION_SET_KEYS:
        for item in action_set.get(key) or []:
            if isinstance(item, Mapping):
                ordered.append(dict(item))
    return ordered


def dynamic_rubric_scores_from_action_set(
    action_set: Mapping[str, Any],
    *,
    matched_required: int | None = None,
    matched_acceptable: int | None = None,
    matched_conditional: int | None = None,
    unsafe_hits: int | None = None,
) -> dict[str, float | None]:
    """Produce routing_metrics-compatible score fields from an action set."""

    required = list(action_set.get("required") or [])
    acceptable = list(action_set.get("acceptable") or [])
    conditional = list(action_set.get("conditional") or [])
    unsafe = list(action_set.get("unsafe") or [])

    return {
        "required_score": _coverage_score(len(required), matched_required),
        "acceptable_score": _coverage_score(len(acceptable), matched_acceptable),
        "defer_score": _coverage_score(len(conditional), matched_conditional),
        "unsafe_score": 0.0
        if unsafe_hits == 0
        else (float(unsafe_hits) / len(unsafe) if unsafe_hits is not None and unsafe else None),
    }


def _coverage_score(total: int, matched: int | None) -> float | None:
    if total == 0:
        return None
    if matched is None:
        return 1.0
    return min(max(float(matched) / total, 0.0), 1.0)


def _resolve_bucket(candidate: Mapping[str, Any], state: Mapping[str, Any]) -> ActionBucket:
    explicit = str(candidate.get("bucket") or "").strip().lower()
    if explicit in ACTION_SET_KEYS:
        bucket = explicit  # type: ignore[assignment]
    elif str(candidate.get("priority", "high")).lower() in {"medium", "low", "should", "optional"}:
        bucket = "acceptable"
    else:
        bucket = "required"

    if bucket == "unsafe":
        return "unsafe"

    depends_on = list(candidate.get("depends_on_evidence") or [])
    only_if_missing = bool(candidate.get("conditional_if_missing", False))
    if depends_on and evidence_missing(state, depends_on):
        if only_if_missing or explicit == "conditional" or candidate.get("condition"):
            return "conditional"
    if explicit == "conditional":
        return "conditional"

    return bucket


def _normalize_action(
    candidate: Mapping[str, Any],
    *,
    bucket: ActionBucket,
    step: int,
    index: int,
) -> dict[str, Any]:
    prefix = {"required": "r", "acceptable": "a", "conditional": "c", "unsafe": "u"}[bucket]
    action_id = str(candidate.get("action_id") or f"s{step}_{prefix}{index}")
    payload: dict[str, Any] = {
        "action_id": action_id,
        "action_type": str(candidate.get("action_type") or _action_type(str(candidate.get("action", "")))),
        "action": str(candidate.get("action", "")),
        "priority": str(candidate.get("priority") or ("high" if bucket == "required" else "medium")),
        "guideline_rule_ids": list(candidate.get("guideline_rule_ids") or []),
        "required_evidence": list(candidate.get("required_evidence") or []),
        "rationale": str(
            candidate.get("rationale")
            or f"Dynamic rubric {bucket} action generated from guideline templates."
        ),
        "bucket": bucket,
    }
    condition = candidate.get("condition")
    if bucket == "conditional" and condition:
        payload["condition"] = str(condition)
    elif bucket == "conditional" and candidate.get("depends_on_evidence"):
        payload["condition"] = _default_condition(candidate.get("depends_on_evidence"))
    return payload


def _default_condition(depends_on: Any) -> str:
    keys = [str(item) for item in depends_on] if isinstance(depends_on, list) else [str(depends_on)]
    joined = ", ".join(keys)
    return f"if {joined} become available and support the action"


def _biomarker_status_unknown(state: Mapping[str, Any]) -> bool:
    molecular = state.get("molecular_status")
    if molecular == "unknown":
        return True
    open_questions = " ".join(state.get("open_questions", [])).lower()
    return "molecular" in open_questions or "biomarker" in open_questions


def _pd_l1_unknown(state: Mapping[str, Any]) -> bool:
    biomarkers = {str(item).upper() for item in state.get("biomarkers_mentioned", [])}
    if "PD-L1" in biomarkers:
        return False
    open_questions = " ".join(state.get("open_questions", [])).lower()
    return "pd-l1" in open_questions


def _gene_status_unknown(state: Mapping[str, Any], gene: str) -> bool:
    biomarkers = {str(item).upper() for item in state.get("biomarkers_mentioned", [])}
    molecular = state.get("molecular_status")
    if isinstance(molecular, Mapping) and gene in molecular:
        value = str(molecular[gene]).lower()
        return value in {"unknown", "missing"}
    if gene.upper() in biomarkers:
        return False
    open_questions = " ".join(state.get("open_questions", [])).lower()
    return gene.lower() in open_questions or "molecular" in open_questions


def _action_type(action: str) -> str:
    lowered = action.lower()
    if "stage" in lowered or "tnm" in lowered:
        return "staging"
    if "molecular" in lowered or "biomarker" in lowered or "pd-l1" in lowered:
        return "molecular_testing"
    if "treatment" in lowered or "therapy" in lowered or "surgery" in lowered or "adjuvant" in lowered:
        return "treatment_planning"
    if "histological" in lowered or "patholog" in lowered:
        return "pathology"
    if "surveillance" in lowered or "surveillance" in lowered:
        return "surveillance"
    if "recurrence" in lowered or "metastatic" in lowered:
        return "recurrence_management"
    if "ecog" in lowered or "performance status" in lowered or "fitness" in lowered:
        return "fitness"
    if "communicat" in lowered:
        return "communication"
    return "diagnostic_workup"
