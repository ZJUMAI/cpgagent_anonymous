"""Deterministic patient-state helpers for dual-agent benchmark runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from guideline_planner.constants import PATIENT_STATE_SCHEMA_VERSION
from guideline_planner.phase_machine import can_commit_phase
from guideline_planner.schemas_v2 import validate_patient_state_v2
from medclaw.utils import read_json
from medclaw_benchmark.planner_skill_adapter import completed_planner_skill_aliases

SkillStateAdapter = str | Callable[..., None]


SKILL_STATE_ADAPTERS: dict[str, SkillStateAdapter] = {
    "clinical.read_summary": "clinical_summary",
    "clinical.read_treatment": "treatment_history",
    "clinical.read_follow_up": "followup",
    "pathology.read_report": "pathology_report",
    "pathology.read_slide_metadata": "pathology_summary",
    "pathology.read_wsi_manifest": "pathology_summary",
    "pathology.conch_patch_roi": "pathology_summary",
    "pathology.ucec_conch_patch_roi": "pathology_summary",
    "pathology.npc_conch_patch_roi": "pathology_summary",
    "radiology.read_ct_manifest": "radiology_summary",
    "radiology.lung_tumor_roi": "radiology_summary",
    "radiology.ucec_mri_roi": "radiology_summary",
    "radiology.npc_mri_roi": "radiology_summary",
    "molecular.query_biomarkers": "molecular",
    "guideline.retrieve": "guideline",
}


def register_skill_state_adapter(
    skill_name: str,
    adapter: SkillStateAdapter,
    *,
    replace: bool = False,
) -> None:
    """Register a deterministic tool-result adapter without changing the updater."""

    name = str(skill_name).strip()
    if not name:
        raise ValueError("skill_name must be non-empty.")
    if name in SKILL_STATE_ADAPTERS and not replace:
        raise ValueError(f"Skill adapter already exists: {name}")
    if not isinstance(adapter, str) and not callable(adapter):
        raise TypeError("Skill adapter must be a built-in adapter name or callable.")
    SKILL_STATE_ADAPTERS[name] = adapter

_AUDIT_ONLY_STATE_FIELDS = {
    "confirmed_evidence",
    "current_decision_stage",
    "evidence_refs",
    "free_text_summary",
    "state_update_summary",
}

_CANONICAL_EVIDENCE_FIELDS = {
    "evidence_refs",
    "guideline_summary",
    "known_biomarkers",
    "known_diagnosis",
    "known_stage",
    "molecular_summary",
    "pathology_summary",
    "radiology_summary",
}


def initialize_patient_state(
    case_dir: str | Path,
    initial_observation: Mapping[str, Any],
    *,
    planner_release: Any | None = None,
) -> dict[str, Any]:
    """Initialize compact state from manifest and the visible initial observation."""

    case_path = Path(case_dir).resolve()
    manifest = _read_optional_object(case_path / "case_manifest.json")
    hidden_metadata = _read_optional_object(case_path / "hidden_state.json")
    visible = dict(initial_observation.get("visible_information", {}))
    case_id = str(
        manifest.get("case_id")
        or hidden_metadata.get("case_id")
        or initial_observation.get("case_id")
        or visible.get("case_id")
        or case_path.name
    )
    cancer_type = _first_text(
        manifest,
        "cancer_type",
        "project_id",
    ) or _first_text(
        hidden_metadata,
        "cancer_type",
        "project_id",
    ) or _infer_cancer_type_from_case(case_path, visible)
    cancer_family, disease_subtype = _cancer_family_and_subtype(cancer_type, visible)
    guideline_default = None
    if planner_release is not None:
        planner_release.require_supported_action(cancer_family, disease_subtype)
        guideline_default = planner_release.guideline_default(
            cancer_family,
            disease_subtype,
        )
    decision_date, guideline_context = _initial_guideline_context(
        manifest,
        initial_observation,
        cancer_family,
        disease_subtype,
        guideline_default=guideline_default,
    )
    available_modalities = _available_modalities(case_path, manifest)
    state = {
        "schema_version": PATIENT_STATE_SCHEMA_VERSION,
        "case_id": case_id,
        "cancer_type": cancer_type or None,
        "cancer_family": cancer_family,
        "disease_subtype": disease_subtype,
        "current_phase": _normalize_phase(initial_observation.get("phase")),
        "decision_date": decision_date,
        "guideline_context": guideline_context,
        "current_time": 0,
        "known_diagnosis": _first_text(
            visible,
            "primary_diagnosis",
            "diagnosis",
            "chief_problem",
        )
        or None,
        "known_stage": _stage_from_mapping(visible),
        "known_biomarkers": {},
        "risk_stratification": {},
        "radiology_summary": None,
        "pathology_summary": None,
        "molecular_summary": None,
        "guideline_summary": None,
        "available_modalities": available_modalities,
        "completed_skills": [],
        "treatment_history": _initial_treatment_history(visible),
        "current_treatment_line": None,
        "evidence_ledger": _baseline_evidence_ledger(visible),
        "last_transition": None,
        "missing_information": [],
        "blocked_pathways": [],
        "blocked_actions": [],
        "evidence_refs": [],
        "warnings": [],
        "confirmed_evidence": {
            key: value
            for key, value in visible.items()
            if value not in (None, "", [], {})
        },
        "unresolved_information": [],
        "completed_actions": [],
        "pending_actions": [],
        "current_decision_stage": _normalize_phase(initial_observation.get("phase")),
        "previous_actions": [],
        "free_text_summary": _visible_summary(visible),
    }
    return validate_patient_state_v2(_drop_none_preserving_required_nulls(state))


def update_patient_state(
    patient_state: Mapping[str, Any],
    tool_records: list[Mapping[str, Any]],
    planner_output: Mapping[str, Any] | None = None,
    *,
    current_time: str | int | None = None,
) -> dict[str, Any]:
    """Update state from trusted structured tool outputs only."""

    before_state = validate_patient_state_v2(patient_state)
    state = deepcopy(before_state)
    if planner_output:
        planner_output = dict(planner_output)
    state.setdefault("completed_skills", [])
    state.setdefault("known_biomarkers", {})
    state.setdefault("evidence_refs", [])
    state.setdefault("warnings", [])
    state.setdefault("confirmed_evidence", {})
    state.setdefault("unresolved_information", [])
    state.setdefault("completed_actions", [])
    state.setdefault("pending_actions", [])
    state.setdefault("previous_actions", [])
    state.setdefault("treatment_history", [])
    state.setdefault("evidence_ledger", [])
    state.setdefault("blocked_actions", [])
    if current_time is not None:
        state["current_time"] = current_time
    if planner_output:
        state["missing_information"] = _string_list(
            planner_output.get("missing_information")
        )
        state["unresolved_information"] = list(state["missing_information"])
        state["blocked_actions"] = [
            dict(item)
            for item in planner_output.get("blocked_actions", [])
            if isinstance(item, Mapping)
        ]
        state["blocked_pathways"] = [
            str(item.get("objective"))
            for item in state["blocked_actions"]
            if str(item.get("objective") or "").strip()
        ]
        suggested = _planner_required_skills(planner_output)
        state["pending_actions"] = suggested
        state["previous_actions"].append(
            {
                "time": state.get("current_time"),
                "actions": deepcopy(list(planner_output.get("actions") or [])),
                "required_skills": suggested,
            }
        )

    changes: dict[str, Any] = {}
    for record in tool_records:
        skill_name = str(record.get("skill_name") or "")
        status = str(record.get("status") or "")
        if skill_name:
            state["completed_actions"].append(
                {
                    "time": state.get("current_time"),
                    "skill_name": skill_name,
                    "call_id": record.get("call_id"),
                    "status": status or None,
                }
            )
        adapter = SKILL_STATE_ADAPTERS.get(skill_name)
        if adapter is None:
            if skill_name:
                state["warnings"].append(f"Untrusted or unknown skill ignored: {skill_name}")
            continue
        if status and status != "success":
            state["warnings"].append(f"Skill did not succeed: {skill_name} ({status})")
            continue
        if skill_name:
            _append_unique(state["completed_skills"], skill_name)
            for alias in completed_planner_skill_aliases(skill_name):
                _append_unique(state["completed_skills"], alias)
        raw = _read_raw_record_output(record)
        data = _finding_data(raw)
        summary = _summary(raw, record)
        if callable(adapter):
            adapter(
                state=state,
                changes=changes,
                record=record,
                raw=raw,
                data=data,
                summary=summary,
            )
        elif adapter == "clinical_summary":
            _set_if_present(
                state,
                changes,
                "known_diagnosis",
                _first_text(data, "primary_diagnosis", "diagnosis", "histology", "tumor_type"),
            )
            _set_if_present(state, changes, "known_stage", _stage_from_mapping(data))
            inferred = _infer_cancer_type_from_data(data)
            _set_if_present(state, changes, "cancer_type", inferred)
            family, subtype = _cancer_family_and_subtype(inferred, data)
            _set_if_present(state, changes, "cancer_family", family)
            _set_if_present(state, changes, "disease_subtype", subtype)
        elif adapter == "treatment_history":
            treatment = _treatment_record(data, summary, record)
            if treatment and treatment not in state["treatment_history"]:
                state["treatment_history"].append(treatment)
                state["current_treatment_line"] = len(state["treatment_history"])
                changes["treatment_history"] = deepcopy(state["treatment_history"])
        elif adapter == "followup":
            _append_evidence_ledger(state, record, summary, field="followup_status")
        elif adapter == "pathology_report":
            _set_if_present(state, changes, "pathology_summary", summary)
            report_text = _first_text(data, "report_text")
            diagnosis = _diagnosis_from_text(report_text)
            _set_if_present(state, changes, "known_diagnosis", diagnosis)
        elif adapter == "radiology_summary":
            _set_if_present(state, changes, "radiology_summary", summary)
        elif adapter == "pathology_summary":
            _set_if_present(state, changes, "pathology_summary", summary)
        elif adapter == "molecular":
            _set_if_present(
                state,
                changes,
                "molecular_summary",
                _first_text(data, "summary", "molecular_subtype", "copy_number_summary")
                or summary,
            )
            biomarkers = _biomarkers_from_data(data)
            if biomarkers:
                state["known_biomarkers"] = {
                    **dict(state.get("known_biomarkers") or {}),
                    **biomarkers,
                }
                changes["known_biomarkers"] = state["known_biomarkers"]
        elif adapter == "guideline":
            _set_if_present(state, changes, "guideline_summary", summary)

        _append_evidence_ref(state, record, summary)
        _append_evidence_ledger(state, record, summary)

    state["state_update_summary"] = changes
    _refresh_canonical_state(state)
    proposed_phase = planner_output.get("proposed_phase") if planner_output else None
    allowed, reason = can_commit_phase(state, proposed_phase)
    if proposed_phase and allowed:
        state["current_phase"] = str(proposed_phase)
    elif proposed_phase and not allowed:
        state["warnings"].append(f"Proposed phase not committed: {reason}")
    state["current_decision_stage"] = state["current_phase"]
    transition_delta = summarize_state_delta(before_state, state)
    state["last_transition"] = {
        "planner_action": deepcopy(dict(planner_output or {})),
        "tool_results": [_compact_tool_result(item) for item in tool_records],
        "state_delta": transition_delta,
    }
    return validate_patient_state_v2(_drop_none_preserving_required_nulls(state))


def summarize_state_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    """Return top-level changed fields for audit."""

    changed: dict[str, Any] = {}
    keys = sorted(set(before) | set(after))
    for key in keys:
        if before.get(key) != after.get(key):
            changed[key] = {
                "before": before.get(key),
                "after": after.get(key),
            }
    return changed


def current_patient_state_view(
    patient_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the current facts without append-only audit history.

    The complete state remains available in ``patient_state_history.jsonl``.
    Model prompts retain the compact previous transition, treatment history,
    completed actions, and evidence ledger required by Planner V2.
    """

    state = deepcopy(dict(patient_state))
    confirmed = state.get("confirmed_evidence")
    baseline_evidence: dict[str, Any] = {}
    if isinstance(confirmed, Mapping):
        baseline_evidence = {
            str(key): deepcopy(value)
            for key, value in confirmed.items()
            if key not in _CANONICAL_EVIDENCE_FIELDS
            and value not in (None, "", [], {})
        }

    view = {
        key: value
        for key, value in state.items()
        if key not in _AUDIT_ONLY_STATE_FIELDS
    }
    view["completed_actions"] = list(state.get("completed_actions", []))[-8:]
    view["previous_actions"] = list(state.get("previous_actions", []))[-2:]
    view["evidence_ledger"] = list(state.get("evidence_ledger", []))[-20:]
    view["treatment_history"] = list(state.get("treatment_history", []))[-12:]
    view["last_transition"] = deepcopy(state.get("last_transition"))
    if baseline_evidence:
        view["baseline_evidence"] = baseline_evidence
    return validate_patient_state_v2(_drop_none_preserving_required_nulls(view))


def _read_optional_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = read_json(path)
    return dict(data) if isinstance(data, Mapping) else {}


def _read_raw_record_output(record: Mapping[str, Any]) -> dict[str, Any]:
    path = record.get("raw_output_path")
    if isinstance(path, str) and Path(path).is_file():
        data = read_json(Path(path))
        return dict(data) if isinstance(data, Mapping) else {}
    result = record.get("result")
    return dict(result) if isinstance(result, Mapping) else {}


def _finding_data(raw: Mapping[str, Any]) -> dict[str, Any]:
    findings = raw.get("findings", {})
    if not isinstance(findings, Mapping):
        return {}
    data = findings.get("data", {})
    return dict(data) if isinstance(data, Mapping) else {}


def _summary(raw: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    findings = raw.get("findings", {})
    if isinstance(findings, Mapping):
        summary = findings.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
    summary = record.get("summary")
    return str(summary).strip() if summary is not None else ""


def _available_modalities(case_dir: Path, manifest: Mapping[str, Any]) -> dict[str, bool]:
    modalities = manifest.get("available_modalities")
    if isinstance(modalities, Mapping):
        return {str(key): bool(value) for key, value in modalities.items()}
    files = manifest.get("files")
    if isinstance(files, Mapping):
        return {
            "evaluation": bool(files.get("evaluation")),
            "radiology": bool(files.get("radiology_nifti")),
            "pathology": bool(files.get("pathology_wsi")),
            "reports": bool(files.get("reports")),
        }
    return {
        "evaluation": (case_dir / "evaluation").is_dir(),
        "radiology": (case_dir / "radiology").is_dir(),
        "pathology": (case_dir / "pathology").is_dir(),
        "reports": (case_dir / "reports").is_dir(),
        "molecular": (case_dir / "molecular").is_dir(),
        "clinical": (case_dir / "clinical").is_dir(),
        "guideline": (case_dir / "guideline").is_dir(),
    }


def _stage_from_mapping(data: Mapping[str, Any]) -> str | None:
    stage_parts = []
    for key in (
        "clinical_stage",
        "pathologic_stage",
        "ajcc_pathologic_stage",
        "pathologic_t_stage",
        "pathologic_n_stage",
        "pathologic_m_stage",
    ):
        value = _first_text(data, key)
        if value:
            stage_parts.append(f"{key}={value}")
    return "; ".join(stage_parts) or None


def _infer_cancer_type_from_case(case_dir: Path, visible: Mapping[str, Any]) -> str | None:
    text = " ".join(
        [
            case_dir.name,
            str(visible.get("project_id") or ""),
            str(visible.get("chief_problem") or ""),
            str(visible.get("primary_diagnosis") or ""),
        ]
    ).lower()
    if "luad" in text or "nsclc" in text or "adenocarcinoma" in text:
        return "nsclc"
    if "sclc" in text or "small cell" in text:
        return "sclc"
    if "ucec" in text or "endometr" in text:
        return "ucec"
    if "npc" in text or "nasopharyn" in text or "鼻咽" in text:
        return "npc"
    return None


def _infer_cancer_type_from_data(data: Mapping[str, Any]) -> str | None:
    text = " ".join(str(value) for value in data.values() if not isinstance(value, (dict, list))).lower()
    if "luad" in text or "nsclc" in text or "adenocarcinoma" in text:
        return "nsclc"
    if "sclc" in text or "small cell" in text:
        return "sclc"
    if "ucec" in text or "endometr" in text:
        return "ucec"
    if "npc" in text or "nasopharyn" in text or "鼻咽" in text:
        return "npc"
    return None


def _diagnosis_from_text(text: str) -> str | None:
    lowered = text.lower()
    if "adenocarcinoma" in lowered:
        return "adenocarcinoma"
    if "squamous" in lowered:
        return "squamous cell carcinoma"
    if "small cell" in lowered:
        return "small cell carcinoma"
    return None


def _biomarkers_from_data(data: Mapping[str, Any]) -> dict[str, Any]:
    details = data.get("biomarker_details")
    if isinstance(details, Mapping):
        return {str(key): value for key, value in details.items()}
    ignored = {
        "case_id",
        "source",
        "extraction_method",
        "extraction_warnings",
        "summary",
        "molecular_subtype",
        "copy_number_summary",
        "high_amplification_genes",
    }
    biomarkers: dict[str, Any] = {}
    for key, value in data.items():
        if key in ignored or value in (None, "", "not_available"):
            continue
        if any(char.isupper() for char in str(key)):
            biomarkers[str(key)] = value
    return biomarkers


def _append_evidence_ref(
    state: dict[str, Any],
    record: Mapping[str, Any],
    summary: str,
) -> None:
    refs = state.setdefault("evidence_refs", [])
    refs.append(
        {
            "call_id": record.get("call_id"),
            "skill_name": record.get("skill_name"),
            "status": record.get("status"),
            "summary": summary,
            "raw_output_path": record.get("raw_output_path"),
            "artifact_paths": list(record.get("artifact_paths", [])),
        }
    )


def _append_evidence_ledger(
    state: dict[str, Any],
    record: Mapping[str, Any],
    summary: str,
    *,
    field: str | None = None,
) -> None:
    if not summary and not record.get("raw_output_path"):
        return
    item = {
        "field": field or _evidence_field_for_skill(str(record.get("skill_name") or "")),
        "value": summary or None,
        "skill_name": record.get("skill_name"),
        "call_id": record.get("call_id"),
        "time": state.get("current_time"),
        "status": record.get("status") or "success",
        "source_ref": (
            record.get("raw_output_path")
            or record.get("call_id")
            or record.get("skill_name")
        ),
    }
    ledger = state.setdefault("evidence_ledger", [])
    identity = (item["field"], item["call_id"], item["source_ref"])
    if any(
        (entry.get("field"), entry.get("call_id"), entry.get("source_ref")) == identity
        for entry in ledger
        if isinstance(entry, Mapping)
    ):
        return
    ledger.append(item)


def _evidence_field_for_skill(skill_name: str) -> str:
    adapter = SKILL_STATE_ADAPTERS.get(skill_name, "unknown")
    if callable(adapter):
        return f"custom:{skill_name}"
    return {
        "clinical_summary": "clinical_summary",
        "treatment_history": "treatment_history",
        "followup": "followup_status",
        "pathology_report": "known_diagnosis",
        "pathology_summary": "pathology_summary",
        "radiology_summary": "radiology_summary",
        "molecular": "known_biomarkers",
        "guideline": "guideline_summary",
    }.get(adapter, adapter)


def _planner_required_skills(planner_output: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for action in planner_output.get("actions", []):
        if not isinstance(action, Mapping):
            continue
        for skill in _string_list(action.get("required_skills")):
            _append_unique(result, skill)
    return result


def _treatment_record(
    data: Mapping[str, Any],
    summary: str,
    record: Mapping[str, Any],
) -> dict[str, Any] | None:
    source = data.get("treatments") or data.get("treatment_history")
    if isinstance(source, list) and source:
        return {
            "status": "observed",
            "items": deepcopy(source),
            "summary": summary or None,
            "source_skill": record.get("skill_name"),
            "call_id": record.get("call_id"),
        }
    if summary:
        return {
            "status": "observed",
            "summary": summary,
            "source_skill": record.get("skill_name"),
            "call_id": record.get("call_id"),
        }
    return None


def _compact_tool_result(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(record.get(key))
        for key in (
            "skill_name",
            "call_id",
            "status",
            "summary",
            "raw_output_path",
            "artifact_paths",
        )
        if record.get(key) not in (None, "", [], {})
    }


def _set_if_present(
    state: dict[str, Any],
    changes: dict[str, Any],
    key: str,
    value: Any,
) -> None:
    if value in (None, "", [], {}):
        return
    if state.get(key) != value:
        state[key] = value
        changes[key] = value


def _append_unique(items: list[Any], value: Any) -> None:
    if value not in items:
        items.append(value)


def _first_text(data: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"not_available", "none", "null", "unknown"}:
            return text
    return ""


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _drop_none(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _drop_none(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


def _drop_none_preserving_required_nulls(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _drop_none(dict(value))
    for key in (
        "disease_subtype",
        "known_diagnosis",
        "known_stage",
        "current_treatment_line",
        "last_transition",
    ):
        if key in value and value[key] is None:
            result[key] = None
    return result


def _refresh_canonical_state(state: dict[str, Any]) -> None:
    evidence = state.setdefault("confirmed_evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
        state["confirmed_evidence"] = evidence
    for key in (
        "known_diagnosis",
        "known_stage",
        "known_biomarkers",
        "radiology_summary",
        "pathology_summary",
        "molecular_summary",
        "guideline_summary",
        "evidence_refs",
    ):
        value = state.get(key)
        if value not in (None, "", [], {}):
            evidence[key] = value
    state["unresolved_information"] = list(state.get("missing_information") or [])
    state["current_decision_stage"] = state.get("current_phase")
    completed = set(state.get("completed_skills") or [])
    state["pending_actions"] = [
        item for item in state.get("pending_actions", []) if item not in completed
    ]
    state["free_text_summary"] = _visible_summary(state)


def _visible_summary(value: Mapping[str, Any]) -> str | None:
    parts: list[str] = []
    for key in (
        "primary_diagnosis",
        "known_diagnosis",
        "known_stage",
        "radiology_summary",
        "pathology_summary",
        "molecular_summary",
    ):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            parts.append(f"{key}: {item.strip()}")
    return "\n".join(parts) or None


def _normalize_phase(value: Any) -> str:
    text = str(value or "diagnostic_workup").strip().lower()
    aliases = {
        "diagnosis_phase": "diagnostic_workup",
        "diagnostic_phase": "diagnostic_workup",
        "diagnosis": "diagnostic_workup",
        "treatment_phase": "treatment_selection",
        "treatment_planning": "treatment_selection",
        "progression_phase": "treatment_monitoring",
        "outcome_phase": "followup",
    }
    return aliases.get(text, text)


def _cancer_family_and_subtype(
    cancer_type: Any,
    context: Mapping[str, Any],
) -> tuple[str, str | None]:
    text = " ".join(
        [
            str(cancer_type or ""),
            *(str(value) for value in context.values() if not isinstance(value, (dict, list))),
        ]
    ).lower()
    if any(token in text for token in ("nasopharyn", "npc", "鼻咽")):
        return "nasopharyngeal", "npc"
    if any(token in text for token in ("ucec", "endometr", "子宫内膜")):
        return "endometrial", "ucec"
    if any(token in text for token in ("sclc", "small cell")) and "nsclc" not in text:
        return "lung", "sclc"
    if any(
        token in text
        for token in ("lung", "luad", "nsclc", "adenocarcinoma", "肺")
    ):
        return "lung", "nsclc" if any(
            token in text for token in ("luad", "nsclc", "adenocarcinoma", "腺癌")
        ) else "unknown"
    raise ValueError(
        "Planner V2 requires cancer_family to be lung, endometrial, or nasopharyngeal."
    )


def _initial_guideline_context(
    manifest: Mapping[str, Any],
    initial_observation: Mapping[str, Any],
    cancer_family: str,
    disease_subtype: str | None,
    *,
    guideline_default: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    supplied = initial_observation.get("guideline_context") or manifest.get(
        "guideline_context"
    )
    explicit_date = (
        initial_observation.get("decision_date")
        or manifest.get("decision_date")
        or (supplied.get("decision_date") if isinstance(supplied, Mapping) else None)
    )
    if guideline_default is None:
        if cancer_family == "lung" and disease_subtype == "nsclc":
            guideline_default = {
                "guideline_id": "NSCLC_2010",
                "version": "2010",
                "decision_date": "2010-12-31",
            }
        elif cancer_family == "endometrial" and disease_subtype == "ucec":
            guideline_default = {
                "guideline_id": "CSCO子宫内膜癌2023",
                "version": "2023",
                "decision_date": "2023-12-31",
            }
        elif cancer_family == "nasopharyngeal" and disease_subtype == "npc":
            guideline_default = {
                "guideline_id": "CSCO鼻咽癌2022",
                "version": "2022",
                "decision_date": "2022-12-31",
            }
        else:
            raise ValueError(
                f"Planner action invocation for {cancer_family}/{disease_subtype or 'unknown'} "
                "is not supported by the current release."
            )
    expected = {
        "guideline_id": str(guideline_default.get("guideline_id") or ""),
        "version": str(guideline_default.get("version") or ""),
        "decision_date": str(guideline_default.get("decision_date") or ""),
    }
    if not all(expected.values()):
        raise ValueError("Planner guideline default is incomplete.")
    if explicit_date is not None and str(explicit_date) != expected["decision_date"]:
        raise ValueError(
            "Explicit decision_date is incompatible with the Planner release: "
            f"expected {expected['decision_date']!r}, got {str(explicit_date)!r}."
        )
    if isinstance(supplied, Mapping) and supplied.get("guidelines"):
        guidelines = supplied["guidelines"]
        if not isinstance(guidelines, list) or len(guidelines) != 1:
            raise ValueError(
                "Planner release requires exactly one explicit target guideline."
            )
        selected = guidelines[0]
        if not isinstance(selected, Mapping):
            raise ValueError("Explicit target guideline must be an object.")
        actual = {
            "guideline_id": str(selected.get("guideline_id") or ""),
            "version": str(selected.get("version") or ""),
        }
        wanted = {
            "guideline_id": expected["guideline_id"],
            "version": expected["version"],
        }
        if actual != wanted:
            raise ValueError(
                "Explicit guideline_context is incompatible with the Planner release: "
                f"expected {wanted!r}, got {actual!r}."
            )
    guideline = {
        "guideline_id": expected["guideline_id"],
        "version": expected["version"],
    }
    return expected["decision_date"], {
        "decision_date": expected["decision_date"],
        "guidelines": [guideline],
    }


def _initial_treatment_history(visible: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = visible.get("treatment_history") or visible.get("observed_treatments")
    if isinstance(value, list):
        return [dict(item) if isinstance(item, Mapping) else {"summary": str(item)} for item in value]
    if isinstance(value, Mapping):
        return [dict(value)]
    return []


def _baseline_evidence_ledger(visible: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "field": str(key),
            "value": deepcopy(value),
            "skill_name": "initial_observation",
            "time": 0,
            "status": "confirmed",
            "source_ref": "initial_observation",
        }
        for key, value in visible.items()
        if value not in (None, "", [], {})
    ]
