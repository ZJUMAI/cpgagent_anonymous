"""Translate Planner V2 semantic skill labels to executable MedClaw tools.

Planner trajectories supervise clinical objectives.  Those labels intentionally
do not have to be identical to the runtime registry names, so the dual-agent
prompt and Patient State need one explicit, audited translation boundary.
"""

from __future__ import annotations

from typing import Any, Mapping


_PLANNER_SKILL_TO_RUNTIME: dict[str, tuple[str, ...]] = {
    "pathology.read_diagnostic_report": ("pathology.read_report",),
    "pathology.read_diagnostic_report.cross_check": ("pathology.read_report",),
    "pathology.read_staging_report": ("pathology.read_report",),
    "pathology.read_staging_report.cross_check": ("pathology.read_report",),
    "molecular.read_biomarker_report": ("molecular.query_biomarkers",),
    "molecular.read_biomarker_report.cross_check": ("molecular.query_biomarkers",),
    "molecular.complete_biomarker_profile": ("molecular.query_biomarkers",),
    "molecular.complete_biomarker_profile.cross_check": (
        "molecular.query_biomarkers",
    ),
    "radiology.review_staging_extent": ("radiology.read_ct_manifest",),
    "radiology.review_staging_extent.cross_check": ("radiology.read_ct_manifest",),
}

_DECISION_ONLY_PREFIXES = ("treatment.",)


def runtime_tools_for_planner_skill(
    planner_skill: str,
    patient_state: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return executable tools for one Planner semantic skill label."""

    skill = str(planner_skill).strip()
    if skill in _PLANNER_SKILL_TO_RUNTIME:
        tools = _PLANNER_SKILL_TO_RUNTIME[skill]
        if skill.startswith("radiology.review_staging_extent"):
            family = str((patient_state or {}).get("cancer_family") or "").lower()
            if family == "endometrial":
                return (*tools, "radiology.ucec_mri_roi")
            if family == "lung":
                return (*tools, "radiology.lung_tumor_roi")
        return tools
    if skill.startswith(_DECISION_ONLY_PREFIXES):
        return ()
    # Labels already present in the runtime registry remain directly callable.
    if skill in {
        "clinical.read_summary",
        "clinical.read_treatment",
        "clinical.read_follow_up",
        "pathology.read_report",
        "pathology.read_slide_metadata",
        "pathology.read_wsi_manifest",
        "pathology.conch_patch_roi",
        "pathology.ucec_conch_patch_roi",
        "radiology.read_ct_manifest",
        "radiology.lung_tumor_roi",
        "radiology.ucec_mri_roi",
        "molecular.query_biomarkers",
        "guideline.retrieve",
    }:
        return (skill,)
    return ()


def planner_skill_execution_map(
    planner_output: Mapping[str, Any],
    patient_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Build prompt-safe execution guidance for all required Planner skills."""

    mapping: dict[str, list[str]] = {}
    decision_only: list[str] = []
    unmapped: list[str] = []
    for action in planner_output.get("actions", []):
        if not isinstance(action, Mapping):
            continue
        for raw_skill in action.get("required_skills", []):
            skill = str(raw_skill).strip()
            if not skill or skill in mapping or skill in decision_only or skill in unmapped:
                continue
            tools = runtime_tools_for_planner_skill(skill, patient_state)
            if tools:
                mapping[skill] = list(tools)
            elif skill.startswith(_DECISION_ONLY_PREFIXES):
                decision_only.append(skill)
            else:
                unmapped.append(skill)
    return {
        "semantic_to_runtime_tools": mapping,
        "decision_only_skills": decision_only,
        "unmapped_skills": unmapped,
        "instruction": (
            "Only names under semantic_to_runtime_tools are executable MedClaw "
            "tools. Decision-only labels describe a clinical decision and must not "
            "be emitted as tool calls. Do not invent a tool for an unmapped label."
        ),
    }


def completed_planner_skill_aliases(runtime_skill: str) -> tuple[str, ...]:
    """Return semantic labels satisfied by a successful runtime tool call."""

    name = str(runtime_skill).strip()
    aliases = [
        planner_skill
        for planner_skill, runtime_tools in _PLANNER_SKILL_TO_RUNTIME.items()
        if name in runtime_tools
    ]
    return tuple(sorted(aliases))
