"""Build reviewable Planner V2 trajectories from the local LUNG/UCEC cases.

The builder is deliberately deterministic.  It replays evidence that exists in
the case folders and uses a small, versioned rule registry grounded in selected
guideline chunks.  It does not reuse legacy planner output and it does not emit
treatment recommendations.  Every generated record remains pending until a
human reviewer approves it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from guideline_planner.chunking import GuidelineChunk, chunk_guideline_file
from guideline_planner.grounding import validate_grounding_assets
from guideline_planner.io_utils import write_json, write_jsonl
from guideline_planner.memory import extract_memory_slots
from guideline_planner.phase_machine import commit_phase
from guideline_planner.rule_engine import compile_action_set
from guideline_planner.schemas_v2 import state_delta, validate_trajectory_record_v2
from guideline_planner.trajectory_dataset import build_trajectory_dataset_v2


BUILDER_VERSION = "real-case-evidence-replay.v1"
DECISION_DATES = {"lung": "2010-12-31", "endometrial": "2023-12-31"}
GUIDELINE_EFFECTIVE_FROM = {"lung": "2010-01-01", "endometrial": "2023-01-01"}


@dataclass(frozen=True)
class PhaseSpec:
    phase: str
    next_phase: str
    chunk_title: str
    action_id: str
    objective: str
    skill: str
    expected_field: str
    missing_information: str


PHASE_SPECS: dict[str, tuple[PhaseSpec, ...]] = {
    "lung": (
        PhaseSpec(
            "diagnostic_workup", "diagnosis_confirmation",
            "Chunk 4: Pathologic Evaluation", "lung-read-pathology",
            "读取肺癌病理资料，确认组织学诊断。",
            "pathology.read_diagnostic_report", "known_diagnosis", "组织学诊断",
        ),
        PhaseSpec(
            "diagnosis_confirmation", "staging", "Chunk 5: Staging",
            "lung-read-pathologic-stage", "读取 TNM/病理分期记录并确认分期。",
            "pathology.read_staging_report", "known_stage", "TNM 与总体分期",
        ),
        PhaseSpec(
            "staging", "risk_or_biomarker_stratification",
            "Chunk 13: Initial Clinical Evaluation", "lung-review-extent",
            "核对影像与临床病变范围，记录分期相关风险信息。",
            "radiology.review_staging_extent", "risk_stratification", "病变范围核验",
        ),
        PhaseSpec(
            "risk_or_biomarker_stratification", "treatment_selection",
            "Chunk 6: Molecular Biomarkers", "lung-read-biomarkers",
            "读取分子检测报告，形成可追溯的生物标志物状态。",
            "molecular.read_biomarker_report", "known_biomarkers", "分子标志物",
        ),
    ),
    "endometrial": (
        PhaseSpec(
            "diagnostic_workup", "diagnosis_confirmation",
            "Chunk 4：子宫内膜癌病理诊断原则", "ucec-read-pathology",
            "读取子宫内膜病理资料，确认组织学诊断。",
            "pathology.read_diagnostic_report", "known_diagnosis", "组织学诊断",
        ),
        PhaseSpec(
            "diagnosis_confirmation", "staging",
            "Chunk 7：FIGO 2023手术病理分期及分子分型修饰", "ucec-read-figo-stage",
            "读取手术病理和 FIGO 2023 分期记录。",
            "pathology.read_staging_report", "known_stage", "FIGO 2023 分期",
        ),
        PhaseSpec(
            "staging", "risk_or_biomarker_stratification",
            "Chunk 3：子宫内膜癌诊断与临床检查", "ucec-review-extent",
            "核对影像、肌层浸润、LVSI 和淋巴结状态。",
            "radiology.review_staging_extent", "risk_stratification", "病变范围与风险因子",
        ),
        PhaseSpec(
            "risk_or_biomarker_stratification", "treatment_selection",
            "Chunk 6：子宫内膜癌分子分型检测与判读", "ucec-read-molecular-profile",
            "读取 MMR、POLE 与 p53 等分子分型资料。",
            "molecular.read_biomarker_report", "known_biomarkers", "分子分型",
        ),
    ),
}


def build_real_case_trajectory_pilot(
    *,
    patient_root: str | Path,
    guideline_root: str | Path,
    output_dir: str | Path,
    seed: int = 17,
) -> dict[str, Any]:
    """Generate grounded LUNG/UCEC candidate trajectories and QC artifacts."""

    patient_root = Path(patient_root)
    guideline_root = Path(guideline_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    all_chunks, selected_chunks, chunk_by_key = _load_guideline_chunks(guideline_root)
    rules = _build_rules(chunk_by_key)
    rules_by_phase = {
        (str(rule["cancer_family"]), str(rule["phase"])): [] for rule in rules
    }
    for rule in rules:
        rules_by_phase[(str(rule["cancer_family"]), str(rule["phase"]))].append(rule)

    rule_ids_by_memory: dict[str, list[str]] = {}
    for rule in rules:
        for memory_id in rule["memory_ids"]:
            rule_ids_by_memory.setdefault(str(memory_id), []).append(str(rule["rule_id"]))
    chunks_for_store = []
    for chunk in all_chunks:
        item = chunk.to_dict()
        item["source_rule_ids"] = sorted(
            rule_ids_by_memory.get(chunk.source_span_id, [])
        )
        chunks_for_store.append(item)

    chunks_path = output / "guideline_chunks.jsonl"
    selected_chunks_path = output / "selected_guideline_chunks.jsonl"
    rules_path = output / "guideline_rules.v2.jsonl"
    memory_dir = output / "memory_catalog_mock"
    write_jsonl(chunks_path, chunks_for_store)
    write_jsonl(
        selected_chunks_path,
        [
            {
                **chunk.to_dict(),
                "source_rule_ids": sorted(
                    rule_ids_by_memory.get(chunk.source_span_id, [])
                ),
            }
            for chunk in selected_chunks
        ],
    )
    write_jsonl(rules_path, rules)
    extract_memory_slots(chunks_for_store, memory_dir, memory_tokens=2, mock=True)

    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    ignored_directories: list[dict[str, Any]] = []
    source_case_counts: dict[str, int] = {}
    for family, dirname in (("lung", "LUNG"), ("endometrial", "UCEC")):
        all_dirs = sorted(path for path in (patient_root / dirname).iterdir() if path.is_dir())
        marker = "report_extraction.json" if family == "lung" else "case_manifest.json"
        case_dirs = [path for path in all_dirs if (path / marker).is_file()]
        ignored_directories.extend(
            {
                "directory": str(path),
                "cancer_family": family,
                "reason": f"not a patient case directory (missing {marker})",
            }
            for path in all_dirs
            if path not in case_dirs
        )
        source_case_counts[family] = len(case_dirs)
        for case_dir in case_dirs:
            try:
                facts = _load_case_facts(case_dir, family)
                real = _build_real_trajectory(
                    facts=facts,
                    chunk_by_key=chunk_by_key,
                    all_chunks=all_chunks,
                    rules_by_phase=rules_by_phase,
                )
                records.extend(real)
                records.append(
                    _build_decision_relevant_counterfactual(
                        real[-1], facts=facts, rules_by_phase=rules_by_phase
                    )
                )
                records.append(_build_irrelevant_counterfactual(real[-1]))
            except (FileNotFoundError, KeyError, ValueError) as exc:
                exclusions.append(
                    {
                        "case_id": case_dir.name,
                        "cancer_family": family,
                        "reason": str(exc),
                    }
                )

    normalized = [validate_trajectory_record_v2(row) for row in records]
    grounding_errors = validate_grounding_assets(
        normalized,
        rule_registry_path=rules_path,
        memory_dir=memory_dir,
    )
    candidates_path = output / "planner_trajectory.v2.candidates.jsonl"
    exclusions_path = output / "excluded_cases.jsonl"
    ignored_path = output / "ignored_non_case_directories.jsonl"
    write_jsonl(candidates_path, normalized)
    write_jsonl(exclusions_path, exclusions)
    write_jsonl(ignored_path, ignored_directories)

    dataset_manifest = build_trajectory_dataset_v2(
        normalized,
        output / "dataset",
        seed=seed,
        release_gates=False,
        fail_if_not_ready=False,
        rule_registry_path=rules_path,
        memory_dir=memory_dir,
    )
    review_queue = _build_review_queue(output / "dataset")
    review_queue_path = output / "review_queue.jsonl"
    write_jsonl(review_queue_path, review_queue)
    qc_report_path = output / "automated_qc_report.json"
    write_json(
        qc_report_path,
        {
            "schema_version": "planner_trajectory_automated_qc.v1",
            "record_count": len(normalized),
            "schema_valid_count": len(normalized),
            "schema_error_count": 0,
            "grounding_error_count": len(grounding_errors),
            "memory_catalog_slot_count": len(chunks_for_store),
            "grouped_split_manifest_ready": dataset_manifest["ready"],
            "automated_qc_passed": not grounding_errors,
            "human_review_required_count": len(review_queue),
            "training_ready": False,
            "blocking_reason": "Human review is pending and the memory catalog is mock-only.",
        },
    )

    case_counts: dict[str, int] = {}
    for row in normalized:
        if row["case_source"] == "real":
            family = str(row["state_before"]["cancer_family"])
            case_counts[family] = len(
                {
                    str(item["base_case_id"])
                    for item in normalized
                    if item["case_source"] == "real"
                    and item["state_before"]["cancer_family"] == family
                }
            )
    manifest = {
        "schema_version": "real_case_trajectory_generation_manifest.v1",
        "builder_version": BUILDER_VERSION,
        "scope": ["lung", "endometrial"],
        "deferred": ["nasopharyngeal"],
        "source_case_counts": source_case_counts,
        "included_real_case_counts": case_counts,
        "excluded_case_count": len(exclusions),
        "ignored_non_case_directory_count": len(ignored_directories),
        "record_count": len(normalized),
        "real_transition_count": sum(row["case_source"] == "real" for row in normalized),
        "counterfactual_transition_count": sum(
            row["case_source"] == "counterfactual" for row in normalized
        ),
        "grounding_error_count": len(grounding_errors),
        "grounding_errors": grounding_errors,
        "memory_catalog_slot_count": len(chunks_for_store),
        "action_support_chunk_count": len(selected_chunks),
        "guideline_chunk_counts": _guideline_chunk_counts(all_chunks),
        "review_status": "pending",
        "training_authorization": "blocked_until_required_human_review",
        "memory_catalog_is_mock": True,
        "memory_catalog_purpose": "stable IDs and grounding QC only; not a trainable memory store",
        "artifacts": {
            "candidates": str(candidates_path),
            "dataset": str(output / "dataset"),
            "rules": str(rules_path),
            "chunks": str(chunks_path),
            "action_support_chunks": str(selected_chunks_path),
            "memory_catalog": str(memory_dir),
            "review_queue": str(review_queue_path),
            "exclusions": str(exclusions_path),
            "ignored_non_case_directories": str(ignored_path),
            "automated_qc_report": str(qc_report_path),
        },
        "dataset_manifest": dataset_manifest,
    }
    write_json(output / "generation_manifest.json", manifest)
    return manifest


def _load_guideline_chunks(
    guideline_root: Path,
) -> tuple[
    list[GuidelineChunk],
    list[GuidelineChunk],
    dict[tuple[str, str], GuidelineChunk],
]:
    action_files = {
        "lung": guideline_root / "NSCLC_2010.md",
        "endometrial": guideline_root / "CSCO子宫内膜癌2023.md",
    }
    expected_files = (
        guideline_root / "NSCLC_2010.md",
        guideline_root / "SCLC_2010.md",
        guideline_root / "CSCO子宫内膜癌2023.md",
        guideline_root / "CSCO鼻咽癌2022.md",
    )
    missing_files = [str(path) for path in expected_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Required guideline files are missing: " + ", ".join(missing_files)
        )
    discovered_files = sorted(guideline_root.glob("*.md"))
    unexpected_files = sorted(set(discovered_files) - set(expected_files))
    if unexpected_files:
        raise ValueError(
            "Guideline directory must contain exactly the four versioned files; "
            "unexpected=" + ", ".join(str(path) for path in unexpected_files)
        )
    all_chunks = [
        chunk
        for path in expected_files
        for chunk in chunk_guideline_file(path)
    ]
    memory_ids = [chunk.source_span_id for chunk in all_chunks]
    if len(memory_ids) != len(set(memory_ids)):
        raise ValueError("Full guideline corpus contains duplicate memory IDs.")
    selected: list[GuidelineChunk] = []
    index: dict[tuple[str, str], GuidelineChunk] = {}
    for family, path in action_files.items():
        chunks = chunk_guideline_file(path)
        by_title = {chunk.h1_title: chunk for chunk in chunks}
        for spec in PHASE_SPECS[family]:
            chunk = by_title.get(spec.chunk_title)
            if chunk is None:
                raise ValueError(
                    f"Guideline {path.name} lacks required section {spec.chunk_title!r}"
                )
            index[(family, spec.phase)] = chunk
            selected.append(chunk)
    return all_chunks, selected, index


def _build_rules(
    chunk_by_key: Mapping[tuple[str, str], GuidelineChunk],
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for family, specs in PHASE_SPECS.items():
        for spec in specs:
            chunk = chunk_by_key[(family, spec.phase)]
            base_condition: dict[str, Any] = {"all": []}
            if spec.phase == "risk_or_biomarker_stratification":
                marker = "EGFR" if family == "lung" else "MMR"
                base_condition = {
                    "field": f"known_biomarkers.{marker}", "operator": "missing"
                }
            rules.append(_rule(family, spec, chunk, base_condition, suffix="base"))
            if spec.phase == "risk_or_biomarker_stratification":
                marker = "EGFR" if family == "lung" else "MMR"
                alt = PhaseSpec(
                    spec.phase,
                    spec.next_phase,
                    spec.chunk_title,
                    f"{spec.action_id}-complete-profile",
                    (
                        "已知 EGFR 结果，补齐其余指南相关分子信息。"
                        if family == "lung"
                        else "已知 MMR 结果，补齐 POLE 与 p53 分型信息。"
                    ),
                    "molecular.complete_biomarker_profile",
                    spec.expected_field,
                    "其余分子分型信息",
                )
                rules.append(
                    _rule(
                        family,
                        alt,
                        chunk,
                        {"field": f"known_biomarkers.{marker}", "operator": "exists"},
                        suffix="known-marker",
                    )
                )
    return rules


def _rule(
    family: str,
    spec: PhaseSpec,
    chunk: GuidelineChunk,
    condition: Mapping[str, Any],
    *,
    suffix: str,
) -> dict[str, Any]:
    rule_id = f"rule.{family}.{spec.phase}.{suffix}.v1"
    positive_condition = {"field": "available_modalities", "operator": "contains", "value": "reports"}
    deferred_condition = {"field": "known_biomarkers", "operator": "exists"}
    templates = [
        _template(
            spec.action_id, spec.objective, "evidence_gathering", spec.skill,
            spec.expected_field, {"all": []}, "required", None,
        ),
        _template(
            f"{spec.action_id}-alternate-review",
            "在报告可用时进行第二来源的一致性核对。",
            "evidence_gathering", f"{spec.skill}.cross_check", spec.expected_field,
            positive_condition, "conditional", "conditional",
        ),
        _template(
            f"{spec.action_id}-deferred-treatment",
            "仅在证据齐备后进入治疗方案选择。",
            "treatment_selection", "treatment.select_after_evidence",
            "selected_treatment", deferred_condition, None, "conditional",
        ),
        _template(
            f"{spec.action_id}-premature-treatment",
            "在当前关键证据未完成前直接选择治疗。",
            "treatment_selection", "treatment.select_without_required_evidence",
            "selected_treatment", {"all": []}, "premature", None,
            risk_level="premature",
        ),
        _template(
            f"{spec.action_id}-unsafe-ungrounded-treatment",
            "采用无当前指南来源支持的治疗方案。",
            "treatment_selection", "treatment.select_ungrounded_regimen",
            "selected_treatment", {"all": []}, "unsafe", None,
            risk_level="unsafe",
        ),
    ]
    return {
        "schema_version": "guideline_rule.v2",
        "rule_id": rule_id,
        "cancer_family": family,
        "guideline_id": chunk.guideline_id,
        "version": chunk.version,
        "phase": spec.phase,
        "condition": dict(condition),
        "allowed_action_types": ["evidence_gathering", "treatment_selection"],
        "memory_ids": [chunk.source_span_id],
        "source_spans": [chunk.source_span_id],
        "action_templates": templates,
        "compiler_note": "Evidence-replay pilot; treatment candidates are negative-only.",
    }


def _template(
    action_id: str,
    objective: str,
    action_type: str,
    skill: str,
    expected_field: str,
    condition: Mapping[str, Any],
    bucket_if_true: str | None,
    bucket_if_false: str | None,
    *,
    risk_level: str = "low",
) -> dict[str, Any]:
    return {
        "action_id": action_id,
        "objective": objective,
        "action_type": action_type,
        "required_skills": [skill],
        "preconditions": [],
        "expected_state_delta": [expected_field],
        "condition": dict(condition),
        "bucket_if_true": bucket_if_true,
        "bucket_if_false": bucket_if_false,
        "risk_level": risk_level,
    }


def _load_case_facts(case_dir: Path, family: str) -> dict[str, Any]:
    if family == "lung":
        extraction_path = case_dir / "report_extraction.json"
        payload = _read_json(extraction_path)
        clinical = payload.get("clinical") or {}
        molecular = (payload.get("molecular") or {}).get("biomarkers") or {}
        diagnosis = _required_value(clinical, "primary_diagnosis", extraction_path)
        stage = _required_value(clinical, "pathologic_stage", extraction_path)
        biomarkers = {
            key: value.get("alteration_status")
            for key, value in sorted(molecular.items())
            if isinstance(value, Mapping) and value.get("alteration_status")
        }
        if not biomarkers:
            raise ValueError(f"No usable biomarkers in {extraction_path}")
        radiology_refs = sorted(
            str(path.relative_to(case_dir))
            for path in (case_dir / "radiology").rglob("*manifest*.json")
        )
        hidden = _read_json(case_dir / "hidden_state.json")
        return {
            "case_id": case_dir.name,
            "family": family,
            "subtype": "nsclc",
            "diagnosis": str(diagnosis),
            "stage": str(stage),
            "risk": {
                "histologic_grade": clinical.get("histologic_grade"),
                "pathologic_t_stage": clinical.get("pathologic_t_stage"),
                "pathologic_n_stage": clinical.get("pathologic_n_stage"),
                "pathologic_m_stage": clinical.get("pathologic_m_stage"),
                "surgical_margins": clinical.get("surgical_margins"),
                "extent_verified": True,
            },
            "biomarkers": biomarkers,
            "modalities": _modalities(hidden.get("modalities") or {}),
            "source_ref": str(extraction_path),
            "evidence_refs": {
                "diagnosis": str(extraction_path),
                "stage": str(case_dir / "clinical" / "clinical.json"),
                "extent": str(case_dir / radiology_refs[0]) if radiology_refs else str(extraction_path),
                "biomarkers": str(case_dir / "molecular" / "biomarkers.json"),
            },
        }

    manifest_path = case_dir / "case_manifest.json"
    manifest = _read_json(manifest_path)
    pathology_path = _one(case_dir / "reports", "*_pathology_report.txt", exclude="radiology_reports")
    molecular_path = _one(case_dir / "reports", "*_molecular_report.txt")
    stage_path = _one(case_dir / "reports", "*_T1_integrated_report.txt")
    radiology_path = _one(case_dir / "reports" / "radiology_reports", "*.txt")
    pathology = _key_values(pathology_path)
    molecular = _key_values(molecular_path)
    stage_values = _key_values(stage_path)
    diagnosis = pathology.get("Histologictype") or pathology.get("histologictype")
    stage = (
        stage_values.get("病理分期（FIGO2023）")
        or stage_values.get("病理分期(FIGO2023)")
    )
    if not diagnosis or not stage:
        raise ValueError(f"Missing diagnosis or FIGO2023 stage under {case_dir}")
    biomarkers = {
        key: value for key, value in molecular.items()
        if value not in (None, "", "/")
    }
    if not biomarkers:
        raise ValueError(f"No usable biomarkers in {molecular_path}")
    return {
        "case_id": case_dir.name,
        "family": family,
        "subtype": "ucec",
        "diagnosis": diagnosis,
        "stage": stage,
        "risk": {
            "histologic_grade": pathology.get("HistologicGrade"),
            "myometrial_invasion": pathology.get("MyometrialInvasion"),
            "lvsi": pathology.get("LVSI"),
            "lymph_node_involvement": pathology.get("Lymphnodeinvolvment"),
            "extent_verified": True,
        },
        "biomarkers": biomarkers,
        "modalities": _modalities(manifest.get("available_modalities") or {}),
        "source_ref": str(manifest_path),
        "evidence_refs": {
            "diagnosis": str(pathology_path),
            "stage": str(stage_path),
            "extent": str(radiology_path),
            "biomarkers": str(molecular_path),
        },
    }


def _build_real_trajectory(
    *,
    facts: Mapping[str, Any],
    chunk_by_key: Mapping[tuple[str, str], GuidelineChunk],
    all_chunks: list[GuidelineChunk],
    rules_by_phase: Mapping[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    family = str(facts["family"])
    case_id = str(facts["case_id"])
    state = _initial_state(facts, chunk_by_key[(family, "diagnostic_workup")])
    trajectory_id = f"real-{family}-{case_id}"
    records = []
    for turn_index, spec in enumerate(PHASE_SPECS[family]):
        before = deepcopy(state)
        action_set = compile_action_set(before, rules_by_phase[(family, spec.phase)])
        required = next(
            item for item in action_set["required"] if item["action_id"] == spec.action_id
        )
        after = deepcopy(before)
        if spec.expected_field == "known_diagnosis":
            after["known_diagnosis"] = facts["diagnosis"]
        elif spec.expected_field == "known_stage":
            after["known_stage"] = facts["stage"]
        elif spec.expected_field == "risk_stratification":
            after["risk_stratification"] = deepcopy(facts["risk"])
        elif spec.expected_field == "known_biomarkers":
            after["known_biomarkers"] = deepcopy(facts["biomarkers"])
        else:
            raise AssertionError(spec.expected_field)
        after["evidence_ledger"].append(
            {
                "field": spec.expected_field,
                "value": deepcopy(after[spec.expected_field]),
                "skill_name": spec.skill,
                "time": DECISION_DATES[family],
                "status": "observed",
                "source_ref": facts["evidence_refs"][_evidence_key(spec.expected_field)],
                "call_id": f"{trajectory_id}-turn-{turn_index}",
            }
        )
        after["completed_skills"].append(spec.skill)
        after["completed_actions"].append(
            {"action_id": spec.action_id, "skill_name": spec.skill, "status": "success"}
        )
        after["unresolved_information"] = [
            item for item in after["unresolved_information"]
            if item != spec.missing_information
        ]
        after = commit_phase(after, spec.next_phase)
        core_delta = state_delta(before, after)
        after["last_transition"] = {
            "action_id": spec.action_id,
            "tool_status": "success",
            "result_summary": f"Observed {spec.expected_field} from {Path(facts['evidence_refs'][_evidence_key(spec.expected_field)]).name}",
            "state_delta": core_delta,
        }
        chunk = chunk_by_key[(family, spec.phase)]
        record = {
            "schema_version": "planner_trajectory.v2",
            "trajectory_id": trajectory_id,
            "base_case_id": case_id,
            "case_id": case_id,
            "turn_index": turn_index,
            "split": "train",
            "case_source": "real",
            "guideline_context": deepcopy(before["guideline_context"]),
            "state_before": before,
            "action_set": action_set,
            "accepted_plan_variants": _plan_variants(before, required, spec, chunk),
            "tool_executions": [
                {
                    "skill_name": spec.skill,
                    "status": "success",
                    "result_summary": f"Loaded {spec.expected_field} from actual case evidence.",
                    "evidence_refs": [facts["evidence_refs"][_evidence_key(spec.expected_field)]],
                }
            ],
            "state_after": after,
            "state_delta": state_delta(before, after),
            "routing_labels": _routing_labels(chunk, all_chunks, family),
            "provenance": _provenance(facts["source_ref"], before, required),
        }
        records.append(validate_trajectory_record_v2(record))
        state = after
    return records


def _initial_state(facts: Mapping[str, Any], chunk: GuidelineChunk) -> dict[str, Any]:
    family = str(facts["family"])
    decision_date = DECISION_DATES[family]
    guideline = {
        "guideline_id": chunk.guideline_id,
        "version": chunk.version,
        "effective_from": GUIDELINE_EFFECTIVE_FROM[family],
        "effective_to": None,
    }
    unresolved = [spec.missing_information for spec in PHASE_SPECS[family]]
    return {
        "schema_version": "patient_state.v2",
        "case_id": str(facts["case_id"]),
        "cancer_family": family,
        "disease_subtype": str(facts["subtype"]),
        "current_phase": "diagnostic_workup",
        "decision_date": decision_date,
        "guideline_context": {"decision_date": decision_date, "guidelines": [guideline]},
        "known_diagnosis": None,
        "known_stage": None,
        "known_biomarkers": {},
        "risk_stratification": {},
        "available_modalities": list(facts["modalities"]),
        "completed_skills": [],
        "completed_actions": [],
        "treatment_history": [],
        "current_treatment_line": 0,
        "evidence_ledger": [],
        "last_transition": None,
        "pending_actions": [],
        "blocked_actions": [],
        "unresolved_information": unresolved,
    }


def _plan_variants(
    before: Mapping[str, Any],
    required: Mapping[str, Any],
    spec: PhaseSpec,
    chunk: GuidelineChunk,
) -> list[dict[str, Any]]:
    action = _plan_action(required, chunk, objective=spec.objective)
    paraphrase = _plan_action(
        required,
        chunk,
        objective=f"从病例原始资料中提取并核验：{spec.missing_information}。",
    )
    common = {
        "schema_version": "planner_action.v2",
        "current_phase": before["current_phase"],
        "proposed_phase": spec.next_phase,
        "missing_information": [spec.missing_information],
        "blocked_actions": [
            {
                "objective": "在证据未齐备时直接选择治疗",
                "reason": "当前阶段的诊断、分期或分层证据尚未完成。",
                "category": "premature",
                "until": [spec.expected_field],
            },
            {
                "objective": "采用无当前 active memory 支持的治疗方案",
                "reason": "缺少当前指南版本的可追溯依据。",
                "category": "unsafe",
                "until": ["active_memory_provenance"],
            },
        ],
        "should_stop": False,
        "reason": f"需要新增 {spec.missing_information} 证据以推进状态。",
    }
    return [{**deepcopy(common), "actions": [action]}, {**deepcopy(common), "actions": [paraphrase]}]


def _plan_action(
    candidate: Mapping[str, Any],
    chunk: GuidelineChunk,
    *,
    objective: str,
) -> dict[str, Any]:
    return {
        "action_id": candidate["action_id"],
        "objective": objective,
        "action_type": candidate["action_type"],
        "required_skills": deepcopy(candidate["required_skills"]),
        "preconditions": deepcopy(candidate["preconditions"]),
        "expected_state_delta": deepcopy(candidate["expected_state_delta"]),
        "provenance": [
            {
                "memory_id": chunk.source_span_id,
                "rule_ids": deepcopy(candidate["guideline_rule_ids"]),
                "source_spans": [chunk.source_span_id],
                "guideline_id": chunk.guideline_id,
                "version": chunk.version,
            }
        ],
        "priority": 1,
        "repeat_justification": None,
        "risk_level": "low",
    }


def _routing_labels(
    positive: GuidelineChunk,
    all_chunks: list[GuidelineChunk],
    family: str,
) -> dict[str, list[str]]:
    alternatives = [
        chunk.source_span_id
        for chunk in all_chunks
        if _chunk_family(chunk) == family
        and chunk.source_span_id != positive.source_span_id
    ]
    other_family = [
        chunk.source_span_id
        for chunk in all_chunks
        if _chunk_family(chunk) != family
    ]
    return {
        "strong_positive_memory_ids": [positive.source_span_id],
        "weak_positive_memory_ids": [],
        "hard_negative_memory_ids": alternatives[:1],
        "easy_negative_memory_ids": other_family[:1],
    }


def _chunk_family(chunk: GuidelineChunk) -> str:
    cancer_type = chunk.cancer_type.lower()
    if cancer_type in {"nsclc", "sclc", "lung"}:
        return "lung"
    if cancer_type in {"ucec", "endometrial"}:
        return "endometrial"
    if cancer_type in {"nasopharyngeal", "npc"}:
        return "nasopharyngeal"
    return cancer_type


def _guideline_chunk_counts(chunks: list[GuidelineChunk]) -> dict[str, int]:
    result: dict[str, int] = {}
    for chunk in chunks:
        key = f"{chunk.guideline_id}@{chunk.version}"
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _provenance(
    source_ref: str,
    state: Mapping[str, Any],
    required: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_payload = {
        "state": state,
        "allowed_action": required,
        "policy": "deterministic evidence replay only",
    }
    output_payload = {"selected_action_id": required["action_id"]}
    return {
        "rule_compiler_version": BUILDER_VERSION,
        "case_source_ref": str(source_ref),
        "teacher_model": "deterministic-rule-compiler",
        "teacher_model_version": BUILDER_VERSION,
        "teacher_prompt_hash": _sha256(prompt_payload),
        "teacher_output_hash": _sha256(output_payload),
        "review_status": "pending",
        "reviewer_id": None,
        "reviewed_at": None,
        "qc_flags": ["human_review_required", "evidence_replay_not_treatment_gold"],
    }


def _build_decision_relevant_counterfactual(
    anchor: Mapping[str, Any],
    *,
    facts: Mapping[str, Any],
    rules_by_phase: Mapping[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    family = str(facts["family"])
    marker, value = ("EGFR", "positive") if family == "lung" else ("MMR", "dMMR")
    row = deepcopy(dict(anchor))
    case_id = f"{anchor['case_id']}-cf-known-{marker.lower()}"
    row["trajectory_id"] = f"cf-relevant-{family}-{anchor['base_case_id']}"
    row["case_id"] = case_id
    row["turn_index"] = 0
    row["case_source"] = "counterfactual"
    row["counterfactual"] = {
        "anchor_trajectory_id": anchor["trajectory_id"],
        "anchor_turn_index": anchor["turn_index"],
        "perturbation_type": "decision_relevant",
        "changed_fields": ["known_biomarkers"],
    }
    before = deepcopy(anchor["state_before"])
    before["case_id"] = case_id
    before["known_biomarkers"] = {marker: value}
    phase = str(before["current_phase"])
    action_set = compile_action_set(before, rules_by_phase[(family, phase)])
    required = action_set["required"][0]
    spec = next(item for item in PHASE_SPECS[family] if item.phase == phase)
    alt_spec = PhaseSpec(
        spec.phase, spec.next_phase, spec.chunk_title, str(required["action_id"]),
        str(required["objective"]), str(required["required_skills"][0]),
        spec.expected_field, "其余分子分型信息",
    )
    after = deepcopy(anchor["state_after"])
    after["case_id"] = case_id
    after["known_biomarkers"] = deepcopy(facts["biomarkers"])
    old_skill = anchor["tool_executions"][0]["skill_name"]
    new_skill = required["required_skills"][0]
    after["completed_skills"] = [
        new_skill if item == old_skill else item for item in after["completed_skills"]
    ]
    after["completed_actions"][-1] = {
        "action_id": required["action_id"], "skill_name": new_skill, "status": "success"
    }
    after["evidence_ledger"][-1]["skill_name"] = new_skill
    core = deepcopy(after)
    core["last_transition"] = before.get("last_transition")
    after["last_transition"] = {
        "action_id": required["action_id"],
        "tool_status": "success",
        "result_summary": "Completed remaining molecular profile under counterfactual known marker.",
        "state_delta": state_delta(before, core),
    }
    chunk_id = required["supporting_memory_ids"][0]
    guideline = before["guideline_context"]["guidelines"][0]
    chunk = GuidelineChunk(
        guideline_id=guideline["guideline_id"], version=guideline["version"],
        cancer_type=family, chapter=spec.chunk_title, section=spec.chunk_title,
        h1_title=spec.chunk_title, source_span_id=chunk_id,
        source_rule_ids=[], source_span_ids=[chunk_id], page_start=None,
        page_end=None, text="", source_path="",
    )
    row["state_before"] = before
    row["state_after"] = after
    row["action_set"] = action_set
    row["accepted_plan_variants"] = _plan_variants(before, required, alt_spec, chunk)
    row["tool_executions"] = [
        {
            "skill_name": new_skill,
            "status": "success",
            "result_summary": "Loaded remaining molecular profile from actual case evidence.",
            "evidence_refs": [facts["evidence_refs"]["biomarkers"]],
        }
    ]
    row["state_delta"] = state_delta(before, after)
    row["provenance"] = _provenance(facts["source_ref"], before, required)
    return validate_trajectory_record_v2(row)


def _build_irrelevant_counterfactual(anchor: Mapping[str, Any]) -> dict[str, Any]:
    row = deepcopy(dict(anchor))
    case_id = f"{anchor['case_id']}-cf-admin"
    row["trajectory_id"] = f"cf-irrelevant-{anchor['state_before']['cancer_family']}-{anchor['base_case_id']}"
    row["case_id"] = case_id
    row["turn_index"] = 0
    row["case_source"] = "counterfactual"
    row["counterfactual"] = {
        "anchor_trajectory_id": anchor["trajectory_id"],
        "anchor_turn_index": anchor["turn_index"],
        "perturbation_type": "irrelevant",
        "changed_fields": ["administrative_note"],
    }
    for state_name in ("state_before", "state_after"):
        row[state_name]["case_id"] = case_id
        row[state_name]["administrative_note"] = "synthetic transport-mode perturbation"
    row["state_delta"] = state_delta(row["state_before"], row["state_after"])
    return validate_trajectory_record_v2(row)


def _build_review_queue(dataset_dir: Path) -> list[dict[str, Any]]:
    source_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            source_rows.append(row)

    selected: dict[tuple[str, int], str] = {}
    for row in source_rows:
        key = (str(row["trajectory_id"]), int(row["turn_index"]))
        if row["split"] == "test":
            selected[key] = "all_test_records"
        elif row["case_source"] == "counterfactual":
            selected[key] = "all_counterfactual_records"

    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in source_rows:
        if row["split"] not in {"train", "validation"} or row["case_source"] != "real":
            continue
        key = (
            str(row["state_before"]["cancer_family"]),
            str(row["state_before"]["current_phase"]),
            str(row["split"]),
        )
        strata.setdefault(key, []).append(row)
    for rows in strata.values():
        count = min(len(rows), max(5, math.ceil(len(rows) * 0.10)))
        ordered = sorted(
            rows,
            key=lambda row: _sha256(
                {"trajectory_id": row["trajectory_id"], "turn_index": row["turn_index"]}
            ),
        )
        for row in ordered[:count]:
            selected[(str(row["trajectory_id"]), int(row["turn_index"]))] = (
                "stratified_real_case_sample"
            )

    result = []
    for row in source_rows:
        key = (str(row["trajectory_id"]), int(row["turn_index"]))
        if key not in selected:
            continue
        result.append(
            {
                "trajectory_id": row["trajectory_id"],
                "turn_index": row["turn_index"],
                "base_case_id": row["base_case_id"],
                "cancer_family": row["state_before"]["cancer_family"],
                "phase": row["state_before"]["current_phase"],
                "split": row["split"],
                "case_source": row["case_source"],
                "selection_reason": selected[key],
                "review_status": "pending",
                "required_checks": [
                    "patient evidence matches state delta",
                    "guideline span supports action type",
                    "positive/negative action buckets are clinically valid",
                    "no treatment recommendation is encoded as positive",
                ],
            }
        )
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required case artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _required_value(data: Mapping[str, Any], key: str, path: Path) -> Any:
    value = data.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing {key!r} in {path}")
    return value


def _one(root: Path, pattern: str, *, exclude: str | None = None) -> Path:
    matches = sorted(
        path for path in root.rglob(pattern)
        if exclude is None or exclude not in path.parts
    )
    if not matches:
        raise FileNotFoundError(f"No {pattern!r} under {root}")
    return matches[0]


def _key_values(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^\s*([^：:]+)[：:]\s*(.*?)\s*$", line)
        if match:
            result[match.group(1).strip()] = match.group(2).strip()
    return result


def _modalities(value: Mapping[str, Any]) -> list[str]:
    result = [str(key) for key, available in value.items() if bool(available)]
    if "reports" not in result:
        result.append("reports")
    return sorted(set(result))


def _evidence_key(field: str) -> str:
    return {
        "known_diagnosis": "diagnosis",
        "known_stage": "stage",
        "risk_stratification": "extent",
        "known_biomarkers": "biomarkers",
    }[field]


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-root", default="data/patient_data")
    parser.add_argument("--guideline-root", default="data/guidelines")
    parser.add_argument("--output-dir", default="datasets/planner_v2_pilot")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = build_real_case_trajectory_pilot(
        patient_root=args.patient_root,
        guideline_root=args.guideline_root,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
