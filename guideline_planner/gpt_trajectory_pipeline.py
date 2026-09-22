"""GPT-authored Planner V2 trajectory preparation and release assembly.

This module intentionally does not call a rule compiler or an LLM.  It builds
content-addressed, case-isolated packets that an external Codex orchestrator
can give to independent teacher/reviewer tasks, validates their returned JSON,
and assembles a new dataset without modifying the original pilot dataset.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any

from guideline_planner.artifacts import sha256_json, sha256_path
from guideline_planner.grounding import validate_grounding_assets
from guideline_planner.io_utils import read_jsonl, write_json, write_jsonl
from guideline_planner.release_scope import (
    PLANNER_V2_LUNG_ENDOMETRIAL_NPC_RELEASE_SCOPE,
    copy_release_scope,
    load_release_scope,
)
from guideline_planner.retrieval import load_memory_metadata_records
from guideline_planner.schemas_v2 import (
    PLANNER_TRAJECTORY_V2_SCHEMA,
    V2SchemaError,
    audit_trajectory_dataset_v2,
    stable_case_splits,
    state_delta,
    validate_trajectory_record_v2,
)
from guideline_planner.trajectory_dataset import build_trajectory_dataset_v2

PIPELINE_SCHEMA_VERSION = "gpt_trajectory_pipeline.v2"
PACKET_SCHEMA_VERSION = "gpt_trajectory_packet.v2"
TEACHER_OUTPUT_SCHEMA_VERSION = "gpt_trajectory_teacher_output.v2"
REVIEW_OUTPUT_SCHEMA_VERSION = "gpt_trajectory_review_output.v2"
REPAIR_OUTPUT_SCHEMA_VERSION = "gpt_trajectory_repair_output.v2"
VERIFIER_OUTPUT_SCHEMA_VERSION = "gpt_trajectory_verifier_output.v2"
CLAIM_SCHEMA_VERSION = "gpt_guideline_claim.v2"
PIPELINE_VERSION = "gpt-direct-trajectory.v2"
MANUAL_APPROVAL_SCHEMA_VERSION = "planner_trajectory_manual_approval.v2"

_ACTION_BUCKETS = ("required", "acceptable", "conditional", "premature", "unsafe")
_VALID_REVIEW_DECISIONS = {"pass_unchanged", "repair_required", "unusable"}
_VALID_VERIFY_DECISIONS = {"approved", "rejected", "repair_required", "unusable"}
_TEXT_SUFFIXES = {".json", ".md", ".txt"}
_EXCLUDED_PARTS = {
    "evaluation",
    "guideline",
    "roi",
    "roi_256",
    "roi_512",
    "tiles",
    "masks",
    "nifti",
    "wsi",
    "qc_audit",
}
_EXCLUDED_NAMES = {
    "hidden_state.json",
    "follow_up.json",
    "report_extraction_cache.json",
    "trajectory_index.jsonl",
}
_EXCLUDED_NAME_FRAGMENTS = (
    "rubric",
    "trajectory",
    "retrieval_result",
    "prompt_score",
    "qwen_prompt",
    "outcome",
    "follow_up",
)

GPT_GUIDELINE_CLAIM_V2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "rule_id",
        "claim_text",
        "provenance_only",
        "guideline_id",
        "version",
        "cancer_family",
        "memory_ids",
        "source_spans",
        "allowed_action_types",
    ],
    "properties": {
        "schema_version": {"const": CLAIM_SCHEMA_VERSION},
        "rule_id": {"type": "string", "minLength": 1},
        "claim_id": {"type": "string", "minLength": 1},
        "claim_text": {"type": "string", "minLength": 1},
        "provenance_only": {"const": True},
        "guideline_id": {"type": "string", "minLength": 1},
        "version": {"type": "string", "minLength": 1},
        "cancer_family": {"type": "string", "minLength": 1},
        "memory_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "source_spans": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "allowed_action_types": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    },
    "additionalProperties": True,
}


@dataclass(frozen=True)
class GPTTrajectoryPipelineConfig:
    """Resolved local paths and fixed release expectations."""

    repo_root: Path
    npc_case_root: Path
    lung_case_root: Path
    ucec_case_root: Path
    source_dataset_root: Path
    npc_workspace: Path
    review_workspace: Path
    merged_dataset_root: Path
    guideline_chunks_path: Path
    memory_dir: Path
    legacy_rule_registry_path: Path
    manual_approval_path: Path
    seed: int = 17
    expected_npc_cases: int = 39
    expected_existing_cases: int = 70
    expected_existing_records: int = 420
    npc_min_real_turns: int = 3
    npc_max_real_turns: int = 6
    counterfactuals_per_case: int = 2

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in list(result.items()):
            if isinstance(value, Path):
                result[key] = _portable_path(value, self.repo_root)
        return result


def load_gpt_trajectory_pipeline_config(
    config: str | Path | Mapping[str, Any] | GPTTrajectoryPipelineConfig,
) -> GPTTrajectoryPipelineConfig:
    """Load JSON/YAML configuration and resolve all paths against ``repo_root``."""

    if isinstance(config, GPTTrajectoryPipelineConfig):
        return config
    source_path: Path | None = None
    if isinstance(config, (str, Path)):
        source_path = Path(config).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"GPT trajectory pipeline config does not exist: {source_path}"
            )
        text = source_path.read_text(encoding="utf-8")
        if source_path.suffix.lower() == ".json":
            payload = json.loads(text)
        else:
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "PyYAML is required to load GPT trajectory YAML config."
                ) from exc
            payload = yaml.safe_load(text)
    else:
        payload = dict(config)
    if not isinstance(payload, Mapping):
        raise TypeError("GPT trajectory pipeline config must be an object.")
    paths = payload.get("paths") if isinstance(payload.get("paths"), Mapping) else {}
    expected = (
        payload.get("expected") if isinstance(payload.get("expected"), Mapping) else {}
    )
    generation = (
        payload.get("generation")
        if isinstance(payload.get("generation"), Mapping)
        else {}
    )
    root_value = payload.get("repo_root") or paths.get("repo_root")
    if root_value:
        repo_root = Path(str(root_value)).expanduser()
        if not repo_root.is_absolute() and source_path is not None:
            repo_root = source_path.parent / repo_root
        repo_root = repo_root.resolve()
    else:
        repo_root = Path.cwd().resolve()

    def resolve_path(name: str, default: str) -> Path:
        raw = payload.get(name) or paths.get(name) or default
        path = Path(str(raw)).expanduser()
        return (path if path.is_absolute() else repo_root / path).resolve()

    def integer(name: str, default: int, group: Mapping[str, Any]) -> int:
        value = payload.get(name, group.get(name, default))
        result = int(value)
        if result < 0:
            raise ValueError(f"{name} must be non-negative.")
        return result

    result = GPTTrajectoryPipelineConfig(
        repo_root=repo_root,
        npc_case_root=resolve_path("npc_case_root", "data/patient_data/NPC"),
        lung_case_root=resolve_path("lung_case_root", "data/patient_data/LUNG"),
        ucec_case_root=resolve_path("ucec_case_root", "data/patient_data/UCEC"),
        source_dataset_root=resolve_path(
            "source_dataset_root", "datasets/planner_v2_pilot"
        ),
        npc_workspace=resolve_path("npc_workspace", "datasets/planner_v2_npc_gpt"),
        review_workspace=resolve_path(
            "review_workspace", "datasets/planner_v2_lung_endometrial_gpt_review"
        ),
        merged_dataset_root=resolve_path(
            "merged_dataset_root", "datasets/planner_v2_lung_endometrial_npc_gpt"
        ),
        guideline_chunks_path=resolve_path(
            "guideline_chunks_path", "datasets/planner_v2_pilot/guideline_chunks.jsonl"
        ),
        memory_dir=resolve_path(
            "memory_dir", "datasets/planner_v2_pilot/memory_catalog_mock"
        ),
        legacy_rule_registry_path=resolve_path(
            "legacy_rule_registry_path",
            "datasets/planner_v2_pilot/guideline_rules.v2.jsonl",
        ),
        manual_approval_path=resolve_path(
            "manual_approval_path",
            "datasets/planner_v2_lung_endometrial_npc_manual_review.json",
        ),
        seed=integer("seed", 17, generation),
        expected_npc_cases=integer("npc_cases", 39, expected),
        expected_existing_cases=integer("existing_cases", 70, expected),
        expected_existing_records=integer("existing_records", 420, expected),
        npc_min_real_turns=integer("npc_min_real_turns", 3, generation),
        npc_max_real_turns=integer("npc_max_real_turns", 6, generation),
        counterfactuals_per_case=integer("counterfactuals_per_case", 2, generation),
    )
    if result.npc_min_real_turns > result.npc_max_real_turns:
        raise ValueError("npc_min_real_turns cannot exceed npc_max_real_turns.")
    _assert_source_destination_separation(result)
    return result


def prepare_gpt_trajectory_packets(
    config: str | Path | Mapping[str, Any] | GPTTrajectoryPipelineConfig,
    *,
    model: str = "gpt-5.6-sol",
    reasoning_effort: str = "high",
    max_repair_cycles: int = 2,
    force: bool = False,
) -> dict[str, Any]:
    """Create case-isolated NPC teacher and Lung/UCEC review packets."""

    cfg = load_gpt_trajectory_pipeline_config(config)
    _require_sources(cfg)
    npc_cases = _discover_cases(cfg.npc_case_root, marker="case_manifest.json")
    existing_rows = _load_existing_rows(cfg.source_dataset_root)
    grouped_existing = _group_existing_records(existing_rows)
    _assert_expected_counts(cfg, npc_cases, grouped_existing, existing_rows)

    npc_splits = stable_case_splits(
        {case.name: "nasopharyngeal" for case in npc_cases}, seed=cfg.seed
    )
    chunks = read_jsonl(cfg.guideline_chunks_path)
    memory_records = load_memory_metadata_records(cfg.memory_dir)
    npc_chunks = _guideline_chunks(chunks, "CSCO鼻咽癌2022", "2022")
    npc_memories = _memory_records(memory_records, "CSCO鼻咽癌2022", "2022")
    if len(npc_chunks) != 13 or len(npc_memories) != 13:
        raise V2SchemaError(
            f"NPC packet preparation requires 13 chunks and memories; got "
            f"{len(npc_chunks)} chunks/{len(npc_memories)} memories."
        )

    teacher_dir = cfg.npc_workspace / "teacher_packets"
    review_dir = cfg.review_workspace / "review_packets"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    for case_dir in npc_cases:
        case_id = case_dir.name
        teacher_path = teacher_dir / f"{case_id}.json"
        teacher_output_path = cfg.npc_workspace / "teacher_outputs" / f"{case_id}.json"
        if teacher_path.is_file() and _case_output_is_ready(
            cfg,
            teacher_path,
            teacher_output_path,
        ):
            # A completed generation is immutable.  Updating prompt guidance may
            # refresh pending packets, but must not invalidate an auditable
            # teacher output that has already passed the author-side contract.
            continue
        packet = {
            "schema_version": PACKET_SCHEMA_VERSION,
            "packet_type": "npc_teacher",
            "pipeline_version": PIPELINE_VERSION,
            "case_id": case_id,
            "base_case_id": case_id,
            "split": npc_splits[case_id],
            "model": model,
            "reasoning_effort": reasoning_effort,
            "guideline_context": _npc_guideline_context(),
            "generation_contract": {
                "output_schema_version": TEACHER_OUTPUT_SCHEMA_VERSION,
                "trajectory_schema_version": "planner_trajectory.v2",
                "real_transition_count": {
                    "minimum": cfg.npc_min_real_turns,
                    "maximum": cfg.npc_max_real_turns,
                },
                "counterfactual_counts": {"decision_relevant": 1, "irrelevant": 1},
                "direct_gpt_actions": True,
                "rule_compiler_forbidden": True,
                "claims_are_provenance_only": True,
                "no_invented_patient_facts_or_tool_results": True,
                "trajectory_schema": PLANNER_TRAJECTORY_V2_SCHEMA,
                "claim_schema": GPT_GUIDELINE_CLAIM_V2_SCHEMA,
            },
            "allowed_skills": _allowed_skills("nasopharyngeal"),
            "phase_vocabulary": [
                "diagnostic_workup",
                "diagnosis_confirmation",
                "staging",
                "risk_or_biomarker_stratification",
                "treatment_selection",
                "treatment_monitoring",
                "surveillance",
            ],
            "patient_evidence": _collect_case_evidence(case_dir, cfg.repo_root),
            "guideline_chunks": npc_chunks,
            "memory_catalog": npc_memories,
            "task_instructions": [
                "Work on this case only and do not read any other case or prior model output.",
                "Author clinical actions directly from patient evidence and the supplied guideline chunks; do not use a deterministic rule compiler or phase template.",
                "Produce 3-6 evidence-dependent real transitions followed by exactly one decision-relevant and one irrelevant counterfactual.",
                "Do not invent patient facts, completed tools, treatment administration, memory IDs, guideline claims, or source spans.",
                "Every cited rule_id must identify a provenance-only gpt_guideline_claim.v2 entry grounded to supplied memory IDs and spans.",
                "Omit claim_id or make it exactly equal to rule_id; provenance-only claims must never contain action_templates.",
                "Use the exact condition DSL key operator (never op) with only eq, ne, in, not_in, exists, missing, contains, gte and lte, or all/any/not composites, against real patient_state.v2 fields.",
                "Set state_before/state_after disease_subtype to npc; record WHO histology in known_diagnosis or the evidence ledger, not in disease_subtype.",
                "Every record must include at least one positive action and at least one premature/unsafe/unsatisfied-conditional negative action.",
                "Every action preconditions field, including actions copied into accepted_plan_variants, must be an array of condition DSL objects and never natural-language strings.",
                "Do not count guideline retrieval, file availability, ROI availability, bookkeeping fields, or a repeated report summary as clinical state progress; each transition must add a patient-specific finding, resolve an unresolved item, or make a supported clinical decision.",
                "Every expected_state_delta must name only fields the action can actually change, and every non-bookkeeping state_delta field must be declared by an accepted action; do not declare deltas that never occur merely to satisfy validation.",
                "Treatment-selection conditions must encode every material evidence gate named by the objective (confirmed diagnosis, stage, pathology/risk factors, biomarkers and fitness when applicable); a broad exists check must not accept unknown, pending, not_reported or placeholder values.",
                "Each counterfactual is a separate trajectory whose turn_index is 0 and whose counterfactual anchor points to an existing real trajectory_id/turn_index in this same output.",
                "When should_stop is true, actions must be empty. Every action should normally cite one supporting memory whose own source_span_ids contain every cited source span.",
                "Return JSON only. Do not store hidden reasoning or prose outside the structured findings fields.",
                "Before finishing, run validate-gpt-planner-output for this packet/output and fix every reported error.",
            ],
            "expected_output": {
                "path": f"teacher_outputs/{case_id}.json",
                "envelope": {
                    "schema_version": TEACHER_OUTPUT_SCHEMA_VERSION,
                    "case_id": case_id,
                    "request_hash": "copy this packet request_hash",
                    "model": model,
                    "model_version": "exact runtime model identifier",
                    "generated_at": "UTC timestamp",
                    "records": "non-empty planner_trajectory.v2 array",
                    "claims": "non-empty provenance-only gpt_guideline_claim.v2 array",
                },
            },
            "excluded_inputs": [
                "evaluation/*rubric*",
                "evaluation/*trajectory*",
                "guideline/relevant_nodes.json",
                "hidden_state.json",
                "follow-up/outcome data",
            ],
        }
        _write_packet(teacher_path, packet, force=force)

    for case_id, records in sorted(grouped_existing.items()):
        review_path = review_dir / f"{case_id}.json"
        review_output_path = cfg.review_workspace / "review_outputs" / f"{case_id}.json"
        if review_path.is_file() and _case_output_is_ready(
            cfg,
            review_path,
            review_output_path,
        ):
            continue
        family = str(records[0]["state_before"]["cancer_family"])
        case_root = cfg.lung_case_root if family == "lung" else cfg.ucec_case_root
        case_dir = case_root / case_id
        guideline_pairs = {
            (str(item["guideline_id"]), str(item["version"]))
            for row in records
            for item in row["guideline_context"]["guidelines"]
        }
        review_chunks = [
            chunk
            for guideline_id, version in sorted(guideline_pairs)
            for chunk in _guideline_chunks(chunks, guideline_id, version)
        ]
        packet = {
            "schema_version": PACKET_SCHEMA_VERSION,
            "packet_type": "existing_case_review",
            "pipeline_version": PIPELINE_VERSION,
            "case_id": case_id,
            "base_case_id": case_id,
            "split": str(records[0]["split"]),
            "model": model,
            "reasoning_effort": reasoning_effort,
            "review_contract": {
                "output_schema_version": REVIEW_OUTPUT_SCHEMA_VERSION,
                "allowed_decisions": sorted(_VALID_REVIEW_DECISIONS),
                "review_whole_case": True,
                "maximum_repair_cycles": max_repair_cycles,
                "pass_unchanged_preserves_record_bytes": True,
                "rule_compiler_forbidden_for_repairs": True,
                "claims_are_provenance_only": True,
            },
            "patient_evidence": _collect_case_evidence(case_dir, cfg.repo_root),
            "guideline_chunks": review_chunks,
            "candidate_records": records,
            "quality_checks": _quality_check_names(),
            "task_instructions": [
                "Review all six records for this case together in an independent context.",
                "Use only supplied patient evidence and guideline chunks; do not infer facts from old model provenance.",
                "Choose pass_unchanged only if current content needs no clinical or supervision correction.",
                "Choose repair_required and list concrete findings when any state, action bucket, counterfactual, routing label, or citation needs repair.",
                "Choose unusable only when available evidence cannot support an admissible trajectory.",
                "Return JSON only and do not store hidden reasoning.",
                "Before finishing, run validate-gpt-planner-output for this packet/output and fix every reported error.",
            ],
            "expected_output": {
                "path": f"review_outputs/{case_id}.json",
                "envelope": {
                    "schema_version": REVIEW_OUTPUT_SCHEMA_VERSION,
                    "case_id": case_id,
                    "request_hash": "copy this packet request_hash",
                    "model": model,
                    "decision": "pass_unchanged | repair_required | unusable",
                    "findings": [],
                    "reviewed_at": "UTC timestamp",
                },
            },
        }
        _write_packet(review_path, packet, force=force)

    manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "ready": False,
        "status": "awaiting_independent_gpt_outputs",
        "prepared_at": _utc_now(),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_repair_cycles": max_repair_cycles,
        "seed": cfg.seed,
        "source_dataset_hash": _source_dataset_hash(
            cfg.source_dataset_root, existing_rows
        ),
        "guideline_chunks_hash": sha256_path(cfg.guideline_chunks_path),
        "memory_catalog_hash": sha256_path(cfg.memory_dir),
        "npc_case_count": len(npc_cases),
        "existing_case_count": len(grouped_existing),
        "existing_record_count": len(existing_rows),
        "npc_split_case_counts": dict(sorted(Counter(npc_splits.values()).items())),
        "teacher_packet_hash": sha256_path(teacher_dir),
        "review_packet_hash": sha256_path(review_dir),
        "config": cfg.public_dict(),
    }
    write_json(cfg.npc_workspace / "preparation_manifest.json", manifest)
    write_json(cfg.review_workspace / "preparation_manifest.json", manifest)
    return manifest


def approve_gpt_trajectory_data_manually(
    config: str | Path | Mapping[str, Any] | GPTTrajectoryPipelineConfig,
    *,
    reviewer_id: str = "professional-clinician-review-team",
    reviewed_at: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Bind a complete professional review decision to the current inputs.

    The approval is content-addressed.  It never fabricates GPT verifier output,
    and any later change to an existing trajectory, NPC teacher response, or
    teacher packet invalidates the approval before validation or merging.
    """

    cfg = load_gpt_trajectory_pipeline_config(config)
    reviewer = reviewer_id.strip()
    if not reviewer:
        raise ValueError("reviewer_id must be non-empty.")
    timestamp = reviewed_at or _utc_now()
    if not _is_utc_timestamp(timestamp):
        raise ValueError("reviewed_at must be an ISO-8601 UTC timestamp.")
    approval_inputs, statistics = _manual_approval_inputs(cfg)
    approval_input_hash = sha256_json(approval_inputs)
    current = _read_json_optional(cfg.manual_approval_path)
    if current:
        unchanged = (
            current.get("approval_input_hash") == approval_input_hash
            and current.get("reviewer_id") == reviewer
            and current.get("ready") is True
        )
        if unchanged:
            return current
        if not force:
            raise FileExistsError(
                "Manual approval exists for different content or reviewer: "
                f"{cfg.manual_approval_path}. Pass --force only after the changed "
                "content has been reviewed again."
            )
    manifest = {
        "schema_version": MANUAL_APPROVAL_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "ready": True,
        "approval_type": "professional_clinician_manual_review",
        "reviewer_id": reviewer,
        "reviewed_at": timestamp,
        "approval_input_hash": approval_input_hash,
        "statistics": statistics,
        "approved_inputs": approval_inputs,
    }
    write_json(cfg.manual_approval_path, manifest)
    return manifest


def _manual_approval_inputs(
    cfg: GPTTrajectoryPipelineConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the exact per-case hashes covered by a manual approval."""

    _require_sources(cfg)
    existing_rows = _load_existing_rows(cfg.source_dataset_root)
    existing_by_case = _group_existing_records(existing_rows)
    npc_ids = _packet_ids(cfg.npc_workspace / "teacher_packets")
    if len(npc_ids) != cfg.expected_npc_cases:
        raise V2SchemaError(
            f"NPC teacher packet count is {len(npc_ids)}; expected {cfg.expected_npc_cases}."
        )
    if len(existing_by_case) != cfg.expected_existing_cases:
        raise V2SchemaError(
            f"Existing case count is {len(existing_by_case)}; expected {cfg.expected_existing_cases}."
        )
    if len(existing_rows) != cfg.expected_existing_records:
        raise V2SchemaError(
            f"Existing record count is {len(existing_rows)}; expected {cfg.expected_existing_records}."
        )
    invalid_existing = {
        case_id: len(rows)
        for case_id, rows in existing_by_case.items()
        if len(rows) != 6
    }
    if invalid_existing:
        raise V2SchemaError(
            f"Existing cases must each contain six records; invalid={invalid_existing}."
        )

    npc_entries: dict[str, Any] = {}
    npc_record_count = 0
    for case_id in npc_ids:
        packet_path = cfg.npc_workspace / "teacher_packets" / f"{case_id}.json"
        output_path = cfg.npc_workspace / "teacher_outputs" / f"{case_id}.json"
        validation = _case_output_validation(cfg, packet_path, output_path)
        if not validation["ready"]:
            raise V2SchemaError(
                f"NPC {case_id} cannot be manually approved: "
                + "; ".join(validation["errors"][:6])
            )
        packet = _read_json(packet_path)
        output = _read_json(output_path)
        records = _normalize_generated_records(_output_records(output))
        claims = _output_claims(output)
        npc_record_count += len(records)
        npc_entries[case_id] = {
            "packet_request_hash": packet.get("request_hash"),
            "packet_hash": sha256_json(packet),
            "teacher_output_hash": sha256_json(output),
            "normalized_records_hash": _records_hash(records),
            "claim_hash": sha256_json(
                sorted(claims, key=lambda item: str(item.get("rule_id")))
            ),
            "record_count": len(records),
        }

    existing_entries = {
        case_id: {
            "records_hash": _records_hash(rows),
            "record_count": len(rows),
        }
        for case_id, rows in sorted(existing_by_case.items())
    }
    inputs = {
        "source_dataset_hash": _source_dataset_hash(
            cfg.source_dataset_root, existing_rows
        ),
        "existing_cases": existing_entries,
        "npc_cases": npc_entries,
    }
    statistics = {
        "case_count": len(existing_entries) + len(npc_entries),
        "existing_case_count": len(existing_entries),
        "existing_record_count": len(existing_rows),
        "npc_case_count": len(npc_entries),
        "npc_record_count": npc_record_count,
        "record_count": len(existing_rows) + npc_record_count,
    }
    return inputs, statistics


def _valid_manual_approval(
    cfg: GPTTrajectoryPipelineConfig,
) -> tuple[dict[str, Any] | None, list[str]]:
    manifest = _read_json_optional(cfg.manual_approval_path)
    if manifest is None:
        return None, []
    errors: list[str] = []
    if manifest.get("schema_version") != MANUAL_APPROVAL_SCHEMA_VERSION:
        errors.append("Manual approval schema_version is invalid.")
    if manifest.get("ready") is not True:
        errors.append("Manual approval is not marked ready.")
    if not str(manifest.get("reviewer_id") or "").strip():
        errors.append("Manual approval reviewer_id is missing.")
    if not _is_utc_timestamp(manifest.get("reviewed_at")):
        errors.append("Manual approval reviewed_at is not a UTC timestamp.")
    try:
        inputs, _ = _manual_approval_inputs(cfg)
    except (OSError, ValueError, TypeError, KeyError, V2SchemaError) as exc:
        errors.append(f"Manual approval inputs cannot be reconstructed: {exc}")
    else:
        if manifest.get("approval_input_hash") != sha256_json(inputs):
            errors.append(
                "Manual approval input hash does not match the current reviewed content."
            )
    return (manifest if not errors else None), errors


def get_gpt_trajectory_status(
    config: str | Path | Mapping[str, Any] | GPTTrajectoryPipelineConfig,
    *,
    max_repair_cycles: int = 2,
) -> dict[str, Any]:
    """Return packet/output progress without changing any artifact."""

    cfg = load_gpt_trajectory_pipeline_config(config)
    npc_ids = _packet_ids(cfg.npc_workspace / "teacher_packets")
    existing_ids = _packet_ids(cfg.review_workspace / "review_packets")
    manual_approval, manual_errors = _valid_manual_approval(cfg)
    if manual_approval:
        npc = {
            "packet_count": len(npc_ids),
            "resolved_cases": len(npc_ids),
            "counts": {"manually_approved": len(npc_ids)},
            "pending": {},
        }
        existing = {
            "packet_count": len(existing_ids),
            "resolved_cases": len(existing_ids),
            "counts": {"manually_approved": len(existing_ids)},
            "pending": {},
        }
    else:
        npc = _status_for_npc(cfg, npc_ids, max_repair_cycles=max_repair_cycles)
        existing = _status_for_existing(
            cfg, existing_ids, max_repair_cycles=max_repair_cycles
        )
    merged_manifest = _read_json_optional(
        cfg.merged_dataset_root / "generation_manifest.json"
    )
    return {
        "schema_version": "gpt_trajectory_status.v2",
        "pipeline_version": PIPELINE_VERSION,
        "npc": npc,
        "existing": existing,
        "manual_approval": {
            "exists": cfg.manual_approval_path.is_file(),
            "valid": manual_approval is not None,
            "path": str(cfg.manual_approval_path),
            "reviewer_id": (manual_approval or {}).get("reviewer_id"),
            "reviewed_at": (manual_approval or {}).get("reviewed_at"),
            "errors": manual_errors,
        },
        "merged": {
            "exists": merged_manifest is not None,
            "ready": bool(merged_manifest and merged_manifest.get("ready")),
            "record_count": int((merged_manifest or {}).get("record_count", 0)),
            "dataset_hash": (merged_manifest or {}).get("dataset_hash"),
        },
        "ready_to_validate": bool(npc_ids)
        and bool(existing_ids)
        and npc["resolved_cases"] == len(npc_ids)
        and existing["resolved_cases"] == len(existing_ids),
    }


def validate_gpt_case_output(
    packet_path: str | Path,
    output_path: str | Path,
    config: str | Path | Mapping[str, Any] | GPTTrajectoryPipelineConfig,
) -> dict[str, Any]:
    """Validate one model response before it is admitted to a follow-up queue."""

    cfg = load_gpt_trajectory_pipeline_config(config)
    packet_file = Path(packet_path)
    output_file = Path(output_path)
    packet = _read_json(packet_file)
    output = _read_json(output_file)
    packet_type = str(packet.get("packet_type") or "")
    case_id = str(packet.get("case_id") or "")
    if packet_type == "npc_teacher":
        label = "teacher"
    elif packet_type == "existing_case_review":
        label = "review"
    elif "repair" in packet_type and "verifier" not in packet_type:
        label = f"repair-{packet.get('repair_cycle', 1)}"
    else:
        label = "verifier"
    errors = _validate_model_output_envelope(output, packet, case_id, label)

    if label == "review":
        decision = str(output.get("decision") or "")
        findings = output.get("findings")
        if not _is_utc_timestamp(output.get("reviewed_at")):
            errors.append(f"{case_id}/review: reviewed_at must be a UTC timestamp.")
        if decision not in _VALID_REVIEW_DECISIONS:
            errors.append(f"{case_id}/review: invalid decision {decision!r}.")
        if not isinstance(findings, list):
            errors.append(f"{case_id}/review: findings must be an array.")
        elif decision == "pass_unchanged" and findings:
            errors.append(f"{case_id}/review: pass_unchanged requires empty findings.")
        elif decision in {"repair_required", "unusable"} and not findings:
            errors.append(f"{case_id}/review: {decision} requires concrete findings.")
    elif label == "verifier":
        decision = str(output.get("decision") or "")
        findings = output.get("findings")
        if not _is_utc_timestamp(output.get("reviewed_at")):
            errors.append(f"{case_id}/verifier: reviewed_at must be a UTC timestamp.")
        if decision not in _VALID_VERIFY_DECISIONS:
            errors.append(f"{case_id}/verifier: invalid decision {decision!r}.")
        if not isinstance(findings, list):
            errors.append(f"{case_id}/verifier: findings must be an array.")
        elif decision == "approved" and findings:
            errors.append(f"{case_id}/verifier: approved requires empty findings.")
        elif decision != "approved" and not findings:
            errors.append(
                f"{case_id}/verifier: a non-approved decision requires findings."
            )
    else:
        if not str(output.get("model_version") or ""):
            errors.append(f"{case_id}/{label}: model_version is required.")
        if not _is_utc_timestamp(output.get("generated_at")):
            errors.append(f"{case_id}/{label}: generated_at must be a UTC timestamp.")
        rows = _normalize_generated_records(_output_records(output))
        claims = _output_claims(output)
        if not rows:
            errors.append(f"{case_id}/{label}: records must be non-empty.")
        if not claims and packet_type.startswith("npc"):
            errors.append(
                f"{case_id}/{label}: NPC output requires GPT guideline claims."
            )
        errors.extend(_validate_case_output_identity(case_id, packet, rows))
        for index, row in enumerate(rows):
            try:
                validate_trajectory_record_v2(row)
            except V2SchemaError as exc:
                errors.append(f"{case_id}/{label}/record[{index}]: {exc}")
        local_audit = audit_trajectory_dataset_v2(
            rows,
            release_gates=False,
            release_scope=_three_cancer_release_scope(),
            require_test_approval=False,
        )
        errors.extend(local_audit.errors)
        errors.extend(
            _validate_provenance_only_claims(
                rows, claims, cfg.memory_dir, cfg.legacy_rule_registry_path
            )
        )
        with tempfile.TemporaryDirectory(prefix="planner-gpt-grounding-") as temp_dir:
            registry = Path(temp_dir) / "grounding_registry.v2.jsonl"
            write_jsonl(
                registry,
                [*read_jsonl(cfg.legacy_rule_registry_path), *claims],
            )
            errors.extend(
                validate_grounding_assets(
                    rows,
                    rule_registry_path=registry,
                    memory_dir=cfg.memory_dir,
                )
            )
        if packet_type.startswith("npc"):
            real_count = sum(row.get("case_source") == "real" for row in rows)
            counterfactuals = Counter(
                str((row.get("counterfactual") or {}).get("perturbation_type") or "")
                for row in rows
                if row.get("case_source") == "counterfactual"
            )
            if not cfg.npc_min_real_turns <= real_count <= cfg.npc_max_real_turns:
                errors.append(
                    f"{case_id}/{label}: requires {cfg.npc_min_real_turns}-"
                    f"{cfg.npc_max_real_turns} real transitions; got {real_count}."
                )
            expected_cf = Counter({"decision_relevant": 1, "irrelevant": 1})
            if counterfactuals != expected_cf:
                errors.append(
                    f"{case_id}/{label}: counterfactual counts are "
                    f"{dict(counterfactuals)}, expected {dict(expected_cf)}."
                )

    result = {
        "schema_version": "gpt_case_output_validation.v2",
        "ready": not errors,
        "case_id": case_id,
        "packet_type": packet_type,
        "packet": str(packet_file),
        "output": str(output_file),
        "errors": list(dict.fromkeys(errors)),
    }
    return result


def validate_gpt_trajectory_outputs(
    config: str | Path | Mapping[str, Any] | GPTTrajectoryPipelineConfig,
    *,
    max_repair_cycles: int = 2,
    write_report: bool = True,
) -> dict[str, Any]:
    """Validate all independent decisions and resolved V2 records.

    This is deliberately stricter than packet status: any missing case, invalid
    hash, unresolved repair, unusable decision, or grounding error makes the
    report non-ready.
    """

    cfg = load_gpt_trajectory_pipeline_config(config)
    errors: list[str] = []
    warnings: list[str] = []
    npc_ids = _packet_ids(cfg.npc_workspace / "teacher_packets")
    existing_ids = _packet_ids(cfg.review_workspace / "review_packets")
    if len(npc_ids) != cfg.expected_npc_cases:
        errors.append(
            f"NPC packet count is {len(npc_ids)}; expected {cfg.expected_npc_cases}."
        )
    if len(existing_ids) != cfg.expected_existing_cases:
        errors.append(
            f"Existing review packet count is {len(existing_ids)}; expected {cfg.expected_existing_cases}."
        )

    resolved_npc, npc_claims, npc_stats, npc_errors = _resolve_npc_cases(
        cfg, npc_ids, max_repair_cycles=max_repair_cycles
    )
    resolved_existing, existing_claims, annotations, existing_stats, existing_errors = (
        _resolve_existing_cases(cfg, existing_ids, max_repair_cycles=max_repair_cycles)
    )
    errors.extend(npc_errors)
    errors.extend(existing_errors)
    all_records = [*resolved_existing, *resolved_npc]
    all_claims = [*existing_claims, *npc_claims]
    errors.extend(_validate_record_id_uniqueness(all_records))
    errors.extend(_validate_case_record_contracts(cfg, resolved_npc, resolved_existing))

    schema_valid = 0
    for index, row in enumerate(all_records):
        try:
            validate_trajectory_record_v2(row)
            schema_valid += 1
        except V2SchemaError as exc:
            errors.append(f"resolved record[{index}] is invalid: {exc}")
    errors.extend(
        _validate_provenance_only_claims(
            all_records,
            all_claims,
            cfg.memory_dir,
            cfg.legacy_rule_registry_path,
        )
    )
    try:
        grounding_claims = _deduplicate_claims(all_claims)
    except V2SchemaError as exc:
        errors.append(str(exc))
        grounding_claims = []
    if all_records:
        with tempfile.TemporaryDirectory(prefix="planner-gpt-grounding-") as temporary:
            registry_path = Path(temporary) / "grounding_registry.v2.jsonl"
            write_jsonl(
                registry_path,
                [*read_jsonl(cfg.legacy_rule_registry_path), *grounding_claims],
            )
            errors.extend(
                validate_grounding_assets(
                    all_records,
                    rule_registry_path=registry_path,
                    memory_dir=cfg.memory_dir,
                )
            )
    quality_findings = _quality_scan(all_records)
    manual_approval, manual_approval_errors = _valid_manual_approval(cfg)
    errors.extend(manual_approval_errors)
    if manual_approval:
        warnings.extend(
            f"Clinician-reviewed quality warning: {finding}"
            for finding in quality_findings
        )
    else:
        errors.extend(quality_findings)

    release_scope = _three_cancer_release_scope()
    audit = audit_trajectory_dataset_v2(
        all_records, release_gates=True, release_scope=release_scope
    )
    errors.extend(audit.errors)
    report = {
        "schema_version": "gpt_trajectory_validation_report.v2",
        "pipeline_version": PIPELINE_VERSION,
        "validated_at": _utc_now(),
        "ready": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "statistics": {
            "record_count": len(all_records),
            "schema_valid_count": schema_valid,
            "case_count": len({str(row["base_case_id"]) for row in all_records}),
            "npc": npc_stats,
            "existing": existing_stats,
            "claim_count": len(all_claims),
            "annotation_count": len(annotations),
            "admission": audit.statistics,
        },
        "resolved_records_hash": _records_hash(all_records),
        "claims_hash": sha256_json(
            sorted(all_claims, key=lambda item: str(item.get("rule_id")))
        ),
        "release_scope": release_scope,
    }
    if write_report:
        write_json(
            cfg.merged_dataset_root.parent / "planner_v2_gpt_validation_report.json",
            report,
        )
    return report


def merge_gpt_trajectory_dataset(
    config: str | Path | Mapping[str, Any] | GPTTrajectoryPipelineConfig,
    *,
    max_repair_cycles: int = 2,
    force: bool = False,
) -> dict[str, Any]:
    """Assemble a new three-cancer release after all GPT checks pass."""

    cfg = load_gpt_trajectory_pipeline_config(config)
    report = validate_gpt_trajectory_outputs(
        cfg, max_repair_cycles=max_repair_cycles, write_report=True
    )
    if not report["ready"]:
        raise V2SchemaError(
            "GPT trajectory data failed validation: " + "; ".join(report["errors"][:12])
        )
    npc_ids = _packet_ids(cfg.npc_workspace / "teacher_packets")
    existing_ids = _packet_ids(cfg.review_workspace / "review_packets")
    npc_records, npc_claims, npc_stats, _ = _resolve_npc_cases(
        cfg, npc_ids, max_repair_cycles=max_repair_cycles
    )
    existing_records, existing_claims, annotations, existing_stats, _ = (
        _resolve_existing_cases(cfg, existing_ids, max_repair_cycles=max_repair_cycles)
    )
    all_records = [*existing_records, *npc_records]
    claims = _deduplicate_claims([*existing_claims, *npc_claims])
    input_hash = sha256_json(
        {
            "records": _records_hash(all_records),
            "claims": claims,
            "annotations": annotations,
            "scope": _three_cancer_release_scope(),
        }
    )
    old_manifest = _read_json_optional(
        cfg.merged_dataset_root / "generation_manifest.json"
    )
    if old_manifest:
        if old_manifest.get("merge_input_hash") == input_hash and old_manifest.get(
            "ready"
        ):
            return old_manifest
        if not force:
            raise FileExistsError(
                f"Merged dataset already exists with different inputs: {cfg.merged_dataset_root}. "
                "Pass force=True to replace only generated artifacts."
            )

    cfg.merged_dataset_root.mkdir(parents=True, exist_ok=True)
    candidates_path = cfg.merged_dataset_root / "planner_trajectory.v2.candidates.jsonl"
    claims_path = cfg.merged_dataset_root / "gpt_guideline_claims.v2.jsonl"
    combined_registry_path = cfg.merged_dataset_root / "grounding_registry.v2.jsonl"
    annotations_path = cfg.merged_dataset_root / "gpt_review_annotations.jsonl"
    revision_diffs_path = cfg.merged_dataset_root / "case_revision_diffs.jsonl"
    normalized_cases_dir = cfg.merged_dataset_root / "normalized_case_outputs"
    scope_path = cfg.merged_dataset_root / "release_scope.json"
    chunks_path = cfg.merged_dataset_root / "guideline_chunks.jsonl"
    write_jsonl(candidates_path, all_records)
    write_jsonl(claims_path, claims)
    write_jsonl(
        combined_registry_path,
        [*read_jsonl(cfg.legacy_rule_registry_path), *claims],
    )
    write_jsonl(annotations_path, annotations)
    claims_by_id = {str(item["rule_id"]): item for item in claims}
    records_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_records:
        records_by_case[str(row["base_case_id"])].append(row)
    for case_id, case_records in sorted(records_by_case.items()):
        used_claim_ids = {
            str(rule_id)
            for row in case_records
            for actions in row["action_set"].values()
            for action in actions
            for rule_id in action["guideline_rule_ids"]
            if str(rule_id) in claims_by_id
        }
        write_json(
            normalized_cases_dir / f"{case_id}.json",
            {
                "schema_version": "gpt_normalized_case_output.v2",
                "case_id": case_id,
                "records": sorted(case_records, key=_record_key),
                "claims": [claims_by_id[item] for item in sorted(used_claim_ids)],
            },
        )
    write_json(scope_path, _three_cancer_release_scope())
    shutil.copy2(cfg.guideline_chunks_path, chunks_path)

    # The grounding layer recognizes provenance-only GPT claims alongside
    # legacy compilable rules.  No GPT claim contains an action template.
    dataset_manifest = build_trajectory_dataset_v2(
        all_records,
        cfg.merged_dataset_root / "dataset",
        seed=cfg.seed,
        release_gates=True,
        fail_if_not_ready=True,
        rule_registry_path=combined_registry_path,
        memory_dir=cfg.memory_dir,
        release_scope=scope_path,
    )
    original_rows = _load_existing_rows(cfg.source_dataset_root)
    revision_diffs = _existing_revision_diffs(original_rows, existing_records)
    write_jsonl(revision_diffs_path, revision_diffs)
    original_by_key = {_record_key(row): row for row in original_rows}
    unchanged = sum(
        _record_key(row) in original_by_key and original_by_key[_record_key(row)] == row
        for row in existing_records
    )
    manual_approval, manual_approval_errors = _valid_manual_approval(cfg)
    if manual_approval_errors:
        raise V2SchemaError("; ".join(manual_approval_errors))
    if manual_approval:
        human_reviewed_count = len(all_records)
        model_reviewed_only_count = 0
    else:
        human_reviewed_count = unchanged
        model_reviewed_only_count = len(all_records) - unchanged
    generation_manifest = {
        "schema_version": "gpt_trajectory_generation_manifest.v2",
        "pipeline_version": PIPELINE_VERSION,
        "ready": True,
        "generated_at": _utc_now(),
        "merge_input_hash": input_hash,
        "source_dataset_root": str(cfg.source_dataset_root),
        "source_dataset_hash": _source_dataset_hash(
            cfg.source_dataset_root, original_rows
        ),
        "record_count": len(all_records),
        "case_count": len({str(row["base_case_id"]) for row in all_records}),
        "npc": npc_stats,
        "existing": existing_stats,
        "review_summary": {
            "pass_unchanged_record_count": unchanged,
            "revised_existing_record_count": len(existing_records) - unchanged,
            "new_npc_record_count": len(npc_records),
            "failed_case_count": 0,
            "human_reviewed_current_content_count": human_reviewed_count,
            "model_reviewed_only_count": model_reviewed_only_count,
            "revised_case_diff_count": len(revision_diffs),
        },
        "manual_approval": (
            {
                "path": str(cfg.manual_approval_path),
                "reviewer_id": manual_approval["reviewer_id"],
                "reviewed_at": manual_approval["reviewed_at"],
                "approval_input_hash": manual_approval["approval_input_hash"],
            }
            if manual_approval
            else None
        ),
        "gpt_claim_count": len(claims),
        "release_scope": _three_cancer_release_scope(),
        "validation_report": report,
        "dataset_manifest": dataset_manifest,
        "artifacts": {
            "candidates": str(candidates_path),
            "claims": str(claims_path),
            "grounding_registry": str(combined_registry_path),
            "review_annotations": str(annotations_path),
            "case_revision_diffs": str(revision_diffs_path),
            "normalized_case_outputs": str(normalized_cases_dir),
            "release_scope": str(scope_path),
            "guideline_chunks": str(chunks_path),
            "dataset": str(cfg.merged_dataset_root / "dataset"),
        },
    }
    write_json(
        cfg.merged_dataset_root / "generation_manifest.json", generation_manifest
    )
    return generation_manifest


def run_gpt_trajectory_stage(
    stage: str,
    config_path: str | Path | Mapping[str, Any] | GPTTrajectoryPipelineConfig,
    *,
    model: str = "gpt-5.6-sol",
    reasoning_effort: str = "high",
    max_parallel: int = 3,
    max_repair_cycles: int = 2,
    reviewer_id: str = "professional-clinician-review-team",
    reviewed_at: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Stable stage dispatcher used by the CLI and orchestration scripts.

    LLM execution is external by design.  ``generate-npc``, ``review-existing``,
    ``repair`` and ``verify`` materialize the next case-isolated packets and
    return pending work; they never silently substitute deterministic labels.
    """

    if max_parallel < 1 or max_parallel > 3:
        raise ValueError("max_parallel must be between 1 and 3.")
    if max_repair_cycles < 0:
        raise ValueError("max_repair_cycles must be non-negative.")
    normalized = stage.strip().lower().replace("_", "-")
    if normalized == "prepare":
        return prepare_gpt_trajectory_packets(
            config_path,
            model=model,
            reasoning_effort=reasoning_effort,
            max_repair_cycles=max_repair_cycles,
            force=force,
        )
    if normalized == "approve-manual":
        return approve_gpt_trajectory_data_manually(
            config_path,
            reviewer_id=reviewer_id,
            reviewed_at=reviewed_at,
            force=force,
        )
    if normalized == "status":
        return get_gpt_trajectory_status(
            config_path, max_repair_cycles=max_repair_cycles
        )
    if normalized == "validate":
        return validate_gpt_trajectory_outputs(
            config_path, max_repair_cycles=max_repair_cycles, write_report=True
        )
    if normalized == "merge":
        return merge_gpt_trajectory_dataset(
            config_path, max_repair_cycles=max_repair_cycles, force=force
        )
    if normalized in {"generate-npc", "review-existing", "repair", "verify"}:
        cfg = load_gpt_trajectory_pipeline_config(config_path)
        _materialize_followup_packets(
            cfg,
            model=model,
            reasoning_effort=reasoning_effort,
            max_repair_cycles=max_repair_cycles,
            force=force,
        )
        status = get_gpt_trajectory_status(cfg, max_repair_cycles=max_repair_cycles)
        status["requested_stage"] = normalized
        status["max_parallel"] = max_parallel
        status["execution"] = "external_codex_tasks_required"
        return status
    raise ValueError(
        "Unknown GPT trajectory stage. Expected prepare, generate-npc, review-existing, "
        "repair, verify, approve-manual, validate, merge, or status."
    )


def _materialize_followup_packets(
    cfg: GPTTrajectoryPipelineConfig,
    *,
    model: str,
    reasoning_effort: str,
    max_repair_cycles: int,
    force: bool,
) -> None:
    """Create verifier/repair packets only after their independent inputs exist."""

    for case_id in _packet_ids(cfg.npc_workspace / "teacher_packets"):
        teacher_packet_path = cfg.npc_workspace / "teacher_packets" / f"{case_id}.json"
        teacher_output_path = cfg.npc_workspace / "teacher_outputs" / f"{case_id}.json"
        teacher_packet = _read_json(teacher_packet_path)
        teacher = _read_json_optional(teacher_output_path)
        if teacher is None or not _case_output_is_ready(
            cfg, teacher_packet_path, teacher_output_path
        ):
            continue
        _write_followup_packet(
            cfg.npc_workspace / "verifier_packets" / f"{case_id}.json",
            packet_type="npc_verifier",
            case_id=case_id,
            model=model,
            reasoning_effort=reasoning_effort,
            source_packet=teacher_packet,
            candidate_output=teacher,
            force=force,
        )
        primary_verifier = _read_json_optional(
            cfg.npc_workspace / "verifier_outputs" / f"{case_id}.json"
        )
        primary_verifier_output_path = (
            cfg.npc_workspace / "verifier_outputs" / f"{case_id}.json"
        )
        if primary_verifier is None or not _case_output_is_ready(
            cfg,
            cfg.npc_workspace / "verifier_packets" / f"{case_id}.json",
            primary_verifier_output_path,
        ):
            continue
        if primary_verifier.get("decision") == "approved":
            continue
        candidate: Mapping[str, Any] = teacher
        prior_verifier: Mapping[str, Any] = primary_verifier
        for cycle in range(1, max_repair_cycles + 1):
            repair_packet_path = (
                cfg.npc_workspace / "repair_packets" / case_id / f"cycle-{cycle}.json"
            )
            _write_followup_packet(
                repair_packet_path,
                packet_type="npc_case_repair",
                case_id=case_id,
                model=model,
                reasoning_effort=reasoning_effort,
                source_packet=teacher_packet,
                candidate_output=candidate,
                force=force,
                extra={
                    "repair_cycle": cycle,
                    "prior_verifier": dict(prior_verifier),
                    "quality_checks": _quality_check_names(),
                },
            )
            repair = _read_json_optional(
                cfg.npc_workspace / "repair_outputs" / case_id / f"cycle-{cycle}.json"
            )
            repair_output_path = (
                cfg.npc_workspace / "repair_outputs" / case_id / f"cycle-{cycle}.json"
            )
            if repair is None or not _case_output_is_ready(
                cfg, repair_packet_path, repair_output_path
            ):
                break
            verifier_path = (
                cfg.npc_workspace / "verifier_packets" / case_id / f"cycle-{cycle}.json"
            )
            _write_followup_packet(
                verifier_path,
                packet_type="npc_repair_verifier",
                case_id=case_id,
                model=model,
                reasoning_effort=reasoning_effort,
                source_packet=repair_packet_path,
                candidate_output=repair,
                force=force,
                extra={"repair_cycle": cycle},
            )
            verifier = _read_json_optional(
                cfg.npc_workspace / "verifier_outputs" / case_id / f"cycle-{cycle}.json"
            )
            verifier_output_path = (
                cfg.npc_workspace / "verifier_outputs" / case_id / f"cycle-{cycle}.json"
            )
            if verifier is None or not _case_output_is_ready(
                cfg, verifier_path, verifier_output_path
            ):
                break
            if verifier.get("decision") == "approved":
                break
            candidate = repair
            prior_verifier = verifier
    for case_id in _packet_ids(cfg.review_workspace / "review_packets"):
        review_packet_path = cfg.review_workspace / "review_packets" / f"{case_id}.json"
        review_output_path = cfg.review_workspace / "review_outputs" / f"{case_id}.json"
        review_packet = _read_json(review_packet_path)
        review = _read_json_optional(review_output_path)
        if review is None or not _case_output_is_ready(
            cfg, review_packet_path, review_output_path
        ):
            continue
        decision = str(review.get("decision") or "")
        if decision != "repair_required":
            continue
        candidate: Mapping[str, Any] = {
            "records": review_packet.get("candidate_records") or [],
            "claims": [],
        }
        prior_verifier: Mapping[str, Any] = review
        for cycle in range(1, max_repair_cycles + 1):
            repair_packet_path = (
                cfg.review_workspace
                / "repair_packets"
                / case_id
                / f"cycle-{cycle}.json"
            )
            _write_followup_packet(
                repair_packet_path,
                packet_type="existing_case_repair",
                case_id=case_id,
                model=model,
                reasoning_effort=reasoning_effort,
                source_packet=review_packet,
                candidate_output=candidate,
                force=force,
                extra={
                    "repair_cycle": cycle,
                    "prior_verifier": dict(prior_verifier),
                    "quality_checks": _quality_check_names(),
                },
            )
            repair = _read_json_optional(
                cfg.review_workspace
                / "repair_outputs"
                / case_id
                / f"cycle-{cycle}.json"
            )
            repair_output_path = (
                cfg.review_workspace
                / "repair_outputs"
                / case_id
                / f"cycle-{cycle}.json"
            )
            if repair is None or not _case_output_is_ready(
                cfg, repair_packet_path, repair_output_path
            ):
                break
            verifier_path = (
                cfg.review_workspace
                / "verifier_packets"
                / case_id
                / f"cycle-{cycle}.json"
            )
            _write_followup_packet(
                verifier_path,
                packet_type="existing_repair_verifier",
                case_id=case_id,
                model=model,
                reasoning_effort=reasoning_effort,
                source_packet=repair_packet_path,
                candidate_output=repair,
                force=force,
                extra={"repair_cycle": cycle},
            )
            verifier = _read_json_optional(
                cfg.review_workspace
                / "verifier_outputs"
                / case_id
                / f"cycle-{cycle}.json"
            )
            verifier_output_path = (
                cfg.review_workspace
                / "verifier_outputs"
                / case_id
                / f"cycle-{cycle}.json"
            )
            if verifier is None or not _case_output_is_ready(
                cfg, verifier_path, verifier_output_path
            ):
                break
            if verifier.get("decision") == "approved":
                break
            candidate = repair
            prior_verifier = verifier


def _resolve_npc_cases(
    cfg: GPTTrajectoryPipelineConfig,
    case_ids: Sequence[str],
    *,
    max_repair_cycles: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str]]:
    records: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    errors: list[str] = []
    approved = 0
    manual_approval, manual_errors = _valid_manual_approval(cfg)
    if manual_errors:
        errors.extend(manual_errors)
        return records, claims, {
            "expected_case_count": cfg.expected_npc_cases,
            "approved_case_count": 0,
            "failed_or_pending_case_count": cfg.expected_npc_cases,
            "record_count": 0,
            "real_transition_count": 0,
            "counterfactual_transition_count": 0,
            "approval_type": "invalid_manual_approval",
        }, errors
    if manual_approval:
        for case_id in case_ids:
            packet_path = cfg.npc_workspace / "teacher_packets" / f"{case_id}.json"
            output_path = cfg.npc_workspace / "teacher_outputs" / f"{case_id}.json"
            packet = _read_json(packet_path)
            output = _read_json(output_path)
            case_records = _normalize_generated_records(_output_records(output))
            errors.extend(_validate_case_output_identity(case_id, packet, case_records))
            records.extend(
                _mark_human_reviewed(
                    case_records,
                    manual_approval,
                    author_output=output,
                    gpt_authored=True,
                )
            )
            claims.extend(_output_claims(output))
            approved += 1
        stats = {
            "expected_case_count": cfg.expected_npc_cases,
            "approved_case_count": approved,
            "failed_or_pending_case_count": cfg.expected_npc_cases - approved,
            "record_count": len(records),
            "real_transition_count": sum(
                row.get("case_source") == "real" for row in records
            ),
            "counterfactual_transition_count": sum(
                row.get("case_source") == "counterfactual" for row in records
            ),
            "approval_type": manual_approval["approval_type"],
            "reviewer_id": manual_approval["reviewer_id"],
        }
        return records, claims, stats, errors
    for case_id in case_ids:
        teacher_packet_path = cfg.npc_workspace / "teacher_packets" / f"{case_id}.json"
        teacher_output_path = cfg.npc_workspace / "teacher_outputs" / f"{case_id}.json"
        packet = _read_json(teacher_packet_path)
        teacher = _read_json_optional(teacher_output_path)
        verifier_packet = _read_json_optional(
            cfg.npc_workspace / "verifier_packets" / f"{case_id}.json"
        )
        verifier = _read_json_optional(
            cfg.npc_workspace / "verifier_outputs" / f"{case_id}.json"
        )
        if teacher is None:
            errors.append(f"NPC {case_id}: teacher output is missing.")
            continue
        teacher_validation = _case_output_validation(
            cfg, teacher_packet_path, teacher_output_path
        )
        if not teacher_validation["ready"]:
            errors.extend(teacher_validation["errors"])
            continue
        if verifier_packet is None or verifier is None:
            errors.append(f"NPC {case_id}: independent verifier output is missing.")
            continue
        verifier_validation = _case_output_validation(
            cfg,
            cfg.npc_workspace / "verifier_packets" / f"{case_id}.json",
            cfg.npc_workspace / "verifier_outputs" / f"{case_id}.json",
        )
        if not verifier_validation["ready"]:
            errors.extend(verifier_validation["errors"])
            continue
        accepted_output: Mapping[str, Any] | None = None
        accepted_verifier: Mapping[str, Any] | None = None
        if verifier.get("decision") == "approved":
            accepted_output = teacher
            accepted_verifier = verifier
        else:
            for cycle in range(1, max_repair_cycles + 1):
                repair_packet = _read_json_optional(
                    cfg.npc_workspace
                    / "repair_packets"
                    / case_id
                    / f"cycle-{cycle}.json"
                )
                repair = _read_json_optional(
                    cfg.npc_workspace
                    / "repair_outputs"
                    / case_id
                    / f"cycle-{cycle}.json"
                )
                cycle_verifier_packet = _read_json_optional(
                    cfg.npc_workspace
                    / "verifier_packets"
                    / case_id
                    / f"cycle-{cycle}.json"
                )
                cycle_verifier = _read_json_optional(
                    cfg.npc_workspace
                    / "verifier_outputs"
                    / case_id
                    / f"cycle-{cycle}.json"
                )
                if not all(
                    (repair_packet, repair, cycle_verifier_packet, cycle_verifier)
                ):
                    continue
                repair_validation = _case_output_validation(
                    cfg,
                    cfg.npc_workspace
                    / "repair_packets"
                    / case_id
                    / f"cycle-{cycle}.json",
                    cfg.npc_workspace
                    / "repair_outputs"
                    / case_id
                    / f"cycle-{cycle}.json",
                )
                cycle_verifier_validation = _case_output_validation(
                    cfg,
                    cfg.npc_workspace
                    / "verifier_packets"
                    / case_id
                    / f"cycle-{cycle}.json",
                    cfg.npc_workspace
                    / "verifier_outputs"
                    / case_id
                    / f"cycle-{cycle}.json",
                )
                if (
                    not repair_validation["ready"]
                    or not cycle_verifier_validation["ready"]
                ):
                    errors.extend(repair_validation["errors"])
                    errors.extend(cycle_verifier_validation["errors"])
                    continue
                if cycle_verifier.get("decision") == "approved":
                    accepted_output = repair
                    accepted_verifier = cycle_verifier
                    break
        if accepted_output is None or accepted_verifier is None:
            errors.append(
                f"NPC {case_id}: no independently approved output within "
                f"{max_repair_cycles} repair cycles."
            )
            continue
        case_records = _normalize_generated_records(_output_records(accepted_output))
        case_claims = _output_claims(accepted_output)
        errors.extend(_validate_case_output_identity(case_id, packet, case_records))
        records.extend(
            _mark_model_reviewed(case_records, accepted_output, accepted_verifier)
        )
        claims.extend(case_claims)
        approved += 1
    stats = {
        "expected_case_count": cfg.expected_npc_cases,
        "approved_case_count": approved,
        "failed_or_pending_case_count": cfg.expected_npc_cases - approved,
        "record_count": len(records),
        "real_transition_count": sum(
            row.get("case_source") == "real" for row in records
        ),
        "counterfactual_transition_count": sum(
            row.get("case_source") == "counterfactual" for row in records
        ),
    }
    return records, claims, stats, errors


def _resolve_existing_cases(
    cfg: GPTTrajectoryPipelineConfig,
    case_ids: Sequence[str],
    *,
    max_repair_cycles: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[str],
]:
    records: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    errors: list[str] = []
    decision_counts: Counter[str] = Counter()
    revised_cases = 0
    resolved_cases = 0
    manual_approval, manual_errors = _valid_manual_approval(cfg)
    if manual_errors:
        errors.extend(manual_errors)
        return records, claims, annotations, {
            "expected_case_count": cfg.expected_existing_cases,
            "resolved_case_count": 0,
            "failed_or_pending_case_count": cfg.expected_existing_cases,
            "decision_counts": {},
            "revised_case_count": 0,
            "record_count": 0,
            "approval_type": "invalid_manual_approval",
        }, errors
    if manual_approval:
        source_by_case = _group_existing_records(
            _load_existing_rows(cfg.source_dataset_root)
        )
        for case_id in case_ids:
            case_records = source_by_case.get(case_id)
            if not case_records:
                errors.append(
                    f"Existing case {case_id}: source records are missing after manual approval."
                )
                continue
            # Preserve the already human-approved source records byte-for-byte.
            # The additional specialist review is recorded as a content-addressed
            # annotation and in the global manual approval manifest.
            records.extend(deepcopy(case_records))
            annotations.append(
                {
                    "schema_version": "gpt_case_review_annotation.v2",
                    "case_id": case_id,
                    "decision": "pass_unchanged",
                    "reviewer_id": manual_approval["reviewer_id"],
                    "reviewed_at": manual_approval["reviewed_at"],
                    "approval_type": manual_approval["approval_type"],
                    "approval_input_hash": manual_approval["approval_input_hash"],
                    "human_reviewed_current_content": True,
                    "findings": [],
                }
            )
            resolved_cases += 1
        stats = {
            "expected_case_count": cfg.expected_existing_cases,
            "resolved_case_count": resolved_cases,
            "failed_or_pending_case_count": cfg.expected_existing_cases
            - resolved_cases,
            "decision_counts": {"manual_pass_unchanged": resolved_cases},
            "revised_case_count": 0,
            "record_count": len(records),
            "approval_type": manual_approval["approval_type"],
            "reviewer_id": manual_approval["reviewer_id"],
        }
        return records, claims, annotations, stats, errors
    for case_id in case_ids:
        review_packet_path = cfg.review_workspace / "review_packets" / f"{case_id}.json"
        review_output_path = cfg.review_workspace / "review_outputs" / f"{case_id}.json"
        packet = _read_json(review_packet_path)
        review = _read_json_optional(review_output_path)
        if review is None:
            errors.append(f"Existing case {case_id}: review output is missing.")
            continue
        review_validation = _case_output_validation(
            cfg, review_packet_path, review_output_path
        )
        if not review_validation["ready"]:
            errors.extend(review_validation["errors"])
            continue
        decision = str(review.get("decision") or "")
        decision_counts[decision] += 1
        if decision not in _VALID_REVIEW_DECISIONS:
            errors.append(
                f"Existing case {case_id}: invalid review decision {decision!r}."
            )
            continue
        annotation = {
            "schema_version": "gpt_case_review_annotation.v2",
            "case_id": case_id,
            "decision": decision,
            "reviewer_model": review.get("model") or "gpt-5.6-sol",
            "request_hash": review.get("request_hash"),
            "output_hash": sha256_json(review),
            "findings": review.get("findings") or [],
            "reviewed_at": review.get("reviewed_at"),
        }
        if decision == "unusable":
            annotations.append(annotation)
            errors.append(
                f"Existing case {case_id}: reviewer marked the case unusable."
            )
            continue
        if decision == "pass_unchanged":
            candidate = packet.get("candidate_records")
            if not isinstance(candidate, list):
                errors.append(
                    f"Existing case {case_id}: review packet lost candidate records."
                )
                continue
            records.extend(deepcopy(candidate))
            annotation["human_reviewed_current_content"] = True
            annotations.append(annotation)
            resolved_cases += 1
            continue
        resolved = False
        for cycle in range(1, max_repair_cycles + 1):
            repair_packet = _read_json_optional(
                cfg.review_workspace
                / "repair_packets"
                / case_id
                / f"cycle-{cycle}.json"
            )
            repair = _read_json_optional(
                cfg.review_workspace
                / "repair_outputs"
                / case_id
                / f"cycle-{cycle}.json"
            )
            verifier_packet = _read_json_optional(
                cfg.review_workspace
                / "verifier_packets"
                / case_id
                / f"cycle-{cycle}.json"
            )
            verifier = _read_json_optional(
                cfg.review_workspace
                / "verifier_outputs"
                / case_id
                / f"cycle-{cycle}.json"
            )
            if not all((repair_packet, repair, verifier_packet, verifier)):
                continue
            repair_validation = _case_output_validation(
                cfg,
                cfg.review_workspace
                / "repair_packets"
                / case_id
                / f"cycle-{cycle}.json",
                cfg.review_workspace
                / "repair_outputs"
                / case_id
                / f"cycle-{cycle}.json",
            )
            verifier_validation = _case_output_validation(
                cfg,
                cfg.review_workspace
                / "verifier_packets"
                / case_id
                / f"cycle-{cycle}.json",
                cfg.review_workspace
                / "verifier_outputs"
                / case_id
                / f"cycle-{cycle}.json",
            )
            if not repair_validation["ready"] or not verifier_validation["ready"]:
                errors.extend(repair_validation["errors"])
                errors.extend(verifier_validation["errors"])
                continue
            if verifier.get("decision") != "approved":
                continue
            case_records = _normalize_generated_records(_output_records(repair))
            errors.extend(_validate_case_output_identity(case_id, packet, case_records))
            records.extend(_mark_model_reviewed(case_records, repair, verifier))
            claims.extend(_output_claims(repair))
            annotation.update(
                {
                    "final_decision": "repaired_and_approved",
                    "repair_cycle": cycle,
                    "repair_output_hash": sha256_json(repair),
                    "verifier_output_hash": sha256_json(verifier),
                    "human_reviewed_current_content": False,
                }
            )
            annotations.append(annotation)
            revised_cases += 1
            resolved_cases += 1
            resolved = True
            break
        if not resolved:
            errors.append(
                f"Existing case {case_id}: no independently approved repair within "
                f"{max_repair_cycles} cycles."
            )
    stats = {
        "expected_case_count": cfg.expected_existing_cases,
        "resolved_case_count": resolved_cases,
        "failed_or_pending_case_count": cfg.expected_existing_cases - resolved_cases,
        "decision_counts": dict(sorted(decision_counts.items())),
        "revised_case_count": revised_cases,
        "record_count": len(records),
    }
    return records, claims, annotations, stats, errors


def _validate_provenance_only_claims(
    records: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    memory_dir: Path,
    legacy_rule_registry_path: Path,
) -> list[str]:
    errors: list[str] = []
    claim_index: dict[str, Mapping[str, Any]] = {}
    for index, claim in enumerate(claims):
        label = f"claim[{index}]"
        required = (
            "rule_id",
            "claim_text",
            "guideline_id",
            "version",
            "cancer_family",
            "memory_ids",
            "source_spans",
            "allowed_action_types",
        )
        if claim.get("schema_version") != CLAIM_SCHEMA_VERSION:
            errors.append(f"{label}: schema_version must be {CLAIM_SCHEMA_VERSION}.")
        if claim.get("provenance_only") is not True:
            errors.append(f"{label}: provenance_only must be true.")
        missing = [key for key in required if not claim.get(key)]
        if missing:
            errors.append(f"{label}: missing fields {missing}.")
            continue
        claim_id = str(claim["rule_id"])
        if claim.get("claim_id") not in (None, claim_id):
            errors.append(f"{label}: optional claim_id must equal rule_id.")
        if claim_id in claim_index and claim_index[claim_id] != claim:
            errors.append(f"Duplicate conflicting GPT claim {claim_id!r}.")
        claim_index[claim_id] = claim
        if "action_templates" in claim:
            errors.append(
                f"{label}: provenance-only claims cannot contain action_templates."
            )
    memories = {
        str(item["guideline_memory_id"]): item
        for item in load_memory_metadata_records(memory_dir)
    }
    legacy_rule_ids = {
        str(item.get("rule_id") or "") for item in read_jsonl(legacy_rule_registry_path)
    }
    for row in records:
        context = {
            (str(item["guideline_id"]), str(item["version"]))
            for item in row["guideline_context"]["guidelines"]
        }
        family = str(row["state_before"]["cancer_family"])
        for bucket, actions in row["action_set"].items():
            for action in actions:
                label = f"{row['trajectory_id']}/{row['turn_index']}/{bucket}/{action['action_id']}"
                for claim_id in action["guideline_rule_ids"]:
                    if str(claim_id) in legacy_rule_ids:
                        continue
                    claim = claim_index.get(str(claim_id))
                    if claim is None:
                        errors.append(f"{label}: unknown GPT claim {claim_id!r}.")
                        continue
                    if claim["cancer_family"] != family:
                        errors.append(f"{label}: GPT claim has wrong cancer family.")
                    if (
                        str(claim["guideline_id"]),
                        str(claim["version"]),
                    ) not in context:
                        errors.append(
                            f"{label}: GPT claim has wrong guideline version."
                        )
                    if action["action_type"] not in claim["allowed_action_types"]:
                        errors.append(
                            f"{label}: action type is not supported by GPT claim."
                        )
                    if not set(action["supporting_memory_ids"]).issubset(
                        claim["memory_ids"]
                    ):
                        errors.append(f"{label}: memory is not supported by GPT claim.")
                    if not set(action["source_spans"]).issubset(claim["source_spans"]):
                        errors.append(f"{label}: span is not supported by GPT claim.")
                for memory_id in action["supporting_memory_ids"]:
                    memory = memories.get(str(memory_id))
                    if memory is None:
                        errors.append(f"{label}: unknown memory {memory_id!r}.")
                        continue
                    spans = set(
                        memory.get("source_span_ids")
                        or memory.get("source_chunk_ids")
                        or []
                    )
                    if not set(action["source_spans"]).issubset(spans):
                        errors.append(
                            f"{label}: source span is absent from memory {memory_id!r}."
                        )
    return list(dict.fromkeys(errors))


def _quality_scan(records: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    action_signatures: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    bookkeeping_fields = {
        "completed_actions",
        "completed_skills",
        "current_phase",
        "evidence_ledger",
        "last_transition",
        "pending_actions",
        "blocked_actions",
        "unresolved_information",
    }
    for row in records:
        case_id = str(row["base_case_id"])
        before = row["state_before"]
        completed = set(before.get("completed_skills") or [])
        positive_actions = [
            *row["action_set"]["required"],
            *row["action_set"]["acceptable"],
            *[
                action
                for action in row["action_set"]["conditional"]
                if action.get("condition_satisfied") is True
            ],
        ]
        for action in positive_actions:
            repeated = completed.intersection(action.get("required_skills") or [])
            if repeated and not action.get("repeat_justification"):
                errors.append(
                    f"{row['trajectory_id']}/{row['turn_index']}: repeats completed skills "
                    f"without justification: {sorted(repeated)}."
                )
            already_confirmed = {
                field
                for field in ("known_diagnosis", "known_stage", "risk_stratification")
                if before.get(field) not in (None, "", {}, [])
                and field in set(action.get("expected_state_delta") or [])
            }
            if already_confirmed and not action.get("repeat_justification"):
                errors.append(
                    f"{row['trajectory_id']}/{row['turn_index']}: requests already "
                    f"confirmed evidence without refinement justification: "
                    f"{sorted(already_confirmed)}."
                )
        expected_progress = {
            str(field)
            for plan in row.get("accepted_plan_variants") or []
            for action in plan.get("actions") or []
            for field in action.get("expected_state_delta") or []
        }
        observed_progress = set(row.get("state_delta") or {}) - bookkeeping_fields
        if not observed_progress:
            errors.append(
                f"{row['trajectory_id']}/{row['turn_index']}: transition has no "
                "observable clinical state progress."
            )
        unsupported = observed_progress - expected_progress
        if unsupported:
            errors.append(
                f"{row['trajectory_id']}/{row['turn_index']}: state changes are not "
                f"declared by accepted actions: {sorted(unsupported)}."
            )
        signature = sha256_json(
            [
                (bucket, action.get("objective"), action.get("action_type"))
                for bucket in _ACTION_BUCKETS
                for action in row["action_set"][bucket]
            ]
        )
        action_signatures[
            (str(before["cancer_family"]), str(before["current_phase"]), signature)
        ].append(case_id)
    for (family, phase, _), cases in action_signatures.items():
        if len(set(cases)) >= 10:
            errors.append(
                f"Template-collapse signal: identical action sets in {len(set(cases))} "
                f"{family}/{phase} cases."
            )
    return list(dict.fromkeys(errors))


def _validate_case_record_contracts(
    cfg: GPTTrajectoryPipelineConfig,
    npc_records: Sequence[Mapping[str, Any]],
    existing_records: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    npc_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in npc_records:
        npc_by_case[str(row["base_case_id"])].append(row)
        pairs = {
            (str(item["guideline_id"]), str(item["version"]))
            for item in row["guideline_context"]["guidelines"]
        }
        if pairs != {("CSCO鼻咽癌2022", "2022")}:
            errors.append(
                f"NPC {row['base_case_id']}: wrong guideline context {sorted(pairs)}."
            )
        if row["guideline_context"].get("decision_date") != "2022-12-31":
            errors.append(
                f"NPC {row['base_case_id']}: decision_date must be 2022-12-31."
            )
    for case_id, rows in npc_by_case.items():
        real_count = sum(row["case_source"] == "real" for row in rows)
        kinds = Counter(
            str(row.get("counterfactual", {}).get("perturbation_type"))
            for row in rows
            if row["case_source"] == "counterfactual"
        )
        if not cfg.npc_min_real_turns <= real_count <= cfg.npc_max_real_turns:
            errors.append(
                f"NPC {case_id}: real transition count {real_count} is outside "
                f"{cfg.npc_min_real_turns}-{cfg.npc_max_real_turns}."
            )
        if kinds != Counter({"decision_relevant": 1, "irrelevant": 1}):
            errors.append(
                f"NPC {case_id}: requires exactly one counterfactual of each kind."
            )
    if len(npc_by_case) != cfg.expected_npc_cases:
        errors.append(
            f"Resolved NPC case count is {len(npc_by_case)}; expected {cfg.expected_npc_cases}."
        )
    if len(existing_records) != cfg.expected_existing_records:
        errors.append(
            f"Resolved existing record count is {len(existing_records)}; "
            f"expected {cfg.expected_existing_records}."
        )
    return errors


def _status_for_npc(
    cfg: GPTTrajectoryPipelineConfig,
    case_ids: Sequence[str],
    *,
    max_repair_cycles: int,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    pending: dict[str, list[str]] = defaultdict(list)
    resolved = 0
    for case_id in case_ids:
        teacher_packet_path = cfg.npc_workspace / "teacher_packets" / f"{case_id}.json"
        teacher_output_path = cfg.npc_workspace / "teacher_outputs" / f"{case_id}.json"
        packet = _read_json(teacher_packet_path)
        teacher = _read_json_optional(teacher_output_path)
        verifier_packet = _read_json_optional(
            cfg.npc_workspace / "verifier_packets" / f"{case_id}.json"
        )
        verifier = _read_json_optional(
            cfg.npc_workspace / "verifier_outputs" / f"{case_id}.json"
        )
        if teacher is None or not _output_matches_packet(teacher, packet):
            counts["teacher_pending"] += 1
            pending["teacher"].append(case_id)
        elif not _case_output_is_ready(cfg, teacher_packet_path, teacher_output_path):
            counts["teacher_invalid"] += 1
            pending["teacher_invalid"].append(case_id)
        elif (
            verifier_packet is None
            or verifier is None
            or not _output_matches_packet(verifier, verifier_packet)
        ):
            counts["verifier_pending"] += 1
            pending["verifier"].append(case_id)
        elif not _case_output_is_ready(
            cfg,
            cfg.npc_workspace / "verifier_packets" / f"{case_id}.json",
            cfg.npc_workspace / "verifier_outputs" / f"{case_id}.json",
        ):
            counts["verifier_invalid"] += 1
            pending["verifier_invalid"].append(case_id)
        elif verifier.get("decision") == "approved":
            counts["approved"] += 1
            resolved += 1
        else:
            approved_cycle = _approved_npc_repair_cycle(cfg, case_id, max_repair_cycles)
            if approved_cycle:
                counts["repaired_and_approved"] += 1
                resolved += 1
            else:
                counts["rejected"] += 1
                pending["repair"].append(case_id)
    return {
        "packet_count": len(case_ids),
        "resolved_cases": resolved,
        "counts": dict(sorted(counts.items())),
        "pending": dict(sorted(pending.items())),
    }


def _status_for_existing(
    cfg: GPTTrajectoryPipelineConfig,
    case_ids: Sequence[str],
    *,
    max_repair_cycles: int,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    pending: dict[str, list[str]] = defaultdict(list)
    resolved = 0
    for case_id in case_ids:
        review_packet_path = cfg.review_workspace / "review_packets" / f"{case_id}.json"
        review_output_path = cfg.review_workspace / "review_outputs" / f"{case_id}.json"
        packet = _read_json(review_packet_path)
        review = _read_json_optional(review_output_path)
        if review is None or not _output_matches_packet(review, packet):
            counts["review_pending"] += 1
            pending["review"].append(case_id)
            continue
        if not _case_output_is_ready(cfg, review_packet_path, review_output_path):
            counts["review_invalid"] += 1
            pending["review_invalid"].append(case_id)
            continue
        decision = str(review.get("decision") or "")
        counts[decision or "invalid"] += 1
        if decision == "pass_unchanged":
            resolved += 1
        elif decision == "unusable":
            pending["unusable"].append(case_id)
        elif decision == "repair_required":
            approved_cycle = _approved_repair_cycle(cfg, case_id, max_repair_cycles)
            if approved_cycle:
                resolved += 1
                counts["repaired_and_approved"] += 1
            else:
                pending["repair"].append(case_id)
    return {
        "packet_count": len(case_ids),
        "resolved_cases": resolved,
        "counts": dict(sorted(counts.items())),
        "pending": dict(sorted(pending.items())),
    }


def _approved_repair_cycle(
    cfg: GPTTrajectoryPipelineConfig, case_id: str, max_repair_cycles: int
) -> int | None:
    for cycle in range(1, max_repair_cycles + 1):
        repair_packet_path = (
            cfg.review_workspace / "repair_packets" / case_id / f"cycle-{cycle}.json"
        )
        repair_output_path = (
            cfg.review_workspace / "repair_outputs" / case_id / f"cycle-{cycle}.json"
        )
        verifier_packet_path = (
            cfg.review_workspace / "verifier_packets" / case_id / f"cycle-{cycle}.json"
        )
        verifier_output_path = (
            cfg.review_workspace / "verifier_outputs" / case_id / f"cycle-{cycle}.json"
        )
        packet = _read_json_optional(verifier_packet_path)
        verifier = _read_json_optional(verifier_output_path)
        if (
            packet
            and verifier
            and _case_output_is_ready(cfg, repair_packet_path, repair_output_path)
            and _case_output_is_ready(cfg, verifier_packet_path, verifier_output_path)
            and verifier.get("decision") == "approved"
        ):
            return cycle
    return None


def _approved_npc_repair_cycle(
    cfg: GPTTrajectoryPipelineConfig, case_id: str, max_repair_cycles: int
) -> int | None:
    for cycle in range(1, max_repair_cycles + 1):
        repair_packet_path = (
            cfg.npc_workspace / "repair_packets" / case_id / f"cycle-{cycle}.json"
        )
        repair_output_path = (
            cfg.npc_workspace / "repair_outputs" / case_id / f"cycle-{cycle}.json"
        )
        verifier_packet_path = (
            cfg.npc_workspace / "verifier_packets" / case_id / f"cycle-{cycle}.json"
        )
        verifier_output_path = (
            cfg.npc_workspace / "verifier_outputs" / case_id / f"cycle-{cycle}.json"
        )
        packet = _read_json_optional(verifier_packet_path)
        verifier = _read_json_optional(verifier_output_path)
        if (
            packet
            and verifier
            and _case_output_is_ready(cfg, repair_packet_path, repair_output_path)
            and _case_output_is_ready(cfg, verifier_packet_path, verifier_output_path)
            and verifier.get("decision") == "approved"
        ):
            return cycle
    return None


def _write_followup_packet(
    path: Path,
    *,
    packet_type: str,
    case_id: str,
    model: str,
    reasoning_effort: str,
    source_packet: Mapping[str, Any] | Path,
    candidate_output: Mapping[str, Any],
    force: bool,
    extra: Mapping[str, Any] | None = None,
) -> None:
    source = (
        _read_json(source_packet)
        if isinstance(source_packet, Path)
        else dict(source_packet)
    )
    packet = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "packet_type": packet_type,
        "pipeline_version": PIPELINE_VERSION,
        "case_id": case_id,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "source_request_hash": source.get("request_hash"),
        "source_packet": source,
        "candidate_output": dict(candidate_output),
        "verification_contract": {
            "independent_context_required": True,
            "hidden_reasoning_must_not_be_stored": True,
            "allowed_decisions": sorted(_VALID_VERIFY_DECISIONS),
            "no_invented_patient_facts_or_tool_results": True,
        },
    }
    if "repair" in packet_type and "verifier" not in packet_type:
        packet["task_instructions"] = [
            "Repair the complete case trajectory using the supplied verifier findings.",
            "Do not use deterministic rule templates and do not invent patient facts or tool results.",
            "Preserve valid content, split membership and case identity; record replaced IDs in supersedes_trajectory_ids when IDs change.",
            "Every preconditions field must contain condition DSL objects using the exact key operator (never op), not natural-language strings; accepted plan actions must preserve the matching candidate action preconditions.",
            "Do not repair missing progress by treating guideline retrieval, file/ROI availability, bookkeeping fields, or repeated report summaries as clinical progress; add only patient-specific supported findings, resolved uncertainties, or supported clinical decisions.",
            "Keep expected_state_delta exact: declare only fields that can change and do not list fields whose before/after values remain identical merely to satisfy validation.",
            "For treatment selection, encode all material gates stated by the objective (confirmed diagnosis, stage, pathology/risk, biomarkers and fitness as applicable); do not use a broad exists condition when unknown, pending, not_reported or placeholders would pass.",
            "A clinically changed or reworded action may cite a legacy rule only when it still exactly matches that rule action_template; otherwise emit and cite a provenance-only gpt_guideline_claim.v2 grounded to real supplied memory IDs/source spans.",
            "Every record must retain at least one positive and one valid negative action, and counterfactual trajectories must retain a real anchor and start at turn_index 0.",
            "Return JSON only and do not store hidden reasoning.",
            "Before finishing, run validate-gpt-planner-output for this packet/output and fix every reported error.",
        ]
        packet["expected_output"] = {
            "path": f"repair_outputs/{case_id}/cycle-{extra.get('repair_cycle', 1) if extra else 1}.json",
            "envelope": {
                "schema_version": REPAIR_OUTPUT_SCHEMA_VERSION,
                "case_id": case_id,
                "request_hash": "copy this packet request_hash",
                "model": model,
                "model_version": "exact runtime model identifier",
                "generated_at": "UTC timestamp",
                "records": "non-empty planner_trajectory.v2 array",
                "claims": "provenance-only gpt_guideline_claim.v2 array",
            },
        }
    else:
        packet["task_instructions"] = [
            "Independently verify the complete candidate case against only the supplied evidence and guideline chunks.",
            "Check every quality_checks/verification_contract condition and reject unsupported facts, tools, transitions, actions or citations.",
            "Reject guideline/file/ROI availability or bookkeeping-only edits presented as clinical progress, expected deltas that do not occur without an execution explanation, and broad treatment gates that admit unknown or placeholder evidence.",
            "Do not repair the candidate in this task; return concrete structured findings for a separate repair task.",
            "Return JSON only and do not store hidden reasoning.",
            "Before finishing, run validate-gpt-planner-output for this packet/output and fix every reported error.",
        ]
        packet["expected_output"] = {
            "path": (
                f"verifier_outputs/{case_id}/cycle-{extra.get('repair_cycle')}.json"
                if extra and extra.get("repair_cycle")
                else f"verifier_outputs/{case_id}.json"
            ),
            "envelope": {
                "schema_version": VERIFIER_OUTPUT_SCHEMA_VERSION,
                "case_id": case_id,
                "request_hash": "copy this packet request_hash",
                "model": model,
                "decision": "approved | rejected | repair_required | unusable",
                "findings": [],
                "reviewed_at": "UTC timestamp",
            },
        }
    if extra:
        packet.update(extra)
    _write_packet(path, packet, force=force)


def _write_packet(path: Path, packet: Mapping[str, Any], *, force: bool) -> None:
    payload = deepcopy(dict(packet))
    payload.pop("request_hash", None)
    payload["request_hash"] = sha256_json(payload)
    existing = _read_json_optional(path)
    if existing == payload:
        return
    if existing is not None and not force:
        # Inputs changing must invalidate old outputs, but packet regeneration is
        # safe.  Preserve the previous packet for audit before writing the new one.
        archive = path.parent / ".superseded" / path.stem
        archive.mkdir(parents=True, exist_ok=True)
        old_hash = str(existing.get("request_hash") or sha256_json(existing))
        archived = archive / f"{old_hash}.json"
        if not archived.exists():
            write_json(archived, existing)
    write_json(path, payload)


def _collect_case_evidence(case_dir: Path, repo_root: Path) -> list[dict[str, Any]]:
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
    result: list[dict[str, Any]] = []
    total_bytes = 0
    maximum_total = 2 * 1024 * 1024
    for path in sorted(item for item in case_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(case_dir)
        lower_parts = {part.lower() for part in relative.parts[:-1]}
        name = path.name.lower()
        if lower_parts.intersection(_EXCLUDED_PARTS):
            continue
        if name in _EXCLUDED_NAMES or any(
            fragment in name for fragment in _EXCLUDED_NAME_FRAGMENTS
        ):
            continue
        if (
            path.suffix.lower() not in _TEXT_SUFFIXES
            or path.stat().st_size > 1024 * 1024
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        encoded_size = len(text.encode("utf-8"))
        if total_bytes + encoded_size > maximum_total:
            continue
        total_bytes += encoded_size
        try:
            value: Any = json.loads(text) if path.suffix.lower() == ".json" else text
        except json.JSONDecodeError:
            value = text
        value = _sanitize_evidence_value(value, case_id=case_dir.name)
        result.append(
            {
                "source_ref": _portable_path(path, repo_root),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "content": value,
            }
        )
    if not result:
        raise V2SchemaError(
            f"No decision-time textual evidence found for case {case_dir.name}."
        )
    return result


def _sanitize_evidence_value(value: Any, *, case_id: str, key: str = "") -> Any:
    """Remove held-out references and host paths embedded inside JSON manifests."""

    normalized_key = key.strip().lower().replace("-", "_")
    if normalized_key in {"evaluation", "hidden_state", "outcome", "follow_up"}:
        return None
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for child_key, child_value in value.items():
            name = str(child_key)
            normalized = name.strip().lower().replace("-", "_")
            if normalized in {"evaluation", "hidden_state", "outcome", "follow_up"}:
                continue
            sanitized = _sanitize_evidence_value(child_value, case_id=case_id, key=name)
            if sanitized is not None:
                result[name] = sanitized
        return result
    if isinstance(value, list):
        result_list = []
        for item in value:
            if isinstance(item, str) and any(
                fragment in item.lower()
                for fragment in (
                    "rubric",
                    "trajectory",
                    "relevant_nodes",
                    "hidden_state",
                )
            ):
                continue
            sanitized = _sanitize_evidence_value(item, case_id=case_id, key=key)
            if sanitized is not None:
                result_list.append(sanitized)
        return result_list
    if isinstance(value, str):
        text = value.strip()
        # Replace Windows, POSIX and UNC host paths with a case-local reference.
        looks_absolute = text.startswith(("/", "\\\\")) or (
            len(text) > 2 and text[1] == ":" and text[2] in {"/", "\\"}
        )
        if looks_absolute:
            normalized = text.replace("\\", "/")
            marker = f"/{case_id}/"
            if marker in normalized:
                return f"case://{case_id}/{normalized.split(marker, 1)[1]}"
            return f"external-file://{Path(normalized).name}"
    return value


def _discover_cases(root: Path, *, marker: str) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Patient root does not exist: {root}")
    return sorted(
        path for path in root.iterdir() if path.is_dir() and (path / marker).is_file()
    )


def _load_existing_rows(root: Path) -> list[dict[str, Any]]:
    # The admitted split files are authoritative.  The root candidate file is
    # a pre-split construction artifact and older releases may contain stale
    # ``split=train`` values for every record.
    dataset = root / "dataset"
    rows = [
        row
        for split in ("train", "validation", "test")
        for row in read_jsonl(dataset / f"{split}.jsonl")
    ]
    if rows:
        return rows
    candidates = root / "planner_trajectory.v2.candidates.jsonl"
    if candidates.is_file():
        return read_jsonl(candidates)
    raise FileNotFoundError(f"No source Planner V2 trajectories found under {root}")


def _group_existing_records(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row.get("base_case_id") or row.get("case_id") or "")].append(
            dict(row)
        )
    result.pop("", None)
    for items in result.values():
        items.sort(
            key=lambda item: (
                str(item.get("trajectory_id")),
                int(item.get("turn_index", 0)),
            )
        )
    return dict(result)


def _require_sources(cfg: GPTTrajectoryPipelineConfig) -> None:
    for label, path in (
        ("NPC cases", cfg.npc_case_root),
        ("Lung cases", cfg.lung_case_root),
        ("UCEC cases", cfg.ucec_case_root),
        ("source dataset", cfg.source_dataset_root),
        ("guideline chunks", cfg.guideline_chunks_path),
        ("memory catalog", cfg.memory_dir),
        ("legacy rule registry", cfg.legacy_rule_registry_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required {label} path does not exist: {path}")


def _assert_expected_counts(
    cfg: GPTTrajectoryPipelineConfig,
    npc_cases: Sequence[Path],
    existing: Mapping[str, Sequence[Mapping[str, Any]]],
    existing_rows: Sequence[Mapping[str, Any]],
) -> None:
    actual = (len(npc_cases), len(existing), len(existing_rows))
    expected = (
        cfg.expected_npc_cases,
        cfg.expected_existing_cases,
        cfg.expected_existing_records,
    )
    if actual != expected:
        raise V2SchemaError(
            "Planner GPT source counts differ from the locked release contract: "
            f"actual NPC/existing cases/records={actual}, expected={expected}."
        )
    invalid = {
        case_id: len(rows) for case_id, rows in existing.items() if len(rows) != 6
    }
    if invalid:
        raise V2SchemaError(
            f"Existing cases must each contain six records; invalid={invalid}."
        )


def _assert_source_destination_separation(cfg: GPTTrajectoryPipelineConfig) -> None:
    source = cfg.source_dataset_root
    for destination in (
        cfg.npc_workspace,
        cfg.review_workspace,
        cfg.merged_dataset_root,
    ):
        if (
            destination == source
            or source in destination.parents
            or destination in source.parents
        ):
            raise ValueError(
                f"GPT workspace {destination} must be physically separate from source dataset {source}."
            )
    if len({cfg.npc_workspace, cfg.review_workspace, cfg.merged_dataset_root}) != 3:
        raise ValueError("NPC, review, and merged workspaces must be distinct.")


def _guideline_chunks(
    chunks: Sequence[Mapping[str, Any]], guideline_id: str, version: str
) -> list[dict[str, Any]]:
    result = [
        dict(item)
        for item in chunks
        if str(item.get("guideline_id")) == guideline_id
        and str(item.get("version")) == version
    ]
    if not result:
        raise V2SchemaError(f"No guideline chunks found for {guideline_id}@{version}.")
    return result


def _memory_records(
    records: Sequence[Mapping[str, Any]], guideline_id: str, version: str
) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in records
        if str(item.get("guideline_id")) == guideline_id
        and str(item.get("version")) == version
    ]


def _npc_guideline_context() -> dict[str, Any]:
    return {
        "decision_date": "2022-12-31",
        "guidelines": [
            {
                "guideline_id": "CSCO鼻咽癌2022",
                "version": "2022",
                "effective_from": "2022-01-01",
                "effective_to": None,
            }
        ],
    }


def _allowed_skills(family: str) -> list[str]:
    common = [
        "guideline.retrieve",
        "pathology.read_diagnostic_report",
        "pathology.read_staging_report",
        "radiology.review_staging_extent",
        "molecular.read_biomarker_report",
        "treatment.select_after_evidence",
        "followup.plan_surveillance",
    ]
    if family == "nasopharyngeal":
        common.extend(["radiology.npc_mri_roi", "pathology.npc_conch_patch_roi"])
    return sorted(set(common))


def _quality_check_names() -> list[str]:
    return [
        "patient_state_conditioning_not_phase_template",
        "action_bucket_clinical_validity",
        "conditional_precondition_expression",
        "completed_skill_and_known_evidence_repetition",
        "observable_state_progress",
        "evidence_gated_phase_transition",
        "memory_claim_source_span_grounding",
        "tool_state_delta_completed_skill_consistency",
        "decision_relevant_counterfactual_sensitivity",
        "irrelevant_counterfactual_invariance",
        "whole_case_trajectory_coherence",
    ]


def _three_cancer_release_scope() -> dict[str, Any]:
    return load_release_scope(
        copy_release_scope(PLANNER_V2_LUNG_ENDOMETRIAL_NPC_RELEASE_SCOPE)
    )


def _validate_model_output_envelope(
    output: Mapping[str, Any],
    packet: Mapping[str, Any],
    case_id: str,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if str(output.get("case_id") or "") != case_id:
        errors.append(f"{case_id}/{label}: output case_id mismatch.")
    if not _output_matches_packet(output, packet):
        errors.append(f"{case_id}/{label}: request_hash does not match its packet.")
    if output.get("model") != "gpt-5.6-sol":
        errors.append(f"{case_id}/{label}: unexpected model {output.get('model')!r}.")
    if (
        label == "teacher"
        and output.get("schema_version") != TEACHER_OUTPUT_SCHEMA_VERSION
    ):
        errors.append(f"{case_id}/{label}: wrong output schema_version.")
    if (
        label == "review"
        and output.get("schema_version") != REVIEW_OUTPUT_SCHEMA_VERSION
    ):
        errors.append(f"{case_id}/{label}: wrong output schema_version.")
    if (
        label.startswith("repair-")
        and output.get("schema_version") != REPAIR_OUTPUT_SCHEMA_VERSION
    ):
        errors.append(f"{case_id}/{label}: wrong output schema_version.")
    if (
        "verifier" in label
        and output.get("schema_version") != VERIFIER_OUTPUT_SCHEMA_VERSION
    ):
        errors.append(f"{case_id}/{label}: wrong output schema_version.")
    return errors


def _validate_case_output_identity(
    case_id: str,
    packet: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    expected_split = _packet_expected_split(packet)
    for index, row in enumerate(records):
        if str(row.get("base_case_id") or "") != case_id:
            errors.append(f"{case_id}: record[{index}] base_case_id mismatch.")
        if str(row.get("split") or "") != expected_split:
            errors.append(f"{case_id}: record[{index}] split mismatch.")
    original_ids = _repair_source_trajectory_ids(packet)
    if original_ids:
        final_ids = {str(row.get("trajectory_id") or "") for row in records}
        removed_ids = original_ids - final_ids
        declared = {
            str(value)
            for row in records
            for value in row.get("supersedes_trajectory_ids") or []
        }
        unknown = declared - original_ids
        if unknown:
            errors.append(
                f"{case_id}: supersedes_trajectory_ids contains unknown IDs: "
                f"{sorted(unknown)}."
            )
        if not removed_ids.issubset(declared):
            errors.append(
                f"{case_id}: replaced trajectory IDs are not declared in "
                f"supersedes_trajectory_ids: {sorted(removed_ids - declared)}."
            )
    return errors


def _repair_source_trajectory_ids(packet: Mapping[str, Any]) -> set[str]:
    packet_type = str(packet.get("packet_type") or "")
    if "repair" not in packet_type or "verifier" in packet_type:
        return set()
    candidate = packet.get("candidate_output")
    if not isinstance(candidate, Mapping):
        return set()
    return {
        str(row.get("trajectory_id") or "")
        for row in candidate.get("records") or []
        if isinstance(row, Mapping) and row.get("trajectory_id")
    }


def _packet_expected_split(packet: Mapping[str, Any]) -> str:
    """Resolve split through nested repair/verifier source packets."""

    current: Mapping[str, Any] | None = packet
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        split = str(current.get("split") or "")
        if split:
            return split
        source = current.get("source_packet")
        current = source if isinstance(source, Mapping) else None
    return ""


def _is_utc_timestamp(value: Any) -> bool:
    text = str(value or "").strip()
    if not text or not text.endswith(("Z", "+00:00")):
        return False
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _case_output_is_ready(
    cfg: GPTTrajectoryPipelineConfig,
    packet_path: Path,
    output_path: Path,
) -> bool:
    """Require the complete single-case admission contract, not only a hash match."""

    return bool(_case_output_validation(cfg, packet_path, output_path)["ready"])


def _case_output_validation(
    cfg: GPTTrajectoryPipelineConfig,
    packet_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Return a stable error object for queue/status and final dataset resolution."""

    if not packet_path.is_file() or not output_path.is_file():
        return {
            "ready": False,
            "errors": [f"Missing packet or output: {packet_path} / {output_path}."],
        }
    try:
        return validate_gpt_case_output(packet_path, output_path, cfg)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        V2SchemaError,
    ) as exc:
        return {"ready": False, "errors": [f"{output_path}: validation failed: {exc}"]}


def _mark_model_reviewed(
    records: Sequence[Mapping[str, Any]],
    author_output: Mapping[str, Any],
    verifier_output: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    reviewed_at = str(verifier_output.get("reviewed_at") or _utc_now())
    for row in records:
        item = deepcopy(dict(row))
        provenance = item.setdefault("provenance", {})
        old_review = {
            key: provenance.get(key)
            for key in ("review_status", "reviewer_id", "reviewed_at")
            if key in provenance
        }
        provenance.update(
            {
                "rule_compiler_version": PIPELINE_VERSION,
                "teacher_model": "gpt-5.6-sol",
                "teacher_model_version": str(
                    author_output.get("model_version") or "gpt-5.6-sol"
                ),
                "teacher_generated_at": str(author_output.get("generated_at") or ""),
                "teacher_prompt_hash": str(author_output.get("request_hash") or ""),
                "teacher_output_hash": sha256_json(author_output),
                "review_status": "approved",
                "reviewer_id": "gpt-5.6-sol-independent-verifier",
                "reviewed_at": reviewed_at,
                "approval_type": "model_review",
                "human_reviewed": False,
                "previous_review": old_review or None,
            }
        )
        flags = [str(value) for value in provenance.get("qc_flags") or []]
        flags = [value for value in flags if value != "human_review_required"]
        flags.extend(
            ["gpt_direct_action_supervision", "model_reviewed_not_human_reviewed"]
        )
        provenance["qc_flags"] = sorted(set(flags))
        result.append(item)
    return result


def _mark_human_reviewed(
    records: Sequence[Mapping[str, Any]],
    approval: Mapping[str, Any],
    *,
    author_output: Mapping[str, Any] | None = None,
    gpt_authored: bool = False,
) -> list[dict[str, Any]]:
    """Mark the exact content bound by a professional manual approval."""

    result: list[dict[str, Any]] = []
    for row in records:
        item = deepcopy(dict(row))
        provenance = item.setdefault("provenance", {})
        previous_review = {
            key: provenance.get(key)
            for key in (
                "review_status",
                "reviewer_id",
                "reviewed_at",
                "approval_type",
                "human_reviewed",
            )
            if key in provenance
        }
        if author_output is not None:
            provenance.update(
                {
                    "teacher_model": str(author_output.get("model") or "gpt-5.6-sol"),
                    "teacher_model_version": str(
                        author_output.get("model_version") or "gpt-5.6-sol"
                    ),
                    "teacher_generated_at": str(
                        author_output.get("generated_at") or ""
                    ),
                    "teacher_prompt_hash": str(
                        author_output.get("request_hash") or ""
                    ),
                    "teacher_output_hash": sha256_json(author_output),
                }
            )
        provenance.update(
            {
                "rule_compiler_version": PIPELINE_VERSION,
                "review_status": "approved",
                "reviewer_id": str(approval["reviewer_id"]),
                "reviewed_at": str(approval["reviewed_at"]),
                "approval_type": str(approval["approval_type"]),
                "human_reviewed": True,
                "manual_approval_input_hash": str(approval["approval_input_hash"]),
                "previous_review": previous_review or None,
            }
        )
        flags = [str(value) for value in provenance.get("qc_flags") or []]
        flags = [
            value
            for value in flags
            if value
            not in {"human_review_required", "model_reviewed_not_human_reviewed"}
        ]
        flags.append("professional_clinician_reviewed")
        if gpt_authored:
            flags.append("gpt_direct_action_supervision")
        provenance["qc_flags"] = sorted(set(flags))
        result.append(item)
    return result


def _output_records(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = output.get("records")
    if not isinstance(value, list) or not value:
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _normalize_generated_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply mechanical normalization without authoring clinical supervision.

    Raw model responses remain untouched in their output files.  The release
    copy recomputes the canonical state delta from the two model-authored
    states and mirrors the non-recursive portion into ``last_transition``.
    """

    normalized: list[dict[str, Any]] = []
    for record in records:
        item = deepcopy(dict(record))
        before = item.get("state_before")
        after = item.get("state_after")
        if isinstance(before, Mapping) and isinstance(after, Mapping):
            delta = state_delta(before, after)
            item["state_delta"] = delta
            last_transition = after.get("last_transition")
            if isinstance(last_transition, Mapping):
                updated_transition = deepcopy(dict(last_transition))
                updated_transition["state_delta"] = {
                    key: deepcopy(value)
                    for key, value in delta.items()
                    if key != "last_transition"
                }
                item["state_after"] = deepcopy(dict(after))
                item["state_after"]["last_transition"] = updated_transition
                item["state_delta"] = state_delta(before, item["state_after"])
        normalized.append(item)
    return normalized


def _output_claims(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = output.get("claims") or output.get("guideline_claims") or []
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _output_matches_packet(
    output: Mapping[str, Any], packet: Mapping[str, Any]
) -> bool:
    return bool(packet.get("request_hash")) and str(
        output.get("request_hash") or ""
    ) == str(packet.get("request_hash"))


def _packet_ids(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return sorted(item.stem for item in path.glob("*.json") if item.is_file())


def _validate_record_id_uniqueness(records: Sequence[Mapping[str, Any]]) -> list[str]:
    counts = Counter(_record_key(row) for row in records)
    return [
        f"Duplicate trajectory record key {key!r}."
        for key, count in counts.items()
        if count > 1
    ]


def _record_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row.get("trajectory_id") or ""), int(row.get("turn_index") or 0)


def _records_hash(records: Iterable[Mapping[str, Any]]) -> str:
    ordered = sorted(records, key=_record_key)
    return sha256_json(ordered)


def _existing_revision_diffs(
    original_records: Sequence[Mapping[str, Any]],
    final_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Describe clinical-supervision changes without altering either revision."""

    original_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    final_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in original_records:
        original_by_case[str(row.get("base_case_id") or "")].append(row)
    for row in final_records:
        final_by_case[str(row.get("base_case_id") or "")].append(row)
    reports: list[dict[str, Any]] = []
    compared_sections = {
        "identity": (
            "trajectory_id",
            "case_id",
            "turn_index",
            "case_source",
            "counterfactual",
        ),
        "state": ("state_before", "state_after", "state_delta", "tool_executions"),
        "actions": ("action_set", "accepted_plan_variants"),
        "routing": ("routing_labels",),
        "provenance": ("provenance",),
    }
    for case_id in sorted(set(original_by_case) | set(final_by_case)):
        order = lambda row: (
            int(row.get("turn_index") or 0),
            str(row.get("trajectory_id") or ""),
        )
        before = sorted(original_by_case[case_id], key=order)
        after = sorted(final_by_case[case_id], key=order)
        if _records_hash(before) == _records_hash(after):
            continue
        changes: list[dict[str, Any]] = []
        section_counts: Counter[str] = Counter()
        for old, new in zip_longest(before, after):
            changed_sections = [
                section
                for section, fields in compared_sections.items()
                if old is None
                or new is None
                or any(old.get(field) != new.get(field) for field in fields)
            ]
            section_counts.update(changed_sections)
            changes.append(
                {
                    "before_trajectory_id": old.get("trajectory_id") if old else None,
                    "after_trajectory_id": new.get("trajectory_id") if new else None,
                    "before_turn_index": old.get("turn_index") if old else None,
                    "after_turn_index": new.get("turn_index") if new else None,
                    "changed_sections": changed_sections,
                    "supersedes_trajectory_ids": (
                        list(new.get("supersedes_trajectory_ids") or []) if new else []
                    ),
                }
            )
        reports.append(
            {
                "schema_version": "planner_case_revision_diff.v2",
                "case_id": case_id,
                "original_record_count": len(before),
                "final_record_count": len(after),
                "original_records_hash": _records_hash(before),
                "final_records_hash": _records_hash(after),
                "changed_section_counts": dict(sorted(section_counts.items())),
                "record_changes": changes,
            }
        )
    return reports


def _source_dataset_hash(root: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    manifest = _read_json_optional(root / "dataset" / "manifest.json")
    return str((manifest or {}).get("dataset_hash") or _records_hash(rows))


def _deduplicate_claims(claims: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for claim in claims:
        claim_id = str(claim.get("rule_id") or claim.get("claim_id") or "")
        item = dict(claim)
        if not claim_id:
            raise V2SchemaError("GPT claim is missing rule_id.")
        if "rule_id" not in item:
            item["rule_id"] = claim_id
        if item.get("claim_id") not in (None, claim_id):
            raise V2SchemaError("GPT claim claim_id must equal rule_id when present.")
        if claim_id in result and result[claim_id] != item:
            raise V2SchemaError(f"Conflicting GPT claims share claim_id {claim_id!r}.")
        result[claim_id] = item
    return [result[key] for key in sorted(result)]


def _portable_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        # Author/verifier tasks may be interrupted between replacing a draft
        # and completing the JSON document.  Queue/status discovery must remain
        # resumable; the strict single-case validator still reports the corrupt
        # output when it is invoked explicitly.
        return None


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


__all__ = [
    "GPTTrajectoryPipelineConfig",
    "approve_gpt_trajectory_data_manually",
    "get_gpt_trajectory_status",
    "load_gpt_trajectory_pipeline_config",
    "merge_gpt_trajectory_dataset",
    "prepare_gpt_trajectory_packets",
    "run_gpt_trajectory_stage",
    "validate_gpt_case_output",
    "validate_gpt_trajectory_outputs",
]
