from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.private_data

from guideline_planner.gpt_trajectory_pipeline import (
    REVIEW_OUTPUT_SCHEMA_VERSION,
    TEACHER_OUTPUT_SCHEMA_VERSION,
    GPTTrajectoryPipelineConfig,
    _existing_revision_diffs,
    _mark_model_reviewed,
    _normalize_generated_records,
    _quality_scan,
    _resolve_existing_cases,
    _valid_manual_approval,
    _validate_case_output_identity,
    approve_gpt_trajectory_data_manually,
    get_gpt_trajectory_status,
    load_gpt_trajectory_pipeline_config,
    merge_gpt_trajectory_dataset,
    prepare_gpt_trajectory_packets,
    run_gpt_trajectory_stage,
    validate_gpt_case_output,
    validate_gpt_trajectory_outputs,
)
from guideline_planner.io_utils import read_jsonl, write_json
from guideline_planner.schemas_v2 import (
    V2SchemaError,
    audit_trajectory_dataset_v2,
    stable_case_splits,
    validate_trajectory_record_v2,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT / "guideline_planner" / "configs" / "planner_v2_lung_endometrial_npc.yaml"
)


@pytest.fixture(scope="module")
def prepared_pipeline(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[GPTTrajectoryPipelineConfig, dict[str, object]]:
    root = tmp_path_factory.mktemp("gpt-trajectory-pipeline")
    base = load_gpt_trajectory_pipeline_config(CONFIG_PATH)
    config = replace(
        base,
        npc_workspace=root / "npc",
        review_workspace=root / "review",
        merged_dataset_root=root / "merged",
        manual_approval_path=root / "manual_approval.json",
    )
    manifest = prepare_gpt_trajectory_packets(config)
    return config, manifest


def test_npc_release_config_keeps_source_dataset_read_only_and_separate() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    paths = config["paths"]
    assert paths["source_dataset_root"] == "datasets/planner_v2_pilot"
    assert paths["merged_dataset_root"] != paths["source_dataset_root"]
    assert paths["npc_workspace"] != paths["source_dataset_root"]
    assert paths["review_workspace"] != paths["source_dataset_root"]
    assert config["expected"] == {
        "npc_cases": 39,
        "existing_cases": 70,
        "existing_records": 420,
    }
    assert config["generation"]["forbid_rule_generated_targets"] is True
    assert config["generation"]["require_independent_verifier"] is True


def test_manual_approval_is_all_or_nothing_and_content_addressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = load_gpt_trajectory_pipeline_config(CONFIG_PATH)
    config = replace(base, manual_approval_path=tmp_path / "manual_review.json")
    inputs = {
        "source_dataset_hash": "source-a",
        "existing_cases": {"existing": {"records_hash": "a", "record_count": 6}},
        "npc_cases": {"npc": {"teacher_output_hash": "b", "record_count": 5}},
    }
    statistics = {
        "case_count": 2,
        "existing_case_count": 1,
        "existing_record_count": 6,
        "npc_case_count": 1,
        "npc_record_count": 5,
        "record_count": 11,
    }
    monkeypatch.setattr(
        "guideline_planner.gpt_trajectory_pipeline._manual_approval_inputs",
        lambda cfg: (deepcopy(inputs), deepcopy(statistics)),
    )

    approval = approve_gpt_trajectory_data_manually(
        config,
        reviewer_id="professional-clinician-review-team",
        reviewed_at="2026-09-04T00:00:00Z",
    )
    assert approval["ready"] is True
    assert approval["statistics"]["record_count"] == 11
    valid, errors = _valid_manual_approval(config)
    assert valid == approval
    assert errors == []

    changed = deepcopy(inputs)
    changed["npc_cases"]["npc"]["teacher_output_hash"] = "changed"
    monkeypatch.setattr(
        "guideline_planner.gpt_trajectory_pipeline._manual_approval_inputs",
        lambda cfg: (deepcopy(changed), deepcopy(statistics)),
    )
    valid, errors = _valid_manual_approval(config)
    assert valid is None
    assert any("does not match" in error for error in errors)
    with pytest.raises(FileExistsError, match="reviewed again"):
        approve_gpt_trajectory_data_manually(config)


def test_npc_39_case_group_split_is_exact_and_deterministic() -> None:
    families = {f"NPC-{index:02d}": "nasopharyngeal" for index in range(39)}

    first = stable_case_splits(families, seed=17)
    second = stable_case_splits(dict(reversed(list(families.items()))), seed=17)

    assert first == second
    assert Counter(first.values()) == {
        "train": 27,
        "validation": 6,
        "test": 6,
    }


def test_prepare_builds_exact_npc_split_and_one_packet_per_case(
    prepared_pipeline: tuple[GPTTrajectoryPipelineConfig, dict[str, object]],
) -> None:
    config, manifest = prepared_pipeline
    packets = sorted((config.npc_workspace / "teacher_packets").glob("*.json"))

    assert len(packets) == 39
    assert manifest["npc_case_count"] == 39
    assert manifest["npc_split_case_counts"] == {
        "test": 6,
        "train": 27,
        "validation": 6,
    }
    assert Counter(
        json.loads(path.read_text(encoding="utf-8"))["split"] for path in packets
    ) == {"train": 27, "validation": 6, "test": 6}


def test_npc_teacher_packets_exclude_rubric_evaluation_and_reference_trajectory(
    prepared_pipeline: tuple[GPTTrajectoryPipelineConfig, dict[str, object]],
) -> None:
    config, _ = prepared_pipeline

    for path in (config.npc_workspace / "teacher_packets").glob("*.json"):
        packet = json.loads(path.read_text(encoding="utf-8"))
        assert packet["packet_type"] == "npc_teacher"
        assert packet["generation_contract"]["rule_compiler_forbidden"] is True
        instructions = " ".join(packet["task_instructions"])
        assert "disease_subtype to npc" in instructions
        assert "never natural-language strings" in instructions
        assert "separate trajectory whose turn_index is 0" in instructions
        assert len(packet["guideline_chunks"]) == 13
        assert len(packet["memory_catalog"]) == 13
        source_refs = [
            item["source_ref"].lower() for item in packet["patient_evidence"]
        ]
        assert source_refs
        assert all("/evaluation/" not in value for value in source_refs)
        assert all("rubric" not in Path(value).name for value in source_refs)
        assert all("trajectory" not in Path(value).name for value in source_refs)


def test_existing_review_packets_group_all_420_records_by_70_cases(
    prepared_pipeline: tuple[GPTTrajectoryPipelineConfig, dict[str, object]],
) -> None:
    config, manifest = prepared_pipeline
    packets = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((config.review_workspace / "review_packets").glob("*.json"))
    ]

    assert manifest["existing_case_count"] == 70
    assert manifest["existing_record_count"] == 420
    assert len(packets) == 70
    assert {len(packet["candidate_records"]) for packet in packets} == {6}
    assert sum(len(packet["candidate_records"]) for packet in packets) == 420
    assert all(packet["review_contract"]["review_whole_case"] for packet in packets)
    assert all(
        {row["base_case_id"] for row in packet["candidate_records"]}
        == {packet["case_id"]}
        for packet in packets
    )


def test_status_rejects_stale_and_schema_invalid_teacher_outputs(
    prepared_pipeline: tuple[GPTTrajectoryPipelineConfig, dict[str, object]],
) -> None:
    config, _ = prepared_pipeline
    packet_path = min((config.npc_workspace / "teacher_packets").glob("*.json"))
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    case_id = packet["case_id"]
    teacher_path = config.npc_workspace / "teacher_outputs" / f"{case_id}.json"

    teacher_path.parent.mkdir(parents=True, exist_ok=True)
    teacher_path.write_text('{"schema_version":', encoding="utf-8")
    interrupted = get_gpt_trajectory_status(config)
    assert case_id in interrupted["npc"]["pending"]["teacher"]

    write_json(
        teacher_path,
        {
            "schema_version": TEACHER_OUTPUT_SCHEMA_VERSION,
            "case_id": case_id,
            "request_hash": "stale-request-hash",
            "model": "gpt-5.6-sol",
            "records": [],
            "claims": [],
        },
    )
    stale = get_gpt_trajectory_status(config)
    assert case_id in stale["npc"]["pending"]["teacher"]

    teacher = {
        "schema_version": TEACHER_OUTPUT_SCHEMA_VERSION,
        "case_id": case_id,
        "request_hash": packet["request_hash"],
        "model": "gpt-5.6-sol",
        "records": [],
        "claims": [],
    }
    write_json(teacher_path, teacher)
    invalid = get_gpt_trajectory_status(config)
    assert case_id in invalid["npc"]["pending"]["teacher_invalid"]
    run_gpt_trajectory_stage("verify", config)
    verifier_packet_path = config.npc_workspace / "verifier_packets" / f"{case_id}.json"
    assert not verifier_packet_path.exists()

    validation = validate_gpt_case_output(packet_path, teacher_path, config)
    assert validation["ready"] is False
    assert any("records must be non-empty" in error for error in validation["errors"])


def test_single_case_review_validator_accepts_complete_review_envelope(
    prepared_pipeline: tuple[GPTTrajectoryPipelineConfig, dict[str, object]],
) -> None:
    config, _ = prepared_pipeline
    packet_path = sorted((config.review_workspace / "review_packets").glob("*.json"))[1]
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    output_path = (
        config.review_workspace / "review_outputs" / f"{packet['case_id']}.json"
    )
    write_json(
        output_path,
        {
            "schema_version": REVIEW_OUTPUT_SCHEMA_VERSION,
            "case_id": packet["case_id"],
            "request_hash": packet["request_hash"],
            "model": "gpt-5.6-sol",
            "decision": "pass_unchanged",
            "findings": [],
            "reviewed_at": "2026-09-03T00:00:00Z",
        },
    )

    assert validate_gpt_case_output(packet_path, output_path, config)["ready"] is True


def test_repair_packet_inherits_split_from_nested_source_packet() -> None:
    packet = {
        "packet_type": "npc_case_repair",
        "source_packet": {"packet_type": "npc_teacher", "split": "validation"},
    }
    assert (
        _validate_case_output_identity(
            "NPC-1",
            packet,
            [{"base_case_id": "NPC-1", "split": "validation"}],
        )
        == []
    )


def test_repair_requires_explicit_superseded_trajectory_ids() -> None:
    packet = {
        "packet_type": "existing_case_repair",
        "split": "train",
        "candidate_output": {
            "records": [
                {"trajectory_id": "old-a"},
                {"trajectory_id": "old-b"},
            ]
        },
    }
    rows = [
        {
            "base_case_id": "CASE-1",
            "trajectory_id": "new-a",
            "split": "train",
            "supersedes_trajectory_ids": ["old-a"],
        }
    ]

    errors = _validate_case_output_identity("CASE-1", packet, rows)

    assert any("old-b" in error for error in errors)

    valid = deepcopy(
        read_jsonl(
            REPO_ROOT
            / "datasets"
            / "planner_v2_pilot"
            / "planner_trajectory.v2.candidates.jsonl"
        )[0]
    )
    valid["supersedes_trajectory_ids"] = ["old-a"]
    assert validate_trajectory_record_v2(valid) == valid


def test_revision_diff_reports_action_state_routing_and_provenance_changes() -> None:
    source = read_jsonl(
        REPO_ROOT
        / "datasets"
        / "planner_v2_pilot"
        / "planner_trajectory.v2.candidates.jsonl"
    )[0]
    revised = deepcopy(source)
    revised["action_set"]["required"][0]["objective"] += "（复核修订）"
    revised["state_after"]["unresolved_information"].append("复核项")
    revised["routing_labels"]["weak_positive_memory_ids"] = ["extra-memory"]
    revised["provenance"]["reviewer_id"] = "gpt-verifier"

    reports = _existing_revision_diffs([source], [revised])

    assert len(reports) == 1
    assert set(reports[0]["record_changes"][0]["changed_sections"]) >= {
        "actions",
        "state",
        "routing",
        "provenance",
    }


def test_quality_scan_rejects_confirmed_evidence_re_request_and_no_progress() -> None:
    source = deepcopy(
        read_jsonl(
            REPO_ROOT
            / "datasets"
            / "planner_v2_pilot"
            / "planner_trajectory.v2.candidates.jsonl"
        )[0]
    )
    source["state_before"]["known_diagnosis"] = "confirmed diagnosis"
    source["action_set"]["required"][0]["expected_state_delta"] = ["known_diagnosis"]
    source["state_delta"] = {
        "last_transition": {"before": None, "after": {"status": "success"}}
    }

    errors = _quality_scan([source])

    assert any("already confirmed evidence" in error for error in errors)
    assert any("no observable clinical state progress" in error for error in errors)


def test_teacher_prevalidation_can_precede_test_split_approval() -> None:
    source = next(
        deepcopy(row)
        for row in read_jsonl(
            REPO_ROOT
            / "datasets"
            / "planner_v2_pilot"
            / "dataset"
            / "test.jsonl"
        )
    )
    source["provenance"]["review_status"] = "pending"
    source["provenance"]["reviewer_id"] = None
    source["provenance"]["reviewed_at"] = None

    report = audit_trajectory_dataset_v2(
        [source],
        release_gates=False,
        require_test_approval=False,
    )

    assert not any("not human-approved" in error for error in report.errors)


def test_pass_unchanged_review_preserves_records_and_human_review_marker(
    prepared_pipeline: tuple[GPTTrajectoryPipelineConfig, dict[str, object]],
) -> None:
    config, _ = prepared_pipeline
    packet_path = min((config.review_workspace / "review_packets").glob("*.json"))
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    case_id = packet["case_id"]
    original = deepcopy(packet["candidate_records"])
    write_json(
        config.review_workspace / "review_outputs" / f"{case_id}.json",
        {
            "schema_version": REVIEW_OUTPUT_SCHEMA_VERSION,
            "case_id": case_id,
            "request_hash": packet["request_hash"],
            "model": "gpt-5.6-sol",
            "decision": "pass_unchanged",
            "findings": [],
            "reviewed_at": "2026-09-03T00:00:00Z",
        },
    )

    records, claims, annotations, stats, errors = _resolve_existing_cases(
        config, [case_id], max_repair_cycles=2
    )

    assert errors == []
    assert claims == []
    assert records == original
    assert stats["decision_counts"] == {"pass_unchanged": 1}
    assert annotations[0]["human_reviewed_current_content"] is True


def test_repaired_content_is_model_reviewed_and_keeps_prior_review_history() -> None:
    source = read_jsonl(
        REPO_ROOT
        / "datasets"
        / "planner_v2_pilot"
        / "planner_trajectory.v2.candidates.jsonl"
    )[0]
    reviewed = _mark_model_reviewed(
        [source],
        {
            "model": "gpt-5.6-sol",
            "model_version": "gpt-5.6-sol",
            "request_hash": "a" * 64,
        },
        {
            "model": "gpt-5.6-sol",
            "reviewed_at": "2026-09-03T00:00:00Z",
        },
    )[0]

    provenance = reviewed["provenance"]
    assert provenance["review_status"] == "approved"
    assert provenance["reviewer_id"] == "gpt-5.6-sol-independent-verifier"
    assert provenance["approval_type"] == "model_review"
    assert provenance["human_reviewed"] is False
    assert provenance["previous_review"]["reviewer_id"] == "manual-review-team"
    assert "model_reviewed_not_human_reviewed" in provenance["qc_flags"]


def test_generated_record_state_delta_is_mechanically_recomputed() -> None:
    source = deepcopy(
        read_jsonl(
            REPO_ROOT
            / "datasets"
            / "planner_v2_pilot"
            / "planner_trajectory.v2.candidates.jsonl"
        )[0]
    )
    source["state_delta"] = {"incorrect": {"before": 1, "after": 2}}
    source["state_after"]["last_transition"]["state_delta"] = {
        "incorrect": {"before": 1, "after": 2}
    }

    normalized = _normalize_generated_records([source])[0]

    assert "incorrect" not in normalized["state_delta"]
    assert (
        "incorrect" not in normalized["state_after"]["last_transition"]["state_delta"]
    )
    assert normalized["state_delta"]["known_diagnosis"]["after"]


def test_failed_qc_blocks_merge_without_touching_source_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    sentinel = source / "planner_trajectory.v2.candidates.jsonl"
    sentinel.write_bytes(b"original-source-bytes\n")
    base = load_gpt_trajectory_pipeline_config(CONFIG_PATH)
    config = replace(
        base,
        source_dataset_root=source,
        npc_workspace=tmp_path / "npc",
        review_workspace=tmp_path / "review",
        merged_dataset_root=tmp_path / "merged",
        manual_approval_path=tmp_path / "manual_approval.json",
    )

    monkeypatch.setattr(
        "guideline_planner.gpt_trajectory_pipeline.validate_gpt_trajectory_outputs",
        lambda *args, **kwargs: {
            "ready": False,
            "errors": ["independent verifier output is missing"],
        },
    )

    with pytest.raises(V2SchemaError, match="failed validation"):
        merge_gpt_trajectory_dataset(config)

    assert sentinel.read_bytes() == b"original-source-bytes\n"
    assert not config.merged_dataset_root.exists()


def test_incomplete_independent_reviews_fail_full_qc(
    prepared_pipeline: tuple[GPTTrajectoryPipelineConfig, dict[str, object]],
) -> None:
    config, _ = prepared_pipeline

    report = validate_gpt_trajectory_outputs(config, write_report=False)

    assert report["ready"] is False
    assert report["statistics"]["npc"]["approved_case_count"] == 0
    assert report["statistics"]["existing"]["record_count"] == 12
    assert any("output is missing" in error for error in report["errors"])
    assert not (config.merged_dataset_root / "dataset").exists()


def test_successful_merge_writes_only_new_dataset_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    sentinel = source / "source.lock"
    sentinel.write_bytes(b"immutable-pilot")
    base = load_gpt_trajectory_pipeline_config(CONFIG_PATH)
    config = replace(
        base,
        source_dataset_root=source,
        npc_workspace=tmp_path / "npc",
        review_workspace=tmp_path / "review",
        merged_dataset_root=tmp_path / "merged",
        manual_approval_path=tmp_path / "manual_approval.json",
    )
    original = read_jsonl(
        REPO_ROOT
        / "datasets"
        / "planner_v2_pilot"
        / "planner_trajectory.v2.candidates.jsonl"
    )[0]

    monkeypatch.setattr(
        "guideline_planner.gpt_trajectory_pipeline.validate_gpt_trajectory_outputs",
        lambda *args, **kwargs: {"ready": True, "errors": []},
    )
    monkeypatch.setattr(
        "guideline_planner.gpt_trajectory_pipeline._packet_ids",
        lambda path: ["existing-case"] if "review" in str(path) else [],
    )
    monkeypatch.setattr(
        "guideline_planner.gpt_trajectory_pipeline._resolve_npc_cases",
        lambda *args, **kwargs: ([], [], {"record_count": 0}, []),
    )
    monkeypatch.setattr(
        "guideline_planner.gpt_trajectory_pipeline._resolve_existing_cases",
        lambda *args, **kwargs: (
            [deepcopy(original)],
            [],
            [{"case_id": original["base_case_id"], "decision": "pass_unchanged"}],
            {"record_count": 1},
            [],
        ),
    )
    monkeypatch.setattr(
        "guideline_planner.gpt_trajectory_pipeline._load_existing_rows",
        lambda *args, **kwargs: [deepcopy(original)],
    )
    monkeypatch.setattr(
        "guideline_planner.gpt_trajectory_pipeline.build_trajectory_dataset_v2",
        lambda *args, **kwargs: {"ready": True, "dataset_hash": "new-dataset"},
    )

    first = merge_gpt_trajectory_dataset(config)
    second = merge_gpt_trajectory_dataset(config)

    assert first == second
    assert first["ready"] is True
    assert sentinel.read_bytes() == b"immutable-pilot"
    assert list(source.iterdir()) == [sentinel]
    assert (config.merged_dataset_root / "generation_manifest.json").is_file()
    assert read_jsonl(
        config.merged_dataset_root / "planner_trajectory.v2.candidates.jsonl"
    ) == [original]
    normalized = json.loads(
        (
            config.merged_dataset_root
            / "normalized_case_outputs"
            / f"{original['base_case_id']}.json"
        ).read_text(encoding="utf-8")
    )
    assert normalized["records"] == [original]
    assert read_jsonl(config.merged_dataset_root / "case_revision_diffs.jsonl") == []
