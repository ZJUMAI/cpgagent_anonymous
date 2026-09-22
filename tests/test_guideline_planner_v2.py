from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from guideline_planner.artifacts import ArtifactBindingError, validate_artifact_hash
from guideline_planner.evaluation_v2 import (
    evaluate_daa_comparison,
    evaluate_planner_predictions,
    evaluate_router_predictions,
)
from guideline_planner.io_utils import read_jsonl, write_jsonl
from guideline_planner.memory import extract_memory_slots
from guideline_planner.phase_machine import can_commit_phase, commit_phase
from guideline_planner.routing_dataset import build_routing_training_data
from guideline_planner.rule_engine import compile_action_set
from guideline_planner.sampling import (
    DeterministicTaskSampler,
    PlannerTrajectorySampler,
    balanced_validation_records,
)
from guideline_planner.schemas_v2 import (
    V2SchemaError,
    stable_case_splits,
    state_delta,
    validate_patient_state_v2,
    validate_planner_action_v2,
)
from guideline_planner.teacher_cache import TeacherResponseCache
from guideline_planner.trajectory_dataset import build_trajectory_dataset_v2
from medclaw_benchmark.patient_state import (
    register_skill_state_adapter,
    update_patient_state,
)
from medclaw_benchmark.planner_skill_adapter import (
    planner_skill_execution_map,
    runtime_tools_for_planner_skill,
)


def test_v2_schema_round_trip_and_v1_rejection() -> None:
    state = _state()
    action = _plan()

    assert validate_patient_state_v2(state) == state
    assert validate_planner_action_v2(
        action,
        patient_state=state,
        active_memories=[_active_memory()],
    ) == action
    with pytest.raises(V2SchemaError, match="schema_version"):
        validate_planner_action_v2(
            {
                "current_phase": "diagnostic_workup",
                "next_step": "legacy",
                "required_skills": [],
            }
        )


def test_planner_semantic_skills_map_to_runtime_tools_and_completed_aliases() -> None:
    state = _state(cancer_family="endometrial", disease_subtype="ucec")
    plan = _plan()
    plan["actions"][0]["required_skills"] = [
        "molecular.complete_biomarker_profile",
        "radiology.review_staging_extent",
        "treatment.select_after_evidence",
    ]
    execution = planner_skill_execution_map(plan, state)
    assert execution["semantic_to_runtime_tools"][
        "molecular.complete_biomarker_profile"
    ] == ["molecular.query_biomarkers"]
    assert execution["semantic_to_runtime_tools"][
        "radiology.review_staging_extent"
    ] == ["radiology.read_ct_manifest", "radiology.ucec_mri_roi"]
    assert execution["decision_only_skills"] == ["treatment.select_after_evidence"]
    assert runtime_tools_for_planner_skill(
        "radiology.review_staging_extent",
        _state(),
    )[-1] == "radiology.lung_tumor_roi"

    updated = update_patient_state(
        _state(),
        [
            {
                "skill_name": "molecular.query_biomarkers",
                "status": "success",
                "result": {"findings": {"summary": "profile complete", "data": {}}},
            }
        ],
        current_time=1,
    )
    assert "molecular.complete_biomarker_profile" in updated["completed_skills"]


def test_v2_validator_allows_reusable_skill_but_rejects_bad_provenance() -> None:
    state = _state(completed_skills=["pathology.read_report"])
    repeated = validate_planner_action_v2(
        _plan(),
        patient_state=state,
        active_memories=[_active_memory()],
    )
    assert repeated["actions"][0]["required_skills"] == ["pathology.read_report"]

    inactive = _plan()
    inactive["actions"][0]["required_skills"] = ["radiology.read_ct_manifest"]
    inactive["actions"][0]["provenance"][0]["memory_id"] = "not-active"
    with pytest.raises(V2SchemaError, match="not active"):
        validate_planner_action_v2(
            inactive,
            patient_state=state,
            active_memories=[_active_memory()],
        )

    wrong_version = _plan()
    wrong_version["actions"][0]["required_skills"] = ["radiology.read_ct_manifest"]
    wrong_version["actions"][0]["provenance"][0]["version"] = "2024"
    with pytest.raises(V2SchemaError, match="version"):
        validate_planner_action_v2(
            wrong_version,
            patient_state=state,
            active_memories=[_active_memory()],
        )

    future_guideline = _state()
    future_guideline["guideline_context"]["guidelines"][0]["effective_from"] = (
        "2026-01-01"
    )
    with pytest.raises(V2SchemaError, match="precedes"):
        validate_patient_state_v2(future_guideline)

    unmet = _plan()
    unmet["actions"][0]["preconditions"] = [
        {"field": "known_stage", "operator": "exists"}
    ]
    with pytest.raises(V2SchemaError, match="preconditions are false"):
        validate_planner_action_v2(
            unmet,
            patient_state=_state(),
            active_memories=[_active_memory()],
        )


def test_runtime_state_update_does_not_apply_planner_action_validator() -> None:
    state = _state(completed_skills=["pathology.read_report"])
    planner_output = _plan()
    planner_output["current_phase"] = "non_taxonomy_runtime_phase"
    planner_output["actions"][0]["provenance"][0]["memory_id"] = "inactive-memory"

    updated = update_patient_state(
        state,
        [],
        planner_output,
        current_time=1,
    )

    assert updated["missing_information"] == planner_output["missing_information"]
    assert updated["previous_actions"][-1]["actions"][0]["provenance"][0][
        "memory_id"
    ] == "inactive-memory"


def test_phase_machine_requires_evidence_and_never_trusts_a_jump() -> None:
    state = _state()
    allowed, reason = can_commit_phase(state, "diagnosis_confirmation")
    assert allowed is False
    assert "diagnosis" in reason.lower()
    with pytest.raises(V2SchemaError, match="Cannot commit"):
        commit_phase(state, "diagnosis_confirmation")

    diagnosed = _state(known_diagnosis="Adenocarcinoma")
    committed = commit_phase(diagnosed, "diagnosis_confirmation")
    assert committed["current_phase"] == "diagnosis_confirmation"
    with pytest.raises(V2SchemaError, match="exactly one"):
        commit_phase(committed, "treatment_selection")


def test_extensible_skill_registry_updates_state_and_audits_unknowns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from medclaw_benchmark import patient_state as patient_state_module

    def custom_adapter(**kwargs: object) -> None:
        kwargs["state"]["risk_stratification"] = {"group": "high"}
        kwargs["changes"]["risk_stratification"] = {"group": "high"}

    monkeypatch.delitem(
        patient_state_module.SKILL_STATE_ADAPTERS,
        "custom.read_risk",
        raising=False,
    )
    register_skill_state_adapter("custom.read_risk", custom_adapter)
    updated = update_patient_state(
        _state(),
        [{"skill_name": "custom.read_risk", "status": "success", "call_id": "c1"}],
    )
    unknown = update_patient_state(
        updated,
        [{"skill_name": "custom.unknown", "status": "success", "call_id": "c2"}],
    )

    assert updated["risk_stratification"] == {"group": "high"}
    assert "custom.read_risk" in updated["completed_skills"]
    assert any("custom.unknown" in item for item in unknown["warnings"])
    assert unknown["completed_actions"][-1]["skill_name"] == "custom.unknown"


def test_case_split_is_stable_grouped_and_stratified() -> None:
    families = {
        f"{family}-{index:02d}": family
        for family in ("lung", "endometrial", "nasopharyngeal")
        for index in range(30)
    }
    first = stable_case_splits(families, seed=17)
    second = stable_case_splits(families, seed=17)
    assert first == second
    for family in set(families.values()):
        values = [first[case_id] for case_id, item in families.items() if item == family]
        assert values.count("train") == 20
        assert values.count("validation") == 5
        assert values.count("test") == 5


def test_memory_sampler_exact_schedule_distributed_and_deterministic() -> None:
    records = [
        {
            "task": task,
            "guideline_id": f"g-{index % 3}",
            "cancer_type": "lung" if index % 2 else "endometrial",
            "id": f"{task}-{index}",
        }
        for task in ("AE", "RETRIEVE", "CONTINUE")
        for index in range(6)
    ]
    sampler = DeterministicTaskSampler(
        records,
        task_ratios={"AE": 5, "RETRIEVE": 5, "CONTINUE": 3},
        seed=11,
    )
    rank0, audit0 = sampler.sample(13, world_size=2, rank=0)
    rank1, audit1 = sampler.sample(13, world_size=2, rank=1)
    assert audit0.exact and audit1.exact
    assert audit0.requested_task_counts == {"AE": 10, "CONTINUE": 6, "RETRIEVE": 10}
    assert len(rank0) == len(rank1) == 13
    repeated, _ = sampler.sample(13, world_size=2, rank=0)
    assert [item["id"] for item in repeated] == [item["id"] for item in rank0]
    next_epoch, _ = sampler.sample(13, world_size=2, rank=0, epoch=1)
    assert [item["id"] for item in next_epoch] != [item["id"] for item in rank0]


def test_planner_sampler_balances_cancer_phase_and_bucket() -> None:
    rows = []
    for family in ("lung", "endometrial", "nasopharyngeal"):
        for phase in ("diagnostic_workup", "staging"):
            row = _trajectory(f"{family}-{phase}", family=family)
            row["state_before"]["current_phase"] = phase
            rows.append(row)
    sampled, audit = PlannerTrajectorySampler(rows, seed=3).sample(30)
    assert audit["realized_family_counts"] == {
        "endometrial": 10,
        "lung": 10,
        "nasopharyngeal": 10,
    }
    assert len(sampled) == 30


def test_generation_validation_balances_all_planner_cancers() -> None:
    rows = []
    for family in ("lung", "endometrial", "nasopharyngeal"):
        for index in range(5):
            row = _trajectory(f"{family}-{index}", family=family)
            row["guideline_context"] = row["state_before"]["guideline_context"]
            rows.append(row)

    selected = balanced_validation_records(rows, 6, seed=17)

    assert {
        family: sum(
            item["state_before"]["cancer_family"] == family for item in selected
        )
        for family in ("lung", "endometrial", "nasopharyngeal")
    } == {"lung": 2, "endometrial": 2, "nasopharyngeal": 2}


def test_rule_compiler_uses_patient_state_conditions() -> None:
    rule = {
        "schema_version": "guideline_rule.v2",
        "rule_id": "rule-lung",
        "cancer_family": "lung",
        "guideline_id": "guide-lung",
        "version": "2025",
        "phase": "diagnostic_workup",
        "condition": {"all": []},
        "allowed_action_types": ["evidence_gathering"],
        "memory_ids": ["memory-lung-positive", "memory-lung-negative"],
        "source_spans": ["memory-lung-positive", "memory-lung-negative"],
        "action_templates": _rule_action_templates(),
    }

    diagnosis_missing = compile_action_set(_state(), [rule])
    diagnosis_known = compile_action_set(
        _state(known_diagnosis="Confirmed carcinoma"),
        [rule],
    )

    assert {item["action_id"] for item in diagnosis_missing["required"]} == {
        "collect-pathology"
    }
    assert {item["action_id"] for item in diagnosis_known["required"]} == {
        "collect-staging"
    }


def test_release_dataset_gates_and_routing_builder(tmp_path: Path) -> None:
    chunks = []
    rules = []
    records = []
    for family in ("lung", "endometrial", "nasopharyngeal"):
        guideline_id = f"guide-{family}"
        positive = f"memory-{family}-positive"
        negative = f"memory-{family}-negative"
        for memory_id in (positive, negative):
            chunks.append(_chunk(memory_id, family, guideline_id))
        rules.append(
            {
                "schema_version": "guideline_rule.v2",
                "rule_id": f"rule-{family}",
                "cancer_family": family,
                "guideline_id": guideline_id,
                "version": "2025",
                "phase": "diagnostic_workup",
                "condition": {"all": []},
                "allowed_action_types": ["evidence_gathering"],
                "memory_ids": [positive, negative],
                "source_spans": [positive, negative],
                "action_templates": _rule_action_templates(),
            }
        )
        for index in range(30):
            pair = _trajectory_pair(
                f"{family}-{index:02d}",
                family=family,
                guideline_id=guideline_id,
                first_memory=positive,
                second_memory=negative,
                rule_id=f"rule-{family}",
            )
            records.extend(pair)
            records.extend(_counterfactual_pair(pair, "decision_relevant"))
            records.extend(_counterfactual_pair(pair, "irrelevant"))
    memory_dir = tmp_path / "memory"
    extract_memory_slots(chunks, memory_dir, memory_tokens=2, mock=True)
    rules_path = tmp_path / "rules.jsonl"
    write_jsonl(rules_path, rules)
    candidates = tmp_path / "candidates.jsonl"
    write_jsonl(candidates, records)

    manifest = build_trajectory_dataset_v2(
        candidates,
        tmp_path / "dataset",
        rule_registry_path=rules_path,
        memory_dir=memory_dir,
        fail_if_not_ready=True,
    )
    assert manifest["ready"] is True
    assert manifest["split_counts"] == {"test": 90, "train": 360, "validation": 90}

    routing_manifest = build_routing_training_data(
        tmp_path / "dataset",
        tmp_path / "routing.jsonl",
    )
    routing = read_jsonl(tmp_path / "routing.jsonl")
    assert routing_manifest["train_example_count"] == 360
    assert all(row["strong_positive_ids"] and row["hard_negative_ids"] for row in routing)
    assert all("next_activation" not in row for row in routing)
    assert all("previous_activation" not in row for row in routing)
    assert {row["split"] for row in routing} == {"train", "validation", "test"}


def test_routing_builder_accepts_current_state_single_turn_labels(tmp_path: Path) -> None:
    record = _trajectory("lung-one-turn")
    source = tmp_path / "train.jsonl"
    write_jsonl(source, [record])

    manifest = build_routing_training_data(
        source,
        tmp_path / "routing.jsonl",
        allow_nonrelease_smoke=True,
    )
    assert manifest["example_count"] == 1


def test_release_dataset_fails_without_rule_and_memory_assets(tmp_path: Path) -> None:
    records = [
        _trajectory(f"lung-{index:02d}", family="lung") for index in range(30)
    ]
    path = tmp_path / "candidates.jsonl"
    write_jsonl(path, records)
    manifest = build_trajectory_dataset_v2(path, tmp_path / "dataset")
    report = json.loads(
        (tmp_path / "dataset" / "admission_report.json").read_text(encoding="utf-8")
    )
    assert manifest["ready"] is False
    assert any("rule registry" in item for item in report["errors"])


def test_artifact_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "adapter.bin"
    artifact.write_bytes(b"v1")
    from guideline_planner.artifacts import sha256_path

    expected = sha256_path(artifact)
    artifact.write_bytes(b"v2")
    with pytest.raises(ArtifactBindingError, match="hash mismatch"):
        validate_artifact_hash(label="adapter", path=artifact, expected_hash=expected)


def test_teacher_cache_is_content_addressed_and_versioned(tmp_path: Path) -> None:
    cache = TeacherResponseCache(tmp_path / "teacher-cache")
    record = cache.put(
        model="teacher",
        model_version="2026-08-01",
        prompt="grounded prompt",
        rule_compiler_version="rules-v2",
        output={"choice": "required-action"},
    )

    assert cache.get(
        model="teacher",
        model_version="2026-08-01",
        prompt="grounded prompt",
        rule_compiler_version="rules-v2",
    ) == record
    assert cache.get(
        model="teacher",
        model_version="different",
        prompt="grounded prompt",
        rule_compiler_version="rules-v2",
    ) is None


def test_fixed_evaluation_gates_include_counterfactuals_and_daa_ci() -> None:
    real = _trajectory_pair(
        "lung-eval",
        family="lung",
        guideline_id="guide-lung",
        first_memory="memory-lung-positive",
        second_memory="memory-lung-negative",
        rule_id="rule-lung",
    )
    trajectories = [
        *real,
        *_counterfactual_pair(real, "decision_relevant"),
        *_counterfactual_pair(real, "irrelevant"),
    ]
    examples = [
        {
            "trajectory": trajectory,
            "prediction": deepcopy(trajectory["accepted_plan_variants"][0]),
            "active_memories": _active_memories_for_trajectory(trajectory),
        }
        for trajectory in trajectories
    ]

    result = evaluate_planner_predictions(examples)
    comparison = evaluate_daa_comparison(
        [0.8, 0.9, 0.7],
        [0.8, 0.9, 0.7],
        samples=200,
    )

    assert result["accepted"] is True
    assert result["counterfactual"] == {
        "decision_relevant_change_rate": 1.0,
        "irrelevant_perturbation_retention_rate": 1.0,
    }
    assert comparison["accepted_as_default"] is True
    assert comparison["bootstrap_95ci_lower"] == pytest.approx(0.0)


def test_reusing_completed_skill_is_valid_when_action_has_state_progress() -> None:
    trajectory = _trajectory_pair(
        "lung-repeatable-skill",
        family="lung",
        guideline_id="guide-lung",
        first_memory="memory-lung-positive",
        second_memory="memory-lung-negative",
        rule_id="rule-lung",
    )[0]
    plan = deepcopy(trajectory["accepted_plan_variants"][0])
    reused_skill = plan["actions"][0]["required_skills"][0]
    trajectory["state_before"]["completed_skills"] = [reused_skill]

    validated = validate_planner_action_v2(
        plan,
        patient_state=trajectory["state_before"],
        active_memories=_active_memories_for_trajectory(trajectory),
    )
    report = evaluate_planner_predictions(
        [
            {
                "trajectory": trajectory,
                "prediction": validated,
                "active_memories": _active_memories_for_trajectory(trajectory),
            }
        ]
    )

    assert report["macro"]["schema_pass_rate"] == 1.0
    assert report["macro"]["completed_skill_repeat_rate"] == 0.0


def test_router_acceptance_uses_only_current_state_memory_labels() -> None:
    def row() -> dict[str, object]:
        return {
            "routing_labels": {
                "strong_positive_memory_ids": ["positive"],
                "weak_positive_memory_ids": [],
            },
            "routing": {
                "active_memories": [
                    {"memory_id": "positive", "weight": 0.8},
                    {"memory_id": "negative", "weight": 0.2},
                ],
                "merged_candidates": [
                    {"memory_id": "positive"},
                    {"memory_id": "negative"},
                ],
                "normalized_gate_entropy": 0.5,
                "candidate_activations": [
                    {"memory_id": "positive", "gate_score": 0.2},
                    {"memory_id": "negative", "gate_score": 0.1},
                ],
            },
        }

    result = evaluate_router_predictions([row(), row()])

    assert result["accepted"] is True
    assert result["metrics"]["positive_recall_at_4"] == 1.0
    assert result["metrics"]["positive_gate_mass"] == pytest.approx(0.8)


def _state(**overrides: object) -> dict[str, object]:
    family = str(overrides.pop("cancer_family", "lung"))
    guideline_id = str(overrides.pop("guideline_id", f"guide-{family}"))
    case_id = str(overrides.pop("case_id", "case-1"))
    state: dict[str, object] = {
        "schema_version": "patient_state.v2",
        "case_id": case_id,
        "cancer_family": family,
        "disease_subtype": "nsclc" if family == "lung" else family,
        "current_phase": "diagnostic_workup",
        "decision_date": "2025-01-01",
        "guideline_context": {
            "decision_date": "2025-01-01",
            "guidelines": [{"guideline_id": guideline_id, "version": "2025"}],
        },
        "known_diagnosis": None,
        "known_stage": None,
        "known_biomarkers": {},
        "risk_stratification": {},
        "available_modalities": ["clinical"],
        "completed_skills": [],
        "completed_actions": [],
        "treatment_history": [],
        "current_treatment_line": 0,
        "evidence_ledger": [],
        "last_transition": None,
        "pending_actions": [],
        "blocked_actions": [],
        "unresolved_information": ["pathology"],
    }
    state.update(overrides)
    return state


def _plan(
    *,
    family: str = "lung",
    guideline_id: str | None = None,
    memory_id: str | None = None,
    rule_id: str | None = None,
    required_skill: str = "pathology.read_report",
    expected_state_delta: str = "known_diagnosis",
    action_id: str = "collect-pathology",
    missing_information: str = "pathology",
) -> dict[str, object]:
    guideline_id = guideline_id or f"guide-{family}"
    memory_id = memory_id or f"memory-{family}-positive"
    rule_id = rule_id or f"rule-{family}"
    return {
        "schema_version": "planner_action.v2",
        "current_phase": "diagnostic_workup",
        "proposed_phase": None,
        "missing_information": [missing_information],
        "actions": [
            {
                "action_id": action_id,
                "objective": "Collect pathology evidence",
                "action_type": "evidence_gathering",
                "required_skills": [required_skill],
                "preconditions": [],
                "expected_state_delta": [expected_state_delta],
                "provenance": [
                    {
                        "memory_id": memory_id,
                        "rule_ids": [rule_id],
                        "source_spans": [memory_id],
                        "guideline_id": guideline_id,
                        "version": "2025",
                    }
                ],
            }
        ],
        "blocked_actions": [],
        "should_stop": False,
        "reason": "Pathology is unresolved.",
    }


def _training_action(
    action_id: str,
    *,
    family: str,
    memory_id: str,
    rule_id: str,
    condition_satisfied: bool | None = None,
    required_skill: str = "pathology.read_report",
    expected_state_delta: str = "known_diagnosis",
) -> dict[str, object]:
    return {
        "action_id": action_id,
        "objective": f"Objective {action_id}",
        "action_type": "evidence_gathering",
        "required_skills": [required_skill],
        "preconditions": [],
        "expected_state_delta": [expected_state_delta],
        "guideline_rule_ids": [rule_id],
        "supporting_memory_ids": [memory_id],
        "source_spans": [memory_id],
        "condition_satisfied": condition_satisfied,
        "risk_level": "high" if "unsafe" in action_id else "low",
    }


def _trajectory(
    case_id: str,
    *,
    family: str = "lung",
    guideline_id: str | None = None,
    positive_memory: str | None = None,
    negative_memory: str | None = None,
    rule_id: str | None = None,
) -> dict[str, object]:
    guideline_id = guideline_id or f"guide-{family}"
    positive_memory = positive_memory or f"memory-{family}-positive"
    negative_memory = negative_memory or f"memory-{family}-negative"
    rule_id = rule_id or f"rule-{family}"
    before = _state(
        case_id=case_id,
        cancer_family=family,
        guideline_id=guideline_id,
    )
    after = deepcopy(before)
    after["known_diagnosis"] = "Confirmed carcinoma"
    after["completed_skills"] = ["pathology.read_report"]
    after["last_transition"] = {
        "action_id": "collect-pathology",
        "tool_status": "success",
        "result_summary": "Diagnosis confirmed",
        "state_delta": {
            "known_diagnosis": {
                "before": None,
                "after": "Confirmed carcinoma",
            }
        },
    }
    buckets = {
        "required": [
            _training_action(
                "collect-pathology",
                family=family,
                memory_id=positive_memory,
                rule_id=rule_id,
            )
        ],
        "acceptable": [],
        "conditional": [
            _training_action(
                "conditional",
                family=family,
                memory_id=positive_memory,
                rule_id=rule_id,
                condition_satisfied=True,
            ),
            _training_action(
                "conditional-deferred",
                family=family,
                memory_id=positive_memory,
                rule_id=rule_id,
                condition_satisfied=False,
                required_skill="treatment.start_if_stage_known",
                expected_state_delta="treatment_history",
            ),
        ],
        "premature": [
            _training_action(
                "premature",
                family=family,
                memory_id=positive_memory,
                rule_id=rule_id,
                required_skill="treatment.start_without_staging",
                expected_state_delta="treatment_history",
            )
        ],
        "unsafe": [
            _training_action(
                "unsafe",
                family=family,
                memory_id=positive_memory,
                rule_id=rule_id,
                required_skill="treatment.start_unsafe_regimen",
                expected_state_delta="treatment_history",
            )
        ],
    }
    return {
        "schema_version": "planner_trajectory.v2",
        "trajectory_id": f"trajectory-{case_id}",
        "base_case_id": case_id,
        "case_id": case_id,
        "turn_index": 0,
        "split": "train",
        "case_source": "real",
        "guideline_context": deepcopy(before["guideline_context"]),
        "state_before": before,
        "action_set": buckets,
        "accepted_plan_variants": [
            _plan(
                family=family,
                guideline_id=guideline_id,
                memory_id=positive_memory,
                rule_id=rule_id,
            )
        ],
        "tool_executions": [
            {"skill_name": "pathology.read_report", "status": "success"}
        ],
        "state_after": after,
        "state_delta": state_delta(before, after),
        "routing_labels": {
            "strong_positive_memory_ids": [positive_memory],
            "weak_positive_memory_ids": [],
            "hard_negative_memory_ids": [negative_memory],
            "easy_negative_memory_ids": [],
        },
        "provenance": {
            "rule_compiler_version": "v2-test",
            "case_source_ref": f"registry://test/{case_id}",
            "teacher_model": "teacher-test",
            "teacher_model_version": "2026-08-01",
            "teacher_prompt_hash": "a" * 64,
            "teacher_output_hash": "b" * 64,
            "review_status": "approved",
            "reviewer_id": "reviewer-test",
            "reviewed_at": "2026-08-18T00:00:00Z",
            "qc_flags": [],
        },
    }


def _active_memory() -> dict[str, object]:
    return {
        "guideline_memory_id": "memory-lung-positive",
        "metadata": {
            "guideline_id": "guide-lung",
            "version": "2025",
            "source_rule_ids": ["rule-lung"],
            "source_span_ids": ["memory-lung-positive"],
        },
    }


def _active_memories_for_trajectory(
    trajectory: dict[str, object],
) -> list[dict[str, object]]:
    plan = trajectory["accepted_plan_variants"][0]
    provenance = plan["actions"][0]["provenance"][0]
    return [
        {
            "guideline_memory_id": provenance["memory_id"],
            "metadata": {
                "guideline_id": provenance["guideline_id"],
                "version": provenance["version"],
                "source_rule_ids": list(provenance["rule_ids"]),
                "source_span_ids": list(provenance["source_spans"]),
            },
        }
    ]


def _trajectory_pair(
    case_id: str,
    *,
    family: str,
    guideline_id: str,
    first_memory: str,
    second_memory: str,
    rule_id: str,
) -> list[dict[str, object]]:
    first = _trajectory(
        case_id,
        family=family,
        guideline_id=guideline_id,
        positive_memory=first_memory,
        negative_memory=second_memory,
        rule_id=rule_id,
    )
    first["state_after"]["unresolved_information"] = ["staging"]  # type: ignore[index]
    first["state_delta"] = state_delta(first["state_before"], first["state_after"])  # type: ignore[arg-type]

    before = deepcopy(first["state_after"])
    after = deepcopy(before)
    after["known_stage"] = "Staging evidence available"
    after["completed_skills"] = [
        *before["completed_skills"],
        "radiology.read_ct_manifest",
    ]
    after["last_transition"] = {
        "action_id": "collect-staging",
        "tool_status": "success",
        "result_summary": "Staging evidence collected",
        "state_delta": {
            "known_stage": {
                "before": None,
                "after": "Staging evidence available",
            }
        },
    }
    action_kwargs = {
        "family": family,
        "memory_id": second_memory,
        "rule_id": rule_id,
        "required_skill": "radiology.read_ct_manifest",
        "expected_state_delta": "known_stage",
    }
    buckets = {
        "required": [_training_action("collect-staging", **action_kwargs)],
        "acceptable": [],
        "conditional": [
            _training_action(
                "conditional-staging",
                condition_satisfied=True,
                **action_kwargs,
            ),
            _training_action(
                "conditional-deferred-staging",
                condition_satisfied=False,
                **{
                    **action_kwargs,
                    "required_skill": "treatment.start_if_stage_known",
                    "expected_state_delta": "treatment_history",
                },
            ),
        ],
        "premature": [
            _training_action(
                "premature-staging",
                **{
                    **action_kwargs,
                    "required_skill": "treatment.start_without_staging",
                    "expected_state_delta": "treatment_history",
                },
            )
        ],
        "unsafe": [
            _training_action(
                "unsafe-staging",
                **{
                    **action_kwargs,
                    "required_skill": "treatment.start_unsafe_regimen",
                    "expected_state_delta": "treatment_history",
                },
            )
        ],
    }
    second: dict[str, object] = {
        "schema_version": "planner_trajectory.v2",
        "trajectory_id": first["trajectory_id"],
        "base_case_id": case_id,
        "case_id": case_id,
        "turn_index": 1,
        "split": "train",
        "case_source": "real",
        "guideline_context": deepcopy(before["guideline_context"]),
        "state_before": before,
        "action_set": buckets,
        "accepted_plan_variants": [
            _plan(
                family=family,
                guideline_id=guideline_id,
                memory_id=second_memory,
                rule_id=rule_id,
                required_skill="radiology.read_ct_manifest",
                expected_state_delta="known_stage",
                action_id="collect-staging",
                missing_information="staging",
            )
        ],
        "tool_executions": [
            {"skill_name": "radiology.read_ct_manifest", "status": "success"}
        ],
        "state_after": after,
        "state_delta": state_delta(before, after),
        "routing_labels": {
            "strong_positive_memory_ids": [second_memory],
            "weak_positive_memory_ids": [],
            "hard_negative_memory_ids": [first_memory],
            "easy_negative_memory_ids": [],
        },
        "provenance": deepcopy(first["provenance"]),
    }
    return [first, second]


def _counterfactual_pair(
    real_pair: list[dict[str, object]],
    perturbation_type: str,
) -> list[dict[str, object]]:
    suffix = "relevant" if perturbation_type == "decision_relevant" else "irrelevant"
    pair = deepcopy(real_pair)
    real_trajectory_id = str(real_pair[0]["trajectory_id"])
    new_trajectory_id = f"{real_trajectory_id}-cf-{suffix}"
    new_case_id = f"{real_pair[0]['case_id']}-cf-{suffix}"
    for real, item in zip(real_pair, pair):
        item["trajectory_id"] = new_trajectory_id
        item["case_id"] = new_case_id
        item["case_source"] = "counterfactual"
        item["counterfactual"] = {
            "anchor_trajectory_id": real_trajectory_id,
            "anchor_turn_index": real["turn_index"],
            "perturbation_type": perturbation_type,
            "changed_fields": [
                "known_biomarkers"
                if perturbation_type == "decision_relevant"
                else "administrative_note"
            ],
        }
        for state_name in ("state_before", "state_after"):
            state = item[state_name]
            state["case_id"] = new_case_id
            if perturbation_type == "decision_relevant":
                state["known_biomarkers"] = {"EGFR": "positive"}
            else:
                state["administrative_note"] = "counterfactual transport mode"
        if perturbation_type == "decision_relevant":
            _swap_pair_memory_labels(item)
        item["state_delta"] = state_delta(item["state_before"], item["state_after"])
    return pair


def _swap_pair_memory_labels(record: dict[str, object]) -> None:
    labels = record["routing_labels"]
    first = labels["strong_positive_memory_ids"][0]
    second = labels["hard_negative_memory_ids"][0]
    replacement = {first: second, second: first}
    for key in (
        "strong_positive_memory_ids",
        "weak_positive_memory_ids",
        "hard_negative_memory_ids",
        "easy_negative_memory_ids",
    ):
        labels[key] = [replacement.get(value, value) for value in labels[key]]
    for bucket in record["action_set"].values():
        for action in bucket:
            action["supporting_memory_ids"] = [
                replacement.get(value, value)
                for value in action["supporting_memory_ids"]
            ]
            action["source_spans"] = [
                replacement.get(value, value) for value in action["source_spans"]
            ]
    for plan in record["accepted_plan_variants"]:
        for action in plan["actions"]:
            for provenance in action["provenance"]:
                provenance["memory_id"] = replacement.get(
                    provenance["memory_id"], provenance["memory_id"]
                )
                provenance["source_spans"] = [
                    replacement.get(value, value)
                    for value in provenance["source_spans"]
                ]


def _chunk(memory_id: str, family: str, guideline_id: str) -> dict[str, object]:
    return {
        "guideline_id": guideline_id,
        "guideline_name": guideline_id,
        "version": "2025",
        "cancer_type": family,
        "chapter": "Diagnosis",
        "section": "Diagnosis",
        "h1_title": memory_id,
        "source_span_id": memory_id,
        "source_span_ids": [memory_id],
        "source_rule_ids": [f"rule-{family}"],
        "page_start": 1,
        "page_end": 1,
        "text": f"Grounded recommendation for {family}.",
    }


def _rule_action_templates() -> list[dict[str, object]]:
    result = []
    for prefix, condition, skill, expected in (
        (
            "pathology",
            {"field": "known_diagnosis", "operator": "missing"},
            "pathology.read_report",
            "known_diagnosis",
        ),
        (
            "staging",
            {"field": "known_diagnosis", "operator": "exists"},
            "radiology.read_ct_manifest",
            "known_stage",
        ),
    ):
        entries = (
            (f"collect-{prefix}", "required", skill, expected),
            (
                f"conditional-{prefix}" if prefix == "staging" else "conditional",
                "conditional",
                skill,
                expected,
            ),
            (
                f"premature-{prefix}" if prefix == "staging" else "premature",
                "premature",
                "treatment.start_without_staging",
                "treatment_history",
            ),
            (
                f"unsafe-{prefix}" if prefix == "staging" else "unsafe",
                "unsafe",
                "treatment.start_unsafe_regimen",
                "treatment_history",
            ),
        )
        for action_id, bucket, required_skill, state_field in entries:
            result.append(
                {
                    "action_id": action_id,
                    "objective": f"Objective {action_id}",
                    "action_type": "evidence_gathering",
                    "required_skills": [required_skill],
                    "preconditions": [],
                    "expected_state_delta": [state_field],
                    "condition": condition,
                    "bucket_if_true": bucket,
                    "bucket_if_false": None,
                }
            )
        result.append(
            {
                "action_id": (
                    "conditional-deferred-staging"
                    if prefix == "staging"
                    else "conditional-deferred"
                ),
                "objective": (
                    "Objective conditional-deferred-staging"
                    if prefix == "staging"
                    else "Objective conditional-deferred"
                ),
                "action_type": "evidence_gathering",
                "required_skills": ["treatment.start_if_stage_known"],
                "preconditions": [],
                "expected_state_delta": ["treatment_history"],
                "condition": {
                    "field": "known_stage",
                    "operator": "exists",
                },
                "bucket_if_true": None,
                "bucket_if_false": "conditional",
            }
        )
    return result
