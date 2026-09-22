from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from guideline_planner.planner import LatentPlannerError
from guideline_planner.routing_types import MemoryActivation, PlannerStepResult
from medclaw.llm.qwen_client import QwenCompletion
from medclaw.registry.skill_registry import SkillRegistry
from medclaw.stores.artifact_store import ArtifactStore

from medclaw_benchmark.batch_runner import (
    BatchRunConfig,
    discover_ready_cases,
    run_batch,
)
from medclaw_benchmark import batch_runner
from medclaw_benchmark.case_builder import CaseBuilder
from medclaw_benchmark.case_simulator import CaseSimulator
from medclaw_benchmark import cli as benchmark_cli
from medclaw_benchmark import dual_agent_runner
from medclaw_benchmark.dual_agent_runner import (
    DualAgentBatchConfig,
    DualAgentBenchmarkRunner,
    run_dual_agent_batch,
)
from medclaw_benchmark.llm_judge import (
    DIMENSION_WEIGHTS,
    LLMRubricJudge,
    _extract_json_object,
)
from medclaw_benchmark.patient_state import (
    current_patient_state_view,
    initialize_patient_state,
    update_patient_state,
)
from medclaw_benchmark.runner import (
    BENCHMARK_AGENT_SYSTEM_PROMPT,
    BENCHMARK_AGENT_USER_PROMPT,
    BenchmarkRunner,
    _bind_release_guideline_arguments,
    _apply_guideline_env_defaults,
    _build_guideline_context_arguments,
)
from medclaw_benchmark.planner_preflight import (
    _rubric_guideline_versions,
    preflight_planner_case,
)
from medclaw_benchmark.skill_resolver import SkillResolver
from medclaw_benchmark.trajectory_scorer import score_trajectory_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "TCGA-38-4626"
SLIDE_STEM = "TCGA-38-4626-01Z-00-DX1.142bc018-bd79-4db9-84e2-f83a617ea92a"


@pytest.fixture(autouse=True)
def disable_case_builder_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDCLAW_CASE_BUILDER_LLM_MODE", "disabled")


def copy_lightweight_case(tmp_path: Path, *, with_modalities: bool = True) -> Path:
    case_dir = tmp_path / "examples" / "cases" / CASE_ID
    case_dir.mkdir(parents=True)
    report_dir = case_dir / "reports" / "integrated_reports"
    report_dir.mkdir(parents=True)
    (report_dir / f"{CASE_ID}_T0_full_report_zh.md").write_text(
        "\n".join(
            [
                f"# {CASE_ID} 完整病例报告",
                "",
                "## 临床、病理与样本报告",
                "",
                "### 病人概览",
                "57岁女性患者，白种人，非西班牙裔或拉丁裔，美国居民。"
                "末次随访时生存状态为存活。",
                "",
                "### 临床与人口学信息",
                "- **性别**: 女性",
                "- **入组时年龄**: 57岁",
                "- **种族**: 白种人",
                "- **民族**: 非西班牙裔或拉丁裔",
                "- **居住国**: 美国",
                "",
                "### 诊断、分期与肿瘤特征",
                "- **年龄 at 诊断**: 23421天",
                "- **肿瘤分类**: subsequent primary",
                "- **是否为原发疾病**: 是",
                "- **诊断前曾接受治疗**: 是",
                "",
                "### 治疗信息",
                "患者接受过手术、药物治疗和放射治疗记录，但均标记为未接受该疗法。",
                "",
                "### 随访与结局",
                "末次随访时间为第3674天，疾病反应为无肿瘤，末次随访时生存状态为存活。",
                "",
                "### 病理详情",
                "#### 病理报告原文",
                "Diagnosis: Lung, left upper lobe, lobectomy. Adenocarcinoma, "
                "poorly differentiated, 5.3 cm diameter. No metastatic carcinoma "
                "identified in sampled lymph nodes.",
                "",
                "### 病理切片信息",
                "**切片1** (原发肿瘤样本，切片TCGA-38-4626-01A-01-BS1):",
                "- 切面位置: BOTTOM",
                "- 肿瘤细胞占比: 95.0%",
                "- 肿瘤细胞核占比: 75.0%",
                "",
                "### 数据追踪信息",
                "- **项目**: TCGA-LUAD",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    evaluation_dir = case_dir / "evaluation"
    evaluation_dir.mkdir()
    (evaluation_dir / f"{CASE_ID}_rubric.json").write_text(
        json.dumps(
            {
                "case_id": CASE_ID,
                "rubric_version": "test_v1",
                "case_summary": ["test synthetic case"],
                "rubric": {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if with_modalities:
        ct_path = case_dir / "radiology" / "nifti" / f"{CASE_ID}_T0_ct_preprocessed.nii.gz"
        ct_path.parent.mkdir(parents=True)
        ct_path.write_bytes(b"fake-ct")
        wsi_path = case_dir / "pathology" / "wsi" / f"{SLIDE_STEM}.svs"
        wsi_path.parent.mkdir(parents=True)
        wsi_path.write_bytes(b"fake-wsi")
        conch_dir = case_dir / "pathology" / "roi_256" / SLIDE_STEM
        conch_dir.mkdir(parents=True)
        (conch_dir / "prompt_scores.json").write_text("{}", encoding="utf-8")
    return case_dir


def test_case_builder_and_simulator_gate_hidden_fields(tmp_path: Path) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=False)

    CaseBuilder(case_dir).build()
    hidden = json.loads((case_dir / "hidden_state.json").read_text(encoding="utf-8"))
    trajectory = json.loads(
        (case_dir / "evaluation" / f"{CASE_ID}_trajectory.json").read_text(
            encoding="utf-8"
        )
    )

    assert hidden["case_id"] == CASE_ID
    assert hidden["initial_prompt"]["age"] == 57
    assert "3674" not in json.dumps(hidden["initial_prompt"], ensure_ascii=False)
    assert "rubric" not in json.dumps(hidden["initial_prompt"], ensure_ascii=False).lower()
    assert trajectory["case_id"] == CASE_ID
    assert trajectory["schema_version"] == "trajectory.dynamic_rubric.v1"
    assert trajectory["patient_event_table"]
    assert trajectory["trajectory"]

    simulator = CaseSimulator(case_dir)
    initial = simulator.get_initial_observation()
    assert initial["visible_information"]["sex"] == "female"

    clinical = simulator.query("clinical.read_summary", {}, "diagnosis_phase")
    clinical_text = json.dumps(clinical, ensure_ascii=False)
    assert "vital_status_at_last_follow_up" not in clinical_text
    assert "3674" not in clinical_text

    follow_up_hidden = simulator.query("clinical.read_follow_up", {}, "diagnosis_phase")
    assert follow_up_hidden["findings"]["data"]["released"] is False

    follow_up = simulator.query("clinical.read_follow_up", {}, "outcome_phase")
    assert follow_up["findings"]["data"]["vital_status"] == "alive"


def test_case_builder_uses_llm_report_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEDCLAW_CASE_BUILDER_LLM_MODE", "required")
    case_dir = copy_lightweight_case(tmp_path, with_modalities=False)
    client = FakeReportExtractionClient()

    CaseBuilder(case_dir, llm_client=client).build()

    assert client.requests
    clinical = json.loads((case_dir / "clinical" / "clinical.json").read_text(encoding="utf-8"))
    assert clinical["extraction_method"] == "llm"
    assert clinical["primary_diagnosis"] == "Adenocarcinoma, NOS"
    assert clinical["pack_years"] == 40
    assert clinical["tumor_size_cm"] == 5.3

    molecular = json.loads((case_dir / "molecular" / "biomarkers.json").read_text(encoding="utf-8"))
    assert molecular["extraction_method"] == "llm"
    assert molecular["molecular_subtype"] == "common-driver-negative / CNV-dominant"
    assert molecular["EGFR"].startswith("alteration_status=negative")
    assert molecular["ALK"].startswith("alteration_status=uncertain")
    assert molecular["biomarker_details"]["KRAS"]["alteration_status"] == "negative"

    extraction = json.loads((case_dir / "report_extraction.json").read_text(encoding="utf-8"))
    assert extraction["status"] == "success"
    assert extraction["model"] == "fake-extractor"


def test_case_builder_reuses_cached_llm_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEDCLAW_CASE_BUILDER_LLM_MODE", "required")
    case_dir = copy_lightweight_case(tmp_path, with_modalities=False)

    first_client = FakeReportExtractionClient()
    CaseBuilder(case_dir, llm_client=first_client).build()
    assert len(first_client.requests) == 1

    second_client = ExplodingReportExtractionClient()
    CaseBuilder(case_dir, llm_client=second_client).build()

    extraction = json.loads((case_dir / "report_extraction.json").read_text(encoding="utf-8"))
    assert extraction["method"] == "llm"
    assert extraction["cache_status"] == "hit_report_extraction"
    assert any("Reused cached" in item for item in extraction["warnings"])


def test_skill_resolver_finds_existing_roi_skills() -> None:
    registry = SkillRegistry(PROJECT_ROOT / "medclaw" / "skills")
    resolver = SkillResolver(registry)

    assert resolver.resolve("radiology.ct_roi") == "radiology.lung_tumor_roi"
    assert resolver.resolve("pathology.conch_v1_5_roi") == "pathology.conch_patch_roi"
    assert resolver.resolve("guideline.csco_retrieve") == "guideline.retrieve"


def test_guideline_context_query_is_not_lung_hardcoded(tmp_path: Path) -> None:
    records = [
        {
            "skill_name": "clinical.read_summary",
            "raw_output_path": str(tmp_path / "clinical.json"),
        },
        {
            "skill_name": "molecular.query_biomarkers",
            "raw_output_path": str(tmp_path / "molecular.json"),
        },
    ]
    (tmp_path / "clinical.json").write_text(
        json.dumps(
            {
                "findings": {
                    "data": {
                        "project_id": "TCGA-UCEC",
                        "primary_diagnosis": "Endometrioid adenocarcinoma",
                        "tumor_location": "Endometrium",
                        "pathologic_stage": "Stage II",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "molecular.json").write_text(
        json.dumps(
            {
                "findings": {
                    "data": {
                        "summary": "POLE mutation detected.",
                        "biomarker_details": {"POLE": {"alteration_status": "positive"}},
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    arguments = _build_guideline_context_arguments("CASE-UCEC", records)

    assert arguments["cancer_type"] == "auto"
    assert "TCGA-UCEC" in arguments["query"]
    assert "POLE" in arguments["query"]
    assert "非小细胞" not in arguments["query"]
    assert "肺癌" not in arguments["query"]


def test_guideline_env_overrides_force_chapter_single_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEDCLAW_GUIDELINE_CHUNK_MODE", "chapter")
    monkeypatch.setenv("MEDCLAW_GUIDELINE_MAX_SNIPPETS", "1")
    monkeypatch.setenv("MEDCLAW_GUIDELINE_RERANK_CANDIDATE_COUNT", "24")

    arguments = _apply_guideline_env_defaults(
        {
            "case_id": "TCGA-38-4626",
            "query": "existing query",
            "max_snippets": 6,
        }
    )

    assert arguments["chunk_mode"] == "chapter"
    assert arguments["max_snippets"] == 1
    assert arguments["rerank_candidate_count"] == 24

    monkeypatch.delenv("MEDCLAW_GUIDELINE_MAX_SNIPPETS")
    arguments = _apply_guideline_env_defaults(
        {
            "case_id": "TCGA-38-4626",
            "query": "existing query",
            "max_snippets": 6,
        }
    )
    assert arguments["chunk_mode"] == "chapter"
    assert arguments["max_snippets"] == 1


def test_guideline_defaults_to_chapter_single_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MEDCLAW_GUIDELINE_CHUNK_MODE", raising=False)
    monkeypatch.delenv("MEDCLAW_GUIDELINE_MAX_SNIPPETS", raising=False)

    arguments = _apply_guideline_env_defaults(
        {
            "case_id": "TCGA-38-4626",
            "query": "existing query",
            "max_snippets": 6,
        }
    )

    assert arguments["chunk_mode"] == "chapter"
    assert arguments["max_snippets"] == 1


def test_guideline_explicit_page_mode_remains_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MEDCLAW_GUIDELINE_CHUNK_MODE", raising=False)
    monkeypatch.delenv("MEDCLAW_GUIDELINE_MAX_SNIPPETS", raising=False)

    arguments = _apply_guideline_env_defaults(
        {
            "case_id": "TCGA-38-4626",
            "query": "legacy retrieval experiment",
            "chunk_mode": "page",
            "max_snippets": 6,
        }
    )

    assert arguments["chunk_mode"] == "page"
    assert arguments["max_snippets"] == 6


def test_benchmark_agent_prompt_requires_guideline_page_citations() -> None:
    prompt = BENCHMARK_AGENT_SYSTEM_PROMPT + "\n" + BENCHMARK_AGENT_USER_PROMPT

    assert "guideline title/source" in prompt
    assert "page number" in prompt
    assert "snippet/chunk id" in prompt
    assert "指南依据及页码" in prompt


def test_batch_discovery_requires_evaluation_dir(tmp_path: Path) -> None:
    cases_root = tmp_path / "processed" / "LUNG"
    ready = cases_root / "TCGA-READY"
    not_ready = cases_root / "TCGA-NOTREADY"
    (ready / "evaluation").mkdir(parents=True)
    not_ready.mkdir(parents=True)

    assert discover_ready_cases(cases_root) == [ready.resolve()]


def test_batch_run_skips_completed_case_and_writes_summary(tmp_path: Path) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=False)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / CASE_ID / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "final_answers.json").write_text("{}", encoding="utf-8")
    (run_dir / "judge_scores.json").write_text("{}", encoding="utf-8")

    summary = run_batch(
        BatchRunConfig(
            cases_root=case_dir.parent,
            runs_root=runs_root,
            run_id="run_001",
            dry_run=False,
        )
    )

    assert summary["counts"] == {
        "total": 1,
        "completed": 0,
        "skipped": 1,
        "failed": 0,
    }
    assert summary["cases"][0]["status"] == "skipped"
    assert (runs_root / "batch_summary_run_001.json").is_file()


def test_batch_run_sets_medclaw_cases_root_for_case_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=False)
    runs_root = tmp_path / "runs"
    run_id = "run_env"
    seen: dict[str, str | None] = {}
    monkeypatch.delenv("MEDCLAW_CASES_ROOT", raising=False)

    class FakeRunner:
        def __init__(
            self,
            *,
            case_dir: Path,
            run_id: str,
            runs_root: Path,
            agent: str,
        ) -> None:
            self.case_id = Path(case_dir).name
            self.run_id = run_id
            self.run_dir = Path(runs_root) / self.case_id / run_id

        def run(self) -> Any:
            seen["MEDCLAW_CASES_ROOT"] = os.environ.get("MEDCLAW_CASES_ROOT")
            self.run_dir.mkdir(parents=True, exist_ok=True)
            return type("FakeResult", (), {"run_dir": self.run_dir})()

    class FakeJudge:
        def __init__(
            self,
            rubric_path: Path,
            run_dir: Path,
            provider: str | None = None,
        ) -> None:
            self.rubric_path = rubric_path
            self.run_dir = run_dir

        def evaluate(self) -> dict[str, Any]:
            return {"status": "success", "final_total": 100}

    monkeypatch.setattr(batch_runner, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(batch_runner, "LLMRubricJudge", FakeJudge)

    summary = run_batch(
        BatchRunConfig(
            cases_root=case_dir.parent,
            runs_root=runs_root,
            run_id=run_id,
            dry_run=False,
        )
    )

    assert summary["counts"]["completed"] == 1
    assert seen["MEDCLAW_CASES_ROOT"] == str(case_dir.parent.resolve())
    assert "MEDCLAW_CASES_ROOT" not in os.environ


def test_batch_default_disables_llm_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=False)
    runs_root = tmp_path / "runs"

    class FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            self.run_dir = (
                Path(kwargs["runs_root"])
                / Path(kwargs["case_dir"]).name
                / kwargs["run_id"]
            )

        def run(self) -> Any:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "final_answers.json").write_text("{}", encoding="utf-8")
            return type("FakeResult", (), {"run_dir": self.run_dir})()

    class ForbiddenJudge:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("LLMRubricJudge must not be created by default")

    monkeypatch.setattr(batch_runner, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(batch_runner, "LLMRubricJudge", ForbiddenJudge)

    summary = run_batch(
        BatchRunConfig(
            cases_root=case_dir.parent,
            runs_root=runs_root,
            run_id="no_eval",
        )
    )

    run_dir = runs_root / CASE_ID / "no_eval"
    assert summary["config"]["evaluation_enabled"] is False
    assert summary["evaluation_enabled"] is False
    assert summary["cases"][0]["execution_mode"] == "inference_only"
    for filename in (
        "judge_prompt.json",
        "judge_raw_response.json",
        "judge_scores.json",
        "trajectory_scores.json",
        "error_analysis.md",
    ):
        assert not (run_dir / filename).exists()


def test_batch_can_evaluate_existing_inference_without_rerunning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=False)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / CASE_ID / "late_eval"
    run_dir.mkdir(parents=True)
    (run_dir / "final_answers.json").write_text("{}", encoding="utf-8")
    judge_calls: list[Path] = []

    class ForbiddenRunner:
        def __init__(self, **kwargs: Any) -> None:
            raise AssertionError("completed inference must not be rerun")

    class FakeJudge:
        def __init__(
            self,
            rubric_path: Path,
            run_dir: Path,
            provider: str | None = None,
        ) -> None:
            self.run_dir = Path(run_dir)

        def evaluate(self) -> dict[str, Any]:
            judge_calls.append(self.run_dir)
            for filename in (
                "judge_prompt.json",
                "judge_raw_response.json",
                "judge_scores.json",
                "trajectory_scores.json",
            ):
                (self.run_dir / filename).write_text("{}", encoding="utf-8")
            (self.run_dir / "error_analysis.md").write_text("# ok\n", encoding="utf-8")
            return {"status": "success", "final_total": 88}

    monkeypatch.setattr(batch_runner, "BenchmarkRunner", ForbiddenRunner)
    monkeypatch.setattr(batch_runner, "LLMRubricJudge", FakeJudge)

    summary = run_batch(
        BatchRunConfig(
            cases_root=case_dir.parent,
            runs_root=runs_root,
            run_id="late_eval",
            evaluate=True,
        )
    )

    assert judge_calls == [run_dir]
    assert summary["cases"][0]["execution_mode"] == "evaluation_only"
    assert summary["evaluation_enabled"] is True
    assert summary["cases"][0]["final_total"] == 88
    assert (run_dir / "error_analysis.md").is_file()


def test_patient_state_initializes_and_updates_from_trusted_tools(tmp_path: Path) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=False)
    (case_dir / "case_manifest.json").write_text(
        json.dumps(
            {
                "case_id": CASE_ID,
                "cancer_type": "nsclc",
                "available_modalities": {
                    "radiology": True,
                    "pathology": True,
                    "reports": True,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    initial = {
        "case_id": CASE_ID,
        "phase": "diagnosis_phase",
        "visible_information": {"age": 57, "chief_problem": "lung mass"},
    }
    state = initialize_patient_state(case_dir, initial)
    assert state["case_id"] == CASE_ID
    assert state["cancer_type"] == "nsclc"
    assert state["available_modalities"]["radiology"] is True

    clinical_output = tmp_path / "clinical_output.json"
    clinical_output.write_text(
        json.dumps(
            {
                "status": "success",
                "findings": {
                    "summary": "clinical summary",
                    "data": {
                        "primary_diagnosis": "Adenocarcinoma, NOS",
                        "pathologic_stage": "Stage IIB",
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    molecular_output = tmp_path / "molecular_output.json"
    molecular_output.write_text(
        json.dumps(
            {
                "status": "success",
                "findings": {
                    "summary": "molecular summary",
                    "data": {
                        "summary": "No canonical driver mutation.",
                        "biomarker_details": {
                            "EGFR": {"alteration_status": "negative"},
                            "ALK": {"alteration_status": "negative"},
                        },
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    guideline = state["guideline_context"]["guidelines"][0]
    updated = update_patient_state(
        state,
        [
            {
                "skill_name": "clinical.read_summary",
                "status": "success",
                "call_id": "c1",
                "raw_output_path": str(clinical_output),
            },
            {
                "skill_name": "molecular.query_biomarkers",
                "status": "success",
                "call_id": "m1",
                "raw_output_path": str(molecular_output),
            },
            {
                "skill_name": "unknown.skill",
                "status": "success",
                "call_id": "u1",
            },
        ],
        {
            "schema_version": "planner_action.v2",
            "current_phase": "diagnostic_workup",
            "proposed_phase": "treatment_selection",
            "missing_information": ["ECOG"],
            "actions": [
                {
                    "action_id": "collect-ecog",
                    "objective": "Collect ECOG performance status",
                    "action_type": "evidence_gathering",
                    "required_skills": ["clinical.read_summary"],
                    "preconditions": [],
                    "expected_state_delta": ["risk_stratification"],
                    "provenance": [
                        {
                            "memory_id": "test-memory",
                            "rule_ids": ["test-rule"],
                            "source_spans": ["test-span"],
                            "guideline_id": guideline["guideline_id"],
                            "version": guideline["version"],
                        }
                    ],
                }
            ],
            "blocked_actions": [],
            "should_stop": False,
            "reason": "ECOG remains unresolved.",
        },
    )

    assert updated["current_phase"] == "diagnostic_workup"
    assert updated["known_diagnosis"] == "Adenocarcinoma, NOS"
    assert "Stage IIB" in updated["known_stage"]
    assert updated["known_biomarkers"]["EGFR"]["alteration_status"] == "negative"
    assert updated["missing_information"] == ["ECOG"]
    assert any("unknown.skill" in warning for warning in updated["warnings"])


def test_current_patient_state_view_excludes_append_only_audit_history() -> None:
    state = {
        "schema_version": "patient_state.v2",
        "case_id": CASE_ID,
        "cancer_family": "lung",
        "disease_subtype": "nsclc",
        "current_phase": "treatment_selection",
        "decision_date": "2025-01-01",
        "guideline_context": {
            "decision_date": "2025-01-01",
            "guidelines": [{"guideline_id": "guide", "version": "2025"}],
        },
        "known_diagnosis": "Adenocarcinoma, NOS",
        "known_stage": "Stage IIB",
        "known_biomarkers": {},
        "risk_stratification": {},
        "available_modalities": ["clinical"],
        "completed_skills": ["clinical.read_summary"],
        "confirmed_evidence": {
            "age": 57,
            "chief_problem": "lung mass",
            "known_diagnosis": "Adenocarcinoma, NOS",
            "known_stage": "Stage IIB",
            "evidence_refs": [{"call_id": "c1"}],
        },
        "evidence_refs": [{"call_id": "c1"}],
        "previous_actions": ["clinical.read_summary"],
        "completed_actions": [{"skill_name": "clinical.read_summary", "status": "success"}],
        "current_decision_stage": "treatment_selection",
        "unresolved_information": ["ECOG"],
        "treatment_history": [],
        "current_treatment_line": 0,
        "evidence_ledger": [],
        "last_transition": None,
        "pending_actions": [],
        "blocked_actions": [],
        "free_text_summary": "known_stage: Stage IIB",
        "state_update_summary": {"known_stage": "Stage IIB"},
    }

    view = current_patient_state_view(state)

    assert view["known_diagnosis"] == "Adenocarcinoma, NOS"
    assert view["known_stage"] == "Stage IIB"
    assert view["baseline_evidence"] == {
        "age": 57,
        "chief_problem": "lung mass",
    }
    for field in (
        "confirmed_evidence",
        "evidence_refs",
        "current_decision_stage",
        "free_text_summary",
        "state_update_summary",
    ):
        assert field not in view
    assert view["previous_actions"] == ["clinical.read_summary"]
    assert view["completed_actions"][0]["skill_name"] == "clinical.read_summary"
    assert view["unresolved_information"] == ["ECOG"]
    assert state["evidence_refs"] == [{"call_id": "c1"}]


def test_dual_agent_runner_writes_planner_and_patient_state_outputs(
    tmp_path: Path,
) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=True)
    runtime = FakeRuntime(tmp_path)
    client = FakeMultimodalAgentClient()
    planner = FakePlanner()

    result = DualAgentBenchmarkRunner(
        case_dir=case_dir,
        runs_root=tmp_path / "runs",
        runtime_manager=runtime,
        llm_client=client,
        planner=planner,
        max_planner_rounds=2,
        max_tool_rounds_per_step=6,
    ).run()

    assert result.final_answer_path.is_file()
    assert (result.run_dir / "planner_outputs.jsonl").is_file()
    assert (result.run_dir / "patient_state_history.jsonl").is_file()
    assert (result.run_dir / "dual_agent_rounds.jsonl").is_file()
    assert (result.run_dir / "tool_calls.jsonl").is_file()
    assert (result.run_dir / "evidence_board.json").is_file()
    assert (result.run_dir / "planner_token_stats.json").is_file()
    final_answer = json.loads(result.final_answer_path.read_text(encoding="utf-8"))
    assert final_answer["planning_mode"] == "latent_decoder_guided_tool_choice_auto"
    assert final_answer["planner_token_stats_path"].endswith("planner_token_stats.json")
    assert final_answer["planner_context"]["planning_mode"] == "latent_decoder"
    context = json.loads((result.run_dir / "agent_context.json").read_text(encoding="utf-8"))
    assert context["final_synthesis_mode"] == "same_agent_loop_no_tools"
    assert (
        context["history_compaction_strategy"]
        == "replace_each_planner_round_user_prompt_with_short_marker_after_audit"
    )
    assert "current_patient_state" in context
    assert "trajectory_plan" not in context
    assert context["tool_record_count"] == 5
    assert "planner_context" not in context
    assert context["retained_evidence"]["structured_tool_results"] is True
    assert context["retained_evidence"]["guideline_snippets_and_pages"] is True
    assert context["retained_evidence"]["image_messages"] is True
    assert context["retained_evidence"]["full_planner_round_prompts"] is False
    for field in (
        "confirmed_evidence",
        "evidence_refs",
        "current_decision_stage",
        "free_text_summary",
        "state_update_summary",
    ):
        assert field not in context["current_patient_state"]
    assert "previous_actions" in context["current_patient_state"]
    assert "completed_actions" in context["current_patient_state"]
    assert "unresolved_information" in context["current_patient_state"]
    assert all(call["trajectory_plan"] == [] for call in planner.calls)
    auto_requests = [
        request
        for request in client.requests
        if request["tools"] and request["tool_choice"] == "auto"
    ]
    assert len(auto_requests[-1]["messages"]) > 2
    assert auto_requests[-1]["messages"][0]["role"] == "system"
    assert any(
        message.get("role") == "tool"
        for message in auto_requests[-1]["messages"][:-1]
    )
    final_request = client.requests[-1]
    assert final_request["tools"] == []
    assert final_request["tool_choice"] == "none"
    final_messages_text = json.dumps(final_request["messages"], ensure_ascii=False)
    assert "NCCN NSCLC 2010" in final_messages_text
    assert "Auditable guideline excerpt." in final_messages_text
    final_tool_payloads = [
        json.loads(message["content"])
        for message in final_request["messages"]
        if message.get("role") == "tool"
    ]
    assert any(
        payload.get("findings", {}).get("snippets", [{}])[0].get("page") == 42
        for payload in final_tool_payloads
        if payload.get("findings", {}).get("snippets")
    )
    assert "Dual-agent planner-guided evidence round:" not in final_messages_text
    assert final_messages_text.count("current_patient_state") == 1
    assert "Evidence acquisition round 1 completed." in final_messages_text
    assert any(
        isinstance(message.get("content"), list)
        and any(
            isinstance(part, Mapping) and part.get("type") == "image_path"
            for part in message["content"]
        )
        for message in final_request["messages"]
    )
    conversation_events = [
        json.loads(line)
        for line in (result.run_dir / "conversation_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert not any(
        item["event_type"] == "conversation_reset" for item in conversation_events
    )
    compaction_events = [
        item
        for item in conversation_events
        if item["event_type"] == "conversation_compaction"
    ]
    assert len(compaction_events) == 2
    assert any(
        "Dual-agent planner-guided evidence round:"
        in json.dumps(item, ensure_ascii=False)
        for item in conversation_events
        if item["event_type"] == "conversation_message"
    )
    history = [
        json.loads(line)
        for line in (result.run_dir / "patient_state_history.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any("patient_state_before" in item for item in history)
    assert any("patient_state_after" in item for item in history)
    rounds = [
        json.loads(line)
        for line in (result.run_dir / "dual_agent_rounds.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all("patient_state_delta" in item for item in rounds)
    assert "rubric" not in (result.run_dir / "planner_outputs.jsonl").read_text(
        encoding="utf-8"
    ).lower()
    assert planner.calls


def test_dual_agent_runner_writes_current_state_routing_audit_without_graph(
    tmp_path: Path,
) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=True)

    class RoutingPlanner:
        def __init__(self) -> None:
            self.trajectory_histories: list[list[Mapping[str, Any]]] = []
            self.calls = 0

        def public_config(self) -> dict[str, Any]:
            return {
                "planning_mode": "latent_decoder_state_conditioned_routing",
                "routing": {"enabled": True, "checkpoint": {"trained": True}},
            }

        def plan(
            self,
            patient_state: Mapping[str, Any],
            trajectory_history: Sequence[Mapping[str, Any]],
        ) -> PlannerStepResult:
            self.calls += 1
            self.trajectory_histories.append(list(trajectory_history))
            memory_id = f"memory_{self.calls}"
            guideline = patient_state["guideline_context"]["guidelines"][0]
            action = {
                "schema_version": "planner_action.v2",
                "current_phase": patient_state["current_phase"],
                "proposed_phase": None,
                "missing_information": ["pathology"],
                "actions": [
                    {
                        "objective": "Collect the next evidence item.",
                        "action_type": "evidence_gathering",
                        "required_skills": ["clinical.read_summary"],
                        "repeat_justification": "Exercise a second current-state routing audit.",
                        "preconditions": [],
                        "expected_state_delta": ["known_diagnosis"],
                        "provenance": [
                            {
                                "memory_id": memory_id,
                                "rule_ids": [f"{memory_id}:rule:1"],
                                "source_spans": [f"{memory_id}:span:1"],
                                "guideline_id": guideline["guideline_id"],
                                "version": guideline["version"],
                            }
                        ],
                    }
                ],
                "blocked_actions": [],
                "should_stop": False,
                "reason": "Evidence remains incomplete.",
            }
            activation = MemoryActivation(
                memory_id,
                1.0,
                seed_score=0.9,
                gate_score=0.9,
                selected=True,
                section_title="Diagnosis",
                source_pages="1-2",
            )
            diagnostics = {
                "seed_candidates": [{"memory_id": memory_id, "seed_score": 0.9}],
                "candidate_activations": [activation.to_dict()],
                "gate_diagnostics": {"entropy": 0.0},
                "fusion_diagnostics": {
                    "strategy": "dynamic_anchor_attention",
                    "anchor_memory_id": memory_id,
                },
                "decoder_prefix_shape": [4, 8],
                "diagnostics": {"profile": "dynamic_anchor"},
            }
            return PlannerStepResult(
                action=action,
                current_objective=action["actions"][0]["objective"],
                expected_state_change=["pathology"],
                active_memories=[activation],
                expert_outputs=[],
                diagnostics=diagnostics,
            )

    planner = RoutingPlanner()
    result = DualAgentBenchmarkRunner(
        case_dir=case_dir,
        runs_root=tmp_path / "runs",
        runtime_manager=FakeRuntime(tmp_path),
        llm_client=FakeMultimodalAgentClient(),
        planner=planner,
        max_planner_rounds=2,
        max_tool_rounds_per_step=6,
    ).run()

    assert planner.trajectory_histories == [[], []]
    assert (result.run_dir / "memory_routing.json").is_file()
    assert not (result.run_dir / "expert_outputs.jsonl").exists()
    assert not (result.run_dir / "transition_trace.json").exists()
    assert (result.run_dir / "routing_metrics.json").is_file()
    routing = json.loads((result.run_dir / "memory_routing.json").read_text(encoding="utf-8"))
    assert routing["steps"][0]["candidate_activations"][0]["source_pages"] == "1-2"


def test_dual_agent_runner_recovers_partial_tools_after_round_limit(
    tmp_path: Path,
) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=True)
    runtime = FakeRuntime(tmp_path)
    client = FakeMultimodalAgentClient()

    result = DualAgentBenchmarkRunner(
        case_dir=case_dir,
        runs_root=tmp_path / "runs",
        runtime_manager=runtime,
        llm_client=client,
        planner=FakePlanner(),
        max_planner_rounds=2,
        max_tool_rounds_per_step=1,
    ).run()

    assert result.final_answer_path.is_file()
    assert result.tool_calls_path.is_file()
    assert result.evidence_board_path.is_file()
    final_answer = json.loads(result.final_answer_path.read_text(encoding="utf-8"))
    assert final_answer["tool_round_count"] == 1
    rounds_text = (result.run_dir / "dual_agent_rounds.jsonl").read_text(encoding="utf-8")
    assert "AgentLoopError" in rounds_text
    trajectory_text = result.trajectory_path.read_text(encoding="utf-8")
    assert "agent_round_error" in trajectory_text
    debug = json.loads(
        (result.run_dir / "agent_round_1_failure_debug.json").read_text(encoding="utf-8")
    )
    assert debug["error"]["tool_debug"]["last_tool"]["skill_name"] == "clinical.read_summary"
    assert debug["error"]["tool_debug"]["tool_result_count"] == 1


def test_dual_agent_runner_logs_planner_failure_attempt(
    tmp_path: Path,
) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=True)

    class FailingPlanner(FakePlanner):
        def __init__(self) -> None:
            super().__init__()
            self.last_attempt = {
                "query": "LUNG Adenocarcinoma",
                "raw_text": "not json",
                "repair_raw_text": "still not json",
                "retrieved_memories": [{"guideline_memory_id": "nsclc_diag"}],
            }

        def next_step(
            self,
            patient_state: Mapping[str, Any],
            trajectory_plan: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            raise LatentPlannerError(
                "Planner decoder did not return JSON.",
                details=self.last_attempt,
            )

    runner = DualAgentBenchmarkRunner(
        case_dir=case_dir,
        runs_root=tmp_path / "runs",
        runtime_manager=FakeRuntime(tmp_path),
        llm_client=FakeMultimodalAgentClient(),
        planner=FailingPlanner(),
    )

    with pytest.raises(LatentPlannerError):
        runner.run()

    planner_lines = (runner.run_dir / "planner_outputs.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    record = json.loads(planner_lines[-1])
    debug = json.loads(
        (runner.run_dir / "planner_round_1_failure_debug.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["status"] == "failed"
    assert record["planner_attempt"]["raw_text"] == "not json"
    assert debug["planner_attempt"]["retrieved_memories"][0]["guideline_memory_id"] == "nsclc_diag"


def test_dual_agent_batch_skips_completed_case(tmp_path: Path) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=False)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / CASE_ID / "dual_agent_001"
    run_dir.mkdir(parents=True)
    for filename in (
        "final_answers.json",
        "judge_scores.json",
        "planner_outputs.jsonl",
        "patient_state_history.jsonl",
    ):
        (run_dir / filename).write_text("{}", encoding="utf-8")

    summary = run_dual_agent_batch(
        DualAgentBatchConfig(
            cases_root=case_dir.parent,
            runs_root=runs_root,
            run_id="dual_agent_001",
        )
    )

    assert summary["counts"]["skipped"] == 1
    assert summary["cases"][0]["status"] == "skipped"
    assert (runs_root / "dual_agent_batch_summary_dual_agent_001.json").is_file()


def test_dual_agent_batch_can_evaluate_existing_inference_without_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=False)
    runs_root = tmp_path / "runs"
    run_id = "dual_late_eval"
    run_dir = runs_root / CASE_ID / run_id
    run_dir.mkdir(parents=True)
    for filename in (
        "final_answers.json",
        "planner_outputs.jsonl",
        "patient_state_history.jsonl",
    ):
        (run_dir / filename).write_text("{}\n", encoding="utf-8")
    judge_calls: list[Path] = []

    class ForbiddenDualRunner:
        def __init__(self, **kwargs: Any) -> None:
            raise AssertionError("completed dual-agent inference must not be rerun")

    class FakeJudge:
        def __init__(
            self,
            rubric_path: Path,
            run_dir: Path,
            provider: str | None = None,
        ) -> None:
            self.run_dir = Path(run_dir)

        def evaluate(self) -> dict[str, Any]:
            judge_calls.append(self.run_dir)
            (self.run_dir / "judge_scores.json").write_text("{}", encoding="utf-8")
            (self.run_dir / "error_analysis.md").write_text("# ok\n", encoding="utf-8")
            return {"status": "success", "final_total": 91}

    monkeypatch.setattr(dual_agent_runner, "DualAgentBenchmarkRunner", ForbiddenDualRunner)
    monkeypatch.setattr(dual_agent_runner, "LLMRubricJudge", FakeJudge)
    monkeypatch.setattr(dual_agent_runner, "write_routing_evaluation", lambda path: {})

    summary = run_dual_agent_batch(
        DualAgentBatchConfig(
            cases_root=case_dir.parent,
            runs_root=runs_root,
            run_id=run_id,
            evaluate=True,
        )
    )

    assert judge_calls == [run_dir]
    assert summary["cases"][0]["execution_mode"] == "evaluation_only"
    assert summary["evaluation_enabled"] is True
    assert summary["cases"][0]["final_total"] == 91


def test_release_guideline_context_overrides_agent_filters() -> None:
    bound = _bind_release_guideline_arguments(
        {
            "query": "use latest lung guideline",
            "cancer_type": "nsclc",
            "guideline_family": "csco",
            "guideline_version": "2025",
        },
        {
            "decision_date": "2023-12-31",
            "guidelines": [
                {"guideline_id": "CSCO子宫内膜癌2023", "version": "2023"}
            ],
        },
    )
    assert bound["cancer_type"] == "ucec"
    assert bound["guideline_family"] == "csco"
    assert bound["guideline_version"] == "2023"


def test_case_builder_uses_ucec_identity_and_runtime_modalities(tmp_path: Path) -> None:
    case_dir = tmp_path / "examples" / "cases" / "UCEC-001"
    report_dir = case_dir / "reports" / "integrated_reports"
    report_dir.mkdir(parents=True)
    (report_dir / "UCEC-001_T0_full_report_zh.md").write_text(
        "# 子宫内膜癌病例\n",
        encoding="utf-8",
    )
    (case_dir / "case.yaml").write_text(
        "clinical:\n  cancer: endometrial carcinoma\n  guideline: CSCO 2023\n",
        encoding="utf-8",
    )
    hidden = CaseBuilder(case_dir).build_hidden_state(extraction={})
    assert hidden["project_id"] == "TCGA-UCEC"
    assert hidden["cancer_type"] == "UCEC"
    assert hidden["modalities"]["pathology_roi_skill"] == (
        "pathology.ucec_conch_patch_roi"
    )
    assert hidden["modalities"]["ct_roi_skill"] == "radiology.ucec_mri_roi"


def test_planner_preflight_rejects_rubric_version_mismatch(tmp_path: Path) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=False)
    rubric_path = next((case_dir / "evaluation").glob("*_rubric.json"))
    rubric_path.write_text(
        json.dumps(
            {"metadata": {"guideline": "CSCO NSCLC 2025"}, "rubric": {}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class FakeRelease:
        def require_supported_action(self, family: str, subtype: str) -> None:
            assert (family, subtype) == ("lung", "nsclc")

        def guideline_default(self, family: str, subtype: str) -> dict[str, str]:
            return {
                "guideline_id": "NSCLC_2010",
                "version": "2010",
                "decision_date": "2010-12-31",
            }

    assert _rubric_guideline_versions(
        {"metadata": {"guideline": "CSCO NSCLC 2025"}}
    ) == {"2025"}
    with pytest.raises(ValueError, match="requires NSCLC_2010@2010"):
        preflight_planner_case(case_dir, FakeRelease())  # type: ignore[arg-type]


def test_dual_agent_batch_reuses_one_release_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_root = tmp_path / "cases"
    case_a = cases_root / "case-a"
    case_b = cases_root / "case-b"
    case_a.mkdir(parents=True)
    case_b.mkdir(parents=True)
    release = type(
        "Release",
        (),
        {
            "memory_dir": tmp_path / "memory",
            "decoder_artifact_dir": tmp_path / "decoder",
            "top_k": 4,
            "routing_config": None,
            "routing_checkpoint": None,
        },
    )()
    created: list[Any] = []
    seen: list[Any] = []

    class FakeSharedPlanner:
        def __init__(self, **kwargs: Any) -> None:
            created.append(self)

    def fake_run_one(
        case_dir: Path,
        config: DualAgentBatchConfig,
        *,
        planner: Any = None,
        planner_release: Any = None,
    ) -> dict[str, Any]:
        seen.append((planner, planner_release))
        return {"case_id": case_dir.name, "status": "completed", "run_dir": str(case_dir)}

    monkeypatch.setattr(dual_agent_runner, "discover_ready_cases", lambda root: [case_a, case_b])
    monkeypatch.setattr(dual_agent_runner, "resolve_planner_release", lambda *a, **k: release)
    monkeypatch.setattr(
        dual_agent_runner,
        "preflight_planner_case",
        lambda case, resolved: {
            "case_id": case.name,
            "cancer_family": "lung",
            "disease_subtype": "nsclc",
            "guideline_id": "NSCLC_2010",
            "version": "2010",
            "status": "ready",
        },
    )
    monkeypatch.setattr(dual_agent_runner, "LatentGuidelinePlanner", FakeSharedPlanner)
    monkeypatch.setattr(dual_agent_runner, "_run_one_dual_case", fake_run_one)

    summary = run_dual_agent_batch(
        DualAgentBatchConfig(
            cases_root=cases_root,
            runs_root=tmp_path / "runs",
            planner_release_dir=tmp_path / "release",
        )
    )
    assert len(created) == 1
    assert len(seen) == 2
    assert seen[0][0] is created[0] is seen[1][0]
    assert all(item[1] is release for item in seen)
    assert summary["counts"]["completed"] == 2


def test_dual_agent_batch_sets_medclaw_cases_root_for_case_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=False)
    runs_root = tmp_path / "runs"
    run_id = "dual_env"
    seen: dict[str, str | None] = {}
    monkeypatch.delenv("MEDCLAW_CASES_ROOT", raising=False)

    class FakeDualRunner:
        def __init__(
            self,
            *,
            case_dir: Path,
            run_id: str,
            runs_root: Path,
            agent: str,
            planner_memory_dir: Path,
            planner_decoder_artifact_dir: Path,
            planner_top_k: int,
            planner_device: str,
            planner_query_encoder_device: str,
            planner_max_new_tokens: int,
            planner_output_mode: str,
            max_planner_rounds: int,
            max_tool_rounds_per_step: int,
        ) -> None:
            self.case_id = Path(case_dir).name
            self.run_id = run_id
            self.run_dir = Path(runs_root) / self.case_id / run_id

        def run(self) -> Any:
            seen["MEDCLAW_CASES_ROOT"] = os.environ.get("MEDCLAW_CASES_ROOT")
            self.run_dir.mkdir(parents=True, exist_ok=True)
            return type("FakeResult", (), {"run_dir": self.run_dir})()

    class FakeJudge:
        def __init__(
            self,
            rubric_path: Path,
            run_dir: Path,
            provider: str | None = None,
        ) -> None:
            self.rubric_path = rubric_path
            self.run_dir = run_dir

        def evaluate(self) -> dict[str, Any]:
            return {"status": "success", "final_total": 100}

    monkeypatch.setattr(dual_agent_runner, "DualAgentBenchmarkRunner", FakeDualRunner)
    monkeypatch.setattr(dual_agent_runner, "LLMRubricJudge", FakeJudge)

    summary = run_dual_agent_batch(
        DualAgentBatchConfig(
            cases_root=case_dir.parent,
            runs_root=runs_root,
            run_id=run_id,
            dry_run=False,
        )
    )

    assert summary["counts"]["completed"] == 1
    assert seen["MEDCLAW_CASES_ROOT"] == str(case_dir.parent.resolve())
    assert "MEDCLAW_CASES_ROOT" not in os.environ


def test_dual_agent_batch_error_includes_tool_failure_debug(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=False)
    runs_root = tmp_path / "runs"
    run_id = "dual_debug"

    class FailingDualRunner:
        def __init__(
            self,
            *,
            case_dir: Path,
            run_id: str,
            runs_root: Path,
            agent: str,
            planner_memory_dir: Path,
            planner_decoder_artifact_dir: Path,
            planner_top_k: int,
            planner_device: str,
            planner_query_encoder_device: str,
            planner_max_new_tokens: int,
            planner_output_mode: str,
            max_planner_rounds: int,
            max_tool_rounds_per_step: int,
        ) -> None:
            self.case_id = Path(case_dir).name
            self.run_id = run_id
            self.run_dir = Path(runs_root) / self.case_id / run_id

        def run(self) -> Any:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            event = {
                "event_type": "agent_turn",
                "turn_id": "turn_debug",
                "status": "failed",
                "tool_results": [
                    {
                        "function_name": "medclaw__radiology__lung_tumor_roi",
                        "skill_name": "radiology.lung_tumor_roi",
                        "arguments": {"case_id": CASE_ID},
                        "result": {
                            "status": "failed",
                            "skill_name": "radiology.lung_tumor_roi",
                            "findings": {
                                "summary": "CT ROI failed",
                                "error": "No module named lungtumormask",
                            },
                            "warnings": ["No module named lungtumormask"],
                            "artifacts": [],
                        },
                    }
                ],
                "error": {
                    "type": "AgentLoopError",
                    "message": "Model exceeded the maximum of 12 tool-calling rounds.",
                },
            }
            (self.run_dir / "conversation_log.jsonl").write_text(
                json.dumps(event, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError("synthetic runner failure")

    monkeypatch.setattr(dual_agent_runner, "DualAgentBenchmarkRunner", FailingDualRunner)

    summary = run_dual_agent_batch(
        DualAgentBatchConfig(
            cases_root=case_dir.parent,
            runs_root=runs_root,
            run_id=run_id,
            dry_run=False,
        )
    )

    run_dir = runs_root / CASE_ID / run_id
    batch_error = json.loads((run_dir / "batch_error.json").read_text(encoding="utf-8"))
    failure_debug = json.loads((run_dir / "failure_debug.json").read_text(encoding="utf-8"))
    failed_tool = failure_debug["conversation_tool_debug"]["failed_tools"][0]
    assert summary["counts"]["failed"] == 1
    assert batch_error["failure_debug_path"].endswith("failure_debug.json")
    assert failed_tool["skill_name"] == "radiology.lung_tumor_roi"
    assert failed_tool["error"] == "No module named lungtumormask"


class FakeReportExtractionClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.config = FakeConfig()

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
        tool_choice: str | Mapping[str, Any] = "auto",
    ) -> QwenCompletion:
        self.requests.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": list(tools),
                "tool_choice": tool_choice,
            }
        )
        content = {
            "clinical": {
                "project_id": "TCGA-LUAD",
                "age": 57,
                "sex": "female",
                "race": "White",
                "primary_diagnosis": "Adenocarcinoma, NOS",
                "tumor_location": "Left upper lobe",
                "tumor_size_cm": 5.3,
                "pack_years": 40,
                "pathologic_n_stage": "pN0",
            },
            "treatment": {
                "treatments": [
                    {
                        "treatment_type": "Surgery",
                        "received": "no",
                        "evidence": "treatment_or_therapy: no",
                    }
                ]
            },
            "follow_up": {
                "vital_status": "alive",
                "last_follow_up_day": 3674,
                "disease_status_at_last_follow_up": "Tumor Free",
            },
            "molecular": {
                "molecular_subtype": "common-driver-negative / CNV-dominant",
                "summary": "No canonical LUAD driver mutation; CNV-dominant profile.",
                "high_amplification_genes": ["BCAT1", "LRMP", "RN7SL38P"],
                "biomarkers": {
                    "EGFR": {
                        "alteration_status": "negative",
                        "mutation": "not detected",
                        "cnv": "neutral",
                        "rna_expression": -0.1494,
                    },
                    "ALK": {
                        "alteration_status": "uncertain",
                        "mutation": "atypical mutation",
                        "interpretation": "no CNV/RNA support for driver role",
                    },
                    "KRAS": {
                        "alteration_status": "negative",
                        "mutation": "not detected",
                    },
                },
            },
            "warnings": [],
        }
        return QwenCompletion(
            message={"role": "assistant", "content": json.dumps(content, ensure_ascii=False)},
            model="fake-extractor",
            usage={"total_tokens": 42},
        )


class ExplodingReportExtractionClient:
    def __init__(self) -> None:
        self.config = None

    def complete(self, **kwargs: Any) -> QwenCompletion:
        raise AssertionError("LLM extractor should not be called when cache exists.")


class FakeRuntime:
    def __init__(self, tmp_path: Path) -> None:
        self.registry = SkillRegistry(PROJECT_ROOT / "medclaw" / "skills")
        self.artifact_store = ArtifactStore(tmp_path / "registered_artifacts")
        self.counter = 0

    def invoke(self, skill_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.counter += 1
        call_id = f"fake_call_{self.counter}"
        if skill_name.startswith("radiology"):
            modality = "ct_roi"
            roles = ["roi"]
        elif skill_name.startswith("pathology"):
            modality = "wsi_patch_roi"
            roles = ["patch_roi", "wsi_overview"]
        else:
            modality = "guideline"
            roles = []
        artifacts = []
        for role in roles:
            source = self.artifact_store.root / f"{call_id}_{role}.png"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"\x89PNG\r\n\x1a\nfake")
            uri = self.artifact_store.register_file(
                source,
                str(arguments["case_id"]),
                skill_name,
                call_id,
                "image",
            )
            artifacts.append(
                {
                    "type": "image",
                    "role": role,
                    "uri": uri,
                    "size_bytes": source.stat().st_size,
                    "sha256": "0" * 64,
                }
            )
        findings = {
            "summary": f"{skill_name} fake success",
            "retrieved": skill_name.startswith("guideline"),
            "query": str(arguments.get("query", "")),
            "selected_cancer_type": str(arguments.get("cancer_type", "")),
            "data": {"case_id": arguments["case_id"]},
        }
        if skill_name == "guideline.retrieve":
            findings["snippets"] = [
                {
                    "guideline": "NCCN NSCLC 2010",
                    "page": 42,
                    "excerpt": "Auditable guideline excerpt.",
                }
            ]
        return {
            "status": "success",
            "call_id": call_id,
            "modality": modality,
            "findings": findings,
            "artifacts": artifacts,
            "warnings": [],
        }


def test_benchmark_runner_writes_auditable_outputs_without_rubric_leak(
    tmp_path: Path,
) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=True)
    runtime = FakeRuntime(tmp_path)
    client = FakeMultimodalAgentClient()
    result = BenchmarkRunner(
        case_dir=case_dir,
        runs_root=tmp_path / "runs",
        runtime_manager=runtime,
        llm_client=client,
    ).run()

    assert result.trajectory_path.is_file()
    assert result.guideline_trajectory_path.is_file()
    assert result.tool_calls_path.is_file()
    assert result.evidence_board_path.is_file()
    assert result.final_answer_path.is_file()
    assert list((result.run_dir / "artifacts" / "ct_roi").glob("*.png"))
    assert list((result.run_dir / "artifacts" / "wsi_roi").glob("*.png"))

    trajectory_text = result.trajectory_path.read_text(encoding="utf-8")
    assert "rubric" not in trajectory_text.lower()
    guideline_trajectory = json.loads(
        result.guideline_trajectory_path.read_text(encoding="utf-8")
    )
    assert guideline_trajectory["schema_version"] == "trajectory.dynamic_rubric.v1"
    tool_calls = result.tool_calls_path.read_text(encoding="utf-8")
    assert "radiology.lung_tumor_roi" in tool_calls
    assert "pathology.conch_patch_roi" in tool_calls
    assert (
        json.loads(result.final_answer_path.read_text(encoding="utf-8"))["agent"]
        == "local_openai"
    )


class FakeMultimodalAgentClient:
    def __init__(self) -> None:
        self.built_messages: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.config = FakeConfig()
        self._round = 0

    def build_user_message(
        self,
        text: str,
        *,
        image_paths=(),
        image_urls=(),
    ) -> dict[str, Any]:
        paths = [Path(path) for path in image_paths]
        self.built_messages.append(
            {
                "text": text,
                "image_paths": [str(path) for path in paths],
                "image_urls": list(image_urls),
            }
        )
        content = [{"type": "text", "text": text}]
        for path in paths:
            content.append({"type": "image_path", "path": str(path)})
        return {"role": "user", "content": content}

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
        tool_choice: str | Mapping[str, Any] = "auto",
    ) -> QwenCompletion:
        self.requests.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": list(tools),
                "tool_choice": tool_choice,
            }
        )
        if tools and tool_choice == "auto":
            self._round += 1
            planned = [
                ("clinical.read_summary", {"case_id": CASE_ID}),
                ("pathology.conch_patch_roi", {"case_id": CASE_ID}),
                ("radiology.lung_tumor_roi", {"case_id": CASE_ID}),
                ("molecular.query_biomarkers", {"case_id": CASE_ID}),
                ("guideline.retrieve", {"case_id": CASE_ID}),
            ]
            if self._round <= len(planned):
                skill_name, arguments = planned[self._round - 1]
                return QwenCompletion(
                    message={
                        "role": "assistant",
                        "content": f"Calling {skill_name}",
                        "tool_calls": [
                            {
                                "id": f"call_{self._round}",
                                "type": "function",
                                "function": {
                                    "name": f"medclaw__{skill_name.replace('.', '__')}",
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                    model="fake-qwen-vl",
                    usage={"total_tokens": 9},
                )
        return QwenCompletion(
            message={
                "role": "assistant",
                "content": (
                    "Final synthesis: the model integrated its self-selected "
                    "clinical, pathology ROI, radiology ROI, molecular, and "
                    "guideline evidence."
                ),
            },
            model="fake-qwen-vl",
            usage={"total_tokens": 9},
        )


def test_qwen_benchmark_runner_sends_roi_images_to_core_model(tmp_path: Path) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=True)
    runtime = FakeRuntime(tmp_path)
    client = FakeMultimodalAgentClient()

    result = BenchmarkRunner(
        case_dir=case_dir,
        runs_root=tmp_path / "runs",
        runtime_manager=runtime,
        agent="qwen",
        llm_client=client,
    ).run()

    final_answer = json.loads(result.final_answer_path.read_text(encoding="utf-8"))
    assert final_answer["agent"] == "qwen"
    assert final_answer["guideline_trajectory_path"].endswith("guideline_trajectory.json")
    assert final_answer["answer_text"].startswith("Final synthesis")
    assert final_answer["model"] == "fake-qwen-vl"
    assert (result.run_dir / "agent_context.json").is_file()
    assert (result.run_dir / "conversation_log.jsonl").is_file()
    assert final_answer["tool_round_count"] == 5

    sent_images = [
        image
        for message in client.built_messages
        for image in message["image_paths"]
    ]
    assert len(sent_images) == 3
    assert list((result.run_dir / "artifacts" / "ct_roi").glob("*.png"))
    assert list((result.run_dir / "artifacts" / "wsi_roi").glob("*.png"))
    image_messages = [message for message in client.built_messages if message["image_paths"]]
    assert len(image_messages) == 2
    first_wsi_message = image_messages[0]
    assert "wsi_overview" in " ".join(first_wsi_message["image_paths"])

    context = json.loads((result.run_dir / "agent_context.json").read_text(encoding="utf-8"))
    assert context["planning_mode"] == "model_driven_tool_choice_auto"
    assert len(context["tool_records"]) == 5
    assert context["conversation_log_path"].endswith("conversation_log.jsonl")


def test_benchmark_runner_accepts_openai_agent_with_injected_client(tmp_path: Path) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=False)
    runtime = FakeRuntime(tmp_path)
    client = FakeMultimodalAgentClient()

    result = BenchmarkRunner(
        case_dir=case_dir,
        runs_root=tmp_path / "runs",
        runtime_manager=runtime,
        agent="openai",
        llm_client=client,
    ).run()

    final_answer = json.loads(result.final_answer_path.read_text(encoding="utf-8"))
    assert final_answer["agent"] == "openai"
    assert (result.run_dir / "agent_context.json").is_file()


class FakeJudgeClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[dict[str, Any]] = []
        self.config = FakeConfig()

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
        tool_choice: str | Mapping[str, Any] = "auto",
    ) -> QwenCompletion:
        self.requests.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": list(tools),
                "tool_choice": tool_choice,
            }
        )
        return QwenCompletion(
            message={"role": "assistant", "content": self.content},
            model="fake-qwen",
            usage={"total_tokens": 1},
        )


class FakeConfig:
    def public_summary(self) -> dict[str, Any]:
        return {"model": "fake-qwen", "base_url": "https://example.invalid/v1"}


class FakePlanner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def next_step(
        self,
        patient_state: Mapping[str, Any],
        trajectory_plan: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "patient_state": dict(patient_state),
                "trajectory_plan": trajectory_plan,
            }
        )
        required_skills = [
            skill
            for skill in (
                "clinical.read_summary",
                "pathology.conch_patch_roi",
                "radiology.lung_tumor_roi",
                "molecular.query_biomarkers",
                "guideline.retrieve",
            )
            if skill not in set(patient_state.get("completed_skills") or [])
        ]
        guideline = patient_state["guideline_context"]["guidelines"][0]
        if not required_skills:
            return {
                "schema_version": "planner_action.v2",
                "current_phase": patient_state["current_phase"],
                "proposed_phase": None,
                "missing_information": [],
                "actions": [],
                "blocked_actions": [],
                "should_stop": True,
                "reason": "All fake evidence tools completed.",
            }
        return {
            "schema_version": "planner_action.v2",
            "current_phase": patient_state.get("current_phase") or "diagnostic_workup",
            "proposed_phase": None,
            "missing_information": ["pathology", "radiology", "molecular"],
            "actions": [
                {
                    "action_id": "collect-fake-evidence",
                    "objective": "Collect pathology, radiology, molecular, and guideline evidence.",
                    "action_type": "evidence_gathering",
                    "required_skills": required_skills,
                    "preconditions": [],
                    "expected_state_delta": [
                        "known_diagnosis",
                        "known_stage",
                        "known_biomarkers",
                    ],
                    "provenance": [
                        {
                            "memory_id": "fake_guideline_memory",
                            "rule_ids": ["rule_001"],
                            "source_spans": ["fake_span"],
                            "guideline_id": guideline["guideline_id"],
                            "version": guideline["version"],
                        }
                    ],
                }
            ],
            "blocked_actions": [],
            "should_stop": False,
            "reason": "fake planner guidance",
        }


def judge_response(score: float = 1.0) -> str:
    return json.dumps(
        {
            "dimensions": {
                dim: {
                    "score": score,
                    "max_score": max_score,
                    "matched": ["matched item"],
                    "missed": [],
                    "rationale": f"{dim} rationale",
                    "needs_manual_review": [],
                    "criteria": [
                        {
                            "criterion": "criterion",
                            "status": "matched",
                            "reason": "ok",
                        }
                    ],
                }
                for dim, max_score in DIMENSION_WEIGHTS.items()
            },
            "overall_rationale": "overall",
        },
        ensure_ascii=False,
    )


def prepare_judge_run(tmp_path: Path, *, final_answer: str = "建议补充分子检测。") -> tuple[Path, Path]:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=True)
    runtime = FakeRuntime(tmp_path)
    run = BenchmarkRunner(
        case_dir=case_dir,
        runs_root=tmp_path / "runs",
        runtime_manager=runtime,
        llm_client=FakeMultimodalAgentClient(),
    ).run()
    final_path = run.run_dir / "final_answers.json"
    final_data = json.loads(final_path.read_text(encoding="utf-8"))
    final_data["answer_text"] = final_answer
    final_path.write_text(json.dumps(final_data, ensure_ascii=False), encoding="utf-8")
    return run.run_dir, case_dir / "evaluation" / f"{CASE_ID}_rubric.json"


def test_llm_rubric_judge_parses_scores_and_writes_outputs(tmp_path: Path) -> None:
    run_dir, rubric_path = prepare_judge_run(tmp_path)
    judge = LLMRubricJudge(
        rubric_path,
        run_dir,
        llm_client=FakeJudgeClient(judge_response(score=2)),
    )

    result = judge.evaluate()

    assert result["status"] == "success"
    assert result["judge_failed"] is False
    assert result["raw_total"] == 12
    assert result["final_total"] == 12
    assert (run_dir / "judge_prompt.json").is_file()
    assert (run_dir / "judge_raw_response.json").is_file()
    assert (run_dir / "judge_scores.json").is_file()
    assert (run_dir / "trajectory_scores.json").is_file()
    assert (run_dir / "error_analysis.md").is_file()
    trajectory_scores = json.loads(
        (run_dir / "trajectory_scores.json").read_text(encoding="utf-8")
    )
    assert trajectory_scores["score_type"] == "dyn_traj_alignment_v1"
    assert "macro_trajectory_total" in trajectory_scores
    assert "legacy_macro" in trajectory_scores
    assert result["trajectory_evaluation"]["macro_trajectory_total"] == trajectory_scores[
        "macro_trajectory_total"
    ]
    assert result["dual_layer_scores"]["micro_final_answer_total"] == 12.0
    error_text = (run_dir / "error_analysis.md").read_text(encoding="utf-8")
    assert "## Trajectory Process Score" in error_text
    assert "Macro trajectory total" in error_text
    assert "DynTraj score" in error_text
    prompt_text = (run_dir / "judge_prompt.json").read_text(encoding="utf-8")
    assert "TCGA-38-4626_rubric" not in (run_dir / "trajectory.jsonl").read_text(
        encoding="utf-8"
    )
    assert "rubric" in prompt_text.lower()


def test_llm_rubric_judge_clamps_out_of_range_scores(tmp_path: Path) -> None:
    run_dir, rubric_path = prepare_judge_run(tmp_path)

    result = LLMRubricJudge(
        rubric_path,
        run_dir,
        llm_client=FakeJudgeClient(judge_response(score=999)),
    ).evaluate()

    assert result["status"] == "success"
    for dim, max_score in DIMENSION_WEIGHTS.items():
        assert result["llm_scores"][dim]["score"] == max_score
    assert result["raw_total"] == 100


def test_llm_rubric_judge_invalid_json_fails_without_fake_score(tmp_path: Path) -> None:
    run_dir, rubric_path = prepare_judge_run(tmp_path)

    result = LLMRubricJudge(
        rubric_path,
        run_dir,
        llm_client=FakeJudgeClient("not-json"),
    ).evaluate()

    assert result["status"] == "judge_failed"
    assert result["judge_failed"] is True
    assert result["final_total"] is None
    assert result["llm_scores"] == {}


def test_extract_json_object_repairs_unescaped_quotes_in_strings() -> None:
    broken = """
    {
      "dimensions": {
        "GM": {
          "score": 0,
          "criteria": [
            {
              "criterion": "回答需说明不属于"孤立转移"范畴。",
              "status": "missed",
              "reason": "未提及"
            }
          ]
        }
      }
    }
    """

    data = _extract_json_object(broken)
    criterion = data["dimensions"]["GM"]["criteria"][0]["criterion"]
    assert "孤立转移" in criterion


def test_extract_json_object_repairs_missing_criteria_object_brace() -> None:
    broken = """
    {
      "dimensions": {
        "GM": {
          "score": 0,
          "criteria": [
            {
              "criterion": "example",
              "reason": "missing brace",
              "status": "missed"
          ],
          "matched": []
        }
      },
      "overall_rationale": "ok"
    }
    """

    data = _extract_json_object(broken)
    assert data["dimensions"]["GM"]["criteria"][0]["status"] == "missed"


def test_extract_json_object_parses_failed_openai_judge_response() -> None:
    raw_path = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "TCGA-50-8459"
        / "run_openai_5.5"
        / "judge_raw_response.json"
    )
    if not raw_path.is_file():
        pytest.skip("sample failed judge response not available")

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    data = _extract_json_object(raw["message"]["content"])
    assert set(data["dimensions"]) >= {"CI", "GM", "RA", "CR", "EG", "COMM"}
    assert data["dimensions"]["GM"]["score"] == 0


def test_cli_run_and_judge_uses_fake_components_without_external_api(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=True)

    monkeypatch.setattr(
        BenchmarkRunner,
        "_build_runtime",
        lambda self: FakeRuntime(tmp_path),
    )
    monkeypatch.setattr(
        BenchmarkRunner,
        "_build_llm_client",
        lambda self: FakeMultimodalAgentClient(),
    )

    class FakeCliJudge:
        def __init__(
            self,
            rubric_path: Path,
            run_dir: Path,
            provider: str | None = None,
        ) -> None:
            self.run_dir = Path(run_dir)

        def evaluate(self) -> dict[str, Any]:
            score_trajectory_run(self.run_dir)
            result = {"status": "success", "final_total": 100}
            (self.run_dir / "judge_scores.json").write_text(
                json.dumps(result, ensure_ascii=False),
                encoding="utf-8",
            )
            (self.run_dir / "error_analysis.md").write_text("# ok\n", encoding="utf-8")
            return result

    monkeypatch.setattr(benchmark_cli, "LLMRubricJudge", FakeCliJudge)

    exit_code = benchmark_cli.main(
        [
            "run-and-judge",
            "--case-dir",
            str(case_dir),
            "--rubric-path",
            str(case_dir / "evaluation" / f"{CASE_ID}_rubric.json"),
            "--judge",
            "llm",
            "--run-id",
            "run_cli",
            "--runs-root",
            str(tmp_path / "runs"),
        ]
    )

    run_dir = tmp_path / "runs" / CASE_ID / "run_cli"
    assert exit_code == 0
    assert (run_dir / "trajectory.jsonl").is_file()
    assert (run_dir / "guideline_trajectory.json").is_file()
    assert (run_dir / "judge_scores.json").is_file()
    assert (run_dir / "trajectory_scores.json").is_file()


def test_batch_cli_evaluate_flag_reaches_both_batch_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_batch(config: BatchRunConfig) -> dict[str, Any]:
        seen["batch"] = config
        return {"status": "success"}

    def fake_dual(config: DualAgentBatchConfig) -> dict[str, Any]:
        seen["dual"] = config
        return {"status": "success"}

    monkeypatch.setattr(batch_runner, "run_batch", fake_batch)
    monkeypatch.setattr(dual_agent_runner, "run_dual_agent_batch", fake_dual)

    assert benchmark_cli.main(
        [
            "batch-run",
            "--cases-root",
            str(tmp_path / "cases"),
            "--evaluate",
        ]
    ) == 0
    assert benchmark_cli.main(
        [
            "dual-agent-batch-run",
            "--cases-root",
            str(tmp_path / "cases"),
            "--evaluate",
        ]
    ) == 0

    assert seen["batch"].evaluate is True
    assert seen["dual"].evaluate is True


def test_cli_rejects_removed_runner_mode_flag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        benchmark_cli.main(
            [
                "batch-run",
                "--cases-root",
                str(tmp_path / "cases"),
                "--runner-mode",
                "mock",
            ]
        )

    assert exc_info.value.code == 2
    assert "unrecognized arguments: --runner-mode mock" in capsys.readouterr().err


def test_dual_agent_cli_run_and_judge_uses_existing_scoring(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case_dir = copy_lightweight_case(tmp_path, with_modalities=False)
    seen: dict[str, Any] = {}

    class FakeDualCliRunner:
        def __init__(
            self,
            *,
            case_dir: Path,
            run_id: str,
            runs_root: Path,
            agent: str,
            planner_memory_dir: Path,
            planner_decoder_artifact_dir: Path,
            planner_top_k: int,
            planner_device: str,
            planner_query_encoder_device: str,
            planner_max_new_tokens: int,
            planner_output_mode: str,
            max_planner_rounds: int,
            max_tool_rounds_per_step: int,
        ) -> None:
            self.case_id = Path(case_dir).name
            self.run_id = run_id
            self.run_dir = Path(runs_root) / self.case_id / run_id

        def run(self) -> Any:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "final_answers.json").write_text("{}", encoding="utf-8")
            return type("FakeResult", (), {"run_dir": self.run_dir})()

    class FakeDualCliJudge:
        def __init__(
            self,
            rubric_path: Path,
            run_dir: Path,
            provider: str | None = None,
        ) -> None:
            seen["rubric_path"] = rubric_path
            seen["run_dir"] = Path(run_dir)

        def evaluate(self) -> dict[str, Any]:
            (seen["run_dir"] / "judge_scores.json").write_text("{}", encoding="utf-8")
            (seen["run_dir"] / "error_analysis.md").write_text("# ok\n", encoding="utf-8")
            return {"status": "success", "final_total": 88}

    monkeypatch.setattr(benchmark_cli, "DualAgentBenchmarkRunner", FakeDualCliRunner)
    monkeypatch.setattr(benchmark_cli, "LLMRubricJudge", FakeDualCliJudge)

    exit_code = benchmark_cli.main(
        [
            "dual-agent-run-and-judge",
            "--case-dir",
            str(case_dir),
            "--run-id",
            "dual_cli",
            "--runs-root",
            str(tmp_path / "runs"),
        ]
    )

    run_dir = tmp_path / "runs" / CASE_ID / "dual_cli"
    assert exit_code == 0
    assert seen["rubric_path"] == (case_dir / "evaluation" / f"{CASE_ID}_rubric.json").resolve()
    assert (run_dir / "judge_scores.json").is_file()
    assert (run_dir / "error_analysis.md").is_file()
