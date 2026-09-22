from __future__ import annotations

import json
from pathlib import Path

from medclaw.trajectory.builder import build_guideline_trajectory
from medclaw.trajectory.report import extract_patient_events
from medclaw.trajectory.verification import score_action_set, verify_action


def _write_report(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# TCGA-TEST",
                "| 字段 | 值 |",
                "|---|---|",
                "| `demographic.age_at_index` | 57 |",
                "| `demographic.gender` | female |",
                "| `demographic.race` | white |",
                "| `cases.primary_site` | Bronchus and lung |",
                "| `cases.disease_type` | Adenomas and Adenocarcinomas |",
                "| `diagnoses.primary_diagnosis` | Adenocarcinoma, NOS |",
                "| `diagnoses.ajcc_pathologic_t` | T2b |",
                "| `diagnoses.ajcc_pathologic_n` | N0 |",
                "| `diagnoses.ajcc_pathologic_m` | M0 |",
                "| `treatments.treatment_intent_type` | Adjuvant |",
                "| `treatments.treatment_or_therapy` | no |",
                "| `treatments.treatment_type` | Pharmaceutical Therapy, NOS |",
                "| `treatments.treatment_intent_type` | Adjuvant |",
                "| `treatments.treatment_or_therapy` | yes |",
                "| `treatments.treatment_type` | Radiation Therapy, NOS |",
                "| `demographic.vital_status` | Alive |",
            ]
        ),
        encoding="utf-8",
    )


def test_extract_patient_events_from_markdown_report(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    _write_report(report)

    events = extract_patient_events("TCGA-TEST", report)

    assert [event.stage for event in events][:3] == [
        "baseline",
        "diagnosis_confirmation",
        "staging",
    ]
    assert any(event.event_type == "therapy" for event in events)
    assert any("T2b" in event.content for event in events)


def test_build_trajectory_hides_future_information(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    output = tmp_path / "trajectory.json"
    _write_report(report)

    data = build_guideline_trajectory("TCGA-TEST", report, output_path=output)

    assert output.is_file()
    assert data["guideline_id"] == "NSCLC_2010"
    assert data["guideline_version"] == "2010"
    assert data["decision_date"] == "2010-12-31"
    assert data["evaluation_mode"] == "fixed_historical_guideline"
    step0 = data["trajectory"][0]
    assert "diagnosis" not in step0["visible_state"]
    assert "pathology" in step0["hidden_future"]
    step2 = data["trajectory"][2]
    assert step2["visible_state"]["T"] == "T2b"
    step3 = data["trajectory"][3]
    assert step3["phase"] == "biomarker_assessment"
    assert step3["visible_state"]["molecular_status"] == "unknown"
    assert step3["visible_state"]["PD-L1"] == "unknown"
    step4 = data["trajectory"][4]
    assert step4["phase"] == "initial_treatment_decision"
    assert step4["action_set"]["required"]
    assert any(
        action["action_id"].startswith("s4_")
        for action in step4["action_set"]["required"]
    )
    assert "molecular status unknown" in data["canonical_state"]["open_questions"]
    assert "expected_next_actions" not in step4
    assert "forbidden_actions" not in step4
    treatment_actions = [
        item["action"] for item in step4["action_set"]["required"]
    ]
    assert "derive AJCC stage group from T2bN0M0" in treatment_actions


def test_trajectory_includes_verification_templates(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    _write_report(report)

    data = build_guideline_trajectory("TCGA-TEST", report)
    step1 = data["trajectory"][1]

    assert step1["model_prediction"] is None
    staging_verification = [
        item
        for item in step1["verification"]
        if item["action"] == "complete TNM staging"
    ][0]
    assert staging_verification["classification"] == "supported_by_report_and_guideline"
    assert staging_verification["evidence_event_ids"]
    assert staging_verification["state_update"]["T"] == "T2b"


def test_treatment_or_therapy_no_is_not_marked_as_observed_treatment(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.md"
    _write_report(report)

    data = build_guideline_trajectory("TCGA-TEST", report)
    treatments = data["canonical_state"]["known_facts"]["observed_treatments"]

    pharmaceutical = [
        item
        for item in treatments
        if item["treatment_type"] == "Pharmaceutical Therapy, NOS"
    ][0]
    assert pharmaceutical["status"] == "recorded_as_no"
    assert pharmaceutical["note"] == "treatment_or_therapy = no"


def test_guideline_supported_unobserved_is_not_wrong(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    _write_report(report)
    data = build_guideline_trajectory("TCGA-TEST", report)
    rules = data["guideline_rubric_rules"]
    events = extract_patient_events("TCGA-TEST", report)

    verification = verify_action(
        {"action_id": "a1", "action": "perform molecular testing for non-squamous NSCLC"},
        future_events=[event for event in events if event.available_at_step > 1],
        rubric_rules=[_rule_obj(rule) for rule in rules],
    )

    assert verification.classification == "guideline_supported_but_unobserved"
    assert verification.report_support == "not_observed"
    assert verification.guideline_support == "supported"


def test_build_trajectory_includes_dynamic_action_set(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    _write_report(report)

    data = build_guideline_trajectory("TCGA-TEST", report)

    assert data["schema_version"] == "trajectory.dynamic_rubric.v1"
    step0 = data["trajectory"][0]
    assert "action_set" in step0
    assert set(step0["action_set"].keys()) == {
        "required",
        "acceptable",
        "conditional",
        "unsafe",
    }
    assert "expected_next_actions" not in step0
    assert "forbidden_actions" not in step0
    assert step0["action_set"]["unsafe"]


def test_action_set_reports_expected_action_coverage(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    _write_report(report)
    data = build_guideline_trajectory("TCGA-TEST", report)

    score = score_action_set(
        [{"action": "obtain pathology confirmation"}],
        data["trajectory"][0]["action_set"]["required"],
    )

    assert 0.0 <= score["coverage"] <= 1.0


def test_free_text_sex_does_not_treat_is_female_zero_as_female(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_text(
        "This case is a 61-year-old white male with T2b N1 M0 disease.\n"
        "| is_female | 0.0 |\n",
        encoding="utf-8",
    )

    events = extract_patient_events("CASE-MALE", report)

    baseline = next(event for event in events if event.stage == "baseline")
    assert baseline.attributes["sex"] == "male"


def test_free_text_sex_supports_chinese_female_and_feature_fallback(tmp_path: Path) -> None:
    chinese = tmp_path / "chinese.md"
    chinese.write_text("本病例为57岁女性肺腺癌患者，病理分期T3N0M0。", encoding="utf-8")
    feature = tmp_path / "feature.md"
    feature.write_text("| is_female | 1.0 |", encoding="utf-8")

    chinese_events = extract_patient_events("CASE-F", chinese)
    feature_events = extract_patient_events("CASE-FEATURE", feature)

    assert chinese_events[0].attributes["sex"] == "female"
    assert feature_events[0].attributes["sex"] == "female"


def test_clinical_json_is_authoritative_and_renders_case_tnm(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    clinical = tmp_path / "clinical.json"
    report.write_text(
        "This case is a 70-year-old Asian male with pathologic stage IIB "
        "(T3 N0 M0) lung adenocarcinoma.",
        encoding="utf-8",
    )
    clinical.write_text(
        json.dumps(
            {
                "case_id": "CASE-T3",
                "age": 70,
                "sex": "male",
                "race": "Asian",
                "primary_site": "Lung",
                "disease_type": "Lung Adenocarcinoma",
                "primary_diagnosis": "Adenocarcinoma",
                "pathologic_t_stage": "T3",
                "pathologic_n_stage": "N0",
                "pathologic_m_stage": "M0",
                "pathologic_stage": "Stage IIB",
            }
        ),
        encoding="utf-8",
    )

    data = build_guideline_trajectory(
        "CASE-T3", report, clinical_path=clinical
    )

    actions = [
        item["action"]
        for item in data["trajectory"][4]["action_set"]["required"]
    ]
    assert "derive AJCC stage group from T3N0M0" in actions
    assert all("T2bN0M0" not in action for action in actions)
    assert data["canonical_state"]["known_facts"]["sex"] == "male"


def test_clinical_report_conflict_fails_closed(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    clinical = tmp_path / "clinical.json"
    report.write_text(
        "This case is a 61-year-old white male with T2b N1 M0 disease.",
        encoding="utf-8",
    )
    clinical.write_text(
        json.dumps(
            {
                "case_id": "CASE-CONFLICT",
                "sex": "female",
                "pathologic_t_stage": "T2b",
                "pathologic_n_stage": "N1",
                "pathologic_m_stage": "M0",
            }
        ),
        encoding="utf-8",
    )

    import pytest

    with pytest.raises(ValueError, match="Sex conflict"):
        extract_patient_events(
            "CASE-CONFLICT", report, clinical_path=clinical
        )


def test_unknown_tnm_component_is_not_treated_as_stageable(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_text(
        "This case is a 61-year-old male with T1b N0 MX lung cancer.",
        encoding="utf-8",
    )

    data = build_guideline_trajectory("CASE-MX", report)

    actions = [
        item["action"]
        for item in data["trajectory"][4]["action_set"]["required"]
    ]
    assert (
        "complete unknown TNM components before deriving AJCC stage group "
        "(currently T1bN0MX)"
    ) in actions


def _rule_obj(data: dict):
    from medclaw.trajectory.schema import RubricRule

    return RubricRule(
        rule_id=data["rule_id"],
        applicable_stage=data["applicable_stage"],
        condition=data["condition"],
        recommended_actions=tuple(data["recommended_actions"]),
        required_evidence=tuple(data["required_evidence"]),
        forbidden_actions=tuple(data["forbidden_actions"]),
        priority=data["priority"],
        guideline_source=data["guideline_source"],
    )
