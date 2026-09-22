"""NCCN 2010-derived rubric rules for NSCLC trajectory construction."""

from __future__ import annotations

from medclaw.trajectory.schema import RubricRule


def nsclc_2010_rubric_rules(
    *, guideline_source: str = "NSCLC_2010"
) -> list[RubricRule]:
    """Return the compact NSCLC 2010 condition-action rubric.

    The rules are deliberately structured as action sets, not exact text
    answers. ``{tnm}`` is expanded from the case-visible state by the trajectory
    builder; it must never contain a patient-specific value in this template.
    """

    return [
        RubricRule(
            rule_id="NSCLC_initial_workup_001",
            applicable_stage="baseline",
            condition={"suspected_or_known_lung_primary": True},
            recommended_actions=(
                "confirm histological subtype",
                "complete clinical staging workup",
                "assess performance status and comorbidities",
            ),
            required_evidence=("pathology", "T stage", "N stage", "M stage"),
            forbidden_actions=("start definitive systemic therapy before staging",),
            priority="must",
            guideline_source=guideline_source,
        ),
        RubricRule(
            rule_id="NSCLC_diagnosis_to_staging_001",
            applicable_stage="diagnosis_confirmation",
            condition={"nsclc_confirmed": True, "stage_known": False},
            recommended_actions=(
                "complete TNM staging",
                "evaluate resectability",
                "perform molecular testing for non-squamous NSCLC",
                "consider PD-L1 testing when systemic therapy may be needed",
            ),
            required_evidence=(
                "T stage",
                "N stage",
                "M stage",
                "driver mutation status",
                "PD-L1 status",
            ),
            forbidden_actions=("choose systemic regimen before stage and biomarkers",),
            priority="must",
            guideline_source=guideline_source,
        ),
        RubricRule(
            rule_id="NSCLC_stage_known_001",
            applicable_stage="staging",
            condition={"tnm_available": True},
            recommended_actions=(
                "determine AJCC stage group",
                "confirm surgical resectability if no distant metastasis",
                "plan surgery-centered treatment if resectable",
                "plan adjuvant therapy according to stage, margins, nodal status, and biomarkers",
            ),
            required_evidence=(
                "AJCC stage group",
                "resectability",
                "surgical margin status",
                "nodal status",
                "driver mutation status",
            ),
            forbidden_actions=("ignore nodal status when planning adjuvant therapy",),
            priority="must",
            guideline_source=guideline_source,
        ),
        RubricRule(
            rule_id="NSCLC_biomarker_missing_001",
            applicable_stage="biomarker_assessment",
            condition={"non_squamous_nsclc": True, "biomarker_status_known": False},
            recommended_actions=(
                "recognize molecular and PD-L1 status are missing",
                "recommend molecular testing if systemic therapy or adjuvant targeted therapy is being considered",
                "document unknown biomarkers as open questions",
            ),
            required_evidence=(
                "EGFR",
                "ALK",
                "ROS1",
                "BRAF",
                "MET",
                "RET",
                "NTRK",
                "KRAS",
                "PD-L1",
            ),
            forbidden_actions=("treat unknown biomarker status as negative",),
            priority="should",
            guideline_source=guideline_source,
        ),
        RubricRule(
            rule_id="NSCLC_initial_treatment_decision_001",
            applicable_stage="initial_treatment_decision",
            condition={"tnm_available": True, "distant_metastasis_absent": True},
            recommended_actions=(
                "derive AJCC stage group from {tnm}",
                "evaluate whether disease is surgically resectable",
                "recommend surgery-centered curative-intent treatment if medically operable",
                "assess need for adjuvant therapy based on final stage, margin status, risk factors, and biomarkers",
                "do not recommend metastatic systemic therapy unless distant metastasis is documented",
            ),
            required_evidence=(
                "T stage",
                "N stage",
                "M stage",
                "AJCC stage group",
                "resectability",
                "margin status",
                "risk factors",
                "biomarkers",
            ),
            forbidden_actions=("recommend metastatic systemic therapy for documented M0 disease",),
            priority="must",
            guideline_source=guideline_source,
        ),
        RubricRule(
            rule_id="NSCLC_treatment_review_001",
            applicable_stage="treatment_observed",
            condition={"treatment_observed": True},
            recommended_actions=(
                "compare observed treatment with the NCCN 2010 recommendation",
                "identify missing staging, margin, biomarker, or performance-status evidence",
                "separate observed care from the NCCN 2010 benchmark recommendation",
            ),
            required_evidence=(
                "observed treatment",
                "diagnosis year",
                "stage",
                "biomarkers",
                "performance status",
            ),
            forbidden_actions=("treat observed historical treatment as the only gold answer",),
            priority="must",
            guideline_source=guideline_source,
        ),
    ]
