import pytest

from medclaw.trajectory.action_set import (
    assemble_action_set,
    canonicalize_guideline_trajectory,
    evidence_missing,
    flatten_visible_state,
    resolve_action_set,
    resolve_verifications,
)


def test_assemble_action_set_routes_missing_ecog_to_conditional() -> None:
    state = {"ECOG": "missing", "open_questions": ["ECOG/performance status missing"]}
    action_set = assemble_action_set(
        [
            {
                "action_id": "s0_a0",
                "action_type": "fitness",
                "action": "identify ECOG performance status or explicitly flag it as missing",
                "priority": "high",
                "guideline_rule_ids": ["rule_1"],
                "required_evidence": ["ECOG"],
                "rationale": "test",
            },
            {
                "action_id": "s0_a1",
                "action_type": "fitness",
                "action": "consider adjuvant chemotherapy only when fitness supports it",
                "priority": "high",
                "guideline_rule_ids": ["rule_1"],
                "required_evidence": ["ECOG"],
                "depends_on_evidence": ["ECOG", "performance status"],
                "rationale": "test",
                "conditional_if_missing": True,
                "condition": "if ECOG supports adjuvant chemotherapy",
            },
            {
                "action_id": "s0_a2",
                "action_type": "communication",
                "action": "document open questions before finalizing recommendations",
                "priority": "medium",
                "bucket": "acceptable",
                "guideline_rule_ids": ["rule_1"],
                "required_evidence": [],
                "rationale": "test",
            },
        ],
        forbidden_actions=["recommend adjuvant treatment without considering ECOG"],
        state=state,
        step=0,
    )

    assert len(action_set["required"]) == 1
    assert len(action_set["conditional"]) == 1
    assert action_set["conditional"][0]["condition"].startswith("if")
    assert len(action_set["acceptable"]) == 1
    assert len(action_set["unsafe"]) == 1


def test_resolve_action_set_reads_dual_lm_planner() -> None:
    resolved = resolve_action_set(
        {
            "planner": {
                "action_set": {
                    "required": [
                        {
                            "action_id": "a0",
                            "action": "confirm surgery",
                            "action_type": "pathology",
                        }
                    ],
                    "acceptable": [
                        {"action_id": "a1", "action": "consider PET/CT if staging incomplete"}
                    ],
                    "conditional": [],
                    "unsafe": [
                        {"action_id": "u0", "action": "recommend metastatic therapy for M0 disease"}
                    ],
                }
            }
        }
    )
    assert len(resolved["required"]) == 1
    assert resolved["required"][0]["action"] == "confirm surgery"
    assert len(resolved["acceptable"]) == 1
    assert len(resolved["unsafe"]) == 1
    assert "metastatic" in resolved["unsafe"][0]["action"]


def test_resolve_action_set_reads_dynamic_rubric_step() -> None:
    resolved = resolve_action_set(
        {
            "action_set": {
                "required": [
                    {
                        "action_id": "s0_a0",
                        "action": "confirm histological subtype",
                        "action_type": "pathology",
                    }
                ],
                "acceptable": [],
                "conditional": [
                    {
                        "action_id": "s0_c0",
                        "action": "consider systemic therapy if staging supports it",
                    }
                ],
                "unsafe": [
                    {
                        "action_id": "s0_u0",
                        "action": "start definitive systemic therapy before staging",
                    }
                ],
            }
        }
    )
    assert resolved["required"][0]["action_id"] == "s0_a0"
    assert resolved["conditional"][0]["action_id"] == "s0_c0"
    assert resolved["unsafe"][0]["action_id"] == "s0_u0"


def test_resolve_verifications_reads_dual_lm_verifier() -> None:
    items = resolve_verifications(
        {
            "verifier": {
                "verifications": [
                    {
                        "action_id": "a0",
                        "classification": "supported_by_report_and_guideline",
                    }
                ]
            }
        }
    )
    assert len(items) == 1
    assert items[0]["classification"] == "supported_by_report_and_guideline"


def test_resolve_verifications_reads_dynamic_rubric_step() -> None:
    items = resolve_verifications(
        {
            "verification": [
                {
                    "action_id": "s0_a0",
                    "classification": "guideline_supported_but_unobserved",
                }
            ]
        }
    )
    assert len(items) == 1
    assert items[0]["action_id"] == "s0_a0"


def test_canonicalize_dual_lm_step_moves_scoring_fields_to_root() -> None:
    trajectory = canonicalize_guideline_trajectory(
        {
            "schema_version": "trajectory.dual_lm.v1",
            "trajectory": [
                {
                    "step": 0,
                    "planner": {
                        "reasoning_summary": "review pathology",
                        "action_set": {
                            "required": [{"action_id": "a0", "action": "confirm pathology"}],
                            "acceptable": [],
                            "conditional": [],
                            "unsafe": [],
                        },
                    },
                    "verifier": {
                        "summary": "supported",
                        "verifications": [{"action_id": "a0", "classification": "supported"}],
                    },
                }
            ],
        }
    )
    step = trajectory["trajectory"][0]
    assert step["action_set"]["required"][0]["action_id"] == "a0"
    assert step["verification"][0]["action_id"] == "a0"
    assert "action_set" not in step["planner"]
    assert "verifications" not in step["verifier"]
    assert step["planner"]["reasoning_summary"] == "review pathology"
    assert step["verifier"]["summary"] == "supported"


def test_conflicting_duplicate_action_sets_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting step.action_set"):
        resolve_action_set(
            {
                "action_set": {
                    "required": [{"action_id": "root", "action": "root action"}],
                },
                "planner": {
                    "action_set": {
                        "required": [
                            {"action_id": "nested", "action": "nested action"}
                        ],
                    }
                },
            }
        )


def test_flatten_visible_state_promotes_known_facts() -> None:
    flat = flatten_visible_state(
        {
            "known_facts": {
                "pathologic_stage": "IA",
                "tnm": "pT1b pN0",
                "histology": "lung adenocarcinoma",
            },
            "open_questions": ["ECOG missing"],
        }
    )
    assert flat["pathologic_stage"] == "IA"
    assert flat["tnm"] == "pT1b pN0"
    assert flat["open_questions"] == ["ECOG missing"]


def test_evidence_missing_detects_ecog_and_pd_l1() -> None:
    state = {
        "ECOG": "missing",
        "biomarkers_mentioned": ["EGFR", "KRAS"],
        "open_questions": ["PD-L1 status missing or not clearly reported."],
    }
    assert evidence_missing(state, ["ECOG"])
    assert evidence_missing(state, ["PD-L1"])
