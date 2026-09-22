"""Tests for dynamic rubric step-alignment trajectory scoring."""

from __future__ import annotations

import json
from pathlib import Path

from medclaw.utils import write_json
from medclaw_benchmark.dyn_traj_scorer import (
    AgentStep,
    GoldStep,
    W_COVER,
    align_steps,
    compute_act_score,
    compute_act_score_detail,
    extract_agent_steps,
    extract_gold_steps,
    score_alignment_pairs,
    score_dyn_trajectory_run,
)
from medclaw_benchmark.io_utils import append_jsonl
from medclaw_benchmark.trajectory_scorer import score_trajectory_run


def _action(action_id: str, action: str, action_type: str = "communication") -> dict:
    return {
        "action_id": action_id,
        "action_type": action_type,
        "action": action,
        "priority": "high",
        "bucket": "required",
    }


def _gold_step(
    step: int,
    *,
    phase: str,
    required: list[dict] | None = None,
    acceptable: list[dict] | None = None,
    conditional: list[dict] | None = None,
    unsafe: list[dict] | None = None,
    visible_state: dict | None = None,
) -> dict:
    if visible_state is None:
        visible_state = {
            "known_facts": {"tnm": "T2a N0 M0", "ECOG": "1"},
            "open_questions": [],
        }
    return {
        "step": step,
        "phase": phase,
        "visible_state": visible_state,
        "planner": {
            "action_set": {
                "required": required or [],
                "acceptable": acceptable or [],
                "conditional": conditional or [],
                "unsafe": unsafe or [],
            }
        },
        "verifier": {"verifications": []},
    }


def _guideline_trajectory(steps: list[dict], case_id: str = "TEST-CASE") -> dict:
    return {
        "case_id": case_id,
        "schema_version": "trajectory.dual_lm.v1",
        "trajectory": steps,
    }


def _dual_round(
    round_index: int,
    *,
    next_step: str,
    answer: str = "",
    phase: str = "postoperative_review",
    tool_skills: list[str] | None = None,
) -> dict:
    return {
        "round_index": round_index,
        "planner_output": {
            "current_phase": phase,
            "next_step": next_step,
        },
        "agent_round_answer": answer,
        "tool_skills": tool_skills or [],
    }


def _write_run_dir(
    tmp_path: Path,
    *,
    gold_steps: list[dict],
    dual_rounds: list[dict] | None = None,
    tool_calls: list[dict] | None = None,
    final_answer: str = "",
) -> Path:
    run_dir = tmp_path / "runs" / "TEST-CASE" / "run_test"
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "guideline_trajectory.json",
        _guideline_trajectory(gold_steps),
    )
    write_json(
        run_dir / "final_answers.json",
        {
            "case_id": "TEST-CASE",
            "run_id": "run_test",
            "answer_text": final_answer,
        },
    )
    if dual_rounds is not None:
        for record in dual_rounds:
            append_jsonl(run_dir / "dual_agent_rounds.jsonl", record)
    if tool_calls is not None:
        for record in tool_calls:
            append_jsonl(run_dir / "tool_calls.jsonl", record)
    return run_dir


def test_extract_gold_and_agent_steps(tmp_path: Path) -> None:
    run_dir = _write_run_dir(
        tmp_path,
        gold_steps=[_gold_step(0, phase="review", required=[_action("a0", "assess ECOG performance status")])],
        dual_rounds=[_dual_round(0, next_step="assess ECOG performance status", answer="ECOG is 1")],
    )
    gold = extract_gold_steps(json.loads((run_dir / "guideline_trajectory.json").read_text()))
    agent = extract_agent_steps(run_dir, json.loads((run_dir / "final_answers.json").read_text()))
    assert len(gold) == 1
    assert len(agent) == 1
    assert agent[0].source == "dual_agent_rounds"
    assert "ecog" in agent[0].action_text.lower()


def test_split_step_equivalence_scores_well(tmp_path: Path) -> None:
    gold_steps = [
        _gold_step(
            0,
            phase="postoperative_review",
            required=[
                _action("a0", "confirm definitive surgery was completed", "surgical_completeness"),
                _action("a1", "identify ECOG performance status or explicitly flag it as missing", "fitness"),
            ],
        )
    ]
    dual_rounds = [
        _dual_round(
            0,
            next_step="confirm definitive surgery was completed",
            answer="Surgery completed with negative margins.",
        ),
        _dual_round(
            1,
            next_step="identify ECOG performance status",
            answer="ECOG performance status is 1.",
        ),
    ]
    run_dir = _write_run_dir(tmp_path, gold_steps=gold_steps, dual_rounds=dual_rounds)
    result = score_dyn_trajectory_run(run_dir)

    assert result["status"] == "success"
    assert result["alignment_size"] >= 1
    assert result["miss_step_count"] == 0
    assert result["macro_trajectory_total"] >= 50


def test_merge_step_equivalence_aligns_without_full_miss(tmp_path: Path) -> None:
    gold_steps = [
        _gold_step(
            0,
            phase="molecular_review",
            required=[_action("a0", "order molecular testing for EGFR and ALK", "molecular_testing")],
        ),
        _gold_step(
            1,
            phase="staging_review",
            required=[_action("a1", "confirm TNM staging", "staging")],
        ),
    ]
    dual_rounds = [
        _dual_round(
            0,
            next_step="order molecular testing for EGFR and ALK; confirm TNM staging",
            answer="Recommend molecular testing for EGFR/ALK and review TNM stage.",
        )
    ]
    run_dir = _write_run_dir(tmp_path, gold_steps=gold_steps, dual_rounds=dual_rounds)
    gold = extract_gold_steps(json.loads((run_dir / "guideline_trajectory.json").read_text()))
    agent = extract_agent_steps(run_dir, json.loads((run_dir / "final_answers.json").read_text()))
    alignment, miss_gold, miss_agent = align_steps(gold, agent)

    assert len(miss_agent) == 0
    assert len(miss_gold) <= 1


def test_redundant_agent_steps_are_penalized(tmp_path: Path) -> None:
    gold_steps = [
        _gold_step(
            0,
            phase="review",
            required=[_action("a0", "assess ECOG performance status", "fitness")],
        )
    ]
    dual_rounds = [
        _dual_round(0, next_step="assess ECOG performance status", answer="ECOG is 1."),
        _dual_round(1, next_step="repeat unrelated imaging review", answer="Repeat CT review again."),
        _dual_round(2, next_step="repeat unrelated imaging review", answer="Repeat CT review again."),
    ]
    run_dir = _write_run_dir(tmp_path, gold_steps=gold_steps, dual_rounds=dual_rounds)
    result = score_dyn_trajectory_run(run_dir)

    assert result["redundant_step_count"] >= 1
    assert result["dyn_traj_score"] < 0.95


def test_cond_aware_scores_conditional_with_if(tmp_path: Path) -> None:
    conditional = [
        {
            "action_id": "c0",
            "action_type": "targeted_therapy",
            "action": "consider Osimertinib adjuvant therapy",
            "condition": "if EGFR mutation is positive",
            "required_evidence": ["EGFR"],
            "bucket": "conditional",
        }
    ]
    action_set = {
        "required": [],
        "acceptable": [],
        "conditional": conditional,
        "unsafe": [],
    }
    good_text = "If EGFR mutation is positive, consider Osimertinib adjuvant therapy."
    bad_text = "Recommend Osimertinib adjuvant therapy now."
    good_score = compute_act_score(action_set, good_text, {"EGFR": "unknown"})
    bad_score = compute_act_score(action_set, bad_text, {"EGFR": "unknown"})
    assert good_score > bad_score


def test_act_score_detail_preserves_components_and_counts() -> None:
    action_set = {
        "required": [_action("r0", "confirm TNM staging", "staging")],
        "acceptable": [_action("a0", "assess ECOG performance status", "fitness")],
        "conditional": [
            {
                "action_id": "c0",
                "action_type": "targeted_therapy",
                "action": "consider Osimertinib adjuvant therapy",
                "condition": "if EGFR mutation is positive",
                "required_evidence": ["EGFR"],
                "bucket": "conditional",
            }
        ],
        "unsafe": [_action("u0", "give thoracic radiation", "treatment_planning")],
    }
    detail = compute_act_score_detail(
        action_set,
        "Confirm TNM stage and ECOG. If EGFR is positive, consider Osimertinib adjuvant therapy.",
        {"EGFR": "unknown"},
    )

    assert detail["required"]["matched_count"] == 1
    assert detail["acceptable"]["matched_count"] == 1
    assert detail["conditional"]["condition_aware_count"] == 1
    assert detail["unsafe"]["hit_count"] == 0
    assert detail["weighted_components"]["required_coverage"] == W_COVER
    assert detail["score"] == compute_act_score(
        action_set,
        "Confirm TNM stage and ECOG. If EGFR is positive, consider Osimertinib adjuvant therapy.",
        {"EGFR": "unknown"},
    )


def test_dynamic_score_persists_all_score_components(tmp_path: Path) -> None:
    run_dir = _write_run_dir(
        tmp_path,
        gold_steps=[
            _gold_step(
                0,
                phase="review",
                required=[_action("a0", "assess ECOG performance status", "fitness")],
            )
        ],
        dual_rounds=[
            _dual_round(
                0,
                next_step="assess ECOG performance status",
                answer="ECOG performance status is 1",
            )
        ],
    )
    result = score_dyn_trajectory_run(run_dir)

    assert "avg_act_score" in result
    assert "avg_match_score" in result
    assert "avg_state_similarity" in result
    assert "avg_action_match" in result
    assert result["score_summary"]["action"]["required_matched_count"] == 1
    assert "miss_penalty" in result["trajectory_score_components"]
    assert "coherence_bonus" in result["trajectory_score_components"]
    pair = result["alignment"][0]
    assert pair["act_score_detail"]["required"]["matched_count"] == 1
    assert "weighted_components" in pair["act_score_detail"]
    assert "state_contribution" in pair["match_score_detail"]
    assert "act_score_contribution" in pair["step_score_detail"]

    persisted = json.loads((run_dir / "trajectory_scores.json").read_text())
    assert persisted["avg_act_score"] == result["avg_act_score"]
    assert persisted["score_summary"] == result["score_summary"]


def test_score_trajectory_run_uses_dyn_alignment(tmp_path: Path) -> None:
    run_dir = _write_run_dir(
        tmp_path,
        gold_steps=[
            _gold_step(
                0,
                phase="review",
                required=[_action("a0", "assess ECOG performance status", "fitness")],
            )
        ],
        dual_rounds=[_dual_round(0, next_step="assess ECOG performance status", answer="ECOG is 1")],
    )
    result = score_trajectory_run(run_dir)
    assert result["score_type"] == "dyn_traj_alignment_v1"
    assert "legacy_macro" in result


def test_fallback_tool_calls_by_phase(tmp_path: Path) -> None:
    run_dir = _write_run_dir(
        tmp_path,
        gold_steps=[
            _gold_step(
                0,
                phase="molecular_review",
                required=[_action("a0", "order molecular testing for EGFR", "molecular_testing")],
            )
        ],
        tool_calls=[
            {
                "phase": "molecular_review",
                "skill_name": "molecular.query_biomarkers",
                "result": {"summary": "order molecular testing for EGFR and ALK"},
            }
        ],
        final_answer="Summarize molecular testing plan.",
    )
    agent = extract_agent_steps(run_dir, json.loads((run_dir / "final_answers.json").read_text()))
    assert len(agent) >= 1
    assert agent[0].source == "tool_calls_by_phase"
