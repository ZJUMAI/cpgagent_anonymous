from __future__ import annotations

import json
from pathlib import Path

from medclaw.utils import write_json
from medclaw_benchmark.trajectory_scorer import combine_micro_macro_scores
from scripts import rejudge_llm_batch


def _prepare(tmp_path: Path) -> tuple[Path, Path, Path]:
    cases_root = tmp_path / "cases" / "LUNG"
    case_dir = cases_root / "CASE-1"
    evaluation = case_dir / "evaluation"
    evaluation.mkdir(parents=True)
    write_json(
        evaluation / "CASE-1_rubric.json",
        {"case_id": "CASE-1", "rubric_version": "test", "rubric": {}},
    )
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "CASE-1" / "planner"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "final_answers.json", {"case_id": "CASE-1", "answer_text": "answer"})
    write_json(run_dir / "evidence_board.json", {"case_id": "CASE-1", "evidence_items": []})
    write_json(run_dir / "guideline_trajectory.json", {"case_id": "CASE-1", "trajectory": []})
    write_json(run_dir / "judge_scores.json", {"final_total": 40})
    return cases_root, runs_root, run_dir


def test_default_combined_score_uses_equal_micro_macro_weights() -> None:
    result = combine_micro_macro_scores(micro_total=80, macro_total=40)
    assert result["combined_total"] == 60
    assert result["weights"] == {"micro": 0.5, "macro": 0.5}


def test_dry_run_discovers_inputs_without_overwriting(tmp_path: Path) -> None:
    cases_root, runs_root, run_dir = _prepare(tmp_path)
    result = rejudge_llm_batch.main(
        [
            "--runs-root",
            str(runs_root),
            "--cases-root",
            str(cases_root),
            "--run-id",
            "planner",
            "--dry-run",
        ]
    )
    assert result == 0
    assert json.loads((run_dir / "judge_scores.json").read_text())["final_total"] == 40
    assert not (runs_root / "llm_rejudge_summary.json").exists()


def test_rejudge_calls_judge_and_backs_up_existing_scores(
    tmp_path: Path, monkeypatch
) -> None:
    cases_root, runs_root, run_dir = _prepare(tmp_path)

    class FakeJudge:
        def __init__(self, *, rubric_path, run_dir, provider=None) -> None:
            self.run_dir = Path(run_dir)

        def evaluate(self):
            result = {
                "status": "success",
                "final_total": 75,
                "dual_layer_scores": {
                    "micro_final_answer_total": 75,
                    "macro_trajectory_total": 50,
                    "combined_total": 62.5,
                },
                "judge_model": {"provider": "fake"},
            }
            write_json(self.run_dir / "judge_scores.json", result)
            return result

    monkeypatch.setattr(rejudge_llm_batch, "LLMRubricJudge", FakeJudge)
    result = rejudge_llm_batch.main(
        [
            "--runs-root",
            str(runs_root),
            "--cases-root",
            str(cases_root),
            "--run-id",
            "planner",
        ]
    )
    assert result == 0
    summary = json.loads((runs_root / "llm_rejudge_summary.json").read_text())
    assert summary["groups"]["planner"]["micro"]["mean"] == 75
    assert summary["groups"]["planner"]["combined"]["mean"] == 62.5
    assert summary["runs"][0]["micro_change"] == 35
    backups = list((runs_root / ".llm_rejudge_backups").rglob("judge_scores.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text())["final_total"] == 40
