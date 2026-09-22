"""Batch benchmark execution over processed case directories."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from medclaw.llm.factory import DEFAULT_PROVIDER
from medclaw.utils import utc_now, write_json

from medclaw_benchmark.case_paths import find_rubric_path
from medclaw_benchmark.llm_judge import LLMRubricJudge
from medclaw_benchmark.runner import BenchmarkRunner


@dataclass(frozen=True)
class BatchRunConfig:
    """Configuration for one resumable benchmark batch."""

    cases_root: Path
    runs_root: Path
    run_id: str = "run_001"
    agent: str = DEFAULT_PROVIDER
    judge: str = "llm"
    judge_provider: str | None = None
    evaluate: bool = False
    limit: int = 0
    force: bool = False
    dry_run: bool = False


def run_batch(config: BatchRunConfig) -> dict[str, Any]:
    """Run all processed cases, skipping completed runs by default."""

    cases = discover_ready_cases(config.cases_root)
    if config.limit > 0:
        cases = cases[: config.limit]

    started_at = utc_now()
    records: list[dict[str, Any]] = []
    counts = {"total": len(cases), "completed": 0, "skipped": 0, "failed": 0}
    for case_dir in cases:
        record = _run_one_case(case_dir, config)
        records.append(record)
        status = str(record.get("status"))
        if status in counts:
            counts[status] += 1

    summary = {
        "status": "success" if counts["failed"] == 0 else "partial_failure",
        "evaluation_enabled": config.evaluate,
        "started_at": started_at,
        "finished_at": utc_now(),
        "config": {
            "cases_root": str(config.cases_root.resolve()),
            "runs_root": str(config.runs_root.resolve()),
            "run_id": config.run_id,
            "agent": config.agent,
            "judge": config.judge,
            "judge_provider": config.judge_provider,
            "evaluation_enabled": config.evaluate,
            "limit": config.limit,
            "force": config.force,
            "dry_run": config.dry_run,
        },
        "counts": counts,
        "cases": records,
    }
    config.runs_root.mkdir(parents=True, exist_ok=True)
    write_json(config.runs_root / f"batch_summary_{config.run_id}.json", summary)
    return summary


def discover_ready_cases(cases_root: Path) -> list[Path]:
    """Return case directories that contain an evaluation folder."""

    root = Path(cases_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cases root does not exist: {root}")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "evaluation").is_dir()
    )


def is_run_inference_complete(case_dir: Path, runs_root: Path, run_id: str) -> bool:
    """Return whether the MedClaw inference artifact is complete."""

    run_dir = Path(runs_root).resolve() / case_dir.name / run_id
    return (run_dir / "final_answers.json").is_file()


def is_run_evaluation_complete(case_dir: Path, runs_root: Path, run_id: str) -> bool:
    run_dir = Path(runs_root).resolve() / case_dir.name / run_id
    return (run_dir / "judge_scores.json").is_file()


def is_run_complete(
    case_dir: Path,
    runs_root: Path,
    run_id: str,
    *,
    evaluate: bool = False,
) -> bool:
    """Return completion according to the requested inference/evaluation scope."""

    return is_run_inference_complete(case_dir, runs_root, run_id) and (
        not evaluate or is_run_evaluation_complete(case_dir, runs_root, run_id)
    )


def _run_one_case(case_dir: Path, config: BatchRunConfig) -> dict[str, Any]:
    run_dir = Path(config.runs_root).resolve() / case_dir.name / config.run_id
    inference_complete = is_run_inference_complete(
        case_dir, config.runs_root, config.run_id
    )
    evaluation_complete = is_run_evaluation_complete(
        case_dir, config.runs_root, config.run_id
    )
    if not config.force and inference_complete and (
        not config.evaluate or evaluation_complete
    ):
        return {
            "case_id": case_dir.name,
            "case_dir": str(case_dir.resolve()),
            "run_dir": str(run_dir),
            "status": "skipped",
            "reason": (
                "existing inference and evaluation outputs"
                if config.evaluate
                else "existing inference output"
            ),
            "evaluation_enabled": config.evaluate,
            "execution_mode": "skipped",
        }
    if config.dry_run:
        return {
            "case_id": case_dir.name,
            "case_dir": str(case_dir.resolve()),
            "run_dir": str(run_dir),
            "status": "skipped",
            "reason": "dry_run",
            "evaluation_enabled": config.evaluate,
            "execution_mode": "skipped",
        }

    evaluation_only = not config.force and inference_complete
    try:
        with _temporary_cases_root(config.cases_root):
            if evaluation_only:
                result_run_dir = run_dir
            else:
                result = BenchmarkRunner(
                    case_dir=case_dir,
                    run_id=config.run_id,
                    runs_root=config.runs_root,
                    agent=config.agent,
                ).run()
                result_run_dir = result.run_dir
            scores: dict[str, Any] | None = None
            if config.evaluate:
                rubric_path = _resolve_rubric_path(case_dir, None)
                scores = LLMRubricJudge(
                    rubric_path,
                    result_run_dir,
                    provider=config.judge_provider,
                ).evaluate()
        summary = _run_summary(result_run_dir)
        record = {
            "case_id": case_dir.name,
            "case_dir": str(case_dir.resolve()),
            "run_dir": str(result_run_dir),
            "status": "completed",
            "evaluation_enabled": config.evaluate,
            "execution_mode": (
                "evaluation_only"
                if evaluation_only
                else (
                    "inference_and_evaluation"
                    if config.evaluate
                    else "inference_only"
                )
            ),
            "outputs": summary,
        }
        if scores is not None:
            record["judge_status"] = scores.get("status")
            record["final_total"] = scores.get("final_total")
        return record
    except Exception as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        error = {
            "case_id": case_dir.name,
            "case_dir": str(case_dir.resolve()),
            "run_dir": str(run_dir),
            "status": "failed",
            "evaluation_enabled": config.evaluate,
            "execution_mode": (
                "evaluation_only"
                if evaluation_only
                else (
                    "inference_and_evaluation"
                    if config.evaluate
                    else "inference_only"
                )
            ),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "failed_at": utc_now(),
        }
        (run_dir / "batch_error.json").write_text(
            json.dumps(error, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return error


@contextmanager
def _temporary_cases_root(cases_root: Path) -> Iterator[None]:
    previous = os.environ.get("MEDCLAW_CASES_ROOT")
    os.environ["MEDCLAW_CASES_ROOT"] = str(Path(cases_root).resolve())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MEDCLAW_CASES_ROOT", None)
        else:
            os.environ["MEDCLAW_CASES_ROOT"] = previous


def _run_summary(run_dir: Path) -> dict[str, str]:
    return {
        "run_dir": str(run_dir.resolve()),
        "trajectory": str((run_dir / "trajectory.jsonl").resolve()),
        "guideline_trajectory": str((run_dir / "guideline_trajectory.json").resolve()),
        "tool_calls": str((run_dir / "tool_calls.jsonl").resolve()),
        "evidence_board": str((run_dir / "evidence_board.json").resolve()),
        "final_answers": str((run_dir / "final_answers.json").resolve()),
        "judge_scores": str((run_dir / "judge_scores.json").resolve()),
        "trajectory_scores": str((run_dir / "trajectory_scores.json").resolve()),
        "error_analysis": str((run_dir / "error_analysis.md").resolve()),
    }


def _resolve_rubric_path(case_dir: Path, rubric_path: Path | None) -> Path:
    return find_rubric_path(case_dir, rubric_path)
