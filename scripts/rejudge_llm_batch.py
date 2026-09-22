#!/usr/bin/env python3
"""Rerun the case-specific LLM rubric judge over existing benchmark runs."""

from __future__ import annotations

import argparse
import random
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medclaw.utils import hash_json, read_json, utc_now, write_json  # noqa: E402
from medclaw_benchmark.case_paths import find_rubric_path  # noqa: E402
from medclaw_benchmark.llm_judge import LLMRubricJudge  # noqa: E402


EVALUATION_FILES = (
    "judge_prompt.json",
    "judge_raw_response.json",
    "judge_scores.json",
    "trajectory_scores.json",
    "error_analysis.md",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Call the LLM Judge again for existing runs, recompute Macro, and "
            "refresh the 50/50 combined score."
        )
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--cases-root",
        type=Path,
        action="append",
        required=True,
        help="Processed case root, such as .../LUNG or .../UCEC. Repeat as needed.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        required=True,
        help="Run group to rejudge. Repeat for multiple groups.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Optional case filter. Repeat to rejudge selected cases only.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        default=[],
        help="Report paired LEFT-minus-RIGHT Micro/Macro/combined differences.",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        help="Judge provider; defaults to MEDCLAW_JUDGE_PROVIDER.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary path; defaults to <runs-root>/llm_rejudge_summary.json.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful rows already recorded in the output summary.",
    )
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs_root = args.runs_root.resolve()
    if not runs_root.is_dir():
        raise SystemExit(f"Runs root does not exist: {runs_root}")
    run_ids = _unique(args.run_id)
    selected_cases = set(args.case_id)
    unknown_pairs = sorted(
        {item for pair in args.pair for item in pair if item not in run_ids}
    )
    if unknown_pairs:
        raise SystemExit(
            "Every --pair run ID must also be supplied with --run-id: "
            + ", ".join(unknown_pairs)
        )

    case_index = _build_case_index(args.cases_root)
    output = (args.output or runs_root / "llm_rejudge_summary.json").resolve()
    prior_rows = _resume_rows(output) if args.resume else {}
    jobs = _discover_jobs(runs_root, run_ids, selected_cases)
    if selected_cases:
        found = {case_id for case_id, _, _ in jobs}
        missing = sorted(selected_cases - found)
        if missing:
            raise SystemExit("Selected cases have no matching run directory: " + ", ".join(missing))

    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = runs_root / ".llm_rejudge_backups" / session
    rows: list[dict[str, Any]] = []
    for index, (case_id, run_id, run_dir) in enumerate(jobs, 1):
        key = (case_id, run_id)
        previous = prior_rows.get(key)
        if previous is not None and previous.get("status") == "success":
            row = dict(previous)
            row["resume_skipped"] = True
            rows.append(row)
            print(f"[{index}/{len(jobs)}] [resume-skip] {case_id}/{run_id}", flush=True)
            continue
        row = _rejudge_one(
            case_id=case_id,
            run_id=run_id,
            run_dir=run_dir,
            case_dir=case_index.get(case_id),
            provider=args.judge_provider,
            backup_root=None if args.no_backup else backup_root,
            dry_run=args.dry_run,
        )
        rows.append(row)
        print(
            f"[{index}/{len(jobs)}] [{row['status']}] {case_id}/{run_id} "
            f"micro={_display(row.get('micro'))} "
            f"macro={_display(row.get('macro'))} "
            f"combined={_display(row.get('combined'))}",
            flush=True,
        )
        if not args.dry_run:
            _write_summary(output, runs_root, run_ids, args, rows, session)
        if args.fail_fast and row["status"] not in {"success", "ready"}:
            break

    summary = _make_summary(runs_root, run_ids, args, rows, session)
    if not args.dry_run:
        write_json(output, summary)
    _print_summary(summary, output if not args.dry_run else None)
    return 2 if any(row.get("status") not in {"success", "ready"} for row in rows) else 0


def _build_case_index(roots: Sequence[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for raw_root in roots:
        root = raw_root.resolve()
        if not root.is_dir():
            raise SystemExit(f"Cases root does not exist: {root}")
        for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if not (case_dir / "evaluation").is_dir():
                continue
            existing = index.get(case_dir.name)
            if existing is not None and existing != case_dir.resolve():
                raise SystemExit(
                    f"Duplicate case ID {case_dir.name!r} in {existing} and {case_dir.resolve()}"
                )
            index[case_dir.name] = case_dir.resolve()
    return index


def _discover_jobs(
    runs_root: Path,
    run_ids: Sequence[str],
    selected_cases: set[str],
) -> list[tuple[str, str, Path]]:
    jobs: list[tuple[str, str, Path]] = []
    for case_run_root in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        case_id = case_run_root.name
        if selected_cases and case_id not in selected_cases:
            continue
        for run_id in run_ids:
            run_dir = case_run_root / run_id
            if run_dir.is_dir():
                jobs.append((case_id, run_id, run_dir.resolve()))
    return jobs


def _rejudge_one(
    *,
    case_id: str,
    run_id: str,
    run_dir: Path,
    case_dir: Path | None,
    provider: str | None,
    backup_root: Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    if case_dir is None:
        return _problem_row(case_id, run_id, run_dir, "missing", "case not found in --cases-root")
    try:
        rubric_path = find_rubric_path(case_dir)
    except Exception as exc:
        return _problem_row(case_id, run_id, run_dir, "missing", f"rubric: {type(exc).__name__}: {exc}")
    required = ("final_answers.json", "evidence_board.json", "guideline_trajectory.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        return _problem_row(case_id, run_id, run_dir, "missing", "missing input: " + ", ".join(missing))
    rubric_hash = hash_json(read_json(rubric_path))
    previous_micro = _existing_micro(run_dir)
    if dry_run:
        return {
            "case_id": case_id,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "rubric_path": str(rubric_path),
            "rubric_hash": rubric_hash,
            "status": "ready",
            "previous_micro": previous_micro,
            "micro": None,
            "macro": None,
            "combined": None,
        }
    if backup_root is not None:
        _backup_evaluation(run_dir, backup_root / case_id / run_id)
    try:
        result = LLMRubricJudge(
            rubric_path=rubric_path,
            run_dir=run_dir,
            provider=provider,
        ).evaluate()
        dual = result.get("dual_layer_scores")
        dual = dual if isinstance(dual, Mapping) else {}
        micro = result.get("final_total")
        macro = dual.get("macro_trajectory_total")
        combined = dual.get("combined_total")
        return {
            "case_id": case_id,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "rubric_path": str(rubric_path),
            "rubric_hash": rubric_hash,
            "status": "success" if result.get("status") == "success" else "judge_failed",
            "reason": result.get("error"),
            "previous_micro": previous_micro,
            "micro": micro,
            "micro_change": (
                float(micro) - float(previous_micro)
                if isinstance(micro, (int, float)) and isinstance(previous_micro, (int, float))
                else None
            ),
            "macro": macro,
            "combined": combined,
            "judge_model": result.get("judge_model"),
        }
    except Exception as exc:
        return _problem_row(case_id, run_id, run_dir, "failed", f"{type(exc).__name__}: {exc}")


def _backup_evaluation(run_dir: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in EVALUATION_FILES:
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, destination / name)


def _existing_micro(run_dir: Path) -> float | int | None:
    path = run_dir / "judge_scores.json"
    if not path.is_file():
        return None
    value = read_json(path)
    if not isinstance(value, Mapping):
        return None
    score = value.get("final_total")
    return score if isinstance(score, (int, float)) else None


def _problem_row(
    case_id: str, run_id: str, run_dir: Path, status: str, reason: str
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": status,
        "reason": reason,
        "previous_micro": _existing_micro(run_dir),
        "micro": None,
        "macro": None,
        "combined": None,
    }


def _resume_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    value = read_json(path)
    if not isinstance(value, Mapping):
        return {}
    rows = value.get("runs")
    if not isinstance(rows, list):
        return {}
    return {
        (str(row["case_id"]), str(row["run_id"])): dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("case_id") and row.get("run_id")
    }


def _write_summary(
    output: Path,
    runs_root: Path,
    run_ids: Sequence[str],
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    session: str,
) -> None:
    write_json(output, _make_summary(runs_root, run_ids, args, rows, session))


def _make_summary(
    runs_root: Path,
    run_ids: Sequence[str],
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    session: str,
) -> dict[str, Any]:
    return {
        "schema_version": "llm_rejudge_batch.v1",
        "generated_at": utc_now(),
        "session_id": session,
        "dry_run": bool(args.dry_run),
        "judge_provider": args.judge_provider,
        "runs_root": str(runs_root),
        "cases_roots": [str(path.resolve()) for path in args.cases_root],
        "run_ids": list(run_ids),
        "groups": {
            run_id: _group_summary([row for row in rows if row.get("run_id") == run_id])
            for run_id in run_ids
        },
        "comparisons": [_paired_summary(rows, left, right) for left, right in args.pair],
        "runs": list(rows),
    }


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "success": sum(row.get("status") == "success" for row in rows),
        "ready": sum(row.get("status") == "ready" for row in rows),
        "missing": sum(row.get("status") == "missing" for row in rows),
        "judge_failed": sum(row.get("status") == "judge_failed" for row in rows),
        "failed": sum(row.get("status") == "failed" for row in rows),
        "micro": _distribution(rows, "micro"),
        "macro": _distribution(rows, "macro"),
        "combined": _distribution(rows, "combined"),
        "micro_change": _distribution(rows, "micro_change"),
    }


def _distribution(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 6) if values else None,
        "median": round(statistics.median(values), 6) if values else None,
    }


def _paired_summary(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> dict[str, Any]:
    indexed = {
        run_id: {
            str(row["case_id"]): row
            for row in rows
            if row.get("run_id") == run_id and row.get("status") == "success"
        }
        for run_id in (left, right)
    }
    pairs: list[dict[str, Any]] = []
    mismatched: list[str] = []
    for case_id in sorted(set(indexed[left]) & set(indexed[right])):
        lhs, rhs = indexed[left][case_id], indexed[right][case_id]
        if lhs.get("rubric_hash") != rhs.get("rubric_hash"):
            mismatched.append(case_id)
            continue
        item: dict[str, Any] = {"case_id": case_id, "rubric_hash": lhs.get("rubric_hash")}
        for metric in ("micro", "macro", "combined"):
            if isinstance(lhs.get(metric), (int, float)) and isinstance(rhs.get(metric), (int, float)):
                item[f"{metric}_difference"] = float(lhs[metric]) - float(rhs[metric])
        pairs.append(item)
    metrics: dict[str, Any] = {}
    for metric in ("micro", "macro", "combined"):
        values = [float(item[f"{metric}_difference"]) for item in pairs if f"{metric}_difference" in item]
        metrics[metric] = {
            "count": len(values),
            "mean_difference": round(statistics.mean(values), 6) if values else None,
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
            "bootstrap_95ci": _bootstrap_ci(values),
        }
    return {
        "left": left,
        "right": right,
        "difference": "left_minus_right",
        "paired_count": len(pairs),
        "rubric_mismatch_count": len(mismatched),
        "rubric_mismatch_cases": mismatched,
        "metrics": metrics,
        "pairs": pairs,
    }


def _bootstrap_ci(values: Sequence[float], samples: int = 2000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(17)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [
        round(means[int(0.025 * (samples - 1))], 6),
        round(means[int(0.975 * (samples - 1))], 6),
    ]


def _print_summary(summary: Mapping[str, Any], output: Path | None) -> None:
    print("\nSummary")
    print("run_id\ttotal\tsuccess\tmissing\tjudge_failed\tfailed\tmicro_mean\tmacro_mean\tcombined_mean\tmicro_change")
    for run_id, group in summary["groups"].items():
        print(
            "\t".join(
                [
                    run_id,
                    str(group["total"]),
                    str(group["success"]),
                    str(group["missing"]),
                    str(group["judge_failed"]),
                    str(group["failed"]),
                    _display(group["micro"]["mean"]),
                    _display(group["macro"]["mean"]),
                    _display(group["combined"]["mean"]),
                    _display(group["micro_change"]["mean"]),
                ]
            )
        )
    for comparison in summary["comparisons"]:
        parts = []
        for metric, detail in comparison["metrics"].items():
            parts.append(
                f"{metric}: n={detail['count']} diff={_display(detail['mean_difference'])} "
                f"W/T/L={detail['wins']}/{detail['ties']}/{detail['losses']} "
                f"CI={detail['bootstrap_95ci']}"
            )
        print(f"pair {comparison['left']} - {comparison['right']}: " + "; ".join(parts))
    if output is not None:
        print(f"summary_json={output}")


def _display(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def _unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
