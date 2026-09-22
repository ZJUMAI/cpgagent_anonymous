#!/usr/bin/env python3
"""Batch-rescore benchmark runs with the repository's current legacy scorer.

This script does not call an LLM. It recomputes ``trajectory_scores.json``,
reuses ``judge_scores.json.final_total`` as the static-rubric Micro score, and
updates the 50/50 combined score. Existing files are backed up once by default.
V2-named files are never read, modified, or deleted.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medclaw.utils import hash_json, read_json, utc_now, write_json  # noqa: E402
from medclaw_benchmark.trajectory_scorer import (  # noqa: E402
    combine_micro_macro_scores,
    score_trajectory_run,
)


BACKUP_SUFFIX = ".before_legacy_rescore.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute trajectory and combined scores with dyn_traj_alignment_v1.",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("data/runs"))
    parser.add_argument(
        "--run-id",
        action="append",
        required=True,
        help="Run group to rescore. Repeat for multiple groups.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        default=[],
        help="Emit a case-paired LEFT-minus-RIGHT comparison. May be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary JSON path; defaults to <runs-root>/trajectory_rescore_legacy.json.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create one-time backups before overwriting legacy score files.",
    )
    parser.add_argument(
        "--no-update-judge",
        action="store_true",
        help="Recompute trajectory_scores.json but leave judge_scores.json unchanged.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and validate run directories without writing anything.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.runs_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Runs root does not exist: {root}")
    run_ids = _unique(args.run_id)
    unknown_pairs = sorted(
        {item for pair in args.pair for item in pair if item not in run_ids}
    )
    if unknown_pairs:
        raise SystemExit(
            "Every --pair run ID must also be supplied with --run-id: "
            + ", ".join(unknown_pairs)
        )

    rows: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for run_id in run_ids:
            run_dir = case_dir / run_id
            if not run_dir.is_dir():
                continue
            row = _rescore_one(
                run_dir,
                case_id=case_dir.name,
                run_id=run_id,
                backup=not args.no_backup,
                update_judge=not args.no_update_judge,
                dry_run=args.dry_run,
            )
            rows.append(row)
            print(
                f"[{row['status']}] {case_dir.name}/{run_id} "
                f"macro={_display(row.get('macro'))} "
                f"micro={_display(row.get('micro'))} "
                f"combined={_display(row.get('combined'))}",
                flush=True,
            )

    groups = {
        run_id: _group_summary([row for row in rows if row["run_id"] == run_id])
        for run_id in run_ids
    }
    comparisons = [
        _paired_summary(rows, left, right) for left, right in args.pair
    ]
    result = {
        "schema_version": "trajectory_rescore_legacy.v1",
        "generated_at": utc_now(),
        "dry_run": bool(args.dry_run),
        "scorer": "dyn_traj_alignment_v1",
        "runs_root": str(root),
        "run_ids": run_ids,
        "groups": groups,
        "comparisons": comparisons,
        "runs": rows,
    }
    output = (args.output or root / "trajectory_rescore_legacy.json").resolve()
    if not args.dry_run:
        write_json(output, result)
    _print_summary(groups, comparisons, output if not args.dry_run else None)

    incomplete = sum(
        group["missing"] + group["failed"] for group in groups.values()
    )
    return 2 if incomplete else 0


def _rescore_one(
    run_dir: Path,
    *,
    case_id: str,
    run_id: str,
    backup: bool,
    update_judge: bool,
    dry_run: bool,
) -> dict[str, Any]:
    required = (run_dir / "guideline_trajectory.json", run_dir / "final_answers.json")
    missing_inputs = [path.name for path in required if not path.is_file()]
    reference_hash = _reference_hash(run_dir / "guideline_trajectory.json")
    if missing_inputs:
        return {
            "case_id": case_id,
            "run_id": run_id,
            "run_dir": str(run_dir.resolve()),
            "status": "missing",
            "reason": "missing input: " + ", ".join(missing_inputs),
            "reference_hash": reference_hash,
            "macro": None,
            "micro": _existing_micro(run_dir),
            "combined": None,
        }
    if dry_run:
        return {
            "case_id": case_id,
            "run_id": run_id,
            "run_dir": str(run_dir.resolve()),
            "status": "ready",
            "reference_hash": reference_hash,
            "macro": None,
            "micro": _existing_micro(run_dir),
            "combined": None,
        }

    trajectory_path = run_dir / "trajectory_scores.json"
    judge_path = run_dir / "judge_scores.json"
    if backup:
        _backup_once(trajectory_path)
        if update_judge:
            _backup_once(judge_path)
    try:
        trajectory = score_trajectory_run(run_dir)
        macro = trajectory.get("macro_trajectory_total")
        micro = _existing_micro(run_dir)
        combined = combine_micro_macro_scores(
            micro_total=micro,
            macro_total=macro,
        )
        judge_updated = False
        if update_judge and judge_path.is_file():
            judge = read_json(judge_path)
            if not isinstance(judge, Mapping):
                raise ValueError(f"judge_scores.json must contain an object: {judge_path}")
            judge_payload = dict(judge)
            judge_payload["trajectory_scores_path"] = str(trajectory_path.resolve())
            judge_payload["trajectory_evaluation"] = _trajectory_evaluation(trajectory)
            judge_payload["dual_layer_scores"] = combined
            judge_payload["legacy_trajectory_rescore"] = {
                "rescored_at": utc_now(),
                "scorer": trajectory.get("score_type"),
                "static_micro_reused": True,
                "reference_hash": reference_hash,
            }
            write_json(judge_path, judge_payload)
            judge_updated = True
        status = str(trajectory.get("status") or "failed")
        return {
            "case_id": case_id,
            "run_id": run_id,
            "run_dir": str(run_dir.resolve()),
            "status": status,
            "reason": trajectory.get("reason"),
            "score_type": trajectory.get("score_type"),
            "reference_hash": reference_hash,
            "macro": macro,
            "micro": micro,
            "combined": combined.get("combined_total"),
            "judge_updated": judge_updated,
            "trajectory_scores": str(trajectory_path.resolve()),
            "judge_scores": str(judge_path.resolve()) if judge_path.is_file() else None,
        }
    except Exception as exc:
        return {
            "case_id": case_id,
            "run_id": run_id,
            "run_dir": str(run_dir.resolve()),
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "reference_hash": reference_hash,
            "macro": None,
            "micro": _existing_micro(run_dir),
            "combined": None,
        }


def _trajectory_evaluation(scores: Mapping[str, Any]) -> dict[str, Any]:
    legacy = scores.get("legacy_macro")
    legacy = legacy if isinstance(legacy, Mapping) else {}
    return {
        "status": scores.get("status"),
        "macro_trajectory_total": scores.get("macro_trajectory_total"),
        "coverage": legacy.get("coverage"),
        "expected_action_count": legacy.get("expected_action_count"),
        "matched_action_count": legacy.get("matched_action_count"),
        "alignment_size": scores.get("alignment_size"),
        "avg_match_score": scores.get("avg_match_score"),
        "avg_state_similarity": scores.get("avg_state_similarity"),
        "avg_action_match": scores.get("avg_action_match"),
        "avg_act_score": scores.get("avg_act_score"),
        "avg_step_score": scores.get("avg_step_score"),
        "miss_step_count": scores.get("miss_step_count"),
        "redundant_step_count": scores.get("redundant_step_count"),
        "coherence": scores.get("coherence"),
        "dyn_traj_score": scores.get("dyn_traj_score"),
        "score_summary": scores.get("score_summary"),
        "trajectory_score_components": scores.get("trajectory_score_components"),
    }


def _existing_micro(run_dir: Path) -> float | int | None:
    path = run_dir / "judge_scores.json"
    if not path.is_file():
        return None
    value = read_json(path)
    if not isinstance(value, Mapping):
        return None
    micro = value.get("final_total")
    return micro if isinstance(micro, (int, float)) else None


def _reference_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = read_json(path)
    return hash_json(value)


def _backup_once(path: Path) -> Path | None:
    if not path.is_file():
        return None
    destination = path.with_name(path.stem + BACKUP_SUFFIX)
    if not destination.exists():
        shutil.copy2(path, destination)
    return destination


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "success": sum(row.get("status") == "success" for row in rows),
        "ready": sum(row.get("status") == "ready" for row in rows),
        "missing": sum(row.get("status") == "missing" for row in rows),
        "failed": sum(row.get("status") == "failed" for row in rows),
        "macro": _distribution(rows, "macro"),
        "micro": _distribution(rows, "micro"),
        "combined": _distribution(rows, "combined"),
    }


def _distribution(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, float | int | None]:
    values = [
        float(row[key]) for row in rows if isinstance(row.get(key), (int, float))
    ]
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
        if lhs.get("reference_hash") != rhs.get("reference_hash"):
            mismatched.append(case_id)
            continue
        item: dict[str, Any] = {
            "case_id": case_id,
            "reference_hash": lhs.get("reference_hash"),
        }
        for metric in ("macro", "micro", "combined"):
            if isinstance(lhs.get(metric), (int, float)) and isinstance(
                rhs.get(metric), (int, float)
            ):
                item[f"{metric}_difference"] = float(lhs[metric]) - float(rhs[metric])
        pairs.append(item)
    macro = [
        float(item["macro_difference"])
        for item in pairs
        if "macro_difference" in item
    ]
    return {
        "left": left,
        "right": right,
        "difference": "left_minus_right",
        "paired_count": len(pairs),
        "reference_mismatch_count": len(mismatched),
        "reference_mismatch_cases": mismatched,
        "wins": sum(value > 0 for value in macro),
        "ties": sum(value == 0 for value in macro),
        "losses": sum(value < 0 for value in macro),
        "macro_mean_difference": round(statistics.mean(macro), 6) if macro else None,
        "macro_bootstrap_95ci": _bootstrap_ci(macro),
        "pairs": pairs,
    }


def _bootstrap_ci(values: Sequence[float], samples: int = 2000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(17)
    means = sorted(
        statistics.mean(rng.choices(values, k=len(values)))
        for _ in range(samples)
    )
    return [
        round(means[int(0.025 * (samples - 1))], 6),
        round(means[int(0.975 * (samples - 1))], 6),
    ]


def _print_summary(
    groups: Mapping[str, Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    output: Path | None,
) -> None:
    print("\nSummary")
    print("run_id\ttotal\tsuccess\tmissing\tfailed\tmacro_mean\tmicro_mean\tcombined_mean")
    for run_id, group in groups.items():
        print(
            "\t".join(
                [
                    run_id,
                    str(group["total"]),
                    str(group["success"]),
                    str(group["missing"]),
                    str(group["failed"]),
                    _display(group["macro"]["mean"]),
                    _display(group["micro"]["mean"]),
                    _display(group["combined"]["mean"]),
                ]
            )
        )
    for comparison in comparisons:
        print(
            f"pair {comparison['left']} - {comparison['right']}: "
            f"n={comparison['paired_count']} "
            f"W/T/L={comparison['wins']}/{comparison['ties']}/{comparison['losses']} "
            f"macro_diff={_display(comparison['macro_mean_difference'])} "
            f"95%CI={comparison['macro_bootstrap_95ci']}"
        )
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
