"""Command line interface for MedClaw trajectory benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from medclaw.llm.factory import DEFAULT_PROVIDER, SUPPORTED_PROVIDERS, resolve_provider
from medclaw_benchmark.case_builder import CaseBuilder
from medclaw_benchmark.case_paths import find_rubric_path
from medclaw_benchmark.dual_agent_runner import (
    LEGACY_PLANNER_DECODER_DIR,
    LEGACY_PLANNER_MEMORY_DIR,
    DualAgentBenchmarkRunner,
)
from medclaw_benchmark.llm_judge import LLMRubricJudge
from medclaw_benchmark.runner import BenchmarkRunner
from guideline_planner.planner import PLANNER_ABLATION_MODES


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-case", help="Build lightweight case files.")
    build.add_argument("--case-dir", required=True, type=Path)
    build.add_argument("--report-path", type=Path)
    build.add_argument("--rubric-path", type=Path)

    run = subparsers.add_parser("run", help="Run a benchmark trajectory.")
    _add_run_args(run)

    dual_run = subparsers.add_parser(
        "dual-agent-run",
        help="Run a planner-guided dual-agent benchmark trajectory.",
    )
    _add_dual_agent_run_args(dual_run)

    dual_both = subparsers.add_parser(
        "dual-agent-run-and-judge",
        help="Run and judge a planner-guided dual-agent benchmark trajectory.",
    )
    _add_dual_agent_run_args(dual_both)
    dual_both.add_argument("--rubric-path", type=Path)
    dual_both.add_argument("--judge", choices=["llm"], default="llm")
    _add_judge_provider_arg(dual_both)

    judge = subparsers.add_parser("judge", help="Judge an existing run.")
    judge.add_argument("--run-dir", required=True, type=Path)
    judge.add_argument("--rubric-path", required=True, type=Path)
    judge.add_argument("--judge", choices=["llm"], default="llm")
    _add_judge_provider_arg(judge)

    both = subparsers.add_parser("run-and-judge", help="Run and judge in one command.")
    _add_run_args(both)
    both.add_argument("--rubric-path", type=Path)
    both.add_argument("--judge", choices=["llm"], default="llm")
    _add_judge_provider_arg(both)

    batch = subparsers.add_parser("batch-run", help="Run a resumable case batch.")
    batch.add_argument(
        "--cases-root",
        type=Path,
        default=Path("/data4/tujiayong/processed/LUNG"),
    )
    batch.add_argument("--run-id", default="run_001")
    batch.add_argument("--runs-root", type=Path, default=Path("runs"))
    batch.add_argument(
        "--agent",
        choices=list(SUPPORTED_PROVIDERS),
        default=resolve_provider(None),
    )
    batch.add_argument("--judge", choices=["llm"], default="llm")
    _add_judge_provider_arg(batch)
    batch.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the LLM Judge after inference; disabled by default.",
    )
    batch.add_argument("--limit", type=int, default=0)
    batch.add_argument("--force", action="store_true")
    batch.add_argument("--dry-run", action="store_true")

    dual_batch = subparsers.add_parser(
        "dual-agent-batch-run",
        help="Run a resumable planner-guided dual-agent case batch.",
    )
    dual_batch.add_argument(
        "--cases-root",
        type=Path,
        default=Path("/data4/share/cpgtrajbench/LUNG"),
    )
    dual_batch.add_argument("--run-id", default="dual_agent_001")
    dual_batch.add_argument("--runs-root", type=Path, default=Path("runs"))
    dual_batch.add_argument(
        "--agent",
        choices=list(SUPPORTED_PROVIDERS),
        default=resolve_provider(None),
    )
    dual_batch.add_argument("--judge", choices=["llm"], default="llm")
    _add_judge_provider_arg(dual_batch)
    dual_batch.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the LLM Judge after inference; disabled by default.",
    )
    dual_batch.add_argument(
        "--planner-release-dir",
        type=Path,
        default=None,
        help="Hash-bound planner_release.v2 directory used for formal runs.",
    )
    dual_batch.add_argument(
        "--planner-mode",
        choices=["latent_topk", "daa_full"],
        default=None,
        help="Override the release default mode for an explicit ablation run.",
    )
    _add_planner_ablation_args(dual_batch)
    dual_batch.add_argument(
        "--planner-memory-dir",
        type=Path,
        default=None,
        help="Explicit debug override; normally resolved from --planner-release-dir.",
    )
    dual_batch.add_argument(
        "--planner-decoder-artifact-dir",
        type=Path,
        default=None,
        help="Explicit debug override; normally resolved from --planner-release-dir.",
    )
    dual_batch.add_argument(
        "--planner-top-k",
        type=int,
        default=None,
        help="Explicit debug override; release default is top-K=4.",
    )
    dual_batch.add_argument("--planner-device", default="auto")
    dual_batch.add_argument("--planner-query-encoder-device", default="cpu")
    dual_batch.add_argument(
        "--planner-max-new-tokens",
        default="512",
        help="Planner generation budget. Use a positive integer or 'auto' to use remaining model context.",
    )
    dual_batch.add_argument(
        "--planner-output-mode",
        choices=["json"],
        default="json",
        help="Planner V2 accepts strict JSON only.",
    )
    dual_batch.add_argument("--max-planner-rounds", type=int, default=4)
    dual_batch.add_argument("--max-tool-rounds-per-step", type=int, default=12)
    _add_planner_routing_args(dual_batch)
    dual_batch.add_argument("--limit", type=int, default=0)
    dual_batch.add_argument("--force", action="store_true")
    dual_batch.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "build-case":
        CaseBuilder(args.case_dir, args.report_path, args.rubric_path).build()
        print(json.dumps({"case_dir": str(args.case_dir.resolve())}, ensure_ascii=False))
        return 0
    if args.command == "run":
        result = _run(args)
        print(json.dumps(_run_summary(result.run_dir), ensure_ascii=False, indent=2))
        return 0
    if args.command == "dual-agent-run":
        result = _dual_agent_run(args)
        print(json.dumps(_dual_run_summary(result.run_dir), ensure_ascii=False, indent=2))
        return 0
    if args.command == "dual-agent-run-and-judge":
        result = _dual_agent_run(args)
        rubric_path = _resolve_rubric_path(args.case_dir, args.rubric_path)
        scores = _judge(args, rubric_path=rubric_path, run_dir=result.run_dir).evaluate()
        from guideline_planner.routing_metrics import write_routing_evaluation

        write_routing_evaluation(result.run_dir)
        summary = _dual_run_summary(result.run_dir)
        summary["judge_status"] = scores["status"]
        summary["final_total"] = scores.get("final_total")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "judge":
        scores = _judge(args).evaluate()
        print(json.dumps({"judge_scores": str((args.run_dir / "judge_scores.json").resolve()), "status": scores["status"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-and-judge":
        result = _run(args)
        rubric_path = _resolve_rubric_path(args.case_dir, args.rubric_path)
        scores = _judge(args, rubric_path=rubric_path, run_dir=result.run_dir).evaluate()
        summary = _run_summary(result.run_dir)
        summary["judge_status"] = scores["status"]
        summary["final_total"] = scores.get("final_total")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "batch-run":
        from medclaw_benchmark.batch_runner import BatchRunConfig, run_batch

        summary = run_batch(
            BatchRunConfig(
                cases_root=args.cases_root,
                runs_root=args.runs_root,
                run_id=args.run_id,
                agent=args.agent,
                judge=args.judge,
                judge_provider=getattr(args, "judge_provider", None),
                evaluate=args.evaluate,
                limit=args.limit,
                force=args.force,
                dry_run=args.dry_run,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "dual-agent-batch-run":
        from medclaw_benchmark.dual_agent_runner import (
            DualAgentBatchConfig,
            run_dual_agent_batch,
        )

        summary = run_dual_agent_batch(
            DualAgentBatchConfig(
                cases_root=args.cases_root,
                runs_root=args.runs_root,
                run_id=args.run_id,
                agent=args.agent,
                judge=args.judge,
                judge_provider=getattr(args, "judge_provider", None),
                evaluate=args.evaluate,
                planner_release_dir=args.planner_release_dir,
                planner_mode=args.planner_mode,
                planner_ablation=args.planner_ablation,
                planner_ablation_seed=args.planner_ablation_seed,
                planner_memory_dir=args.planner_memory_dir,
                planner_decoder_artifact_dir=args.planner_decoder_artifact_dir,
                planner_top_k=args.planner_top_k,
                planner_device=args.planner_device,
                planner_query_encoder_device=args.planner_query_encoder_device,
                planner_max_new_tokens=args.planner_max_new_tokens,
                planner_output_mode=args.planner_output_mode,
                planner_routing_config=args.planner_routing_config,
                planner_routing_checkpoint=args.planner_routing_checkpoint,
                max_planner_rounds=args.max_planner_rounds,
                max_tool_rounds_per_step=args.max_tool_rounds_per_step,
                limit=args.limit,
                force=args.force,
                dry_run=args.dry_run,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    parser.error(f"Unsupported command {args.command!r}")
    return 2


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument(
        "--agent",
        choices=list(SUPPORTED_PROVIDERS),
        default=resolve_provider(None),
    )
    parser.add_argument("--run-id", default="run_001")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))


def _add_dual_agent_run_args(parser: argparse.ArgumentParser) -> None:
    _add_run_args(parser)
    parser.set_defaults(run_id="dual_agent_001")
    parser.add_argument(
        "--planner-release-dir",
        type=Path,
        default=None,
        help="Hash-bound planner_release.v2 directory used for formal runs.",
    )
    parser.add_argument(
        "--planner-mode",
        choices=["latent_topk", "daa_full"],
        default=None,
        help="Override the release default mode for an explicit ablation run.",
    )
    _add_planner_ablation_args(parser)
    parser.add_argument(
        "--planner-memory-dir",
        type=Path,
        default=None,
        help="Explicit debug override; normally resolved from --planner-release-dir.",
    )
    parser.add_argument(
        "--planner-decoder-artifact-dir",
        type=Path,
        default=None,
        help="Explicit debug override; normally resolved from --planner-release-dir.",
    )
    parser.add_argument(
        "--planner-top-k",
        type=int,
        default=None,
        help="Explicit debug override; release default is top-K=4.",
    )
    parser.add_argument("--planner-device", default="auto")
    parser.add_argument("--planner-query-encoder-device", default="cpu")
    parser.add_argument(
        "--planner-max-new-tokens",
        default="512",
        help="Planner generation budget. Use a positive integer or 'auto' to use remaining model context.",
    )
    parser.add_argument(
        "--planner-output-mode",
        choices=["json"],
        default="json",
        help="Planner V2 accepts strict JSON only.",
    )
    parser.add_argument("--max-planner-rounds", type=int, default=4)
    parser.add_argument("--max-tool-rounds-per-step", type=int, default=12)
    _add_planner_routing_args(parser)


def _add_planner_routing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--planner-routing-config",
        type=Path,
        default=None,
        help="YAML/JSON config enabling patient-state-conditioned latent routing.",
    )
    parser.add_argument(
        "--planner-routing-checkpoint",
        type=Path,
        default=None,
        help="Trained routing checkpoint bound to the selected memory store.",
    )


def _add_planner_ablation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--planner-ablation",
        choices=list(PLANNER_ABLATION_MODES),
        default="none",
        help="Inference-only Planner V2 memory/routing/fusion ablation.",
    )
    parser.add_argument(
        "--planner-ablation-seed",
        type=int,
        default=17,
        help="Deterministic seed used by random_memory_global.",
    )


def _add_judge_provider_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--judge-provider",
        choices=list(SUPPORTED_PROVIDERS),
        default=None,
        help=(
            "LLM provider used for rubric judging; defaults to "
            f"MEDCLAW_JUDGE_PROVIDER or {DEFAULT_PROVIDER}."
        ),
    )


def _run(args: argparse.Namespace):
    return BenchmarkRunner(
        case_dir=args.case_dir,
        run_id=args.run_id,
        runs_root=args.runs_root,
        agent=args.agent,
    ).run()


def _dual_agent_run(args: argparse.Namespace):
    kwargs = {
        "case_dir": args.case_dir,
        "run_id": args.run_id,
        "runs_root": args.runs_root,
        "agent": args.agent,
        "planner_device": args.planner_device,
        "planner_query_encoder_device": args.planner_query_encoder_device,
        "planner_max_new_tokens": args.planner_max_new_tokens,
        "planner_output_mode": args.planner_output_mode,
        "max_planner_rounds": args.max_planner_rounds,
        "max_tool_rounds_per_step": args.max_tool_rounds_per_step,
    }
    if args.planner_ablation != "none":
        kwargs["planner_ablation"] = args.planner_ablation
        kwargs["planner_ablation_seed"] = args.planner_ablation_seed
    if args.planner_release_dir is not None:
        kwargs["planner_release_dir"] = args.planner_release_dir
        if args.planner_mode is not None:
            kwargs["planner_mode"] = args.planner_mode
    else:
        # Preserve the legacy direct-artifact CLI when no formal release was
        # requested. These are defaults, not hidden overrides of a release.
        kwargs["planner_memory_dir"] = (
            args.planner_memory_dir or LEGACY_PLANNER_MEMORY_DIR
        )
        kwargs["planner_decoder_artifact_dir"] = (
            args.planner_decoder_artifact_dir or LEGACY_PLANNER_DECODER_DIR
        )
        kwargs["planner_top_k"] = args.planner_top_k or 3
    if args.planner_memory_dir is not None and args.planner_release_dir is not None:
        kwargs["planner_memory_dir"] = args.planner_memory_dir
    if (
        args.planner_decoder_artifact_dir is not None
        and args.planner_release_dir is not None
    ):
        kwargs["planner_decoder_artifact_dir"] = args.planner_decoder_artifact_dir
    if args.planner_top_k is not None and args.planner_release_dir is not None:
        kwargs["planner_top_k"] = args.planner_top_k
    if args.planner_routing_config is not None:
        kwargs["planner_routing_config"] = args.planner_routing_config
    if args.planner_routing_checkpoint is not None:
        kwargs["planner_routing_checkpoint"] = args.planner_routing_checkpoint
    return DualAgentBenchmarkRunner(**kwargs).run()


def _judge(
    args: argparse.Namespace,
    *,
    rubric_path: Path | None = None,
    run_dir: Path | None = None,
) -> LLMRubricJudge:
    return LLMRubricJudge(
        rubric_path or args.rubric_path,
        run_dir or args.run_dir,
        provider=getattr(args, "judge_provider", None),
    )


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


def _dual_run_summary(run_dir: Path) -> dict[str, str]:
    summary = _run_summary(run_dir)
    summary.update(
        {
            "planner_outputs": str((run_dir / "planner_outputs.jsonl").resolve()),
            "patient_state_history": str(
                (run_dir / "patient_state_history.jsonl").resolve()
            ),
            "dual_agent_rounds": str((run_dir / "dual_agent_rounds.jsonl").resolve()),
            "memory_routing": str((run_dir / "memory_routing.json").resolve()),
            "transition_trace": str((run_dir / "transition_trace.json").resolve()),
            "routing_metrics": str((run_dir / "routing_metrics.json").resolve()),
        }
    )
    return summary


def _resolve_rubric_path(case_dir: Path, rubric_path: Path | None) -> Path:
    return find_rubric_path(case_dir, rubric_path)


if __name__ == "__main__":
    raise SystemExit(main())
