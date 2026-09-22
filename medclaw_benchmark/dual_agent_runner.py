"""Dual-agent benchmark runner: latent planner plus MedClaw API agent."""

from __future__ import annotations

import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from guideline_planner.planner import LatentGuidelinePlanner
from guideline_planner.release import ResolvedPlannerRelease, resolve_planner_release
from guideline_planner.routing_metrics import write_routing_evaluation
from guideline_planner.routing_types import PlannerStepResult
from medclaw.core.agent_loop import AgentLoop, AgentLoopError
from medclaw.runtime.runtime_manager import RuntimeManager
from medclaw.utils import utc_now, write_json

from medclaw_benchmark.batch_runner import (
    BatchRunConfig,
    _resolve_rubric_path,
    _run_summary,
    discover_ready_cases,
)
from medclaw_benchmark.case_builder import CaseBuilder
from medclaw_benchmark.case_simulator import CaseSimulator
from medclaw_benchmark.io_utils import append_jsonl, safe_id
from medclaw_benchmark.llm_judge import LLMRubricJudge
from medclaw_benchmark.patient_state import (
    current_patient_state_view,
    initialize_patient_state,
    summarize_state_delta,
    update_patient_state,
)
from medclaw_benchmark.planner_skill_adapter import planner_skill_execution_map
from medclaw_benchmark.planner_preflight import preflight_planner_case
from medclaw_benchmark.runner import (
    BENCHMARK_AGENT_SYSTEM_PROMPT,
    BenchmarkRunner,
    BenchmarkDynamicToolRouter,
    BenchmarkRunResult,
    _AgentEvidenceSink,
    _RunConversationLog,
    _json_text,
    _phase_for_skill,
    _public_llm_config,
    _summary,
)


DUAL_AGENT_SYSTEM_PROMPT = (
    BENCHMARK_AGENT_SYSTEM_PROMPT
    + "\n\nYou are working inside a dual-agent loop. A separate latent "
    "guideline-memory planner provides non-binding next-step guidance. Treat "
    "that guidance as a planning prior, not as evidence. You may still choose "
    "the tools that are necessary for the case. Never expose or infer any hidden "
    "rubric."
)

LEGACY_PLANNER_MEMORY_DIR = Path("guideline_planner/outputs/memory_store")
LEGACY_PLANNER_DECODER_DIR = Path("guideline_planner/outputs/planner_decoder")


@dataclass(frozen=True)
class DualAgentBatchConfig(BatchRunConfig):
    """Configuration for resumable dual-agent batch runs."""

    planner_release_dir: Path | None = None
    planner_mode: str | None = None
    planner_ablation: str = "none"
    planner_ablation_seed: int = 17
    planner_memory_dir: Path | None = None
    planner_decoder_artifact_dir: Path | None = None
    planner_top_k: int | None = None
    planner_device: str = "auto"
    planner_query_encoder_device: str = "cpu"
    planner_max_new_tokens: int | str = 512
    planner_output_mode: str = "json"
    planner_routing_config: Path | None = None
    planner_routing_checkpoint: Path | None = None
    max_planner_rounds: int = 4
    max_tool_rounds_per_step: int = 12


class DualAgentBenchmarkRunner(BenchmarkRunner):
    """Run a planner-guided, model-executed MedClaw benchmark trajectory."""

    def __init__(
        self,
        *,
        case_dir: str | Path,
        run_id: str = "dual_agent_001",
        runs_root: str | Path = "runs",
        project_root: str | Path | None = None,
        agent: str | None = None,
        planner_release_dir: str | Path | None = None,
        planner_mode: str | None = None,
        planner_ablation: str = "none",
        planner_ablation_seed: int = 17,
        planner_memory_dir: str | Path | None = None,
        planner_decoder_artifact_dir: str | Path | None = None,
        planner_top_k: int | None = None,
        planner_device: str = "auto",
        planner_query_encoder_device: str = "cpu",
        planner_max_new_tokens: int | str = 512,
        planner_output_mode: str = "json",
        planner_routing_config: str | Path | Mapping[str, Any] | None = None,
        planner_routing_checkpoint: str | Path | None = None,
        max_planner_rounds: int = 4,
        max_tool_rounds_per_step: int = 12,
        planner: Any | None = None,
        planner_release: ResolvedPlannerRelease | None = None,
        runtime_manager: RuntimeManager | Any | None = None,
        artifact_resolver: Any | None = None,
        llm_client: Any | None = None,
        llm_config: Any | None = None,
        qwen_config: Any | None = None,
    ) -> None:
        super().__init__(
            case_dir=case_dir,
            run_id=run_id,
            runs_root=runs_root,
            project_root=project_root,
            agent=agent,
            runtime_manager=runtime_manager,
            artifact_resolver=artifact_resolver,
            llm_client=llm_client,
            llm_config=llm_config,
            qwen_config=qwen_config,
        )
        self.planner_release: ResolvedPlannerRelease | None = planner_release
        self.planner_release_dir = (
            Path(planner_release_dir) if planner_release_dir is not None else None
        )
        self.planner_mode = planner_mode
        self.planner_ablation = str(planner_ablation)
        self.planner_ablation_seed = int(planner_ablation_seed)
        if self.planner_release is not None:
            self.planner_memory_dir = self.planner_release.memory_dir
            self.planner_decoder_artifact_dir = self.planner_release.decoder_artifact_dir
            self.planner_top_k = self.planner_release.top_k
            self.planner_routing_config = self.planner_release.routing_config
            self.planner_routing_checkpoint = self.planner_release.routing_checkpoint
        elif self.planner_release_dir is not None:
            debug_overrides = {
                "memory_dir": planner_memory_dir,
                "decoder_artifact_dir": planner_decoder_artifact_dir,
                "routing_config": planner_routing_config,
                "routing_checkpoint": planner_routing_checkpoint,
                "top_k": planner_top_k,
            }
            self.planner_release = resolve_planner_release(
                self.planner_release_dir,
                mode=planner_mode,
                overrides={
                    key: value
                    for key, value in debug_overrides.items()
                    if value is not None
                },
            )
            self.planner_memory_dir = self.planner_release.memory_dir
            self.planner_decoder_artifact_dir = (
                self.planner_release.decoder_artifact_dir
            )
            self.planner_top_k = self.planner_release.top_k
            self.planner_routing_config = self.planner_release.routing_config
            self.planner_routing_checkpoint = (
                self.planner_release.routing_checkpoint
            )
        else:
            self.planner_memory_dir = Path(
                planner_memory_dir or LEGACY_PLANNER_MEMORY_DIR
            )
            self.planner_decoder_artifact_dir = Path(
                planner_decoder_artifact_dir or LEGACY_PLANNER_DECODER_DIR
            )
            self.planner_top_k = int(planner_top_k or 3)
            self.planner_routing_config = planner_routing_config
            self.planner_routing_checkpoint = (
                Path(planner_routing_checkpoint)
                if planner_routing_checkpoint is not None
                else None
            )
        self.planner_device = planner_device
        self.planner_query_encoder_device = planner_query_encoder_device
        self.planner_max_new_tokens = planner_max_new_tokens
        self.planner_output_mode = planner_output_mode
        self.max_planner_rounds = int(max_planner_rounds)
        self.max_tool_rounds_per_step = int(max_tool_rounds_per_step)
        self.planner = planner

    def run(self) -> BenchmarkRunResult:
        """Run planner-agent loop and write all audit artifacts."""

        preflight_record = None
        if self.planner_release is not None:
            preflight_record = preflight_planner_case(
                self.case_dir,
                self.planner_release,
            )
        CaseBuilder(
            self.case_dir,
            llm_client=self.llm_client,
            llm_config=self.llm_config,
            provider=self.agent,
            qwen_config=self.llm_config,
        ).build()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if preflight_record is not None:
            write_json(self.run_dir / "planner_preflight.json", preflight_record)
        (self.run_dir / "artifacts" / "ct_roi").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "artifacts" / "wsi_roi").mkdir(parents=True, exist_ok=True)
        guideline_trajectory_path = self._copy_guideline_trajectory()

        simulator = CaseSimulator(self.case_dir)
        initial = simulator.get_initial_observation()
        self._record_trajectory(
            "observation",
            {
                "phase": simulator.current_phase(),
                "content": initial,
            },
        )

        runtime = self.runtime_manager or self._build_runtime()
        if self.artifact_resolver is None and hasattr(runtime, "artifact_store"):
            self.artifact_resolver = runtime.artifact_store.resolve_uri

        planner = self.planner or LatentGuidelinePlanner(
            memory_dir=self.planner_memory_dir,
            decoder_artifact_dir=self.planner_decoder_artifact_dir,
            top_k=self.planner_top_k,
            device=self.planner_device,
            query_encoder_device=self.planner_query_encoder_device,
            max_new_tokens=self.planner_max_new_tokens,
            output_mode=self.planner_output_mode,
            routing_config=self.planner_routing_config,
            routing_checkpoint=self.planner_routing_checkpoint,
            release=self.planner_release,
            planner_ablation=self.planner_ablation,
            planner_ablation_seed=self.planner_ablation_seed,
        )
        planner_context = _planner_public_context(planner)
        patient_state = initialize_patient_state(
            self.case_dir,
            initial,
            planner_release=self.planner_release,
        )
        router = BenchmarkDynamicToolRouter(
            simulator=simulator,
            runtime=runtime,
            case_id=self.case_id,
            guideline_context=patient_state.get("guideline_context"),
        )
        agent_loop = AgentLoop(
            router,
            _AgentEvidenceSink(self.case_id, self.run_dir),
            llm_client=self._build_llm_client(),
            system_prompt=DUAL_AGENT_SYSTEM_PROMPT.format(case_id=self.case_id),
            max_tool_rounds=max(self.max_tool_rounds_per_step, 1),
            artifact_uri_resolver=self.artifact_resolver,
            conversation_log=_RunConversationLog(self.run_dir / "conversation_log.jsonl"),
        )
        _write_state_event(
            self.run_dir,
            self.case_id,
            self.run_id,
            {
                "event_type": "patient_state_initialized",
                "round_index": 0,
                "patient_state": patient_state,
            },
        )

        collected: list[dict[str, Any]] = []
        trajectory_plan: list[dict[str, Any]] = []
        routing_events: list[dict[str, Any]] = []
        for round_index in range(1, max(self.max_planner_rounds, 1) + 1):
            state_before = dict(patient_state)
            planner_state = current_patient_state_view(patient_state)
            planner_step_result: PlannerStepResult | None = None
            try:
                if hasattr(planner, "plan"):
                    candidate_result = planner.plan(
                        planner_state,
                        trajectory_history=[],
                    )
                    if isinstance(candidate_result, PlannerStepResult):
                        planner_step_result = candidate_result
                        planner_output = candidate_result.action
                    elif isinstance(candidate_result, Mapping):
                        planner_output = dict(candidate_result)
                    else:
                        raise TypeError(
                            "Planner.plan() must return PlannerStepResult or a mapping."
                        )
                else:
                    planner_output = planner.next_step(planner_state, [])
            except Exception as exc:
                attempt = _planner_last_attempt(planner)
                planner_failure_record = {
                    "round_index": round_index,
                    "case_id": self.case_id,
                    "run_id": self.run_id,
                    "timestamp": utc_now(),
                    "status": "failed",
                    "patient_state_before": state_before,
                    "planner_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "planner_attempt": attempt,
                }
                append_jsonl(self.run_dir / "planner_outputs.jsonl", planner_failure_record)
                _write_planner_token_stats(self.run_dir)
                write_json(
                    self.run_dir / f"planner_round_{round_index}_failure_debug.json",
                    planner_failure_record,
                )
                self._record_trajectory("planner_error", planner_failure_record)
                raise
            planner_record = {
                "round_index": round_index,
                "case_id": self.case_id,
                "run_id": self.run_id,
                "timestamp": utc_now(),
                "patient_state_before": state_before,
                "planner_output": planner_output,
                "planner_attempt": _planner_last_attempt(planner),
                "planner_step_result": (
                    planner_step_result.to_dict() if planner_step_result else None
                ),
            }
            if planner_step_result is not None:
                planner_record.update(_planner_step_audit_fields(planner_step_result))
            append_jsonl(self.run_dir / "planner_outputs.jsonl", planner_record)
            _write_planner_token_stats(self.run_dir)
            self._record_trajectory("planner_output", planner_record)
            if planner_step_result is not None:
                routing_event = _routing_event(
                    self.case_id,
                    self.run_id,
                    round_index,
                    planner_step_result,
                )
                routing_events.append(routing_event)
                tensor_path = _save_routing_tensors(
                    planner,
                    self.run_dir,
                    round_index,
                )
                if tensor_path is not None:
                    routing_event["tensor_path"] = str(tensor_path)
                write_json(
                    self.run_dir / "memory_routing.json",
                    {
                        "schema_version": "memory_routing.v1",
                        "case_id": self.case_id,
                        "run_id": self.run_id,
                        "steps": routing_events,
                    },
                )

            turn_error: dict[str, Any] | None = None
            try:
                turn = agent_loop.chat(
                    _planner_guided_prompt(
                        current_patient_state_view(patient_state),
                        planner_output,
                        round_index,
                    ),
                    retained_user_text=_round_completion_marker(round_index),
                )
            except AgentLoopError as exc:
                turn = _partial_turn_from_conversation_log(
                    agent_loop.conversation_log_path,
                    error=exc,
                )
                tool_debug = _tool_debug_from_tool_results(turn.tool_results)
                turn_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "tool_debug": tool_debug,
                }
                write_json(
                    self.run_dir / f"agent_round_{round_index}_failure_debug.json",
                    {
                        "case_id": self.case_id,
                        "run_id": self.run_id,
                        "round_index": round_index,
                        "error": turn_error,
                        "planner_output": planner_output,
                    },
                )
                self._record_trajectory(
                    "agent_round_error",
                    {
                        "round_index": round_index,
                        "phase": planner_output.get("current_phase"),
                        "error": turn_error,
                        "recovered_tool_result_count": len(turn.tool_results),
                    },
                )
            if (
                turn_error is None
                and bool(getattr(turn, "tool_round_limit_reached", False))
            ):
                turn_error = {
                    "type": "AgentLoopError",
                    "message": (
                        "Model reached the configured tool-calling round limit; "
                        "the partial turn was finalized for audit."
                    ),
                    "tool_debug": _tool_debug_from_tool_results(turn.tool_results),
                }
                write_json(
                    self.run_dir / f"agent_round_{round_index}_failure_debug.json",
                    {
                        "case_id": self.case_id,
                        "run_id": self.run_id,
                        "round_index": round_index,
                        "error": turn_error,
                        "planner_output": planner_output,
                    },
                )
                self._record_trajectory(
                    "agent_round_error",
                    {
                        "round_index": round_index,
                        "phase": planner_output.get("current_phase"),
                        "error": turn_error,
                        "recovered_tool_result_count": len(turn.tool_results),
                    },
                )
            round_records = self._collect_turn_tool_records(
                turn.tool_results,
                round_index=round_index,
            )
            collected.extend(round_records)
            patient_state = update_patient_state(
                patient_state,
                round_records,
                planner_output,
                current_time=round_index,
            )
            delta = summarize_state_delta(state_before, patient_state)
            if routing_events and routing_events[-1].get("round_index") == round_index:
                routing_events[-1].update(
                    {
                        "state_progress": bool(delta),
                        "state_delta_fields": sorted(delta),
                        "tool_skills": [
                            record.get("skill_name") for record in round_records
                        ],
                    }
                )
                write_json(
                    self.run_dir / "memory_routing.json",
                    {
                        "schema_version": "memory_routing.v1",
                        "case_id": self.case_id,
                        "run_id": self.run_id,
                        "steps": routing_events,
                    },
                )
            round_summary = {
                "round_index": round_index,
                "case_id": self.case_id,
                "run_id": self.run_id,
                "timestamp": utc_now(),
                "planner_output": planner_output,
                "tool_call_count": len(round_records),
                "tool_skills": [record.get("skill_name") for record in round_records],
                "agent_round_answer": turn.content,
                "agent_round_error": turn_error,
                "tool_round_limit_reached": bool(
                    getattr(turn, "tool_round_limit_reached", False)
                ),
                "forced_finalization": bool(
                    getattr(turn, "forced_finalization", False)
                ),
                "patient_state_delta": delta,
            }
            append_jsonl(self.run_dir / "dual_agent_rounds.jsonl", round_summary)
            self._record_trajectory("dual_agent_round", round_summary)
            _write_state_event(
                self.run_dir,
                self.case_id,
                self.run_id,
                {
                    "event_type": "patient_state_updated",
                    "round_index": round_index,
                    "patient_state_before": state_before,
                    "patient_state_after": patient_state,
                    "delta": delta,
                },
            )
            trajectory_plan.append(
                {
                    "round_index": round_index,
                    "planner_output": planner_output,
                    "tool_skills": [record.get("skill_name") for record in round_records],
                    "agent_summary": turn.content,
                    "agent_error": turn_error,
                    "patient_state_delta": delta,
                }
            )
            if (
                turn_error is not None
                or not round_records
                or bool(getattr(turn, "tool_round_limit_reached", False))
            ):
                break

        planner_token_stats = _write_planner_token_stats(self.run_dir)
        self._write_evidence_board()
        final_text, final_meta = self._finalize_with_agent_loop(
            agent_loop,
            patient_state,
            collected,
        )
        last_step = getattr(planner, "last_step_result", None)
        last_step_diagnostics = (
            last_step.diagnostics
            if isinstance(last_step, PlannerStepResult)
            else {}
        )
        final_ablation_audit = _ablation_audit_fields(last_step_diagnostics)
        configured_ablation = planner_context.get("ablation") or {}
        for key in ("ablation_id", "planner_ablation", "ablation_seed", "decoder_role"):
            if final_ablation_audit.get(key) is None:
                final_ablation_audit[key] = configured_ablation.get(key)
        final_payload = {
            "case_id": self.case_id,
            "run_id": self.run_id,
            "agent": self.agent,
            "answer_text": final_text,
            "generated_at": utc_now(),
            "provider": self.agent,
            "model": final_meta.get("model"),
            "usage": final_meta.get("usage"),
            "conversation_log_path": str(self.run_dir / "conversation_log.jsonl"),
            "agent_context_path": str(self.run_dir / "agent_context.json"),
            "guideline_trajectory_path": str(guideline_trajectory_path),
            "tool_round_count": len(collected),
            "planner_round_count": len(trajectory_plan),
            "planning_mode": _dual_planning_mode(planner_context),
            "planner_context": planner_context,
            "patient_state_path": str(self.run_dir / "patient_state_history.jsonl"),
            "planner_outputs_path": str(self.run_dir / "planner_outputs.jsonl"),
            "planner_token_stats_path": str(self.run_dir / "planner_token_stats.json"),
            "planner_token_stats": planner_token_stats,
            **final_ablation_audit,
        }
        final_answer_path = self.run_dir / "final_answers.json"
        write_json(final_answer_path, final_payload)
        self._record_trajectory("final_answer", final_payload)
        evidence_board_path = self._write_evidence_board()
        write_routing_evaluation(self.run_dir)

        return BenchmarkRunResult(
            case_id=self.case_id,
            run_id=self.run_id,
            run_dir=self.run_dir,
            final_answer_path=final_answer_path,
            evidence_board_path=evidence_board_path,
            trajectory_path=self.run_dir / "trajectory.jsonl",
            tool_calls_path=self.run_dir / "tool_calls.jsonl",
            guideline_trajectory_path=guideline_trajectory_path,
        )

    def _collect_turn_tool_records(
        self,
        tool_results: tuple[dict[str, Any], ...],
        *,
        round_index: int,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for tool_record in tool_results:
            result = tool_record.get("result", {})
            if not isinstance(result, Mapping):
                result = {
                    "status": "failed",
                    "findings": {"error": "Tool returned a non-object result."},
                    "artifacts": [],
                    "warnings": ["Tool returned a non-object result."],
                }
            skill_name = str(
                tool_record.get("skill_name")
                or tool_record.get("function_name")
                or "unknown"
            )
            arguments = tool_record.get("arguments", {})
            if not isinstance(arguments, Mapping):
                arguments = {}
            phase = str(result.get("phase") or _phase_for_skill(skill_name))
            record = self._record_tool_call(phase, skill_name, arguments, result)
            record["planner_round_index"] = round_index
            records.append(record)
            self._record_trajectory(
                "tool_result",
                {
                    "phase": phase,
                    "skill_name": skill_name,
                    "status": result.get("status"),
                    "summary": _summary(result),
                    "planner_round_index": round_index,
                },
            )
        return records

    def _finalize_with_agent_loop(
        self,
        agent_loop: AgentLoop,
        patient_state: Mapping[str, Any],
        tool_records: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        state_view = current_patient_state_view(patient_state)
        final_prompt = _final_answer_prompt(state_view)
        turn = agent_loop.finalize(final_prompt)
        client_config = getattr(agent_loop.llm_client, "config", self.llm_config)
        context = {
            "case_id": self.case_id,
            "run_id": self.run_id,
            "agent": self.agent,
            "provider": self.agent,
            "final_synthesis_mode": "same_agent_loop_no_tools",
            "history_compaction_strategy": (
                "replace_each_planner_round_user_prompt_with_short_marker_after_audit"
            ),
            "current_patient_state": state_view,
            "conversation_log_path": str(agent_loop.conversation_log_path),
            "message_count": agent_loop.message_count,
            "tool_record_count": len(tool_records),
            "retained_evidence": {
                "structured_tool_results": True,
                "guideline_snippets_and_pages": True,
                "assistant_round_summaries": True,
                "image_messages": True,
                "full_planner_round_prompts": False,
                "latest_patient_state_in_final_prompt": True,
            },
        }
        write_json(self.run_dir / "agent_context.json", context)
        write_json(
            self.run_dir / "agent_raw_response.json",
            {
                "message": turn.raw_message
                or {"role": "assistant", "content": turn.content},
                "model": turn.model,
                "usage": turn.usage,
                "config": _public_llm_config(client_config),
                "turn_id": turn.turn_id,
                "final_synthesis_mode": "same_agent_loop_no_tools",
            },
        )
        return turn.content, {"model": turn.model, "usage": turn.usage}


def run_dual_agent_batch(config: DualAgentBatchConfig) -> dict[str, Any]:
    """Run all ready cases with dual-agent runner, skipping completed runs."""

    cases = discover_ready_cases(config.cases_root)
    if config.limit > 0:
        cases = cases[: config.limit]

    resolved_release: ResolvedPlannerRelease | None = None
    shared_planner: Any | None = None
    preflight_records: list[dict[str, Any]] = []
    runnable_cases = [
        case_dir
        for case_dir in cases
        if config.force
        or not is_dual_agent_inference_complete(
            case_dir,
            config.runs_root,
            config.run_id,
        )
    ]
    if config.planner_release_dir is not None:
        debug_overrides = {
            "memory_dir": config.planner_memory_dir,
            "decoder_artifact_dir": config.planner_decoder_artifact_dir,
            "routing_config": config.planner_routing_config,
            "routing_checkpoint": config.planner_routing_checkpoint,
            "top_k": config.planner_top_k,
        }
        resolved_release = resolve_planner_release(
            config.planner_release_dir,
            mode=config.planner_mode,
            overrides={
                key: value for key, value in debug_overrides.items() if value is not None
            },
        )
        for case_dir in runnable_cases:
            record = preflight_planner_case(case_dir, resolved_release)
            preflight_records.append(record)
            print(
                f"[preflight] {case_dir.name}: {record['cancer_family']}/"
                f"{record['disease_subtype']} -> {record['guideline_id']}@"
                f"{record['version']} [ready]",
                file=sys.stderr,
                flush=True,
            )
        if runnable_cases and not config.dry_run:
            print(
                "[planner] Loading the hash-bound Planner release once for this batch...",
                file=sys.stderr,
                flush=True,
            )
            shared_planner = LatentGuidelinePlanner(
                memory_dir=resolved_release.memory_dir,
                decoder_artifact_dir=resolved_release.decoder_artifact_dir,
                top_k=resolved_release.top_k,
                device=config.planner_device,
                query_encoder_device=config.planner_query_encoder_device,
                max_new_tokens=config.planner_max_new_tokens,
                output_mode=config.planner_output_mode,
                routing_config=resolved_release.routing_config,
                routing_checkpoint=resolved_release.routing_checkpoint,
                release=resolved_release,
                planner_ablation=config.planner_ablation,
                planner_ablation_seed=config.planner_ablation_seed,
            )
            print("[planner] Planner release loaded.", file=sys.stderr, flush=True)

    started_at = utc_now()
    records: list[dict[str, Any]] = []
    counts = {"total": len(cases), "completed": 0, "skipped": 0, "failed": 0}
    for case_index, case_dir in enumerate(cases, 1):
        print(
            f"[batch {case_index}/{len(cases)}] {case_dir.name}",
            file=sys.stderr,
            flush=True,
        )
        record = _run_one_dual_case(
            case_dir,
            config,
            planner=shared_planner,
            planner_release=resolved_release,
        )
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
            "planner_release_dir": (
                str(config.planner_release_dir)
                if config.planner_release_dir is not None
                else None
            ),
            "planner_mode": config.planner_mode,
            "planner_ablation": config.planner_ablation,
            "planner_ablation_seed": config.planner_ablation_seed,
            "planner_memory_dir": (
                str(config.planner_memory_dir)
                if config.planner_memory_dir is not None
                else None
            ),
            "planner_decoder_artifact_dir": (
                str(config.planner_decoder_artifact_dir)
                if config.planner_decoder_artifact_dir is not None
                else None
            ),
            "planner_top_k": config.planner_top_k,
            "planner_device": config.planner_device,
            "planner_query_encoder_device": config.planner_query_encoder_device,
            "planner_max_new_tokens": config.planner_max_new_tokens,
            "planner_output_mode": config.planner_output_mode,
            "planner_routing_config": (
                str(config.planner_routing_config)
                if config.planner_routing_config is not None
                else None
            ),
            "planner_routing_checkpoint": (
                str(config.planner_routing_checkpoint)
                if config.planner_routing_checkpoint is not None
                else None
            ),
            "max_planner_rounds": config.max_planner_rounds,
            "max_tool_rounds_per_step": config.max_tool_rounds_per_step,
            "limit": config.limit,
            "force": config.force,
            "dry_run": config.dry_run,
        },
        "counts": counts,
        "cases": records,
        "preflight": preflight_records,
        "planner_token_stats_path": str(
            (
                config.runs_root
                / f"dual_agent_planner_token_stats_{config.run_id}.json"
            ).resolve()
        ),
    }
    config.runs_root.mkdir(parents=True, exist_ok=True)
    planner_token_stats = _aggregate_batch_planner_token_stats(records)
    summary["planner_token_stats"] = planner_token_stats
    write_json(
        config.runs_root / f"dual_agent_planner_token_stats_{config.run_id}.json",
        planner_token_stats,
    )
    write_json(config.runs_root / f"dual_agent_batch_summary_{config.run_id}.json", summary)
    return summary


def is_dual_agent_inference_complete(
    case_dir: Path,
    runs_root: Path,
    run_id: str,
) -> bool:
    run_dir = Path(runs_root).resolve() / case_dir.name / safe_id(run_id, "dual_agent_001")
    return all(
        (run_dir / filename).is_file()
        for filename in (
            "final_answers.json",
            "planner_outputs.jsonl",
            "patient_state_history.jsonl",
        )
    )


def is_dual_agent_evaluation_complete(
    case_dir: Path,
    runs_root: Path,
    run_id: str,
) -> bool:
    run_dir = Path(runs_root).resolve() / case_dir.name / safe_id(run_id, "dual_agent_001")
    return (run_dir / "judge_scores.json").is_file()


def is_dual_agent_run_complete(
    case_dir: Path,
    runs_root: Path,
    run_id: str,
    *,
    evaluate: bool = False,
) -> bool:
    return is_dual_agent_inference_complete(case_dir, runs_root, run_id) and (
        not evaluate
        or is_dual_agent_evaluation_complete(case_dir, runs_root, run_id)
    )


def _run_one_dual_case(
    case_dir: Path,
    config: DualAgentBatchConfig,
    *,
    planner: Any | None = None,
    planner_release: ResolvedPlannerRelease | None = None,
) -> dict[str, Any]:
    run_dir = Path(config.runs_root).resolve() / case_dir.name / safe_id(config.run_id, "dual_agent_001")
    batch_ablation = _batch_ablation_fields(config, run_dir=run_dir)
    inference_complete = is_dual_agent_inference_complete(
        case_dir, config.runs_root, config.run_id
    )
    evaluation_complete = is_dual_agent_evaluation_complete(
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
                "existing dual-agent inference and evaluation outputs"
                if config.evaluate
                else "existing dual-agent inference outputs"
            ),
            "evaluation_enabled": config.evaluate,
            "execution_mode": "skipped",
            **batch_ablation,
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
            **batch_ablation,
        }
    evaluation_only = not config.force and inference_complete
    try:
        with _temporary_cases_root(config.cases_root):
            if evaluation_only:
                scores = LLMRubricJudge(
                    _resolve_rubric_path(case_dir, None),
                    run_dir,
                    provider=config.judge_provider,
                ).evaluate()
                write_routing_evaluation(run_dir)
                summary = _dual_run_summary(run_dir)
                return {
                    "case_id": case_dir.name,
                    "case_dir": str(case_dir.resolve()),
                    "run_dir": str(run_dir),
                    "status": "completed",
                    "evaluation_enabled": True,
                    "execution_mode": "evaluation_only",
                    "judge_status": scores.get("status"),
                    "final_total": scores.get("final_total"),
                    "outputs": summary,
                    **_batch_ablation_fields(config, run_dir=run_dir),
                }
            runner_kwargs: dict[str, Any] = {
                "case_dir": case_dir,
                "run_id": config.run_id,
                "runs_root": config.runs_root,
                "agent": config.agent,
                "planner_memory_dir": (
                    config.planner_memory_dir
                    if config.planner_memory_dir is not None
                    else (
                        None
                        if config.planner_release_dir is not None
                        else LEGACY_PLANNER_MEMORY_DIR
                    )
                ),
                "planner_decoder_artifact_dir": (
                    config.planner_decoder_artifact_dir
                    if config.planner_decoder_artifact_dir is not None
                    else (
                        None
                        if config.planner_release_dir is not None
                        else LEGACY_PLANNER_DECODER_DIR
                    )
                ),
                "planner_top_k": (
                    config.planner_top_k
                    if config.planner_top_k is not None
                    else (None if config.planner_release_dir is not None else 3)
                ),
                "planner_device": config.planner_device,
                "planner_query_encoder_device": config.planner_query_encoder_device,
                "planner_max_new_tokens": config.planner_max_new_tokens,
                "planner_output_mode": config.planner_output_mode,
                "max_planner_rounds": config.max_planner_rounds,
                "max_tool_rounds_per_step": config.max_tool_rounds_per_step,
            }
            if planner is not None:
                runner_kwargs["planner"] = planner
            if planner_release is not None:
                runner_kwargs["planner_release"] = planner_release
            if config.planner_release_dir is not None:
                runner_kwargs["planner_release_dir"] = config.planner_release_dir
            if config.planner_mode is not None:
                runner_kwargs["planner_mode"] = config.planner_mode
            if config.planner_ablation != "none":
                runner_kwargs["planner_ablation"] = config.planner_ablation
                runner_kwargs["planner_ablation_seed"] = (
                    config.planner_ablation_seed
                )
            if config.planner_routing_config is not None:
                runner_kwargs["planner_routing_config"] = config.planner_routing_config
            if config.planner_routing_checkpoint is not None:
                runner_kwargs["planner_routing_checkpoint"] = (
                    config.planner_routing_checkpoint
                )
            result = DualAgentBenchmarkRunner(**runner_kwargs).run()
            scores: dict[str, Any] | None = None
            if config.evaluate:
                scores = LLMRubricJudge(
                    _resolve_rubric_path(case_dir, None),
                    result.run_dir,
                    provider=config.judge_provider,
                ).evaluate()
                write_routing_evaluation(result.run_dir)
        summary = _dual_run_summary(result.run_dir)
        record = {
            "case_id": case_dir.name,
            "case_dir": str(case_dir.resolve()),
            "run_dir": str(result.run_dir),
            "status": "completed",
            "evaluation_enabled": config.evaluate,
            "execution_mode": (
                "inference_and_evaluation"
                if config.evaluate
                else "inference_only"
            ),
            "outputs": summary,
            **_batch_ablation_fields(config, run_dir=result.run_dir),
        }
        if scores is not None:
            record["judge_status"] = scores.get("status")
            record["final_total"] = scores.get("final_total")
        return record
    except Exception as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        failure_debug = _failure_debug_from_run_dir(run_dir, exc)
        failure_debug_path = run_dir / "failure_debug.json"
        write_json(failure_debug_path, failure_debug)
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
            **batch_ablation,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "failure_debug_path": str(failure_debug_path),
            "failure_debug": failure_debug,
            "failed_at": utc_now(),
        }
        write_json(run_dir / "batch_error.json", error)
        return error


def _planner_guided_prompt(
    patient_state: Mapping[str, Any],
    planner_output: Mapping[str, Any],
    round_index: int,
) -> str:
    payload = {
        "round_index": round_index,
        "current_patient_state": patient_state,
        "planner_output": planner_output,
        "planner_skill_execution_map": planner_skill_execution_map(
            planner_output,
            patient_state,
        ),
        "instruction": (
            "Use the planner output as non-binding guidance. Call any MedClaw "
            "tools needed to fill missing evidence. Consider the ordered V2 actions "
            "and their required_skills when they are relevant and available, but you may "
            "revise the plan based on tool results. Do not give the final answer "
            "unless no more evidence is useful; this round is primarily for "
            "evidence acquisition and state refinement."
        ),
    }
    return "Dual-agent planner-guided evidence round:\n" + _json_text(payload)


def _round_completion_marker(round_index: int) -> str:
    return (
        f"Evidence acquisition round {round_index} completed. The detailed patient "
        "state and planner guidance for this round are retained in the audit log."
    )


def _final_answer_prompt(patient_state: Mapping[str, Any]) -> str:
    payload = {
        "current_patient_state": patient_state,
        "instructions": [
            (
                "Generate the final Chinese oncology benchmark answer now. Do not "
                "call any more tools."
            ),
            (
                "Use the complete structured role=tool results, guideline snippets, "
                "page metadata, ROI image feedback, and CancerClaw round summaries "
                "already retained in this conversation as the evidence context."
            ),
            (
                "Treat structured tool results as primary evidence. Planner guidance "
                "and earlier assistant interpretations are not patient facts."
            ),
            (
                "Cite the actual guideline title or source and page number returned "
                "by guideline tools. If a supporting snippet has no page number, "
                "state explicitly that no page number was provided."
            ),
            (
                "Clearly separate known clinical background, diagnosis and evidence, "
                "reviewed modalities, guideline-matched management, and uncertainty "
                "or missing information. Do not invent staging, biomarkers, pathology, "
                "treatment history, or guideline details."
            ),
        ],
    }
    return "Final answer request:\n" + _json_text(payload)


def _planner_public_context(planner: Any) -> dict[str, Any]:
    if hasattr(planner, "public_config"):
        value = planner.public_config()
        if isinstance(value, Mapping):
            return dict(value)
    return {"planning_mode": "latent_decoder", "planner": type(planner).__name__}


def _dual_planning_mode(planner_context: Mapping[str, Any]) -> str:
    if planner_context.get("planning_mode") == "latent_decoder_state_conditioned_routing":
        return "latent_decoder_state_conditioned_routing_guided_tool_choice_auto"
    return "latent_decoder_guided_tool_choice_auto"


def _routing_event(
    case_id: str,
    run_id: str,
    round_index: int,
    result: PlannerStepResult,
) -> dict[str, Any]:
    diagnostics = result.diagnostics
    gate = diagnostics.get("gate_diagnostics", {})
    return {
        "case_id": case_id,
        "run_id": run_id,
        "time": f"T{round_index}",
        "round_index": round_index,
        "timestamp": utc_now(),
        "seed_candidates": diagnostics.get("seed_candidates", []),
        "candidate_activations": diagnostics.get("candidate_activations", []),
        "active_memories": [item.to_dict() for item in result.active_memories],
        "gate_entropy": gate.get("entropy"),
        "selected_action": dict(result.action),
        "current_objective": result.current_objective,
        "expected_state_change": list(result.expected_state_change),
        "trajectory_quality": None,
        **_ablation_audit_fields(diagnostics),
        "routing_diagnostics": {
            "profile": diagnostics.get("diagnostics", {}).get("profile"),
            "decoder_prefix_shape": diagnostics.get("decoder_prefix_shape"),
            "fusion": diagnostics.get("fusion_diagnostics", {}),
            "checkpoint": diagnostics.get("diagnostics", {}).get("checkpoint", {}),
        },
    }


def _planner_step_audit_fields(result: PlannerStepResult) -> dict[str, Any]:
    diagnostics = result.diagnostics
    return {
        **_ablation_audit_fields(diagnostics),
        "seed_candidates": diagnostics.get("seed_candidates", []),
        "active_memories": [item.to_dict() for item in result.active_memories],
        "source_pages": [
            {
                "memory_id": item.memory_id,
                "section_title": item.section_title,
                "source_pages": item.source_pages,
            }
            for item in result.active_memories
        ],
        "routing_checkpoint": diagnostics.get("diagnostics", {}).get(
            "checkpoint",
            {},
        ),
    }


def _ablation_audit_fields(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: diagnostics.get(key)
        for key in (
            "ablation_id",
            "planner_ablation",
            "ablation_seed",
            "decoder_role",
            "selected_memory_policy",
            "original_active_memory_ids",
            "effective_active_memory_ids",
            "original_gate_weights",
            "effective_gate_weights",
            "random_memory_derived_seed",
            "daa_bypassed",
        )
    }


def _batch_ablation_fields(
    config: DualAgentBatchConfig,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    fields = {
        "ablation_id": None,
        "planner_ablation": config.planner_ablation,
        "ablation_seed": config.planner_ablation_seed,
        "decoder_role": "runtime" if config.planner_mode == "daa_full" else None,
        "selected_memory_policy": None,
        "original_active_memory_ids": None,
        "effective_active_memory_ids": None,
        "original_gate_weights": None,
        "effective_gate_weights": None,
    }
    final_path = Path(run_dir) / "final_answers.json"
    if final_path.is_file():
        try:
            payload = json.loads(final_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        if isinstance(payload, Mapping):
            for key in fields:
                if payload.get(key) is not None:
                    fields[key] = payload[key]
    return fields


def _save_routing_tensors(
    planner: Any,
    run_dir: Path,
    round_index: int,
) -> Path | None:
    config = getattr(planner, "routing_config", None)
    runtime = getattr(config, "runtime", None)
    if not bool(getattr(runtime, "save_daa_attention_tensors", False)):
        return None
    routing = getattr(planner, "last_routing_result", None)
    if routing is None:
        return None
    import torch

    output_dir = run_dir / "routing_tensors"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"round_{round_index:03d}.pt"
    payload = {
        "state_repr": routing.state_repr.detach().cpu(),
        "decoder_prefix": routing.decoder_prefix.detach().cpu(),
        "fused_memory": routing.fused_memory.slots.detach().cpu(),
        "memory_attention_mass": (
            routing.fused_memory.memory_attention_mass.detach().cpu()
            if routing.fused_memory.memory_attention_mass is not None
            else None
        ),
        "attention_weights": (
            routing.fused_memory.attention_weights.detach().cpu()
            if routing.fused_memory.attention_weights is not None
            else None
        ),
        "anchor_slots": (
            routing.fused_memory.anchor_slots.detach().cpu()
            if routing.fused_memory.anchor_slots is not None
            else None
        ),
    }
    torch.save(payload, output_path)
    return output_path.resolve()


def _planner_last_attempt(planner: Any) -> dict[str, Any]:
    value = getattr(planner, "last_attempt", None)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _write_planner_token_stats(run_dir: Path) -> dict[str, Any]:
    records = _read_jsonl_records(run_dir / "planner_outputs.jsonl")
    generation_events: list[dict[str, Any]] = []
    output_events: list[dict[str, Any]] = []
    for record in records:
        attempt = record.get("planner_attempt")
        if not isinstance(attempt, Mapping):
            continue
        round_index = record.get("round_index")
        status = attempt.get("status") or record.get("status")
        generations = attempt.get("generation_attempts")
        if isinstance(generations, list):
            for generation in generations:
                if not isinstance(generation, Mapping):
                    continue
                token_count = generation.get("token_count")
                if not isinstance(token_count, int):
                    continue
                generation_events.append(
                    {
                        "round_index": round_index,
                        "status": status,
                        "label": generation.get("label"),
                        "token_count": token_count,
                        "char_count": generation.get("char_count"),
                        "byte_count": generation.get("byte_count"),
                        "line_count": generation.get("line_count"),
                        "token_count_method": generation.get("token_count_method"),
                    }
                )
        output_metrics = attempt.get("planner_output_text_metrics")
        if isinstance(output_metrics, Mapping):
            token_count = output_metrics.get("token_count")
            if isinstance(token_count, int):
                output_events.append(
                    {
                        "round_index": round_index,
                        "status": status,
                        "source_generation_label": output_metrics.get(
                            "source_generation_label"
                        ),
                        "token_count": token_count,
                        "char_count": output_metrics.get("char_count"),
                        "byte_count": output_metrics.get("byte_count"),
                        "line_count": output_metrics.get("line_count"),
                        "token_count_method": output_metrics.get("token_count_method"),
                    }
                )
    stats = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "planner_outputs_path": str((run_dir / "planner_outputs.jsonl").resolve()),
        "generation_attempts": _token_event_summary(generation_events),
        "normalized_planner_outputs": _token_event_summary(output_events),
        "events": {
            "generation_attempts": generation_events,
            "normalized_planner_outputs": output_events,
        },
        "notes": [
            "generation_attempts are raw decoder outputs; JSON mode may also include repair attempts.",
            "normalized_planner_outputs are strict planner_action.v2 JSON objects.",
        ],
    }
    write_json(run_dir / "planner_token_stats.json", stats)
    return stats


def _token_event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [
        int(event["token_count"])
        for event in events
        if isinstance(event.get("token_count"), int)
    ]
    summary: dict[str, Any] = {
        "count": len(counts),
        "by_label": {},
    }
    if not counts:
        return summary
    summary.update(_token_count_distribution(counts))
    summary["recommended_decoder_tokens"] = {
        "p90_rounded_128": _round_up_to_multiple(int(summary["p90"]), 128),
        "p95_rounded_128": _round_up_to_multiple(int(summary["p95"]), 128),
        "max_rounded_128": _round_up_to_multiple(int(summary["max"]), 128),
        "p95_plus_20pct_rounded_128": _round_up_to_multiple(
            math.ceil(float(summary["p95"]) * 1.2),
            128,
        ),
    }
    by_label: dict[str, list[int]] = {}
    for event in events:
        token_count = event.get("token_count")
        if not isinstance(token_count, int):
            continue
        label = str(event.get("label") or event.get("source_generation_label") or "unknown")
        by_label.setdefault(label, []).append(token_count)
    summary["by_label"] = {
        label: _token_count_distribution(values)
        for label, values in sorted(by_label.items())
    }
    return summary


def _token_count_distribution(counts: list[int]) -> dict[str, Any]:
    ordered = sorted(int(value) for value in counts)
    return {
        "min": ordered[0],
        "p50": _nearest_rank_percentile(ordered, 50),
        "p75": _nearest_rank_percentile(ordered, 75),
        "p90": _nearest_rank_percentile(ordered, 90),
        "p95": _nearest_rank_percentile(ordered, 95),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def _nearest_rank_percentile(ordered_counts: list[int], percentile: int) -> int:
    if not ordered_counts:
        return 0
    rank = max(1, math.ceil((percentile / 100) * len(ordered_counts)))
    return ordered_counts[min(rank - 1, len(ordered_counts) - 1)]


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if value <= 0:
        return multiple
    return int(math.ceil(value / multiple) * multiple)


def _write_state_event(
    run_dir: Path,
    case_id: str,
    run_id: str,
    payload: Mapping[str, Any],
) -> None:
    append_jsonl(
        run_dir / "patient_state_history.jsonl",
        {
            "case_id": case_id,
            "run_id": run_id,
            "timestamp": utc_now(),
            **dict(payload),
        },
    )


@dataclass(frozen=True)
class _PartialAgentTurn:
    content: str
    tool_results: tuple[dict[str, Any], ...]
    model: str | None
    usage: dict[str, Any] | None
    turn_id: str
    conversation_log_path: str
    tool_round_limit_reached: bool = False
    forced_finalization: bool = False


def _partial_turn_from_conversation_log(
    conversation_log_path: Path,
    *,
    error: Exception,
) -> _PartialAgentTurn:
    """Recover already-executed tools from an AgentLoop failed turn audit record."""

    latest: dict[str, Any] | None = None
    if conversation_log_path.is_file():
        for line in conversation_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event_type") == "agent_turn":
                latest = record
    tool_results = latest.get("tool_results", []) if latest else []
    if not isinstance(tool_results, list):
        tool_results = []
    model_calls = latest.get("model_calls", []) if latest else []
    model = None
    usage = None
    if isinstance(model_calls, list) and model_calls:
        last_call = model_calls[-1]
        if isinstance(last_call, Mapping):
            model = str(last_call.get("model")) if last_call.get("model") else None
            usage_value = last_call.get("usage")
            usage = dict(usage_value) if isinstance(usage_value, Mapping) else None
    turn_id = str(latest.get("turn_id")) if latest and latest.get("turn_id") else "partial_turn"
    content = (
        "Agent round stopped before a final assistant answer because the tool-calling "
        f"round limit was reached or another AgentLoopError occurred: {error}"
    )
    return _PartialAgentTurn(
        content=content,
        tool_results=tuple(dict(item) for item in tool_results if isinstance(item, Mapping)),
        model=model,
        usage=usage,
        turn_id=turn_id,
        conversation_log_path=str(conversation_log_path),
    )


def _failure_debug_from_run_dir(run_dir: Path, error: Exception) -> dict[str, Any]:
    """Build a compact failure report that points at likely failing tools."""

    conversation = _latest_agent_turn_from_log(run_dir / "conversation_log.jsonl")
    conversation_tool_results = conversation.get("tool_results", []) if conversation else []
    if not isinstance(conversation_tool_results, list):
        conversation_tool_results = []
    tool_call_records = _read_jsonl_records(run_dir / "tool_calls.jsonl")
    planner_records = _read_jsonl_records(run_dir / "planner_outputs.jsonl")
    debug = {
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "conversation_error": conversation.get("error") if conversation else None,
        "conversation_tool_debug": _tool_debug_from_tool_results(
            tuple(
                dict(item)
                for item in conversation_tool_results
                if isinstance(item, Mapping)
            )
        ),
        "recorded_tool_call_debug": _tool_debug_from_tool_call_records(tool_call_records),
        "last_planner_output": planner_records[-1] if planner_records else None,
        "available_files": sorted(path.name for path in run_dir.iterdir()) if run_dir.is_dir() else [],
        "debug_hint": (
            "Check failed_tools/warning_tools first. If those are empty and the "
            "error is a tool-round limit, inspect repeated_skill_counts and "
            "recent_tools to find the tool the model repeatedly called or retried."
        ),
    }
    return debug


def _latest_agent_turn_from_log(path: Path) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    if not path.is_file():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, Mapping) and record.get("event_type") == "agent_turn":
            latest = dict(record)
    return latest


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, Mapping):
            records.append(dict(record))
    return records


def _tool_debug_from_tool_results(
    tool_results: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    compact = [_compact_tool_result(item) for item in tool_results]
    return _tool_debug_from_compact_items(compact)


def _tool_debug_from_tool_call_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    compact = []
    for record in records:
        compact.append(
            {
                "function_name": record.get("function_name"),
                "skill_name": record.get("skill_name"),
                "arguments": record.get("arguments", {}),
                "status": record.get("status"),
                "summary": record.get("summary"),
                "warnings": [],
                "error": None,
                "call_id": record.get("call_id"),
                "phase": record.get("phase"),
                "raw_output_path": record.get("raw_output_path"),
            }
        )
    return _tool_debug_from_compact_items(compact)


def _compact_tool_result(tool_record: Mapping[str, Any]) -> dict[str, Any]:
    result = tool_record.get("result", {})
    result = result if isinstance(result, Mapping) else {}
    findings = result.get("findings", {})
    findings = findings if isinstance(findings, Mapping) else {}
    warnings = result.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = [warnings]
    status = result.get("status")
    return {
        "function_name": tool_record.get("function_name"),
        "skill_name": tool_record.get("skill_name") or result.get("skill_name"),
        "arguments": tool_record.get("arguments", {}),
        "status": status,
        "summary": findings.get("summary") or result.get("summary") or status,
        "warnings": [str(item) for item in warnings if str(item).strip()],
        "error": findings.get("error"),
        "call_id": result.get("call_id"),
        "phase": result.get("phase"),
        "artifact_count": len(result.get("artifacts", []))
        if isinstance(result.get("artifacts"), list)
        else 0,
    }


def _tool_debug_from_compact_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    skill_counts: dict[str, int] = {}
    failed = []
    warning = []
    for item in items:
        status = str(item.get("status") or "unknown")
        skill_name = str(item.get("skill_name") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        skill_counts[skill_name] = skill_counts.get(skill_name, 0) + 1
        has_error = item.get("error") not in (None, "", [], {})
        has_warning = bool(item.get("warnings"))
        if status not in {"success", "ok"} or has_error:
            failed.append(item)
        elif has_warning:
            warning.append(item)
    repeated = {
        skill: count for skill, count in sorted(skill_counts.items()) if count > 1
    }
    return {
        "tool_result_count": len(items),
        "last_tool": items[-1] if items else None,
        "failed_tools": failed,
        "warning_tools": warning,
        "status_counts": status_counts,
        "skill_counts": skill_counts,
        "repeated_skill_counts": repeated,
        "recent_tools": items[-8:],
    }


def _compact_tool_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "call_id": record.get("call_id"),
            "phase": record.get("phase"),
            "skill_name": record.get("skill_name"),
            "status": record.get("status"),
            "summary": record.get("summary"),
            "artifact_paths": record.get("artifact_paths", []),
            "image_artifact_paths": record.get("image_artifact_paths", []),
            "raw_output_path": record.get("raw_output_path"),
        }
        for record in records
    ]


def _image_review_summaries(
    trajectory_plan: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    image_tools_by_round: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        if not record.get("image_artifact_paths"):
            continue
        round_index = int(record.get("planner_round_index") or 0)
        image_tools_by_round.setdefault(round_index, []).append(
            {
                "call_id": record.get("call_id"),
                "skill_name": record.get("skill_name"),
            }
        )

    summaries = []
    for item in trajectory_plan:
        round_index = int(item.get("round_index") or 0)
        tools = image_tools_by_round.get(round_index, [])
        summary = item.get("agent_summary")
        if tools and isinstance(summary, str) and summary.strip():
            summaries.append(
                {
                    "round_index": round_index,
                    "tools": tools,
                    "summary_text": summary.strip(),
                }
            )
    return summaries


def _dual_run_summary(run_dir: Path) -> dict[str, str]:
    summary = _run_summary(run_dir)
    summary.update(
        {
            "planner_outputs": str((run_dir / "planner_outputs.jsonl").resolve()),
            "planner_token_stats": str((run_dir / "planner_token_stats.json").resolve()),
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


def _aggregate_batch_planner_token_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    generation_events: list[dict[str, Any]] = []
    output_events: list[dict[str, Any]] = []
    for record in records:
        run_dir_value = record.get("run_dir")
        if not run_dir_value:
            continue
        stats_path = Path(str(run_dir_value)) / "planner_token_stats.json"
        if not stats_path.is_file():
            continue
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        events = stats.get("events")
        if not isinstance(events, Mapping):
            continue
        for source_key, target in (
            ("generation_attempts", generation_events),
            ("normalized_planner_outputs", output_events),
        ):
            values = events.get(source_key)
            if not isinstance(values, list):
                continue
            for event in values:
                if not isinstance(event, Mapping):
                    continue
                enriched = dict(event)
                enriched["case_id"] = record.get("case_id")
                enriched["run_dir"] = str(run_dir_value)
                target.append(enriched)
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "case_count_with_token_stats": len(
            {
                event.get("case_id")
                for event in generation_events + output_events
                if event.get("case_id")
            }
        ),
        "generation_attempts": _token_event_summary(generation_events),
        "normalized_planner_outputs": _token_event_summary(output_events),
        "events": {
            "generation_attempts": generation_events,
            "normalized_planner_outputs": output_events,
        },
        "notes": [
            "Planner V2 records strict structured-output token counts and generation attempts.",
            "generation_attempts include failed/repair outputs and are useful for debugging truncation or verbose decoder behavior.",
        ],
    }


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
