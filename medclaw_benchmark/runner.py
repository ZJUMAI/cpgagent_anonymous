"""Benchmark runner that records trajectory, tool calls, and evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

from medclaw.core.agent_loop import AgentLoop
from medclaw.core.tool_router import ToolRouter
from medclaw.llm.factory import SUPPORTED_PROVIDERS, create_llm_client, resolve_provider
from medclaw.registry.skill_registry import SkillRegistry
from medclaw.runtime.runtime_manager import RuntimeManager
from medclaw.runtime.skill_runner import SkillRunner
from medclaw.stores.artifact_store import ArtifactStore
from medclaw.stores.audit_log import AuditLog
from medclaw.stores.conversation_log import sanitize_for_audit
from medclaw.trajectory.action_set import canonicalize_guideline_trajectory
from medclaw.utils import read_json, utc_now, write_json

from medclaw_benchmark.case_builder import CaseBuilder
from medclaw_benchmark.case_paths import find_gold_trajectory, should_generate_gold_trajectory
from medclaw_benchmark.case_simulator import CaseSimulator
from medclaw_benchmark.io_utils import append_jsonl, safe_id


BENCHMARK_AGENT_SYSTEM_PROMPT = """You are MedClaw, a multimodal oncology benchmark agent.
You are evaluated on diagnostic reasoning, evidence gathering, modality selection,
guideline use, and communication. The rubric is hidden from you.

You may call available tools in any order and revise your plan after each result.
Use tools only when they help answer the case. Inspect image artifacts immediately
when a tool returns them; do not claim that you reviewed a modality unless the
tool evidence supports it. If you need image-producing ROI tools, call one such
tool at a time and inspect its returned images before requesting another image
ROI tool. Do not fabricate biomarkers, staging, pathology, or guideline claims.
When you use guideline evidence, cite the returned guideline title/source and
page number in the final answer. If a snippet has no page number, state that the
page number was not provided instead of inventing one. 最终回答中必须明确写出
指南依据及页码。
This is benchmark research, not clinical advice.

The current case_id is {case_id}. Pass this exact case_id whenever a tool accepts
case_id. Produce the final answer in Chinese, with explicit uncertainty and
missing information."""


BENCHMARK_AGENT_USER_PROMPT = """Initial visible case information:
{initial_observation}

Plan and execute your own diagnostic workflow. You can request clinical,
pathology, radiology, molecular, ROI/image, and guideline evidence through tools.
After each tool result, decide whether to continue, change direction, or stop.

When you have enough evidence, provide a final Chinese answer covering:
1. concise patient background;
2. diagnostic impression and supporting evidence;
3. pathology/radiology/molecular findings actually reviewed;
4. guideline-matched treatment or management recommendation, explicitly citing
   guideline title/source, page number, and snippet/chunk id when available
   （必须指出指南依据及页码）;
5. key missing information, limitations, and recommended verification."""


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Paths produced by one benchmark run."""

    case_id: str
    run_id: str
    run_dir: Path
    final_answer_path: Path
    evidence_board_path: Path
    trajectory_path: Path
    tool_calls_path: Path
    guideline_trajectory_path: Path


class BenchmarkRunner:
    """Drive a benchmark agent over a simulator and MedClaw skills."""

    def __init__(
        self,
        *,
        case_dir: str | Path,
        run_id: str = "run_001",
        runs_root: str | Path = "runs",
        project_root: str | Path | None = None,
        agent: str | None = None,
        runtime_manager: RuntimeManager | Any | None = None,
        artifact_resolver: Callable[[str], Path] | None = None,
        llm_client: Any | None = None,
        llm_config: Any | None = None,
        qwen_config: Any | None = None,
    ) -> None:
        self.agent = resolve_provider(agent)
        if self.agent not in SUPPORTED_PROVIDERS:
            supported = ", ".join(SUPPORTED_PROVIDERS)
            raise ValueError(f"Unsupported agent provider {agent!r}. Supported: {supported}.")
        self.case_dir = Path(case_dir).resolve()
        self.case_id = self.case_dir.name
        self.run_id = safe_id(run_id, "run_001")
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
        self.run_dir = Path(runs_root).resolve() / self.case_id / self.run_id
        self.runtime_manager = runtime_manager
        self.artifact_resolver = artifact_resolver
        self.llm_client = llm_client
        self.llm_config = llm_config or qwen_config
        self.qwen_config = self.llm_config
        self.evidence_items: list[dict[str, Any]] = []

    def run(self) -> BenchmarkRunResult:
        """Run the benchmark agent and write all audit files."""

        CaseBuilder(
            self.case_dir,
            llm_client=self.llm_client,
            llm_config=self.llm_config,
            provider=self.agent,
            qwen_config=self.llm_config,
        ).build()
        self.run_dir.mkdir(parents=True, exist_ok=True)
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

        router = BenchmarkDynamicToolRouter(
            simulator=simulator,
            runtime=runtime,
            case_id=self.case_id,
        )
        agent_loop = AgentLoop(
            router,
            _AgentEvidenceSink(self.case_id, self.run_dir),
            llm_client=self._build_llm_client(),
            system_prompt=BENCHMARK_AGENT_SYSTEM_PROMPT.format(case_id=self.case_id),
            max_tool_rounds=12,
            artifact_uri_resolver=self.artifact_resolver,
            conversation_log=_RunConversationLog(self.run_dir / "conversation_log.jsonl"),
        )

        turn = agent_loop.chat(
            BENCHMARK_AGENT_USER_PROMPT.format(
                initial_observation=_json_text(initial.get("visible_information", {})),
            )
        )

        collected: list[dict[str, Any]] = []
        for tool_record in turn.tool_results:
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
            collected.append(record)
            self._record_trajectory(
                "tool_result",
                {
                    "phase": phase,
                    "skill_name": skill_name,
                    "status": result.get("status"),
                    "summary": _summary(result),
                },
            )

        self._write_agent_context(initial, collected, turn)
        final_payload = {
            "case_id": self.case_id,
            "run_id": self.run_id,
            "agent": self.agent,
            "answer_text": turn.content,
            "generated_at": utc_now(),
            "provider": self.agent,
            "model": turn.model,
            "usage": turn.usage,
            "conversation_log_path": turn.conversation_log_path,
            "agent_context_path": str(self.run_dir / "agent_context.json"),
            "guideline_trajectory_path": str(guideline_trajectory_path),
            "tool_round_count": len(turn.tool_results),
            "tool_round_limit_reached": bool(
                getattr(turn, "tool_round_limit_reached", False)
            ),
            "forced_finalization": bool(getattr(turn, "forced_finalization", False)),
        }
        final_answer_path = self.run_dir / "final_answers.json"
        write_json(final_answer_path, final_payload)
        self._record_trajectory("final_answer", final_payload)
        evidence_board_path = self._write_evidence_board()

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

    def _record_tool_call(
        self,
        phase: str,
        skill_name: str,
        arguments: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        call_id = str(result.get("call_id") or f"bench_{uuid4().hex}")
        raw_output_path = self.run_dir / "artifacts" / "tool_outputs" / f"{call_id}_{safe_id(skill_name)}.json"
        write_json(raw_output_path, dict(result))
        artifact_paths, image_artifact_paths = self._collect_artifacts(
            skill_name,
            call_id,
            result,
        )
        record = {
            "call_id": call_id,
            "case_id": self.case_id,
            "run_id": self.run_id,
            "phase": phase,
            "skill_name": skill_name,
            "arguments": dict(arguments),
            "status": result.get("status"),
            "summary": _summary(result),
            "raw_output_path": str(raw_output_path),
            "artifact_paths": artifact_paths,
            "image_artifact_paths": image_artifact_paths,
            "timestamp": utc_now(),
        }
        append_jsonl(self.run_dir / "tool_calls.jsonl", record)
        self._add_evidence(record, result)
        return record

    def _collect_artifacts(
        self,
        skill_name: str,
        call_id: str,
        result: Mapping[str, Any],
    ) -> tuple[list[str], list[str]]:
        artifacts = result.get("artifacts", [])
        if not isinstance(artifacts, list) or self.artifact_resolver is None:
            return [], []
        if "radiology" in skill_name:
            modality_dir = "ct_roi"
        elif "pathology" in skill_name:
            modality_dir = "wsi_roi"
        elif "guideline" in skill_name:
            modality_dir = "guideline"
        else:
            modality_dir = "tool_outputs"
        destination_dir = self.run_dir / "artifacts" / modality_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        image_items: list[tuple[str, str]] = []
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                continue
            uri = artifact.get("uri")
            if not isinstance(uri, str) or not uri.startswith("artifact://"):
                continue
            try:
                source = self.artifact_resolver(uri)
            except (OSError, ValueError):
                continue
            destination = destination_dir / f"{call_id}_{source.name}"
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            paths.append(str(destination))
            if artifact.get("type") == "image":
                image_items.append((str(artifact.get("role") or ""), str(destination)))
        image_paths = [
            path
            for _, path in sorted(
                image_items,
                key=lambda item: (_image_role_priority(item[0]), item[1]),
            )
        ]
        return paths, image_paths

    def _add_evidence(self, record: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        skill_name = str(record["skill_name"])
        modality = str(result.get("modality") or _modality_for_skill(skill_name))
        evidence_id = f"E{len(self.evidence_items) + 1:03d}"
        self.evidence_items.append(
            {
                "evidence_id": evidence_id,
                "step": len(self.evidence_items) + 1,
                "source_skill": skill_name,
                "modality": modality,
                "status": record.get("status"),
                "fact": _summary(result),
                "supports": _supports_for_modality(modality),
                "artifact_paths": list(record.get("artifact_paths", [])),
                "image_artifact_paths": list(record.get("image_artifact_paths", [])),
                "raw_output_path": record.get("raw_output_path"),
                "call_id": record.get("call_id"),
            }
        )

    def _write_evidence_board(self) -> Path:
        path = self.run_dir / "evidence_board.json"
        write_json(
            path,
            {
                "case_id": self.case_id,
                "run_id": self.run_id,
                "evidence_items": self.evidence_items,
            },
        )
        return path

    def _copy_guideline_trajectory(self) -> Path:
        destination = self.run_dir / "guideline_trajectory.json"
        source = find_gold_trajectory(self.case_dir)
        if source is None and should_generate_gold_trajectory(self.case_dir):
            source = CaseBuilder(
                self.case_dir,
                llm_client=self.llm_client,
                llm_config=self.llm_config,
                provider=self.agent,
                qwen_config=self.llm_config,
            ).build_trajectory_file()
        if source is None or not Path(source).is_file():
            self._record_trajectory(
                "guideline_trajectory",
                {
                    "phase": "case_setup",
                    "source_path": None,
                    "output_path": str(destination),
                    "summary": (
                        "No gold trajectory at evaluation/{case_id}_trajectory.json; "
                        "skipped copy. UCEC and NPC packs may still be empty."
                    ),
                },
            )
            return destination
        source_trajectory = read_json(source)
        if not isinstance(source_trajectory, Mapping):
            raise ValueError(f"Gold trajectory must contain an object: {source}")
        write_json(
            destination,
            canonicalize_guideline_trajectory(source_trajectory),
        )
        self._record_trajectory(
            "guideline_trajectory",
            {
                "phase": "case_setup",
                "source_path": str(source),
                "output_path": str(destination),
                "summary": (
                    "Guideline-grounded state-machine trajectory copied into "
                    "the run directory. This is distinct from trajectory.jsonl, "
                    "which records the live agent run log."
                ),
            },
        )
        return destination

    def _record_trajectory(self, event_type: str, payload: Mapping[str, Any]) -> None:
        append_jsonl(
            self.run_dir / "trajectory.jsonl",
            {
                "event_type": event_type,
                "case_id": self.case_id,
                "run_id": self.run_id,
                "timestamp": utc_now(),
                **dict(payload),
            },
        )

    def _write_agent_context(
        self,
        initial_observation: Mapping[str, Any],
        tool_records: list[dict[str, Any]],
        turn: Any,
    ) -> Path:
        path = self.run_dir / "agent_context.json"
        write_json(
            path,
            {
                "case_id": self.case_id,
                "run_id": self.run_id,
                "agent": self.agent,
                "provider": self.agent,
                "planning_mode": "model_driven_tool_choice_auto",
                "initial_observation": initial_observation.get(
                    "visible_information", {}
                ),
                "tool_records": tool_records,
                "conversation_log_path": turn.conversation_log_path,
                "final_model": turn.model,
                "final_usage": turn.usage,
                "tool_round_limit_reached": bool(
                    getattr(turn, "tool_round_limit_reached", False)
                ),
                "forced_finalization": bool(
                    getattr(turn, "forced_finalization", False)
                ),
                "guideline_trajectory_path": str(
                    self.run_dir / "guideline_trajectory.json"
                ),
                "note": (
                    "The core model selected tools dynamically. Image artifacts "
                    "were returned to the model immediately after the producing "
                    "tool call through AgentLoop artifact feedback."
                ),
            },
        )
        return path

    def _llm_final_answer(
        self,
        initial_observation: Mapping[str, Any],
        tool_records: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        """Review image-producing tools immediately, then synthesize text only."""

        client = self._build_llm_client()
        image_reviews = self._review_image_tools(client, tool_records)
        context_items = self._build_context_items(
            initial_observation,
            tool_records,
            image_reviews,
        )
        write_json(
            self.run_dir / "agent_context.json",
            {
                "case_id": self.case_id,
                "run_id": self.run_id,
                "agent": self.agent,
                "provider": self.agent,
                "items": context_items,
                "image_review_rounds": image_reviews,
                "note": (
                    "Final synthesis is text-only. ROI pixels are sent only in "
                    "per-tool image review rounds."
                ),
            },
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are MedClaw, a multimodal medical benchmark agent. "
                    "Use the provided clinical, pathology, radiology, molecular, guideline, "
                    "and prior image-review summaries to produce a final Chinese answer. "
                    "The final synthesis is text-only; do not ask for or assume additional images. "
                    "Do not claim a modality was reviewed unless the supplied evidence supports it. "
                    "Do not fabricate biomarkers. This is research output, not clinical advice. "
                    "The rubric is hidden from you."
                ),
            }
        ]
        for item in context_items:
            messages.append(
                _build_model_user_message(
                    client,
                    str(item.get("text", "")),
                    image_paths=[],
                )
            )

        completion = client.complete(messages=messages, tools=[], tool_choice="none")
        raw_response = {
            "message": completion.message,
            "model": completion.model,
            "usage": completion.usage,
            "config": _public_llm_config(getattr(client, "config", self.llm_config)),
        }
        write_json(self.run_dir / "agent_raw_response.json", raw_response)
        content = completion.message.get("content")
        if not isinstance(content, str) or not content.strip():
            content = "核心多模态模型未返回可用文本答案。"
        return (
            content,
            {
                "provider": self.agent,
                "model": completion.model,
                "usage": completion.usage,
                "agent_context_path": str(self.run_dir / "agent_context.json"),
                "agent_raw_response_path": str(
                    self.run_dir / "agent_raw_response.json"
                ),
                "image_review_count": len(image_reviews),
            },
        )

    def _build_context_items(
        self,
        initial_observation: Mapping[str, Any],
        tool_records: list[dict[str, Any]],
        image_reviews: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        reviews_by_call_id = {
            str(review.get("call_id")): review for review in image_reviews
        }
        items = [
            {
                "kind": "initial_observation",
                "phase": "diagnosis_phase",
                "skill_name": None,
                "text": (
                    "初始可见病例信息："
                    + _json_text(initial_observation.get("visible_information", {}))
                ),
                "images": [],
            }
        ]
        for record in tool_records:
            raw = _read_raw_output(record)
            review = reviews_by_call_id.get(str(record.get("call_id")))
            items.append(
                {
                    "kind": "tool_result",
                    "phase": record.get("phase"),
                    "skill_name": record.get("skill_name"),
                    "status": record.get("status"),
                    "text": _tool_context_text(record, raw, image_review=review),
                    "images": [],
                }
            )
        items.append(
            {
                "kind": "final_instruction",
                "phase": "treatment_phase",
                "skill_name": None,
                "text": (
                    "请综合以上分阶段文本证据和各工具已经生成的图像观察摘要，输出最终诊疗建议。"
                    "不要重新查看或假设更多图片；请明确区分已知事实、缺失信息、图像证据能支持什么、不能支持什么、"
                    "指南匹配、推荐行动、风险控制和人工复核点。"
                ),
                "images": [],
            }
        )
        return items

    def _review_image_tools(
        self,
        client: Any,
        tool_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Review each image-producing tool immediately and keep text summaries."""

        reviews: list[dict[str, Any]] = []
        for record in tool_records:
            image_paths = [
                Path(path)
                for path in record.get("image_artifact_paths", [])
                if Path(str(path)).is_file()
            ]
            image_limit = getattr(
                getattr(client, "config", None),
                "max_images_per_message",
                None,
            )
            if isinstance(image_limit, int) and image_limit > 0:
                image_paths = image_paths[:image_limit]
            if not image_paths:
                continue
            raw = _read_raw_output(record)
            prompt = _image_review_prompt(record, raw)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是 MedClaw 的多模态医学证据观察员。"
                        "请只分析当前工具刚产生的这一批 ROI/patch 图片，输出结构化中文观察摘要。"
                        "不要做最终治疗决策，不要把研究性 ROI 当作临床诊断金标准。"
                    ),
                },
                _build_model_user_message(client, prompt, image_paths=image_paths),
            ]
            completion = client.complete(messages=messages, tools=[], tool_choice="none")
            content = completion.message.get("content")
            if not isinstance(content, str) or not content.strip():
                content = "图像观察未返回可用文本。"
            review = {
                "call_id": record.get("call_id"),
                "phase": record.get("phase"),
                "skill_name": record.get("skill_name"),
                "status": "success",
                "summary_text": content,
                "image_count": len(image_paths),
                "images": [_image_reference(path) for path in image_paths],
                "prompt_text": prompt,
                "model": completion.model,
                "usage": completion.usage,
                "raw_response": {
                    "message": completion.message,
                    "model": completion.model,
                    "usage": completion.usage,
                },
                "timestamp": utc_now(),
            }
            reviews.append(review)
            self._record_trajectory(
                "image_review",
                {
                    "phase": record.get("phase"),
                    "skill_name": record.get("skill_name"),
                    "call_id": record.get("call_id"),
                    "image_count": len(image_paths),
                    "summary": content,
                },
            )

        write_json(
            self.run_dir / "image_reviews.json",
            {
                "case_id": self.case_id,
                "run_id": self.run_id,
                "provider": self.agent,
                "reviews": reviews,
                "note": (
                    "Images are reviewed immediately after each image-producing tool call; "
                    "final synthesis uses these text summaries."
                ),
            },
        )
        return reviews

    def _build_llm_client(self) -> Any:
        if self.llm_client is not None:
            return self.llm_client
        self.llm_client = create_llm_client(self.agent, config=self.llm_config)
        self.llm_config = getattr(self.llm_client, "config", self.llm_config)
        self.qwen_config = self.llm_config
        return self.llm_client

    def _build_runtime(self) -> RuntimeManager:
        registry = SkillRegistry(self.project_root / "medclaw" / "skills")
        artifact_store = ArtifactStore(self.run_dir / "artifacts" / "registered")
        return RuntimeManager(
            registry,
            SkillRunner(),
            artifact_store,
            AuditLog(self.run_dir / "skill_audit"),
            self.run_dir / "skill_runs",
        )


class BenchmarkDynamicToolRouter:
    """Expose simulator data tools and executable MedClaw skills to the model."""

    def __init__(
        self,
        *,
        simulator: CaseSimulator,
        runtime: Any,
        case_id: str,
        guideline_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.simulator = simulator
        self.runtime = runtime
        self.case_id = case_id
        self.guideline_context = dict(guideline_context or {})
        self.registry = runtime.registry if hasattr(runtime, "registry") else runtime
        self._function_to_skill = self._build_function_mapping()
        self._call_history: list[dict[str, Any]] = []

    def list_function_tools(self) -> list[dict[str, Any]]:
        tools = []
        for card in [*_simulator_tool_cards(), *self.registry.list_tools_for_agent()]:
            skill_name = str(card["name"])
            function_name = self.function_name_for_skill(skill_name)
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": (
                            f"{card['description']} "
                            f"This function executes the MedClaw skill {skill_name!r}."
                        ),
                        "parameters": dict(card["input_schema"]),
                    },
                }
            )
        return tools

    def call(self, skill_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        arguments = dict(arguments)
        arguments.setdefault("case_id", self.case_id)
        if skill_name in _simulator_skill_names():
            phase = _phase_for_skill(skill_name)
            self.simulator.advance_phase(phase)
            result = self.simulator.query(skill_name, arguments, phase)
        else:
            if skill_name == "guideline.retrieve":
                arguments = self._guideline_arguments(arguments)
            result = self.runtime.invoke(skill_name, arguments)

        result.setdefault("phase", _phase_for_skill(skill_name))
        result.setdefault("skill_name", skill_name)
        self._call_history.append(
            {
                "skill_name": skill_name,
                "arguments": arguments,
                "result": result,
            }
        )
        return result

    def skill_name_for_function(self, function_name: str) -> str:
        try:
            return self._function_to_skill[function_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._function_to_skill)) or "<none>"
            raise ValueError(
                f"Unknown tool function {function_name!r}. Available functions: {available}"
            ) from exc

    @staticmethod
    def function_name_for_skill(skill_name: str) -> str:
        return ToolRouter.function_name_for_skill(skill_name)

    def _build_function_mapping(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for card in [*_simulator_tool_cards(), *self.registry.list_tools_for_agent()]:
            skill_name = str(card["name"])
            function_name = self.function_name_for_skill(skill_name)
            if function_name in mapping:
                raise ValueError(
                    f"Duplicate benchmark tool function {function_name!r} for "
                    f"{mapping[function_name]!r} and {skill_name!r}."
                )
            mapping[function_name] = skill_name
        return mapping

    def _guideline_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if isinstance(query, str) and query.strip():
            arguments.setdefault("case_id", self.case_id)
            return _apply_guideline_env_defaults(
                _bind_release_guideline_arguments(arguments, self.guideline_context)
            )

        records = [
            {
                "skill_name": item.get("skill_name"),
                "result": item.get("result", {}),
            }
            for item in self._call_history
        ]
        generated = _build_guideline_context_arguments_from_results(
            self.case_id,
            records,
        )
        generated.update({key: value for key, value in arguments.items() if value})
        return _apply_guideline_env_defaults(
            _bind_release_guideline_arguments(generated, self.guideline_context)
        )


def _bind_release_guideline_arguments(
    arguments: Mapping[str, Any],
    guideline_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Force guideline.retrieve filters to the Patient State V2 release context."""

    result = dict(arguments)
    context = dict(guideline_context or {})
    guidelines = context.get("guidelines")
    if not isinstance(guidelines, list) or not guidelines:
        return result
    selected = guidelines[0]
    if not isinstance(selected, Mapping):
        return result
    guideline_id = str(selected.get("guideline_id") or "").strip()
    version = str(selected.get("version") or "").strip()
    filters = {
        "NSCLC_2010": ("nsclc", "nccn"),
        "SCLC_2010": ("sclc", "nccn"),
        "CSCO子宫内膜癌2023": ("ucec", "csco"),
        "CSCO鼻咽癌2022": ("npc", "csco"),
    }.get(guideline_id)
    if filters is None:
        raise ValueError(
            f"Unsupported release guideline for guideline.retrieve: {guideline_id!r}"
        )
    result["cancer_type"], result["guideline_family"] = filters
    result["guideline_version"] = version
    return result


class _AgentEvidenceSink:
    """Small AgentLoop evidence sink; benchmark writes its own evidence board."""

    def __init__(self, case_id: str, runs_root: Path) -> None:
        self.case_id = case_id
        self.runs_root = Path(runs_root)
        self.evidence: list[dict[str, Any]] = []

    def add_result(self, skill_name: str, result: Mapping[str, Any]) -> None:
        self.evidence.append({"skill_name": skill_name, "result": dict(result)})


class _RunConversationLog:
    """Conversation log scoped to one benchmark run directory."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def path_for_case(self, case_id: str) -> Path:
        return self.path

    def append(self, case_id: str, record: Mapping[str, Any]) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sanitized = sanitize_for_audit(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sanitized, ensure_ascii=False))
            handle.write("\n")
        return self.path


def _simulator_tool_cards() -> list[dict[str, Any]]:
    schema = {
        "type": "object",
        "properties": {
            "case_id": {
                "type": "string",
                "description": "Benchmark case identifier. If omitted, the current case is used.",
            }
        },
        "additionalProperties": False,
    }
    cards = [
        (
            "clinical.read_summary",
            "Read released case-level clinical baseline, diagnosis, staging, and tumor background fields.",
            "clinical",
        ),
        (
            "pathology.read_report",
            "Read released pathology report text for the current benchmark case.",
            "pathology_text",
        ),
        (
            "pathology.read_slide_metadata",
            "Read slide-level pathology metadata such as sample type and tumor-cell annotations.",
            "pathology_slide_metadata",
        ),
        (
            "pathology.read_wsi_manifest",
            "Check whether whole-slide images are available for pathology ROI tools.",
            "wsi_manifest",
        ),
        (
            "radiology.read_ct_manifest",
            "Check whether radiology volumes are available for radiology ROI tools.",
            "ct_manifest",
        ),
        (
            "molecular.query_biomarkers",
            "Read released molecular and biomarker results for treatment planning.",
            "molecular",
        ),
    ]
    return [
        {
            "name": name,
            "version": "benchmark",
            "description": description,
            "modalities": {"primary": modality},
            "input_schema": schema,
        }
        for name, description, modality in cards
    ]


def _simulator_skill_names() -> set[str]:
    return {str(card["name"]) for card in _simulator_tool_cards()}


def _phase_for_skill(skill_name: str) -> str:
    if skill_name == "clinical.read_follow_up":
        return "progression_phase"
    if skill_name.startswith("clinical."):
        return "diagnosis_phase"
    if skill_name in {"pathology.read_wsi_manifest", "pathology.conch_patch_roi"}:
        return "pathology_image_phase"
    if skill_name.startswith("pathology."):
        if "roi" in skill_name or "wsi" in skill_name:
            return "pathology_image_phase"
        return "pathology_text_phase"
    if skill_name.startswith("radiology."):
        return "radiology_phase"
    if skill_name.startswith("molecular.") or skill_name.startswith("guideline."):
        return "treatment_phase"
    return "diagnosis_phase"


def _missing(skill_name: str, modality: str, message: str) -> dict[str, Any]:
    return {
        "status": "missing",
        "skill_name": skill_name,
        "modality": modality,
        "findings": {
            "summary": message,
            "data": {"available": False},
        },
        "artifacts": [],
        "warnings": [message],
    }


def _summary(result: Mapping[str, Any]) -> str:
    findings = result.get("findings", {})
    if isinstance(findings, Mapping):
        summary = findings.get("summary")
        if isinstance(summary, str) and summary:
            return summary
    return str(result.get("status", "unknown"))


def _finding_data(result: Mapping[str, Any]) -> dict[str, Any]:
    findings = result.get("findings", {})
    if isinstance(findings, Mapping):
        data = findings.get("data", {})
        return dict(data) if isinstance(data, Mapping) else {}
    return {}


def _data_from_record(records: list[dict[str, Any]], skill_name: str) -> dict[str, Any]:
    for record in reversed(records):
        if record.get("skill_name") != skill_name:
            continue
        path = record.get("raw_output_path")
        if not isinstance(path, str) or not Path(path).is_file():
            continue
        data = read_json(Path(path))
        if not isinstance(data, Mapping):
            continue
        return _finding_data(data)
    return {}


def _data_from_result_records(
    records: list[dict[str, Any]],
    skill_name: str,
) -> dict[str, Any]:
    for record in reversed(records):
        if record.get("skill_name") != skill_name:
            continue
        result = record.get("result")
        if isinstance(result, Mapping):
            return _finding_data(result)
    return {}


def _build_guideline_context_arguments(
    case_id: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    clinical = _data_from_record(records, "clinical.read_summary")
    pathology = _data_from_record(records, "pathology.read_report")
    molecular = _data_from_record(records, "molecular.query_biomarkers")
    return _build_guideline_context_arguments_from_data(
        case_id,
        clinical=clinical,
        pathology=pathology,
        molecular=molecular,
    )
    context = {
        "case_id": case_id,
        "project_id": _first_text(clinical, "project_id"),
        "primary_diagnosis": _first_text(
            clinical,
            "primary_diagnosis",
            "diagnosis",
            "histology",
            "tumor_type",
        ),
        "primary_site": _first_text(
            clinical,
            "tumor_location",
            "tissue_or_organ_of_origin",
            "site_of_resection_or_biopsy",
            "primary_site",
        ),
        "stage": {
            key: _first_text(clinical, key)
            for key in (
                "clinical_stage",
                "pathologic_stage",
                "ajcc_pathologic_stage",
                "pathologic_t_stage",
                "pathologic_n_stage",
                "pathologic_m_stage",
            )
            if _first_text(clinical, key)
        },
        "residual_disease": _first_text(clinical, "residual_disease"),
        "molecular_summary": _first_text(
            molecular,
            "summary",
            "molecular_subtype",
            "copy_number_summary",
        ),
        "available_biomarkers": _available_biomarkers(molecular),
        "pathology_report_excerpt": _truncate_text(
            _first_text(pathology, "report_text"),
            500,
        ),
    }
    query = (
        "请在当前已配置的肿瘤诊疗指南语料中，检索与该病例的诊断确认、分期评估、"
        "病理/影像/分子证据整合、治疗决策和随访监测相关的指南证据片段。"
        "不要假设固定癌种；请根据病例上下文和指南语料本身选择最相关内容。"
        "病例上下文："
        + _json_text(_drop_empty_values(context))
    )
    return {
        "case_id": case_id,
        "query": query[:1200],
        "cancer_type": "auto",
        "max_snippets": 6,
    }


def _build_guideline_context_arguments_from_results(
    case_id: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    clinical = _data_from_result_records(records, "clinical.read_summary")
    pathology = _data_from_result_records(records, "pathology.read_report")
    molecular = _data_from_result_records(records, "molecular.query_biomarkers")
    return _build_guideline_context_arguments_from_data(
        case_id,
        clinical=clinical,
        pathology=pathology,
        molecular=molecular,
    )


def _build_guideline_context_arguments_from_data(
    case_id: str,
    *,
    clinical: Mapping[str, Any],
    pathology: Mapping[str, Any],
    molecular: Mapping[str, Any],
) -> dict[str, Any]:
    context = {
        "case_id": case_id,
        "project_id": _first_text(clinical, "project_id"),
        "primary_diagnosis": _first_text(
            clinical,
            "primary_diagnosis",
            "diagnosis",
            "histology",
            "tumor_type",
        ),
        "primary_site": _first_text(
            clinical,
            "tumor_location",
            "tissue_or_organ_of_origin",
            "site_of_resection_or_biopsy",
            "primary_site",
        ),
        "stage": {
            key: _first_text(clinical, key)
            for key in (
                "clinical_stage",
                "pathologic_stage",
                "ajcc_pathologic_stage",
                "pathologic_t_stage",
                "pathologic_n_stage",
                "pathologic_m_stage",
            )
            if _first_text(clinical, key)
        },
        "residual_disease": _first_text(clinical, "residual_disease"),
        "molecular_summary": _first_text(
            molecular,
            "summary",
            "molecular_subtype",
            "copy_number_summary",
        ),
        "available_biomarkers": _available_biomarkers(molecular),
        "pathology_report_excerpt": _truncate_text(
            _first_text(pathology, "report_text"),
            500,
        ),
    }
    query = (
        "Search the configured oncology guideline corpus for evidence relevant "
        "to this case's diagnostic confirmation, staging, pathology/radiology/"
        "molecular integration, treatment decision, and surveillance. Do not "
        "assume a fixed cancer type; select the most relevant guideline content "
        "from the case context and corpus. Case context: "
        + _json_text(_drop_empty_values(context))
    )
    return {
        "case_id": case_id,
        "query": query[:1200],
        "cancer_type": "auto",
        "max_snippets": 6,
    }


def _apply_guideline_env_defaults(arguments: dict[str, Any]) -> dict[str, Any]:
    configured = dict(arguments)
    _set_env_override(configured, "chunk_mode", "MEDCLAW_GUIDELINE_CHUNK_MODE")
    configured.setdefault("chunk_mode", "chapter")
    _set_env_override(configured, "retrieval_mode", "MEDCLAW_GUIDELINE_RETRIEVAL_MODE")
    _set_env_override(configured, "rerank_mode", "MEDCLAW_GUIDELINE_RERANK_MODE")
    _set_env_int_override(configured, "max_snippets", "MEDCLAW_GUIDELINE_MAX_SNIPPETS")
    _set_env_int_override(
        configured,
        "rerank_candidate_count",
        "MEDCLAW_GUIDELINE_RERANK_CANDIDATE_COUNT",
    )
    if configured.get("chunk_mode") == "chapter" and not os.environ.get(
        "MEDCLAW_GUIDELINE_MAX_SNIPPETS"
    ):
        configured["max_snippets"] = 1
    return configured


def _set_env_override(arguments: dict[str, Any], key: str, env_name: str) -> None:
    value = os.environ.get(env_name)
    if value:
        arguments[key] = value


def _set_env_int_override(arguments: dict[str, Any], key: str, env_name: str) -> None:
    value = os.environ.get(env_name)
    if not value:
        return
    try:
        arguments[key] = int(value)
    except ValueError:
        return


def _guideline_arguments_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    findings = result.get("findings", {})
    if not isinstance(findings, Mapping):
        return {}
    arguments: dict[str, Any] = {}
    for source, target in (
        ("query", "query"),
        ("selected_cancer_type", "cancer_type"),
        ("selected_guideline_family", "guideline_family"),
        ("selected_guideline_version", "guideline_version"),
    ):
        value = findings.get(source)
        if isinstance(value, str) and value:
            arguments[target] = value
    snippets = findings.get("snippets")
    if isinstance(snippets, list) and snippets:
        arguments["max_snippets"] = len(snippets)
    return arguments


def _first_text(data: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            text = _json_text(value)
        else:
            text = str(value)
        text = text.strip()
        if text and text.lower() not in {"not_available", "none", "null"}:
            return text
    return ""


def _available_biomarkers(data: Mapping[str, Any]) -> list[str]:
    details = data.get("biomarker_details")
    if isinstance(details, Mapping):
        return sorted(str(key) for key in details if str(key).strip())
    ignored = {
        "case_id",
        "source",
        "extraction_method",
        "extraction_warnings",
        "summary",
        "molecular_subtype",
        "copy_number_summary",
        "high_amplification_genes",
    }
    biomarkers = []
    for key, value in data.items():
        if key in ignored or value in (None, "", "not_available"):
            continue
        if any(char.isupper() for char in str(key)):
            biomarkers.append(str(key))
    return sorted(set(biomarkers))


def _drop_empty_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := _drop_empty_values(item)) not in ("", [], {})
        }
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _drop_empty_values(item)) not in ("", [], {})]
    return value


def _truncate_text(value: str, max_chars: int) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _read_raw_output(record: Mapping[str, Any]) -> dict[str, Any]:
    path = record.get("raw_output_path")
    if not isinstance(path, str) or not Path(path).is_file():
        return {}
    data = read_json(Path(path))
    return dict(data) if isinstance(data, Mapping) else {}


def _tool_context_text(
    record: Mapping[str, Any],
    raw: Mapping[str, Any],
    *,
    image_review: Mapping[str, Any] | None = None,
) -> str:
    findings = raw.get("findings", {})
    payload: dict[str, Any] = {
        "phase": record.get("phase"),
        "skill_name": record.get("skill_name"),
        "status": record.get("status"),
        "summary": record.get("summary"),
        "warnings": raw.get("warnings", []),
    }
    if isinstance(findings, Mapping):
        payload["findings"] = findings
    if image_review is not None:
        payload["image_review_summary"] = {
            "image_count": image_review.get("image_count"),
            "summary_text": image_review.get("summary_text"),
            "model": image_review.get("model"),
        }
    return "工具调用结果：\n" + _json_text(payload)


def _image_review_prompt(record: Mapping[str, Any], raw: Mapping[str, Any]) -> str:
    payload = {
        "phase": record.get("phase"),
        "skill_name": record.get("skill_name"),
        "status": record.get("status"),
        "summary": record.get("summary"),
        "findings": raw.get("findings", {}),
        "warnings": raw.get("warnings", []),
        "image_artifact_paths": record.get("image_artifact_paths", []),
    }
    return (
        "请立即分析当前工具刚输出的图像证据。"
        "请按以下结构用中文回答：\n"
        "1. 图像/ROI 类型和数量；\n"
        "2. 肉眼可见的关键观察；\n"
        "3. 这些观察能支持哪些诊疗信息；\n"
        "4. 不能从这些图像推出什么，尤其不要过度诊断或完整分期；\n"
        "5. 后续最终回答中应如何引用这批证据。\n\n"
        "工具结构化输出如下：\n"
        + _json_text(payload)
    )


def _build_model_user_message(
    client: Any,
    text: str,
    *,
    image_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    builder = getattr(client, "build_user_message", None)
    if callable(builder):
        return builder(text, image_paths=image_paths)
    return {"role": "user", "content": text}


def _image_reference(path: Path) -> dict[str, Any]:
    reference = {"path": str(path)}
    if not path.is_file():
        reference["exists"] = False
        return reference
    reference["exists"] = True
    reference["size_bytes"] = path.stat().st_size
    reference["sha256"] = _sha256_file(path)
    return reference


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _public_llm_config(config: Any) -> dict[str, Any]:
    summary = getattr(config, "public_summary", None)
    if callable(summary):
        return dict(summary())
    return {"client_type": type(config).__name__ if config is not None else None}


def _modality_for_skill(skill_name: str) -> str:
    if skill_name.startswith("clinical."):
        return "clinical"
    if skill_name in {
        "pathology.conch_patch_roi",
        "pathology.ucec_conch_patch_roi",
        "pathology.npc_conch_patch_roi",
    }:
        return "wsi_patch_roi"
    if skill_name.startswith("pathology."):
        return "pathology_text"
    if skill_name == "radiology.lung_tumor_roi":
        return "ct_roi"
    if skill_name in {"radiology.ucec_mri_roi", "radiology.npc_mri_roi"}:
        return "mri_roi"
    if skill_name.startswith("radiology."):
        return "radiology"
    if skill_name.startswith("molecular."):
        return "molecular"
    if skill_name.startswith("guideline."):
        return "guideline"
    return "unknown"


def _supports_for_modality(modality: str) -> list[str]:
    mapping = {
        "clinical": ["case_information"],
        "pathology_text": ["diagnosis", "pathology_understanding"],
        "pathology_slide_metadata": ["pathology_context"],
        "wsi_patch_roi": ["pathology_image_review"],
        "ct_roi": ["radiology_review", "staging"],
        "mri_roi": ["radiology_review", "staging"],
        "molecular": ["biomarker_status"],
        "guideline": ["guideline_matching", "recommended_actions"],
    }
    return mapping.get(modality, ["benchmark_trace"])


def _image_role_priority(role: str) -> int:
    priorities = {
        "wsi_overview": 0,
        "patch_contact_sheet": 1,
        "patch_roi": 2,
        "roi": 3,
        "full_slice": 4,
        "overlay": 5,
    }
    return priorities.get(role, 99)
