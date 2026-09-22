"""Deterministic and Qwen-driven agent loops."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4

from medclaw.core.evidence_board import EvidenceBoard
from medclaw.core.tool_router import ToolRouter, ToolRoutingError
from medclaw.llm.protocols import ChatCompletion, ChatModel
from medclaw.stores.conversation_log import ConversationLog
from medclaw.utils import utc_now


DEFAULT_SYSTEM_PROMPT = """You are MedClaw, a multimodal medical benchmark agent.
You can inspect user-provided images directly and call registered MedClaw tools to
collect reproducible evidence. Do not fabricate tool outputs. Use tools when they
are needed, summarize uncertainty, and clearly distinguish mock evidence from
real clinical evidence. When using guideline evidence, cite the returned
guideline title/source and page number in the answer; if a snippet has no page
number, state that the page number was not provided. 最终回答中必须明确写出指南依据
及页码。This runtime is for benchmark research, not clinical care. The current
case_id is {case_id}; pass this exact value to tools that require case_id."""


class AgentLoopError(RuntimeError):
    """Raised when the model-driven agent loop cannot complete a turn."""


@dataclass(frozen=True)
class AgentTurnResult:
    """Final answer and tool activity for one user turn."""

    content: str
    tool_results: tuple[dict[str, Any], ...]
    model: str | None
    usage: dict[str, Any] | None
    turn_id: str
    conversation_log_path: str
    raw_message: dict[str, Any] | None = None
    tool_round_limit_reached: bool = False
    forced_finalization: bool = False


class AgentLoop:
    """Run explicit plans or let a multimodal chat model choose skills."""

    def __init__(
        self,
        tool_router: ToolRouter,
        evidence_board: EvidenceBoard,
        *,
        llm_client: ChatModel | None = None,
        system_prompt: str | None = None,
        max_tool_rounds: int = 8,
        artifact_uri_resolver: Callable[[str], Path] | None = None,
        conversation_log: ConversationLog | None = None,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        self.tool_router = tool_router
        self.evidence_board = evidence_board
        self.llm_client = llm_client
        self.max_tool_rounds = max_tool_rounds
        self.artifact_uri_resolver = artifact_uri_resolver
        self.conversation_log = conversation_log or ConversationLog(
            evidence_board.runs_root
        )
        self.conversation_id = f"conversation_{uuid4().hex}"
        self._conversation_started_logged = False
        prompt = system_prompt or DEFAULT_SYSTEM_PROMPT.format(case_id=evidence_board.case_id)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": prompt}]

    def call_skill(self, skill_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        result = self.tool_router.call(skill_name, arguments)
        self.evidence_board.add_result(skill_name, result)
        return result

    def run_plan(self, calls: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for call in calls:
            results.append(self.call_skill(str(call["skill_name"]), call["arguments"]))
        return results

    def chat(
        self,
        user_text: str,
        *,
        image_paths: Iterable[Path] = (),
        image_urls: Iterable[str] = (),
        retained_user_text: str | None = None,
    ) -> AgentTurnResult:
        """Run one multimodal conversation turn with model-selected tools.

        When ``retained_user_text`` is provided, the model sees ``user_text``
        throughout this turn, while later turns retain only the replacement
        text. The complete original prompt remains in the conversation audit.
        """

        self._ensure_conversation_started()
        turn_id = f"turn_{uuid4().hex}"
        started_at = utc_now()
        message_start_index = len(self.messages)
        paths = [Path(path) for path in image_paths]
        urls = list(image_urls)
        user_input = {
            "text": user_text,
            "local_images": [self._local_image_reference(path) for path in paths],
            "image_urls": urls,
        }
        tool_results: list[dict[str, Any]] = []
        model_calls: list[dict[str, Any]] = []
        turn_result: AgentTurnResult | None = None

        try:
            if self.llm_client is None:
                raise AgentLoopError("No LLM client is configured for this AgentLoop.")

            user_message = self.llm_client.build_user_message(
                user_text,
                image_paths=paths,
                image_urls=urls,
            )
            self.messages.append(user_message)
            tools = self.tool_router.list_function_tools()

            for round_index in range(self.max_tool_rounds):
                completion = self.llm_client.complete(
                    messages=self.messages,
                    tools=tools,
                    tool_choice="auto",
                )
                model_calls.append(
                    {
                        "round_index": round_index,
                        "model": completion.model,
                        "usage": completion.usage,
                    }
                )
                assistant_message = dict(completion.message)
                assistant_message.setdefault("role", "assistant")
                self.messages.append(assistant_message)

                tool_calls = assistant_message.get("tool_calls") or []
                if not tool_calls:
                    content = assistant_message.get("content")
                    turn_result = AgentTurnResult(
                        content=content if isinstance(content, str) else "",
                        tool_results=tuple(tool_results),
                        model=completion.model,
                        usage=completion.usage,
                        turn_id=turn_id,
                        conversation_log_path=str(self.conversation_log_path),
                        raw_message=assistant_message,
                    )
                    break

                if not isinstance(tool_calls, list):
                    raise AgentLoopError("Model response field 'tool_calls' must be a list.")

                image_artifacts: list[Mapping[str, Any]] = []
                for tool_call in tool_calls:
                    tool_record, tool_message = self._execute_tool_call(tool_call)
                    tool_results.append(tool_record)
                    self.messages.append(tool_message)
                    image_artifacts.extend(self._image_artifacts(tool_record["result"]))

                artifact_message = self._build_artifact_feedback_message(image_artifacts)
                if artifact_message is not None:
                    self.messages.append(artifact_message)

            if turn_result is None:
                turn_result = self._finalize_after_tool_round_limit(
                    turn_id=turn_id,
                    tool_results=tool_results,
                    model_calls=model_calls,
                )
        except Exception as exc:
            self._record_turn(
                turn_id=turn_id,
                status="failed",
                started_at=started_at,
                message_start_index=message_start_index,
                user_input=user_input,
                tool_results=tool_results,
                model_calls=model_calls,
                error=exc,
            )
            self._compact_turn_user_message(
                turn_id=turn_id,
                message_index=message_start_index,
                original_user_text=user_text,
                retained_user_text=retained_user_text,
            )
            raise

        self._record_turn(
            turn_id=turn_id,
            status="success",
            started_at=started_at,
            message_start_index=message_start_index,
            user_input=user_input,
            tool_results=tool_results,
            model_calls=model_calls,
            turn_result=turn_result,
        )
        self._compact_turn_user_message(
            turn_id=turn_id,
            message_index=message_start_index,
            original_user_text=user_text,
            retained_user_text=retained_user_text,
        )
        return turn_result

    def finalize(self, user_text: str) -> AgentTurnResult:
        """Generate a final answer from the retained history without tools."""

        self._ensure_conversation_started()
        turn_id = f"turn_{uuid4().hex}"
        started_at = utc_now()
        message_start_index = len(self.messages)
        user_input = {
            "text": user_text,
            "local_images": [],
            "image_urls": [],
        }
        model_calls: list[dict[str, Any]] = []
        turn_result: AgentTurnResult | None = None

        try:
            if self.llm_client is None:
                raise AgentLoopError("No LLM client is configured for this AgentLoop.")

            self.messages.append(self.llm_client.build_user_message(user_text))
            completion = self.llm_client.complete(
                messages=self.messages,
                tools=[],
                tool_choice="none",
            )
            model_calls.append(
                {
                    "round_index": 0,
                    "model": completion.model,
                    "usage": completion.usage,
                    "finalization": True,
                    "reason": "explicit_no_tools",
                }
            )
            assistant_message = dict(completion.message)
            assistant_message.setdefault("role", "assistant")
            self.messages.append(assistant_message)
            content = assistant_message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise AgentLoopError(
                    "Final no-tools model response did not contain usable text."
                )
            turn_result = AgentTurnResult(
                content=content,
                tool_results=(),
                model=completion.model,
                usage=completion.usage,
                turn_id=turn_id,
                conversation_log_path=str(self.conversation_log_path),
                raw_message=assistant_message,
            )
        except Exception as exc:
            self._record_turn(
                turn_id=turn_id,
                status="failed",
                started_at=started_at,
                message_start_index=message_start_index,
                user_input=user_input,
                tool_results=(),
                model_calls=model_calls,
                error=exc,
            )
            raise

        self._record_turn(
            turn_id=turn_id,
            status="success",
            started_at=started_at,
            message_start_index=message_start_index,
            user_input=user_input,
            tool_results=(),
            model_calls=model_calls,
            turn_result=turn_result,
        )
        return turn_result

    def reset_conversation(self, *, keep_system_prompt: bool = True) -> None:
        """Clear dialogue history while optionally preserving the system prompt."""

        message_count_before = len(self.messages)
        if keep_system_prompt and self.messages:
            self.messages = [self.messages[0]]
        else:
            self.messages = []
        if self._conversation_started_logged:
            self._append_conversation_event(
                {
                    "event_type": "conversation_reset",
                    "timestamp": utc_now(),
                    "keep_system_prompt": keep_system_prompt,
                    "message_count_before": message_count_before,
                    "message_count_after": len(self.messages),
                }
            )

    @property
    def conversation_log_path(self) -> Path:
        """Return the durable conversation audit log path for this case."""

        return self.conversation_log.path_for_case(self.evidence_board.case_id)

    @property
    def message_count(self) -> int:
        """Return the number of messages retained in model context."""

        return len(self.messages)

    def _ensure_conversation_started(self) -> None:
        if self._conversation_started_logged:
            return
        self._append_conversation_event(
            {
                "event_type": "conversation_started",
                "started_at": utc_now(),
                "system_message": self.messages[0] if self.messages else None,
                "llm": self._llm_audit_summary(),
            }
        )
        self._conversation_started_logged = True

    def _record_turn(
        self,
        *,
        turn_id: str,
        status: str,
        started_at: str,
        message_start_index: int,
        user_input: Mapping[str, Any],
        tool_results: Sequence[Mapping[str, Any]],
        model_calls: Sequence[Mapping[str, Any]],
        turn_result: AgentTurnResult | None = None,
        error: Exception | None = None,
    ) -> None:
        finished_at = utc_now()
        messages = self.messages[message_start_index:]
        for offset, message in enumerate(messages, start=1):
            role = message.get("role") if isinstance(message, Mapping) else None
            name = message.get("name") if isinstance(message, Mapping) else None
            self._append_conversation_event(
                {
                    "event_type": "conversation_message",
                    "turn_id": turn_id,
                    "status": status,
                    "message_index": offset,
                    "absolute_message_index": message_start_index + offset,
                    "role": role,
                    "name": name,
                    "message": message,
                    "recorded_at": finished_at,
                }
            )

        record: dict[str, Any] = {
            "event_type": "agent_turn",
            "turn_id": turn_id,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "message_count_before": message_start_index,
            "message_count_after": len(self.messages),
            "user_input": dict(user_input),
            "message_event_count": len(messages),
            "message_event_type": "conversation_message",
            "model_calls": list(model_calls),
            "tool_results": list(tool_results),
            "final_answer": turn_result.content if turn_result else None,
            "model": turn_result.model if turn_result else None,
            "usage": turn_result.usage if turn_result else None,
            "tool_round_limit_reached": (
                turn_result.tool_round_limit_reached if turn_result else False
            ),
            "forced_finalization": (
                turn_result.forced_finalization if turn_result else False
            ),
        }
        if error is not None:
            record["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        self._append_conversation_event(record)

    def _compact_turn_user_message(
        self,
        *,
        turn_id: str,
        message_index: int,
        original_user_text: str,
        retained_user_text: str | None,
    ) -> None:
        if retained_user_text is None or self.llm_client is None:
            return
        if message_index < 0 or message_index >= len(self.messages):
            return
        original_message = self.messages[message_index]
        if original_message.get("role") != "user":
            return

        replacement = self.llm_client.build_user_message(retained_user_text)
        self.messages[message_index] = replacement
        self._append_conversation_event(
            {
                "event_type": "conversation_compaction",
                "turn_id": turn_id,
                "timestamp": utc_now(),
                "strategy": "replace_turn_user_message",
                "absolute_message_index": message_index + 1,
                "reason": "retain_round_marker_after_full_prompt_audit",
                "original_user_text": {
                    "length": len(original_user_text),
                    "sha256": hashlib.sha256(
                        original_user_text.encode("utf-8")
                    ).hexdigest(),
                },
                "retained_message": replacement,
            }
        )

    def _finalize_after_tool_round_limit(
        self,
        *,
        turn_id: str,
        tool_results: Sequence[Mapping[str, Any]],
        model_calls: list[dict[str, Any]],
    ) -> AgentTurnResult:
        if self.llm_client is None:
            raise AgentLoopError("No LLM client is configured for this AgentLoop.")

        finalization_prompt = (
            f"The maximum tool-calling budget of {self.max_tool_rounds} rounds has "
            "been reached. Do not call any more tools. Produce a final answer now "
            "using only the audited tool results, image-review summaries, and "
            "conversation context already available. Explicitly state uncertainty "
            "and missing information instead of requesting more tools."
        )
        self.messages.append(self.llm_client.build_user_message(finalization_prompt))
        completion = self.llm_client.complete(
            messages=self.messages,
            tools=[],
            tool_choice="none",
        )
        model_calls.append(
            {
                "round_index": self.max_tool_rounds,
                "model": completion.model,
                "usage": completion.usage,
                "finalization": True,
                "reason": "tool_round_limit",
            }
        )
        assistant_message = dict(completion.message)
        assistant_message.setdefault("role", "assistant")
        self.messages.append(assistant_message)
        content = assistant_message.get("content")
        return AgentTurnResult(
            content=content if isinstance(content, str) else "",
            tool_results=tuple(dict(item) for item in tool_results),
            model=completion.model,
            usage=completion.usage,
            turn_id=turn_id,
            conversation_log_path=str(self.conversation_log_path),
            raw_message=assistant_message,
            tool_round_limit_reached=True,
            forced_finalization=True,
        )

    def _append_conversation_event(self, event: Mapping[str, Any]) -> None:
        record = {
            "schema_version": "1.0",
            "case_id": self.evidence_board.case_id,
            "conversation_id": self.conversation_id,
            **dict(event),
        }
        self.conversation_log.append(self.evidence_board.case_id, record)

    def _llm_audit_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "client_type": (
                type(self.llm_client).__name__ if self.llm_client is not None else None
            )
        }
        config = getattr(self.llm_client, "config", None)
        public_summary = getattr(config, "public_summary", None)
        if callable(public_summary):
            summary["config"] = public_summary()
        return summary

    @staticmethod
    def _local_image_reference(path: Path) -> dict[str, Any]:
        resolved = Path(path).resolve()
        reference: dict[str, Any] = {"path": str(resolved)}
        try:
            if not resolved.is_file():
                reference["exists"] = False
                return reference
            reference.update(
                {
                    "exists": True,
                    "size_bytes": resolved.stat().st_size,
                    "sha256": AgentLoop._sha256_file(resolved),
                }
            )
        except OSError as exc:
            reference["error"] = str(exc)
        return reference

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _execute_tool_call(
        self,
        tool_call: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(tool_call, Mapping):
            raise AgentLoopError("Each model tool call must be an object.")

        tool_call_id = tool_call.get("id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            tool_call_id = f"tool_call_{uuid4().hex}"

        function = tool_call.get("function")
        if not isinstance(function, Mapping):
            result = self._tool_error("Model tool call is missing a function object.")
            return (
                {
                    "function_name": None,
                    "skill_name": None,
                    "arguments": {},
                    "result": result,
                },
                self._tool_message(tool_call_id, "unknown", result),
            )

        function_name = function.get("name")
        if not isinstance(function_name, str) or not function_name:
            result = self._tool_error("Model tool call is missing a function name.")
            return (
                {
                    "function_name": None,
                    "skill_name": None,
                    "arguments": {},
                    "result": result,
                },
                self._tool_message(tool_call_id, "unknown", result),
            )

        arguments: dict[str, Any] = {}
        skill_name: str | None = None
        try:
            arguments = self._parse_tool_arguments(function.get("arguments", "{}"))
            skill_name = self.tool_router.skill_name_for_function(function_name)
            result = self.call_skill(skill_name, arguments)
        except (ToolRoutingError, ValueError) as exc:
            result = self._tool_error(str(exc))

        return (
            {
                "function_name": function_name,
                "skill_name": skill_name,
                "arguments": arguments,
                "result": result,
            },
            self._tool_message(
                tool_call_id,
                function_name,
                self._result_for_model(result),
            ),
        )

    @staticmethod
    def _parse_tool_arguments(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        if not isinstance(value, str):
            raise ValueError("Tool arguments must be a JSON object or JSON string.")
        try:
            arguments = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool arguments are not valid JSON: {exc.msg}") from exc
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must decode to a JSON object.")
        return arguments

    @staticmethod
    def _tool_message(
        tool_call_id: str,
        function_name: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": function_name,
            "content": json.dumps(result, ensure_ascii=False, sort_keys=True),
        }

    @staticmethod
    def _tool_error(message: str) -> dict[str, Any]:
        return {
            "status": "failed",
            "findings": {"error": message},
            "artifacts": [],
            "warnings": [message],
        }

    @staticmethod
    def _result_for_model(result: Mapping[str, Any]) -> dict[str, Any]:
        """Remove non-image artifact references from external model messages."""

        filtered = dict(result)
        artifacts = result.get("artifacts", [])
        if isinstance(artifacts, list):
            filtered["artifacts"] = [
                dict(artifact)
                for artifact in artifacts
                if isinstance(artifact, Mapping) and artifact.get("type") == "image"
            ]
        else:
            filtered["artifacts"] = []
        return filtered

    @staticmethod
    def _image_artifacts(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        artifacts = result.get("artifacts", [])
        if not isinstance(artifacts, list):
            return []
        return [
            artifact
            for artifact in artifacts
            if isinstance(artifact, Mapping) and artifact.get("type") == "image"
        ]

    def _build_artifact_feedback_message(
        self,
        artifacts: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        if not artifacts or self.artifact_uri_resolver is None or self.llm_client is None:
            return None

        role_priority = {
            "wsi_overview": 0,
            "patch_contact_sheet": 1,
            "patch_roi": 2,
            "roi": 3,
            "full_slice": 4,
            "overlay": 5,
        }
        ordered = sorted(
            artifacts,
            key=lambda artifact: role_priority.get(str(artifact.get("role")), 99),
        )
        image_limit = getattr(
            getattr(self.llm_client, "config", None),
            "max_images_per_message",
            None,
        )
        if isinstance(image_limit, int) and image_limit > 0:
            ordered = ordered[:image_limit]
        paths: list[Path] = []
        labels: list[str] = []
        for artifact in ordered:
            uri = artifact.get("uri")
            if not isinstance(uri, str) or not uri.startswith("artifact://"):
                continue
            try:
                path = self.artifact_uri_resolver(uri)
            except (FileNotFoundError, OSError, ValueError):
                continue
            paths.append(path)
            labels.append(str(artifact.get("role") or path.name))

        if not paths:
            return None
        text = (
            "The tool generated the following image artifacts in this order: "
            f"{', '.join(labels)}. Inspect these images before producing the final answer. "
            "The ROI is derived from a research segmentation model and is not clinical advice."
        )
        return self.llm_client.build_user_message(text, image_paths=paths)
