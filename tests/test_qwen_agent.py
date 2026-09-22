from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import pytest

from medclaw.core.agent_loop import AgentLoop, AgentLoopError
from medclaw.core.evidence_board import EvidenceBoard
from medclaw.core.tool_router import ToolRouter
from medclaw.llm.qwen_client import QwenClient, QwenCompletion, QwenConfig, image_to_data_url
from medclaw.registry.skill_registry import SkillRegistry
from medclaw.runtime.runtime_manager import RuntimeManager
from medclaw.runtime.skill_runner import SkillRunner
from medclaw.stores.artifact_store import ArtifactStore
from medclaw.stores.audit_log import AuditLog
from medclaw.stores.conversation_log import ConversationLog


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeQwenClient:
    def __init__(self, completions: list[QwenCompletion]) -> None:
        self.completions = completions
        self.requests: list[dict[str, Any]] = []

    def build_user_message(
        self,
        text: str,
        *,
        image_paths: Iterable[Path] = (),
        image_urls: Iterable[str] = (),
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for path in image_paths:
            content.append(
                {"type": "image_url", "image_url": {"url": image_to_data_url(path)}}
            )
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": text})
        return {"role": "user", "content": content}

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
        tool_choice: str | Mapping[str, Any] = "auto",
    ) -> QwenCompletion:
        self.requests.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": [dict(tool) for tool in tools],
                "tool_choice": tool_choice,
            }
        )
        return self.completions.pop(0)


def build_agent(tmp_path: Path, llm_client: FakeQwenClient) -> tuple[AgentLoop, ToolRouter]:
    registry = SkillRegistry(PROJECT_ROOT / "medclaw" / "skills")
    runs_root = tmp_path / "runs"
    router = ToolRouter(
        registry,
        RuntimeManager(
            registry,
            SkillRunner(),
            ArtifactStore(tmp_path / "artifacts"),
            AuditLog(runs_root),
            runs_root,
        ),
    )
    agent = AgentLoop(
        router,
        EvidenceBoard("TCGA-38-4626", runs_root, load_existing=False),
        llm_client=llm_client,
    )
    return agent, router


def test_qwen_config_reads_key_without_exposing_it() -> None:
    config = QwenConfig.from_env({"DASHSCOPE_API_KEY": "secret-key"})

    assert config.model == "qwen3.7-plus"
    assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config.enable_thinking is True
    assert config.stream is True
    assert "secret-key" not in repr(config)
    assert "api_key" not in config.public_summary()


def test_qwen_client_builds_dashscope_streaming_tool_request() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.request: dict[str, Any] | None = None

        def create(self, **kwargs: Any) -> Any:
            self.request = kwargs
            return iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta={
                                    "role": "assistant",
                                    "reasoning_content": "thinking",
                                }
                            )
                        ],
                        model="qwen3.7-plus",
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta={"content": "done"})],
                        model="qwen3.7-plus",
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[],
                        model="qwen3.7-plus",
                        usage={"total_tokens": 3},
                    ),
                ]
            )

    completions = FakeCompletions()
    fake_openai = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = QwenClient(
        QwenConfig(api_key="secret-key"),
        client=fake_openai,
    )

    completion = client.complete(
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "medclaw__test",
                    "description": "test",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    assert completion.message["content"] == "done"
    assert completion.message["reasoning_content"] == "thinking"
    assert completion.model == "qwen3.7-plus"
    assert completion.usage == {"total_tokens": 3}
    assert completions.request is not None
    assert completions.request["model"] == "qwen3.7-plus"
    assert completions.request["tool_choice"] == "auto"
    assert completions.request["extra_body"] == {"enable_thinking": True}
    assert completions.request["stream"] is True
    assert completions.request["stream_options"] == {"include_usage": True}
    assert "secret-key" not in repr(completions.request)


def test_qwen_client_reassembles_streaming_tool_calls() -> None:
    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            return iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta={
                                    "role": "assistant",
                                    "reasoning_content": "use tool",
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "medclaw__radiology__",
                                                "arguments": '{"case_id":',
                                            },
                                        }
                                    ],
                                }
                            )
                        ],
                        model="qwen3.7-plus",
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta={
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "name": "lung_tumor_roi",
                                                "arguments": '"TCGA-38-4626"}',
                                            },
                                        }
                                    ]
                                }
                            )
                        ],
                        model="qwen3.7-plus",
                        usage=None,
                    ),
                ]
            )

    fake_openai = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    client = QwenClient(QwenConfig(api_key="secret-key"), client=fake_openai)

    completion = client.complete(messages=[{"role": "user", "content": "segment"}])

    assert completion.message == {
        "role": "assistant",
        "reasoning_content": "use tool",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "medclaw__radiology__lung_tumor_roi",
                    "arguments": '{"case_id":"TCGA-38-4626"}',
                },
            }
        ],
    }


def test_multimodal_agent_exposes_roi_tools_and_returns_answer(
    tmp_path: Path,
) -> None:
    fake_client = FakeQwenClient(
        [
            QwenCompletion(
                message={"role": "assistant", "content": "Ready to inspect CT."},
                model="qwen3.7-plus",
                usage={"total_tokens": 42},
            ),
        ]
    )
    agent, router = build_agent(tmp_path, fake_client)
    image_path = tmp_path / "example.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nmock")

    result = agent.chat(
        "Inspect this image and decide whether tools are needed.",
        image_paths=[image_path],
    )

    assert result.content == "Ready to inspect CT."
    assert result.model == "qwen3.7-plus"
    assert result.usage == {"total_tokens": 42}
    assert result.tool_results == ()
    assert len(agent.evidence_board.summary()["evidence"]) == 0

    tools = fake_client.requests[0]["tools"]
    tool_names = {tool["function"]["name"] for tool in tools}
    assert tool_names == {
        router.function_name_for_skill("guideline.retrieve"),
        router.function_name_for_skill("pathology.conch_patch_roi"),
        router.function_name_for_skill("pathology.npc_conch_patch_roi"),
        router.function_name_for_skill("pathology.ucec_conch_patch_roi"),
        router.function_name_for_skill("radiology.lung_tumor_roi"),
        router.function_name_for_skill("radiology.npc_mri_roi"),
        router.function_name_for_skill("radiology.ucec_mri_roi"),
    }

    user_content = fake_client.requests[0]["messages"][-1]["content"]
    assert user_content[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_generated_image_artifacts_are_sent_back_to_qwen_without_nifti(
    tmp_path: Path,
) -> None:
    skill_name = "radiology.lung_tumor_roi"
    function_name = ToolRouter.function_name_for_skill(skill_name)
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    artifact_specs = [
        ("image", "overlay", "overlay.png", b"overlay"),
        ("nifti", "largest_tumor_mask", "mask.nii.gz", b"nifti"),
        ("json", "roi_metadata", "metadata.json", b"json"),
        ("image", "roi", "roi.png", b"roi"),
        ("image", "full_slice", "full.png", b"full"),
    ]
    artifacts = []
    for artifact_type, role, filename, payload in artifact_specs:
        path = source_dir / filename
        path.write_bytes(payload)
        artifacts.append(
            {
                "type": artifact_type,
                "role": role,
                "uri": artifact_store.register_file(
                    path,
                    "TCGA-38-4626",
                    skill_name,
                    "call-test",
                    artifact_type,
                ),
            }
        )

    result = {
        "status": "success",
        "findings": {"summary": "ROI exported.", "tumor_detected": True},
        "artifacts": artifacts,
        "warnings": [],
    }

    class FakeImageToolRouter:
        def list_function_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": "Segment a lung tumor and export an ROI.",
                        "parameters": {
                            "type": "object",
                            "required": ["case_id"],
                            "properties": {"case_id": {"type": "string"}},
                        },
                    },
                }
            ]

        def skill_name_for_function(self, name: str) -> str:
            assert name == function_name
            return skill_name

        def call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
            assert name == skill_name
            assert arguments == {"case_id": "TCGA-38-4626"}
            return result

    fake_client = FakeQwenClient(
        [
            QwenCompletion(
                message={
                    "role": "assistant",
                    "reasoning_content": "private segmentation reasoning",
                    "tool_calls": [
                        {
                            "id": "tool-roi",
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": '{"case_id":"TCGA-38-4626"}',
                            },
                        }
                    ],
                },
                model="qwen3.7-plus",
            ),
            QwenCompletion(
                message={"role": "assistant", "content": "ROI images inspected."},
                model="qwen3.7-plus",
            ),
        ]
    )
    runs_root = tmp_path / "runs"
    agent = AgentLoop(
        FakeImageToolRouter(),  # type: ignore[arg-type]
        EvidenceBoard("TCGA-38-4626", runs_root, load_existing=False),
        llm_client=fake_client,
        artifact_uri_resolver=artifact_store.resolve_uri,
    )

    turn = agent.chat("Segment the tumor and inspect the generated images.")

    assert turn.content == "ROI images inspected."
    assert turn.tool_results[0]["skill_name"] == skill_name
    second_request_messages = fake_client.requests[1]["messages"]
    feedback = second_request_messages[-1]
    assert feedback["role"] == "user"
    content = feedback["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    text_parts = [part for part in content if part["type"] == "text"]
    assert len(image_parts) == 3
    assert len(text_parts) == 1
    assert "roi, full_slice, overlay" in text_parts[0]["text"]
    external_messages = repr(second_request_messages).lower()
    assert "nifti" not in external_messages
    assert "mask.nii.gz" not in external_messages
    assert "metadata.json" not in external_messages
    assert len(turn.tool_results[0]["result"]["artifacts"]) == 5

    log_text = agent.conversation_log_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in log_text.splitlines()]
    turn_record = next(
        record for record in records if record["event_type"] == "agent_turn"
    )
    message_records = [
        record for record in records if record["event_type"] == "conversation_message"
    ]
    assert turn_record["turn_id"] == turn.turn_id
    assert turn_record["status"] == "success"
    assert turn_record["final_answer"] == "ROI images inspected."
    assert turn_record["message_event_count"] == len(message_records)
    assert "messages" not in turn_record
    assert message_records[0]["role"] == "user"
    assert message_records[-1]["role"] == "assistant"
    assert message_records[-1]["message"]["content"] == "ROI images inspected."
    assert len(turn_record["tool_results"][0]["result"]["artifacts"]) == 5
    assert "mask.nii.gz" in log_text
    assert "data:image" not in log_text
    assert "private segmentation reasoning" not in log_text
    assert "reasoning_content_redacted" in log_text


def test_tool_round_limit_forces_no_tool_final_answer(tmp_path: Path) -> None:
    function_name = "medclaw__guideline__retrieve"
    skill_name = "guideline.retrieve"

    class FakeLoopingToolRouter:
        def list_function_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": "Retrieve guideline evidence.",
                        "parameters": {
                            "type": "object",
                            "required": ["case_id"],
                            "properties": {"case_id": {"type": "string"}},
                        },
                    },
                }
            ]

        def skill_name_for_function(self, name: str) -> str:
            assert name == function_name
            return skill_name

        def call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
            assert name == skill_name
            return {
                "status": "success",
                "findings": {
                    "summary": "Retrieved one chapter-level guideline snippet.",
                    "snippets": [{"chunk_id": "guideline::h1::0001"}],
                },
                "artifacts": [],
                "warnings": [],
            }

    fake_client = FakeQwenClient(
        [
            QwenCompletion(
                message={
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tool-guideline",
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": '{"case_id":"TCGA-38-4626"}',
                            },
                        }
                    ],
                },
                model="qwen3.7-plus",
            ),
            QwenCompletion(
                message={"role": "assistant", "content": "Forced final answer."},
                model="qwen3.7-plus",
            ),
        ]
    )
    runs_root = tmp_path / "runs"
    agent = AgentLoop(
        FakeLoopingToolRouter(),  # type: ignore[arg-type]
        EvidenceBoard("TCGA-38-4626", runs_root, load_existing=False),
        llm_client=fake_client,
        max_tool_rounds=1,
    )

    turn = agent.chat("Inspect the case.")

    assert turn.content == "Forced final answer."
    assert turn.tool_round_limit_reached is True
    assert turn.forced_finalization is True
    assert len(turn.tool_results) == 1
    assert fake_client.requests[1]["tools"] == []
    assert fake_client.requests[1]["tool_choice"] == "none"
    finalization_message = fake_client.requests[1]["messages"][-1]
    assert finalization_message["role"] == "user"
    assert "maximum tool-calling budget" in finalization_message["content"][-1]["text"]
    records = [
        json.loads(line)
        for line in agent.conversation_log_path.read_text(encoding="utf-8").splitlines()
    ]
    turn_record = records[-1]
    assert turn_record["event_type"] == "agent_turn"
    assert turn_record["status"] == "success"
    assert turn_record["tool_round_limit_reached"] is True
    assert turn_record["forced_finalization"] is True


def test_chat_compacts_full_user_prompt_after_auditing(tmp_path: Path) -> None:
    fake_client = FakeQwenClient(
        [
            QwenCompletion(
                message={"role": "assistant", "content": "Round evidence collected."},
                model="fake-qwen",
            )
        ]
    )
    agent, _ = build_agent(tmp_path, fake_client)
    full_prompt = "FULL_PATIENT_STATE_FOR_ROUND_1"
    marker = "Evidence acquisition round 1 completed."

    agent.chat(full_prompt, retained_user_text=marker)

    sent_messages = json.dumps(
        fake_client.requests[0]["messages"],
        ensure_ascii=False,
    )
    retained_messages = json.dumps(agent.messages, ensure_ascii=False)
    assert full_prompt in sent_messages
    assert marker not in sent_messages
    assert full_prompt not in retained_messages
    assert marker in retained_messages

    records = [
        json.loads(line)
        for line in agent.conversation_log_path.read_text(encoding="utf-8").splitlines()
    ]
    original_message = next(
        record
        for record in records
        if record["event_type"] == "conversation_message"
        and record["role"] == "user"
    )
    compaction = next(
        record
        for record in records
        if record["event_type"] == "conversation_compaction"
    )
    assert full_prompt in json.dumps(original_message, ensure_ascii=False)
    assert marker in json.dumps(compaction["retained_message"], ensure_ascii=False)
    assert compaction["original_user_text"]["length"] == len(full_prompt)
    assert len(compaction["original_user_text"]["sha256"]) == 64


def test_finalize_reuses_full_tool_history_without_tools(tmp_path: Path) -> None:
    function_name = "medclaw__guideline__retrieve"
    skill_name = "guideline.retrieve"

    class GuidelineRouter:
        def list_function_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": "Retrieve guideline evidence.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

        def skill_name_for_function(self, name: str) -> str:
            assert name == function_name
            return skill_name

        def call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
            assert name == skill_name
            return {
                "status": "success",
                "call_id": "guideline-call-1",
                "findings": {
                    "summary": "NCCN evidence retrieved.",
                    "snippets": [
                        {
                            "guideline": "NCCN NSCLC 2010",
                            "page": 42,
                            "excerpt": "Auditable guideline excerpt.",
                        }
                    ],
                },
                "artifacts": [],
                "warnings": [],
            }

    fake_client = FakeQwenClient(
        [
            QwenCompletion(
                message={
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "guideline-tool-call",
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                model="fake-qwen",
            ),
            QwenCompletion(
                message={"role": "assistant", "content": "Round guideline summary."},
                model="fake-qwen",
            ),
            QwenCompletion(
                message={"role": "assistant", "content": "Audited final answer."},
                model="fake-qwen",
            ),
        ]
    )
    agent = AgentLoop(
        GuidelineRouter(),  # type: ignore[arg-type]
        EvidenceBoard("TCGA-38-4626", tmp_path / "runs", load_existing=False),
        llm_client=fake_client,
    )
    agent.chat(
        "OLD_FULL_PATIENT_STATE",
        retained_user_text="Evidence acquisition round 1 completed.",
    )

    final = agent.finalize("LATEST_PATIENT_STATE_ONCE")

    assert final.content == "Audited final answer."
    final_request = fake_client.requests[-1]
    assert final_request["tools"] == []
    assert final_request["tool_choice"] == "none"
    serialized = json.dumps(final_request["messages"], ensure_ascii=False)
    assert "OLD_FULL_PATIENT_STATE" not in serialized
    assert serialized.count("LATEST_PATIENT_STATE_ONCE") == 1
    assert "Round guideline summary." in serialized
    assert "NCCN NSCLC 2010" in serialized
    assert "Auditable guideline excerpt." in serialized
    tool_payloads = [
        json.loads(message["content"])
        for message in final_request["messages"]
        if message.get("role") == "tool"
    ]
    assert tool_payloads[0]["findings"]["snippets"][0]["page"] == 42


def test_finalize_rejects_empty_model_response(tmp_path: Path) -> None:
    agent, _ = build_agent(
        tmp_path,
        FakeQwenClient(
            [
                QwenCompletion(
                    message={"role": "assistant", "content": "   "},
                    model="fake-qwen",
                )
            ]
        ),
    )

    with pytest.raises(AgentLoopError, match="did not contain usable text"):
        agent.finalize("Generate the final answer.")

    records = [
        json.loads(line)
        for line in agent.conversation_log_path.read_text(encoding="utf-8").splitlines()
    ]
    turn_record = records[-1]
    assert turn_record["event_type"] == "agent_turn"
    assert turn_record["status"] == "failed"
    assert turn_record["error"]["type"] == "AgentLoopError"


def test_conversation_log_redacts_secrets_and_inline_images(tmp_path: Path) -> None:
    log = ConversationLog(tmp_path / "runs")
    secret = "sk-secretvalue123"
    log.append(
        "TCGA-38-4626",
        {
            "event_type": "agent_turn",
            "api_key": secret,
            "content": f"DASHSCOPE_API_KEY={secret}",
            "local_config": f"MEDCLAW_LOCAL_API_KEY={secret}",
            "reasoning_content": "do not persist this",
            "image": "data:image/png;base64,aW1hZ2U=",
        },
    )

    text = log.path_for_case("TCGA-38-4626").read_text(encoding="utf-8")
    record = json.loads(text)
    assert secret not in text
    assert "do not persist this" not in text
    assert "data:image" not in text
    assert record["api_key"] == "<redacted>"
    assert record["local_config"].endswith("<redacted>")
    assert record["reasoning_content_redacted"] is True
    assert record["image"]["sha256"]


def test_failed_model_turn_is_audited(tmp_path: Path) -> None:
    class FailingQwenClient(FakeQwenClient):
        def complete(
            self,
            *,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[Mapping[str, Any]] = (),
            tool_choice: str | Mapping[str, Any] = "auto",
        ) -> QwenCompletion:
            raise RuntimeError("simulated API failure")

    agent, _ = build_agent(tmp_path, FailingQwenClient([]))

    with pytest.raises(RuntimeError, match="simulated API failure"):
        agent.chat("Inspect the case.")

    records = [
        json.loads(line)
        for line in agent.conversation_log_path.read_text(encoding="utf-8").splitlines()
    ]
    turn_record = records[-1]
    assert turn_record["event_type"] == "agent_turn"
    assert turn_record["status"] == "failed"
    assert turn_record["message_event_count"] == 1
    assert "messages" not in turn_record
    assert turn_record["error"]["type"] == "RuntimeError"
    assert turn_record["error"]["message"] == "simulated API failure"
