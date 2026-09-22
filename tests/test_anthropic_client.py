"""Tests for Anthropic message conversion used by the Claude client."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from medclaw.llm.anthropic_client import (
    AnthropicClient,
    AnthropicConfig,
    _from_anthropic_response,
    _to_anthropic_messages,
)


def test_to_anthropic_messages_converts_assistant_tool_calls() -> None:
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Call a tool"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "toolu_abc123",
                    "type": "function",
                    "function": {
                        "name": "lookup_case",
                        "arguments": json.dumps({"case_id": "CASE-1"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_abc123",
            "name": "lookup_case",
            "content": json.dumps({"status": "success"}),
        },
    ]

    system_prompt, converted = _to_anthropic_messages(messages)

    assert system_prompt == "You are helpful."
    assert len(converted) == 3
    assert converted[0] == {"role": "user", "content": [{"type": "text", "text": "Call a tool"}]}
    assert converted[1]["role"] == "assistant"
    assert converted[1]["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_abc123",
            "name": "lookup_case",
            "input": {"case_id": "CASE-1"},
        }
    ]
    assert converted[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_abc123",
                "content": json.dumps({"status": "success"}),
            }
        ],
    }


def test_to_anthropic_messages_merges_multiple_tool_results() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "Running tools",
            "tool_calls": [
                {
                    "id": "toolu_one",
                    "type": "function",
                    "function": {"name": "tool_a", "arguments": "{}"},
                },
                {
                    "id": "toolu_two",
                    "type": "function",
                    "function": {"name": "tool_b", "arguments": '{"x": 1}'},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_one",
            "name": "tool_a",
            "content": '{"status": "ok"}',
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_two",
            "name": "tool_b",
            "content": '{"status": "ok"}',
        },
    ]

    _, converted = _to_anthropic_messages(messages)

    assert converted[0]["content"][0] == {"type": "text", "text": "Running tools"}
    assert converted[0]["content"][1]["id"] == "toolu_one"
    assert converted[0]["content"][2]["id"] == "toolu_two"
    assert converted[1]["role"] == "user"
    assert len(converted[1]["content"]) == 2
    assert converted[1]["content"][0]["tool_use_id"] == "toolu_one"
    assert converted[1]["content"][1]["tool_use_id"] == "toolu_two"


def test_from_anthropic_response_round_trip_tool_calls() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Need data"),
            SimpleNamespace(
                type="tool_use",
                id="toolu_xyz",
                name="fetch",
                input={"case_id": "CASE-2"},
            ),
        ],
        model="claude-test",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )

    message = _from_anthropic_response(response)

    assert message["content"] == "Need data"
    assert message["tool_calls"][0]["id"] == "toolu_xyz"
    assert message["tool_calls"][0]["function"]["name"] == "fetch"

    _, converted = _to_anthropic_messages(
        [
            {"role": "user", "content": "go"},
            message,
            {
                "role": "tool",
                "tool_call_id": "toolu_xyz",
                "name": "fetch",
                "content": '{"status": "success"}',
            },
        ]
    )

    assert converted[1]["content"][-1]["type"] == "tool_use"
    assert converted[1]["content"][-1]["id"] == "toolu_xyz"
    assert converted[2]["content"][0]["tool_use_id"] == "toolu_xyz"


def test_to_anthropic_messages_parallel_tool_calls_from_failed_run() -> None:
    """Regression for TCGA-38-4625 claude run: text + 3 parallel tool_calls."""

    messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": (
                "I'll systematically gather evidence for this case. "
                "Let me start by collecting clinical, pathology, and molecular data simultaneously."
            ),
            "tool_calls": [
                {
                    "id": "toolu_0187G6uWZoDgii3np3DntwFV",
                    "type": "function",
                    "function": {
                        "name": "medclaw__clinical__read_summary",
                        "arguments": '{"case_id": "TCGA-38-4625"}',
                    },
                },
                {
                    "id": "toolu_01LHRXDqkvqLvhZtVeWVgFhm",
                    "type": "function",
                    "function": {
                        "name": "medclaw__pathology__read_report",
                        "arguments": '{"case_id": "TCGA-38-4625"}',
                    },
                },
                {
                    "id": "toolu_01HchBPRqCa39c1iEWMTbomZ",
                    "type": "function",
                    "function": {
                        "name": "medclaw__molecular__query_biomarkers",
                        "arguments": '{"case_id": "TCGA-38-4625"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_0187G6uWZoDgii3np3DntwFV",
            "name": "medclaw__clinical__read_summary",
            "content": '{"status": "success"}',
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_01LHRXDqkvqLvhZtVeWVgFhm",
            "name": "medclaw__pathology__read_report",
            "content": '{"status": "success"}',
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_01HchBPRqCa39c1iEWMTbomZ",
            "name": "medclaw__molecular__query_biomarkers",
            "content": '{"status": "success"}',
        },
    ]

    _, converted = _to_anthropic_messages(messages)

    assert converted[1]["content"][0]["type"] == "text"
    assert [block["id"] for block in converted[1]["content"] if block["type"] == "tool_use"] == [
        "toolu_0187G6uWZoDgii3np3DntwFV",
        "toolu_01LHRXDqkvqLvhZtVeWVgFhm",
        "toolu_01HchBPRqCa39c1iEWMTbomZ",
    ]
    assert len(converted[2]["content"]) == 3
    assert all(block["type"] == "tool_result" for block in converted[2]["content"])


def test_complete_sends_tool_use_before_tool_result() -> None:
    mock_api = MagicMock()
    mock_api.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="done")],
        model="claude-test",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    client = AnthropicClient(
        AnthropicConfig(provider="claude", api_key="test-key", model="claude-test"),
        client=mock_api,
    )

    client.complete(
        messages=[
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "toolu_roundtrip",
                        "type": "function",
                        "function": {
                            "name": "demo_tool",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_roundtrip",
                "name": "demo_tool",
                "content": '{"ok": true}',
            },
        ],
        tools=[],
    )

    request = mock_api.messages.create.call_args.kwargs
    messages = request["messages"]
    assert messages[1]["content"][0]["type"] == "tool_use"
    assert messages[1]["content"][0]["id"] == "toolu_roundtrip"
    assert messages[2]["content"][0]["tool_use_id"] == "toolu_roundtrip"
