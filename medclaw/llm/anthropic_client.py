"""Anthropic-native chat client implementing the shared ChatModel protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from medclaw.llm.image_utils import image_to_data_url, is_supported_image_url
from medclaw.llm.protocols import ChatCompletion, LLMAPIError, LLMConfigurationError
from medclaw.llm.retry_utils import call_with_llm_retries


@dataclass(frozen=True)
class AnthropicConfig:
    """Configuration for the Anthropic Claude API client."""

    provider: str
    api_key: str
    model: str
    timeout_sec: float = 120.0
    max_tokens: int = 8192

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise LLMConfigurationError(f"{self.provider} API key must not be empty.")
        if not self.model.strip():
            raise LLMConfigurationError(f"{self.provider} model name must not be empty.")
        if self.timeout_sec <= 0:
            raise LLMConfigurationError(f"{self.provider} timeout must be greater than zero.")
        if self.max_tokens <= 0:
            raise LLMConfigurationError(f"{self.provider} max_tokens must be greater than zero.")

    def public_summary(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "timeout_sec": self.timeout_sec,
            "max_tokens": self.max_tokens,
        }


class AnthropicClient:
    """Call Claude through the Anthropic Messages API."""

    def __init__(self, config: AnthropicConfig, *, client: Any | None = None) -> None:
        self.config = config
        if client is not None:
            self._client = client
            return
        try:
            import anthropic
        except ImportError as exc:
            raise LLMConfigurationError(
                "Anthropic provider requires the optional 'anthropic' package."
            ) from exc
        self._client = anthropic.Anthropic(
            api_key=config.api_key,
            timeout=config.timeout_sec,
        )

    def build_user_message(
        self,
        text: str,
        *,
        image_paths: Iterable[Path] = (),
        image_urls: Iterable[str] = (),
    ) -> dict[str, Any]:
        paths = [Path(path) for path in image_paths]
        urls = list(image_urls)
        if not text.strip() and not paths and not urls:
            raise ValueError("A user message must contain text or at least one image.")

        if not paths and not urls:
            return {"role": "user", "content": text}

        content: list[dict[str, Any]] = []
        for path in paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(path)},
                }
            )
        for url in urls:
            if not is_supported_image_url(url):
                raise ValueError(
                    "Image URLs must use HTTPS or be a valid base64 data URL."
                )
            content.append({"type": "image_url", "image_url": {"url": url}})
        if text.strip():
            content.append({"type": "text", "text": text})
        return {"role": "user", "content": content}

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
        tool_choice: str | Mapping[str, Any] = "auto",
    ) -> ChatCompletion:
        system_prompt, anthropic_messages = _to_anthropic_messages(messages)
        request: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": anthropic_messages,
        }
        if system_prompt:
            request["system"] = system_prompt
        if tools:
            request["tools"] = [_to_anthropic_tool(tool) for tool in tools]
            request["tool_choice"] = _to_anthropic_tool_choice(tool_choice)

        provider = self.config.provider

        def _create() -> ChatCompletion:
            response = self._client.messages.create(**request)
            message = _from_anthropic_response(response)
            usage = _anthropic_usage(response)
            return ChatCompletion(
                message=message,
                model=getattr(response, "model", None),
                usage=usage,
            )

        return call_with_llm_retries(_create, provider=provider)


def _to_anthropic_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            converted.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        if role == "system":
            if isinstance(content, str) and content.strip():
                system_parts.append(content)
            continue
        if role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": str(message.get("tool_call_id", "")),
                    "content": _stringify_content(content),
                }
            )
            continue

        flush_tool_results()

        if role not in {"user", "assistant"}:
            role = "user"
        blocks = _to_anthropic_content_blocks(content, role=role)
        if role == "assistant":
            tool_use_blocks = _tool_calls_to_tool_use_blocks(message.get("tool_calls"))
            if tool_use_blocks:
                blocks = [
                    block
                    for block in blocks
                    if not (
                        block.get("type") == "text"
                        and block.get("text") in {"", "null"}
                    )
                ]
                blocks.extend(tool_use_blocks)
        if not blocks:
            blocks = [{"type": "text", "text": ""}]
        converted.append({"role": role, "content": blocks})

    flush_tool_results()
    return ("\n\n".join(system_parts) or None, converted)


def _tool_calls_to_tool_use_blocks(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []

    blocks: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, Mapping):
            continue
        function = tool_call.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        tool_id = tool_call.get("id")
        if not isinstance(tool_id, str) or not tool_id:
            continue
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                input_data = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                input_data = {}
        elif isinstance(arguments, Mapping):
            input_data = dict(arguments)
        else:
            input_data = {}
        if not isinstance(input_data, dict):
            input_data = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": input_data,
            }
        )
    return blocks


def _to_anthropic_content_blocks(content: Any, *, role: str) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if not isinstance(content, list):
        return [{"type": "text", "text": _stringify_content(content)}]

    blocks: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                blocks.append({"type": "text", "text": text})
            continue
        if item_type == "image_url":
            image_url = item.get("image_url")
            url = image_url.get("url") if isinstance(image_url, Mapping) else None
            if isinstance(url, str) and url.startswith("data:"):
                header, _, payload = url.partition(",")
                mime_type = header[5:].split(";", 1)[0].strip()
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": payload,
                        },
                    }
                )
            elif isinstance(url, str) and url.startswith("https://"):
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    }
                )
            continue
        if item_type == "tool_use" and role == "assistant":
            blocks.append(dict(item))
            continue
        if item_type == "tool_result" and role == "user":
            blocks.append(dict(item))
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


def _from_anthropic_response(response: Any) -> dict[str, Any]:
    content_blocks = getattr(response, "content", None) or []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", "")
            if isinstance(text, str) and text:
                text_parts.append(text)
            continue
        if block_type == "tool_use":
            tool_calls.append(
                {
                    "id": getattr(block, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(block, "name", ""),
                        "arguments": json.dumps(getattr(block, "input", {}), ensure_ascii=False),
                    },
                }
            )
    message: dict[str, Any] = {"role": "assistant"}
    if text_parts:
        message["content"] = "".join(text_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _anthropic_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


def _to_anthropic_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    if not isinstance(function, Mapping):
        raise LLMAPIError("Anthropic tools must use OpenAI-style function definitions.")
    return {
        "name": function.get("name"),
        "description": function.get("description", ""),
        "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
    }


def _to_anthropic_tool_choice(tool_choice: str | Mapping[str, Any]) -> dict[str, Any] | str:
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "none":
            return {"type": "none"}
        if tool_choice == "required":
            return {"type": "any"}
        return {"type": "auto"}
    if isinstance(tool_choice, Mapping):
        function = tool_choice.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            if isinstance(name, str) and name:
                return {"type": "tool", "name": name}
    return {"type": "auto"}


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)
