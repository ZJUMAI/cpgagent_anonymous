"""OpenAI-compatible chat client shared by multiple LLM providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

from openai import OpenAI, OpenAIError

from medclaw.llm.image_utils import image_to_data_url, is_supported_image_url
from medclaw.llm.protocols import ChatCompletion, LLMAPIError, LLMConfigurationError
from medclaw.llm.retry_utils import call_with_llm_retries


class ProviderHooks(Protocol):
    """Provider-specific request and response customization."""

    provider_name: str

    def augment_request(self, request: dict[str, Any], config: Any) -> dict[str, Any]: ...

    def normalize_message(self, message: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DefaultProviderHooks:
    """No-op hooks for providers that need no request customization."""

    provider_name: str

    def augment_request(self, request: dict[str, Any], config: Any) -> dict[str, Any]:
        return request

    def normalize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        return message


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """Base configuration for OpenAI-compatible chat APIs."""

    provider: str
    api_key: str
    base_url: str
    model: str
    timeout_sec: float = 120.0
    stream: bool = True

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise LLMConfigurationError(f"{self.provider} API key must not be empty.")
        if not self.model.strip():
            raise LLMConfigurationError(f"{self.provider} model name must not be empty.")
        if self.timeout_sec <= 0:
            raise LLMConfigurationError(f"{self.provider} timeout must be greater than zero.")

        from medclaw.llm.env_utils import is_safe_base_url

        base_url = self.base_url.rstrip("/")
        if not is_safe_base_url(base_url):
            raise LLMConfigurationError(
                f"{self.provider} base URL must use HTTPS, except for localhost development endpoints."
            )
        object.__setattr__(self, "base_url", base_url)

    def public_summary(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "timeout_sec": self.timeout_sec,
            "stream": self.stream,
        }


class OpenAICompatibleClient:
    """Call chat models through an OpenAI-compatible HTTP API."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        hooks: ProviderHooks | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self._hooks = hooks or DefaultProviderHooks(provider_name=config.provider)
        self._client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_sec,
        )

    def build_user_message(
        self,
        text: str,
        *,
        image_paths: Iterable[Any] = (),
        image_urls: Iterable[str] = (),
    ) -> dict[str, Any]:
        from pathlib import Path

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
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            "stream": self.config.stream,
        }
        if self.config.stream:
            request["stream_options"] = {"include_usage": True}
        if tools:
            request["tools"] = [dict(tool) for tool in tools]
            request["tool_choice"] = tool_choice
            request["parallel_tool_calls"] = False

        request = self._hooks.augment_request(request, self.config)
        provider = self._hooks.provider_name

        def _complete() -> ChatCompletion:
            try:
                response = self._client.chat.completions.create(**request)
            except OpenAIError as exc:
                raise LLMAPIError(f"{provider} API request failed: {exc}") from exc
            except Exception as exc:
                raise LLMAPIError(f"{provider} API request failed: {exc}") from exc

            if self.config.stream:
                return _consume_stream(response, provider_name=provider, hooks=self._hooks)

            if not getattr(response, "choices", None):
                raise LLMAPIError(f"{provider} API response did not contain any choices.")

            message_object = response.choices[0].message
            message = _model_dump_or_mapping(message_object)
            if message is None:
                raise LLMAPIError(
                    f"{provider} API response contained an unsupported message object."
                )
            message.setdefault("role", "assistant")
            message = self._hooks.normalize_message(message)

            usage_object = getattr(response, "usage", None)
            usage = _model_dump_or_mapping(usage_object)

            return ChatCompletion(
                message=message,
                model=getattr(response, "model", None),
                usage=usage,
            )

        return call_with_llm_retries(_complete, provider=provider)


def _consume_stream(
    response: Any,
    *,
    provider_name: str,
    hooks: ProviderHooks,
) -> ChatCompletion:
    role = "assistant"
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    model: str | None = None
    usage: dict[str, Any] | None = None
    saw_choice = False

    try:
        for chunk in response:
            model = getattr(chunk, "model", None) or model
            usage_object = getattr(chunk, "usage", None)
            normalized_usage = _model_dump_or_mapping(usage_object)
            if normalized_usage is not None:
                usage = normalized_usage

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            saw_choice = True
            delta = choices[0].delta
            delta_data = _model_dump_or_mapping(delta)
            if delta_data is None:
                raise LLMAPIError(f"{provider_name} API stream contained an unsupported delta object.")

            delta_role = delta_data.get("role")
            if isinstance(delta_role, str) and delta_role:
                role = delta_role
            delta_content = delta_data.get("content")
            if isinstance(delta_content, str):
                content_parts.append(delta_content)
            delta_reasoning = delta_data.get("reasoning_content")
            if isinstance(delta_reasoning, str):
                reasoning_parts.append(delta_reasoning)
            _merge_tool_call_deltas(tool_calls, delta_data.get("tool_calls"))
    except LLMAPIError:
        raise
    except Exception as exc:
        raise LLMAPIError(f"{provider_name} API stream failed: {exc}") from exc

    if not saw_choice:
        raise LLMAPIError(f"{provider_name} API stream did not contain any choices.")

    message: dict[str, Any] = {"role": role}
    content = "".join(content_parts)
    reasoning_content = "".join(reasoning_parts)
    if content:
        message["content"] = content
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    message = hooks.normalize_message(message)

    return ChatCompletion(message=message, model=model, usage=usage)


def _merge_tool_call_deltas(
    accumulated: dict[int, dict[str, Any]],
    deltas: Any,
) -> None:
    if not isinstance(deltas, list):
        return

    for fallback_index, delta in enumerate(deltas):
        data = _model_dump_or_mapping(delta)
        if data is None:
            continue
        index = data.get("index", fallback_index)
        if not isinstance(index, int):
            index = fallback_index
        call = accumulated.setdefault(
            index,
            {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )
        call_id = data.get("id")
        if isinstance(call_id, str):
            call["id"] += call_id
        call_type = data.get("type")
        if isinstance(call_type, str) and call_type:
            call["type"] = call_type

        function_data = _model_dump_or_mapping(data.get("function"))
        if function_data is None:
            continue
        function = call["function"]
        name = function_data.get("name")
        if isinstance(name, str):
            function["name"] += name
        arguments = function_data.get("arguments")
        if isinstance(arguments, str):
            function["arguments"] += arguments


def _model_dump_or_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        data = value.model_dump(exclude_none=True)
        return dict(data) if isinstance(data, Mapping) else None
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        data = {key: item for key, item in vars(value).items() if not key.startswith("_")}
        return data or None
    return None
