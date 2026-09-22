"""Shared chat-model protocols and normalized completion types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


class LLMConfigurationError(ValueError):
    """Raised when an LLM client configuration is missing or unsafe."""


class LLMAPIError(RuntimeError):
    """Raised when a remote LLM API request fails."""


@dataclass(frozen=True)
class ChatCompletion:
    """Normalized response from one chat completion request."""

    message: dict[str, Any]
    model: str | None = None
    usage: dict[str, Any] | None = None


class ChatModel(Protocol):
    """Minimal interface required by agent loops and benchmark runners."""

    config: Any

    def build_user_message(
        self,
        text: str,
        *,
        image_paths: Iterable[Path] = (),
        image_urls: Iterable[str] = (),
    ) -> dict[str, Any]: ...

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
        tool_choice: str | Mapping[str, Any] = "auto",
    ) -> ChatCompletion: ...


# Backward-compatible aliases used by existing imports and tests.
QwenCompletion = ChatCompletion
QwenConfigurationError = LLMConfigurationError
QwenAPIError = LLMAPIError
