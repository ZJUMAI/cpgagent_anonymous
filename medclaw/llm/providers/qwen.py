"""Qwen provider for DashScope OpenAI-compatible API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, field
from typing import Any, Mapping

from medclaw.llm.env_utils import parse_bool, parse_float
from medclaw.llm.openai_compatible import (
    DefaultProviderHooks,
    OpenAICompatibleClient,
    OpenAICompatibleConfig,
)
from medclaw.llm.protocols import LLMConfigurationError


DEFAULT_QWEN_MODEL = "qwen3.7-plus"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PROVIDER_NAME = "qwen"


@dataclass(frozen=True)
class QwenConfig(OpenAICompatibleConfig):
    """Configuration for the Qwen API client."""

    provider: str = PROVIDER_NAME
    api_key: str = field(repr=False, default="")
    base_url: str = DEFAULT_QWEN_BASE_URL
    model: str = DEFAULT_QWEN_MODEL
    enable_thinking: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "QwenConfig":
        values = os.environ if environ is None else environ
        api_key = values.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise LLMConfigurationError(
                "Missing Qwen API key. Set DASHSCOPE_API_KEY in the process environment."
            )

        return cls(
            provider=PROVIDER_NAME,
            api_key=api_key,
            base_url=values.get("MEDCLAW_QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL),
            model=values.get("MEDCLAW_QWEN_MODEL", DEFAULT_QWEN_MODEL),
            timeout_sec=parse_float(
                values.get("MEDCLAW_QWEN_TIMEOUT_SEC", "120"),
                "MEDCLAW_QWEN_TIMEOUT_SEC",
                120.0,
            ),
            enable_thinking=parse_bool(
                values.get("MEDCLAW_QWEN_ENABLE_THINKING", "true"),
                "MEDCLAW_QWEN_ENABLE_THINKING",
            ),
            stream=parse_bool(
                values.get("MEDCLAW_QWEN_STREAM", "true"),
                "MEDCLAW_QWEN_STREAM",
            ),
        )

    def public_summary(self) -> dict[str, Any]:
        summary = super().public_summary()
        summary["enable_thinking"] = self.enable_thinking
        return summary


@dataclass(frozen=True)
class QwenProviderHooks(DefaultProviderHooks):
    provider_name: str = PROVIDER_NAME

    def augment_request(self, request: dict[str, Any], config: Any) -> dict[str, Any]:
        request = dict(request)
        request["extra_body"] = {"enable_thinking": getattr(config, "enable_thinking", True)}
        return request


class QwenClient(OpenAICompatibleClient):
    """Call Qwen through the DashScope OpenAI-compatible API."""

    def __init__(self, config: QwenConfig, *, client: Any | None = None) -> None:
        super().__init__(config, hooks=QwenProviderHooks(), client=client)


# Backward-compatible constants re-exported from the legacy module path.
MAX_BASE64_IMAGE_BYTES = 7 * 1024 * 1024
