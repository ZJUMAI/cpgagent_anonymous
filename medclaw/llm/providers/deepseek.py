"""DeepSeek provider."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from medclaw.llm.env_utils import parse_bool, parse_float
from medclaw.llm.openai_compatible import OpenAICompatibleClient, OpenAICompatibleConfig
from medclaw.llm.protocols import LLMConfigurationError


DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
PROVIDER_NAME = "deepseek"


@dataclass(frozen=True)
class DeepSeekConfig(OpenAICompatibleConfig):
    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "DeepSeekConfig":
        values = os.environ if environ is None else environ
        api_key = values.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise LLMConfigurationError(
                "Missing DeepSeek API key. Set DEEPSEEK_API_KEY in the process environment."
            )
        return cls(
            provider=PROVIDER_NAME,
            api_key=api_key,
            base_url=values.get("MEDCLAW_DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
            model=values.get("MEDCLAW_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
            timeout_sec=parse_float(
                values.get("MEDCLAW_DEEPSEEK_TIMEOUT_SEC", "120"),
                "MEDCLAW_DEEPSEEK_TIMEOUT_SEC",
                120.0,
            ),
            stream=parse_bool(
                values.get("MEDCLAW_DEEPSEEK_STREAM", "false"),
                "MEDCLAW_DEEPSEEK_STREAM",
            ),
        )


class DeepSeekClient(OpenAICompatibleClient):
    def __init__(self, config: DeepSeekConfig, *, client: Any | None = None) -> None:
        super().__init__(config, client=client)
