"""OpenAI / ChatGPT provider."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from medclaw.llm.env_utils import parse_bool, parse_float
from medclaw.llm.openai_compatible import OpenAICompatibleClient, OpenAICompatibleConfig
from medclaw.llm.protocols import LLMConfigurationError


DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
PROVIDER_NAME = "openai"


@dataclass(frozen=True)
class OpenAIConfig(OpenAICompatibleConfig):
    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "OpenAIConfig":
        values = os.environ if environ is None else environ
        api_key = values.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMConfigurationError(
                "Missing OpenAI API key. Set OPENAI_API_KEY in the process environment."
            )
        return cls(
            provider=PROVIDER_NAME,
            api_key=api_key,
            base_url=values.get("MEDCLAW_OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            model=values.get("MEDCLAW_OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            timeout_sec=parse_float(
                values.get("MEDCLAW_OPENAI_TIMEOUT_SEC", "120"),
                "MEDCLAW_OPENAI_TIMEOUT_SEC",
                120.0,
            ),
            stream=parse_bool(
                values.get("MEDCLAW_OPENAI_STREAM", "false"),
                "MEDCLAW_OPENAI_STREAM",
            ),
        )


class OpenAIClient(OpenAICompatibleClient):
    def __init__(self, config: OpenAIConfig, *, client: Any | None = None) -> None:
        super().__init__(config, client=client)
