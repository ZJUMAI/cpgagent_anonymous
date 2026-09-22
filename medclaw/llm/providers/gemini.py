"""Google Gemini provider via OpenAI-compatible endpoint."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from medclaw.llm.env_utils import parse_bool, parse_float
from medclaw.llm.openai_compatible import OpenAICompatibleClient, OpenAICompatibleConfig
from medclaw.llm.protocols import LLMConfigurationError


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
PROVIDER_NAME = "gemini"


@dataclass(frozen=True)
class GeminiConfig(OpenAICompatibleConfig):
    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "GeminiConfig":
        values = os.environ if environ is None else environ
        api_key = values.get("GEMINI_API_KEY") or values.get("GOOGLE_API_KEY")
        if not api_key:
            raise LLMConfigurationError(
                "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY."
            )
        return cls(
            provider=PROVIDER_NAME,
            api_key=api_key,
            base_url=values.get("MEDCLAW_GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL),
            model=values.get("MEDCLAW_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            timeout_sec=parse_float(
                values.get("MEDCLAW_GEMINI_TIMEOUT_SEC", "120"),
                "MEDCLAW_GEMINI_TIMEOUT_SEC",
                120.0,
            ),
            stream=parse_bool(
                values.get("MEDCLAW_GEMINI_STREAM", "false"),
                "MEDCLAW_GEMINI_STREAM",
            ),
        )


class GeminiClient(OpenAICompatibleClient):
    def __init__(self, config: GeminiConfig, *, client: Any | None = None) -> None:
        super().__init__(config, client=client)
