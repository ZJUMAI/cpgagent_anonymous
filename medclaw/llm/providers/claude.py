"""Claude provider via Anthropic Messages API."""

from __future__ import annotations

import os
from typing import Any, Mapping

from medclaw.llm.anthropic_client import AnthropicClient, AnthropicConfig
from medclaw.llm.env_utils import parse_float
from medclaw.llm.protocols import LLMConfigurationError


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
PROVIDER_NAME = "claude"


class ClaudeConfig(AnthropicConfig):
    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ClaudeConfig":
        values = os.environ if environ is None else environ
        api_key = values.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMConfigurationError(
                "Missing Anthropic API key. Set ANTHROPIC_API_KEY in the process environment."
            )
        return cls(
            provider=PROVIDER_NAME,
            api_key=api_key,
            model=values.get("MEDCLAW_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
            timeout_sec=parse_float(
                values.get("MEDCLAW_CLAUDE_TIMEOUT_SEC", "120"),
                "MEDCLAW_CLAUDE_TIMEOUT_SEC",
                120.0,
            ),
            max_tokens=int(
                parse_float(
                    values.get("MEDCLAW_CLAUDE_MAX_TOKENS", "8192"),
                    "MEDCLAW_CLAUDE_MAX_TOKENS",
                    8192.0,
                )
            ),
        )


class ClaudeClient(AnthropicClient):
    def __init__(self, config: ClaudeConfig, *, client: Any | None = None) -> None:
        super().__init__(config, client=client)
