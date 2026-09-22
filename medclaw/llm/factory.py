"""Factory for creating provider-specific LLM clients."""

from __future__ import annotations

import os
from typing import Any, Mapping

from medclaw.llm.protocols import ChatModel, LLMConfigurationError
from medclaw.llm.providers.claude import ClaudeClient, ClaudeConfig
from medclaw.llm.providers.deepseek import DeepSeekClient, DeepSeekConfig
from medclaw.llm.providers.gemini import GeminiClient, GeminiConfig
from medclaw.llm.providers.local_openai import (
    LocalOpenAIClient,
    LocalOpenAIConfig,
)
from medclaw.llm.providers.openai import OpenAIClient, OpenAIConfig
from medclaw.llm.providers.qwen import QwenClient, QwenConfig


DEFAULT_PROVIDER = "local_openai"
SUPPORTED_PROVIDERS = (
    "local_openai",
    "qwen",
    "openai",
    "gemini",
    "deepseek",
    "claude",
)

_CONFIG_LOADERS = {
    "local_openai": LocalOpenAIConfig.from_env,
    "qwen": QwenConfig.from_env,
    "openai": OpenAIConfig.from_env,
    "gemini": GeminiConfig.from_env,
    "deepseek": DeepSeekConfig.from_env,
    "claude": ClaudeConfig.from_env,
}

_CLIENT_TYPES = {
    "local_openai": LocalOpenAIClient,
    "qwen": QwenClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
    "deepseek": DeepSeekClient,
    "claude": ClaudeClient,
}


def normalize_provider(name: str) -> str:
    provider = name.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise LLMConfigurationError(
            f"Unsupported LLM provider {name!r}. Supported providers: {supported}."
        )
    return provider


def resolve_provider(
    name: str | None,
    *,
    env_key: str = "MEDCLAW_LLM_PROVIDER",
    default: str = DEFAULT_PROVIDER,
) -> str:
    candidate = name or os.environ.get(env_key) or default
    return normalize_provider(candidate)


def load_config_from_env(
    provider: str,
    environ: Mapping[str, str] | None = None,
) -> Any:
    normalized = normalize_provider(provider)
    return _CONFIG_LOADERS[normalized](environ)


def create_llm_client(
    provider: str,
    *,
    config: Any | None = None,
    client: Any | None = None,
    environ: Mapping[str, str] | None = None,
) -> ChatModel:
    normalized = normalize_provider(provider)
    resolved_config = config or load_config_from_env(normalized, environ)
    client_type = _CLIENT_TYPES[normalized]
    return client_type(resolved_config, client=client)
