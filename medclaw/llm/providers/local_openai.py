"""Local OpenAI-compatible provider, such as a vLLM chat server."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from medclaw.llm.env_utils import parse_bool, parse_float
from medclaw.llm.openai_compatible import OpenAICompatibleClient, OpenAICompatibleConfig
from medclaw.llm.protocols import LLMConfigurationError


DEFAULT_LOCAL_MODEL = "qwen3.6-27b"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8087/v1"
PROVIDER_NAME = "local_openai"


@dataclass(frozen=True)
class LocalOpenAIConfig(OpenAICompatibleConfig):
    """Configuration for an OpenAI-compatible model on the local machine."""

    provider: str = PROVIDER_NAME
    api_key: str = field(repr=False, default="")
    base_url: str = DEFAULT_LOCAL_BASE_URL
    model: str = DEFAULT_LOCAL_MODEL
    max_images_per_message: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_images_per_message <= 0:
            raise LLMConfigurationError(
                "MEDCLAW_LOCAL_MAX_IMAGES_PER_MESSAGE must be greater than zero."
            )

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "LocalOpenAIConfig":
        values = os.environ if environ is None else environ
        api_key = values.get("MEDCLAW_LOCAL_API_KEY") or values.get("VLLM_API_KEY")
        if not api_key:
            raise LLMConfigurationError(
                "Missing local LLM API key. Set MEDCLAW_LOCAL_API_KEY "
                "(or VLLM_API_KEY) to the value passed to vllm serve --api-key."
            )
        return cls(
            provider=PROVIDER_NAME,
            api_key=api_key,
            base_url=values.get("MEDCLAW_LOCAL_BASE_URL", DEFAULT_LOCAL_BASE_URL),
            model=values.get("MEDCLAW_LOCAL_MODEL", DEFAULT_LOCAL_MODEL),
            timeout_sec=parse_float(
                values.get("MEDCLAW_LOCAL_TIMEOUT_SEC", "300"),
                "MEDCLAW_LOCAL_TIMEOUT_SEC",
                300.0,
            ),
            stream=parse_bool(
                values.get("MEDCLAW_LOCAL_STREAM", "true"),
                "MEDCLAW_LOCAL_STREAM",
            ),
            max_images_per_message=_positive_int(
                values.get("MEDCLAW_LOCAL_MAX_IMAGES_PER_MESSAGE", "4"),
                "MEDCLAW_LOCAL_MAX_IMAGES_PER_MESSAGE",
            ),
        )

    def public_summary(self) -> dict[str, Any]:
        summary = super().public_summary()
        summary["max_images_per_message"] = self.max_images_per_message
        return summary


class LocalOpenAIClient(OpenAICompatibleClient):
    """Call a local vLLM server through its OpenAI-compatible endpoint."""

    def __init__(
        self,
        config: LocalOpenAIConfig,
        *,
        client: Any | None = None,
    ) -> None:
        super().__init__(config, client=client)


def _positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be an integer, got {value!r}.") from exc
    if parsed <= 0:
        raise LLMConfigurationError(f"{name} must be greater than zero.")
    return parsed
