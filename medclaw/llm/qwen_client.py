"""OpenAI-compatible client for the Qwen3.7-Plus multimodal model on DashScope."""

from __future__ import annotations

from medclaw.llm.image_utils import (
    MAX_BASE64_IMAGE_BYTES,
    SUPPORTED_IMAGE_MIME_TYPES,
    image_to_data_url,
)
from medclaw.llm.protocols import (
    QwenAPIError,
    QwenCompletion,
    QwenConfigurationError,
)
from medclaw.llm.providers.qwen import (
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL,
    QwenClient,
    QwenConfig,
)

__all__ = [
    "DEFAULT_QWEN_BASE_URL",
    "DEFAULT_QWEN_MODEL",
    "MAX_BASE64_IMAGE_BYTES",
    "QwenAPIError",
    "QwenClient",
    "QwenCompletion",
    "QwenConfig",
    "QwenConfigurationError",
    "SUPPORTED_IMAGE_MIME_TYPES",
    "image_to_data_url",
]
