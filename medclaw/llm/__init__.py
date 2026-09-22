"""Large-model clients used by the MedClaw agent."""

from medclaw.llm.factory import (
    DEFAULT_PROVIDER,
    SUPPORTED_PROVIDERS,
    create_llm_client,
    load_config_from_env,
    normalize_provider,
    resolve_provider,
)
from medclaw.llm.providers.local_openai import (
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    LocalOpenAIClient,
    LocalOpenAIConfig,
)
from medclaw.llm.image_utils import MAX_BASE64_IMAGE_BYTES, image_to_data_url
from medclaw.llm.protocols import (
    ChatCompletion,
    ChatModel,
    LLMAPIError,
    LLMConfigurationError,
    QwenAPIError,
    QwenCompletion,
    QwenConfigurationError,
)
from medclaw.llm.qwen_client import (
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL,
    QwenClient,
    QwenConfig,
)

__all__ = [
    "ChatCompletion",
    "ChatModel",
    "DEFAULT_LOCAL_BASE_URL",
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_PROVIDER",
    "DEFAULT_QWEN_BASE_URL",
    "DEFAULT_QWEN_MODEL",
    "LLMAPIError",
    "LLMConfigurationError",
    "LocalOpenAIClient",
    "LocalOpenAIConfig",
    "MAX_BASE64_IMAGE_BYTES",
    "QwenAPIError",
    "QwenClient",
    "QwenCompletion",
    "QwenConfig",
    "QwenConfigurationError",
    "SUPPORTED_PROVIDERS",
    "create_llm_client",
    "image_to_data_url",
    "load_config_from_env",
    "normalize_provider",
    "resolve_provider",
]
