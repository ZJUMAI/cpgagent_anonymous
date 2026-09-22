from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from medclaw.llm.factory import (
    SUPPORTED_PROVIDERS,
    create_llm_client,
    load_config_from_env,
    normalize_provider,
    resolve_provider,
)
from medclaw.llm.openai_compatible import OpenAICompatibleClient
from medclaw.llm.protocols import LLMConfigurationError
from medclaw.llm.providers.claude import ClaudeClient, ClaudeConfig
from medclaw.llm.providers.deepseek import DeepSeekClient, DeepSeekConfig
from medclaw.llm.providers.gemini import GeminiClient, GeminiConfig
from medclaw.llm.providers.local_openai import (
    LocalOpenAIClient,
    LocalOpenAIConfig,
)
from medclaw.llm.providers.openai import OpenAIClient, OpenAIConfig
from medclaw.llm.providers.qwen import QwenClient, QwenConfig


@pytest.mark.parametrize("provider", SUPPORTED_PROVIDERS)
def test_normalize_provider_accepts_supported_names(provider: str) -> None:
    assert normalize_provider(provider) == provider
    assert normalize_provider(provider.upper()) == provider


def test_normalize_provider_rejects_unknown_name() -> None:
    with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider"):
        normalize_provider("unknown-model")


@pytest.mark.parametrize(
    ("provider", "env", "config_type", "client_type"),
    [
        (
            "local_openai",
            {"MEDCLAW_LOCAL_API_KEY": "secret"},
            LocalOpenAIConfig,
            LocalOpenAIClient,
        ),
        ("qwen", {"DASHSCOPE_API_KEY": "secret"}, QwenConfig, QwenClient),
        ("openai", {"OPENAI_API_KEY": "secret"}, OpenAIConfig, OpenAIClient),
        ("gemini", {"GEMINI_API_KEY": "secret"}, GeminiConfig, GeminiClient),
        ("deepseek", {"DEEPSEEK_API_KEY": "secret"}, DeepSeekConfig, DeepSeekClient),
        ("claude", {"ANTHROPIC_API_KEY": "secret"}, ClaudeConfig, ClaudeClient),
    ],
)
def test_load_config_from_env(provider: str, env: dict[str, str], config_type: type, client_type: type) -> None:
    config = load_config_from_env(provider, env)
    assert isinstance(config, config_type)
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: None))
    client = create_llm_client(provider, config=config, client=fake_client)
    assert isinstance(client, client_type)


@pytest.mark.parametrize(
    ("provider", "missing_env"),
    [
        ("local_openai", {}),
        ("qwen", {}),
        ("openai", {}),
        ("gemini", {}),
        ("deepseek", {}),
        ("claude", {}),
    ],
)
def test_load_config_from_env_requires_api_key(provider: str, missing_env: dict[str, str]) -> None:
    with pytest.raises(LLMConfigurationError):
        load_config_from_env(provider, missing_env)


def test_resolve_provider_prefers_explicit_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDCLAW_LLM_PROVIDER", "openai")
    assert resolve_provider("deepseek") == "deepseek"


def test_resolve_provider_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDCLAW_LLM_PROVIDER", "gemini")
    assert resolve_provider(None) == "gemini"


def test_resolve_provider_defaults_to_local_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MEDCLAW_LLM_PROVIDER", raising=False)
    assert resolve_provider(None) == "local_openai"


def test_local_openai_provider_uses_vllm_defaults_without_exposing_key() -> None:
    config = LocalOpenAIConfig.from_env({"MEDCLAW_LOCAL_API_KEY": "local-secret"})
    assert config.provider == "local_openai"
    assert config.model == "qwen3.6-27b"
    assert config.base_url == "http://127.0.0.1:8087/v1"
    assert config.stream is True
    assert "local-secret" not in repr(config)
    assert "api_key" not in config.public_summary()


def test_openai_provider_uses_expected_defaults() -> None:
    config = OpenAIConfig.from_env({"OPENAI_API_KEY": "secret-key"})
    assert config.provider == "openai"
    assert config.model == "gpt-4o"
    assert config.base_url == "https://api.openai.com/v1"


def test_qwen_provider_applies_thinking_hook() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.request: dict[str, Any] | None = None

        def create(self, **kwargs: Any) -> Any:
            self.request = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                model="qwen3.7-plus",
                usage=None,
            )

    fake_openai = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    config = QwenConfig.from_env({"DASHSCOPE_API_KEY": "secret-key", "MEDCLAW_QWEN_STREAM": "false"})
    client = QwenClient(config, client=fake_openai)
    completion = client.complete(messages=[{"role": "user", "content": "hello"}], tools=[])

    assert completion.message["content"] == "ok"
    assert fake_openai.chat.completions.request is not None
    assert fake_openai.chat.completions.request["extra_body"] == {"enable_thinking": True}


def test_openai_compatible_client_builds_multimodal_message(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\x9cc``\x00"
        b"\x00\x00\x04\x00\x01\x05\x57\x18\x8d\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    config = OpenAIConfig.from_env({"OPENAI_API_KEY": "secret-key"})
    client = OpenAICompatibleClient(config)
    message = client.build_user_message("inspect", image_paths=[image_path])
    assert message["role"] == "user"
    assert isinstance(message["content"], list)
    assert message["content"][0]["type"] == "image_url"


def test_claude_client_requires_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ClaudeConfig.from_env({"ANTHROPIC_API_KEY": "secret-key"})
    monkeypatch.setitem(os.environ, "ANTHROPIC_API_KEY", "secret-key")
    try:
        import anthropic  # noqa: F401
    except ImportError:
        with pytest.raises(LLMConfigurationError, match="optional 'anthropic' package"):
            ClaudeClient(config)
    else:
        client = ClaudeClient(config, client=SimpleNamespace(messages=SimpleNamespace(create=lambda **_: None)))
        assert isinstance(client, ClaudeClient)
