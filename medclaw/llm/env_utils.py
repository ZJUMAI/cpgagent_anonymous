"""Shared environment parsing helpers for LLM providers."""

from __future__ import annotations

from medclaw.llm.protocols import LLMConfigurationError


def parse_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise LLMConfigurationError(f"{name} must be true or false, got {value!r}.")


def parse_float(value: str, name: str, default: float) -> float:
    if not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be numeric, got {value!r}.") from exc
    return parsed


def is_safe_base_url(value: str) -> bool:
    return (
        value.startswith("https://")
        or value.startswith("http://localhost")
        or value.startswith("http://127.0.0.1")
    )
