"""Retry helpers for transient LLM API connection failures."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import TypeVar

from medclaw.llm.env_utils import parse_float
from medclaw.llm.protocols import LLMAPIError


T = TypeVar("T")

_RETRYABLE_TYPE_NAMES = frozenset(
    {
        "APIConnectionError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "RemoteProtocolError",
        "ProtocolError",
        "ConnectionError",
        "TimeoutError",
    }
)

_RETRYABLE_MESSAGE_FRAGMENTS = (
    "connection error",
    "connection reset",
    "server disconnected",
    "unexpected_eof",
    "eof occurred",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
)


def llm_retry_settings() -> tuple[int, float]:
    """Return (max_attempts, base_delay_sec) from environment."""

    max_attempts = int(
        parse_float(
            os.environ.get("MEDCLAW_LLM_MAX_RETRIES", "4"),
            "MEDCLAW_LLM_MAX_RETRIES",
            4.0,
        )
    )
    base_delay_sec = parse_float(
        os.environ.get("MEDCLAW_LLM_RETRY_BASE_SEC", "2"),
        "MEDCLAW_LLM_RETRY_BASE_SEC",
        2.0,
    )
    if max_attempts < 1:
        raise LLMAPIError("MEDCLAW_LLM_MAX_RETRIES must be at least 1.")
    if base_delay_sec < 0:
        raise LLMAPIError("MEDCLAW_LLM_RETRY_BASE_SEC must be non-negative.")
    return max_attempts, base_delay_sec


def is_retryable_llm_error(exc: BaseException) -> bool:
    """Return True when the error is likely transient network/proxy noise."""

    if isinstance(exc, LLMAPIError) and exc.__cause__ is not None:
        return is_retryable_llm_error(exc.__cause__)

    type_name = type(exc).__name__
    if type_name in _RETRYABLE_TYPE_NAMES:
        return True

    message = str(exc).lower()
    return any(fragment in message for fragment in _RETRYABLE_MESSAGE_FRAGMENTS)


def call_with_llm_retries(
    operation: Callable[[], T],
    *,
    provider: str,
    max_attempts: int | None = None,
    base_delay_sec: float | None = None,
) -> T:
    """Call an LLM API operation with exponential backoff on transient failures."""

    if max_attempts is None or base_delay_sec is None:
        default_attempts, default_delay = llm_retry_settings()
        if max_attempts is None:
            max_attempts = default_attempts
        if base_delay_sec is None:
            base_delay_sec = default_delay

    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts or not is_retryable_llm_error(exc):
                break
            delay = base_delay_sec * (2 ** (attempt - 1))
            time.sleep(delay)

    assert last_exc is not None
    raise LLMAPIError(f"{provider} API request failed: {last_exc}") from last_exc
