"""Tests for transient LLM API retry helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from medclaw.llm.protocols import LLMAPIError
from medclaw.llm.retry_utils import call_with_llm_retries, is_retryable_llm_error


class _FakeConnectError(Exception):
    pass


def test_is_retryable_llm_error_detects_connection_failures() -> None:
    assert is_retryable_llm_error(_FakeConnectError("Connection error."))
    assert is_retryable_llm_error(
        LLMAPIError("claude API request failed: Server disconnected without sending a response.")
    )


def test_is_retryable_llm_error_ignores_validation_errors() -> None:
    assert not is_retryable_llm_error(
        LLMAPIError(
            "claude API request failed: Error code: 400 - invalid_request_error"
        )
    )


def test_call_with_llm_retries_eventually_succeeds() -> None:
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise _FakeConnectError("connection reset")
        return "ok"

    with patch("medclaw.llm.retry_utils.time.sleep"):
        result = call_with_llm_retries(
            flaky,
            provider="claude",
            max_attempts=4,
            base_delay_sec=0.0,
        )

    assert result == "ok"
    assert attempts["count"] == 3


def test_call_with_llm_retries_raises_after_exhaustion() -> None:
    def always_fail() -> str:
        raise _FakeConnectError("connection reset")

    with patch("medclaw.llm.retry_utils.time.sleep"):
        with pytest.raises(LLMAPIError, match="connection reset"):
            call_with_llm_retries(
                always_fail,
                provider="openai",
                max_attempts=2,
                base_delay_sec=0.0,
            )
