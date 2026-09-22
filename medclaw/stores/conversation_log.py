"""Append-only, sanitized JSONL logging for model conversations."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from medclaw.utils import DataIOError, safe_component


_SECRET_KEYS = {
    "api_key",
    "authorization",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "token",
}
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:DASHSCOPE_API_KEY|OPENAI_API_KEY|MEDCLAW_LOCAL_API_KEY|"
    r"VLLM_API_KEY|api[_-]?key|authorization|"
    r"access[_-]?token|refresh[_-]?token|password|secret)\b[\"']?\s*[:=]\s*"
    r"[\"']?(?:bearer\s+)?)([^\"'\s,;}]+)"
)
_SK_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_IMAGE_DATA_URL_RE = re.compile(
    r"^data:(image/[A-Za-z0-9.+-]+);base64,(.*)$",
    re.DOTALL,
)


class ConversationLog:
    """Write reconstructable conversation events without storing image bodies."""

    def __init__(self, runs_root: Path) -> None:
        self.runs_root = Path(runs_root).resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path_for_case(self, case_id: str) -> Path:
        case_id = safe_component(case_id, "case_id")
        return self.runs_root / case_id / "conversation_log.jsonl"

    def append(self, case_id: str, record: Mapping[str, Any]) -> Path:
        path = self.path_for_case(case_id)
        sanitized = sanitize_for_audit(record)
        try:
            line = json.dumps(sanitized, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise DataIOError(
                f"Conversation audit record is not JSON serializable: {exc}"
            ) from exc

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.write("\n")
        except OSError as exc:
            raise DataIOError(
                f"Could not append conversation audit log {path}: {exc}"
            ) from exc
        return path


def sanitize_for_audit(value: Any) -> Any:
    """Remove secrets, reasoning text, and inline image payloads from audit data."""

    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            normalized = key_text.lower()
            if normalized == "reasoning_content":
                sanitized["reasoning_content_redacted"] = bool(item)
            elif normalized in _SECRET_KEYS:
                sanitized[key_text] = "<redacted>"
            else:
                sanitized[key_text] = sanitize_for_audit(item)
        return sanitized

    if isinstance(value, (list, tuple)):
        return [sanitize_for_audit(item) for item in value]

    if isinstance(value, str):
        image_match = _IMAGE_DATA_URL_RE.match(value)
        if image_match:
            return _redacted_image_reference(
                mime_type=image_match.group(1),
                payload=image_match.group(2),
            )

        sanitized = _strip_url_query(value)
        sanitized = _SECRET_ASSIGNMENT_RE.sub(r"\1<redacted>", sanitized)
        return _SK_KEY_RE.sub("<redacted-api-key>", sanitized)

    return value


def _redacted_image_reference(*, mime_type: str, payload: str) -> dict[str, Any]:
    reference: dict[str, Any] = {
        "redacted": "base64_image_data",
        "mime_type": mime_type,
        "encoded_length": len(payload),
    }
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        reference["valid_base64"] = False
        return reference

    reference.update(
        {
            "valid_base64": True,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
    return reference


def _strip_url_query(value: str) -> str:
    if not value.startswith(("http://", "https://")):
        return value
    parts = urlsplit(value)
    if not parts.query and not parts.fragment:
        return value
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
