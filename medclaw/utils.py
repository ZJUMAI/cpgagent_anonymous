"""Small, shared helpers for deterministic IO and provenance."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class DataIOError(RuntimeError):
    """Raised when a JSON or YAML file cannot be read or written."""


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def safe_component(value: str, label: str = "path component") -> str:
    """Validate a value before using it as one filesystem path component."""

    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"Invalid {label} {value!r}; use letters, numbers, '.', '_', or '-'."
        )
    return value


def canonical_json_bytes(data: Any) -> bytes:
    """Serialize JSON-compatible data deterministically."""

    try:
        text = json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise DataIOError(f"Value is not JSON serializable: {exc}") from exc
    return text.encode("utf-8")


def hash_json(data: Any) -> str:
    """Return a SHA-256 hash for JSON-compatible data."""

    return hashlib.sha256(canonical_json_bytes(data)).hexdigest()


def read_json(path: Path) -> Any:
    """Read JSON with a path-rich error message."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise DataIOError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataIOError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise DataIOError(f"Could not read JSON file {path}: {exc}") from exc


def write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """Write JSON with a path-rich error message."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=indent, sort_keys=True)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise DataIOError(f"Could not write JSON file {path}: {exc}") from exc


def read_yaml(path: Path) -> Any:
    """Read YAML with a path-rich error message."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise DataIOError(f"YAML file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise DataIOError(f"Invalid YAML in {path}: {exc}") from exc
    except OSError as exc:
        raise DataIOError(f"Could not read YAML file {path}: {exc}") from exc
