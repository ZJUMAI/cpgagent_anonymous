"""Small IO helpers for benchmark outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from medclaw.utils import DataIOError, read_json, write_json


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object and fail with a path-rich message otherwise."""

    data = read_json(path)
    if not isinstance(data, dict):
        raise DataIOError(f"JSON file must contain an object: {path}")
    return data


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON record to a JSONL file."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise DataIOError(f"Could not append JSONL file {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of objects."""

    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise DataIOError(f"Line {line_no} is not a JSON object: {path}")
            records.append(data)
    except json.JSONDecodeError as exc:
        raise DataIOError(
            f"Invalid JSONL in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise DataIOError(f"Could not read JSONL file {path}: {exc}") from exc
    return records


def safe_id(value: str, fallback: str = "item") -> str:
    """Return a filesystem-safe identifier without rejecting human labels."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or fallback


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text with parent directory creation."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise DataIOError(f"Could not write text file {path}: {exc}") from exc


__all__ = [
    "append_jsonl",
    "read_json_object",
    "read_jsonl",
    "safe_id",
    "write_json",
    "write_text",
]
