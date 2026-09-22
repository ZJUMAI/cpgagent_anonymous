"""Small JSONL and filesystem helpers used by the planner package."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def safe_filename(value: str, *, fallback: str = "untitled", max_length: int = 96) -> str:
    """Return a readable cross-platform filename stem while preserving Unicode."""

    value = re.sub(r"^[#\s]+", "", value).strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("._ ")
    if not value:
        value = fallback
    if len(value) > max_length:
        value = value[:max_length].rstrip("._ ")
    return value or fallback


def stable_short_id(value: str, length: int = 10) -> str:
    import hashlib

    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]
