"""Append-only JSONL audit logging for skill calls."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping

from medclaw.utils import DataIOError, safe_component


class AuditLog:
    """Write one reproducibility record per skill invocation."""

    def __init__(self, runs_root: Path) -> None:
        self.runs_root = Path(runs_root).resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path_for_case(self, case_id: str) -> Path:
        case_id = safe_component(case_id, "case_id")
        return self.runs_root / case_id / "audit_log.jsonl"

    def append(self, case_id: str, record: Mapping[str, Any]) -> Path:
        path = self.path_for_case(case_id)
        try:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise DataIOError(f"Audit record is not JSON serializable: {exc}") from exc

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.write("\n")
        except OSError as exc:
            raise DataIOError(f"Could not append audit log {path}: {exc}") from exc
        return path
