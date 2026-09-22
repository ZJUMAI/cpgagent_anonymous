"""Dependency-free terminal progress reporting for long Planner V2 stages."""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Callable, Mapping, TextIO


class ProgressReporter:
    """Render one updating TTY line and sparse line-oriented CI logs.

    Progress is written to stderr so command JSON summaries on stdout remain
    machine readable.  TTY output refreshes in place; redirected output is
    emitted at a lower frequency and always flushed immediately.
    """

    def __init__(
        self,
        label: str,
        total: int,
        *,
        unit: str = "item",
        enabled: bool = True,
        stream: TextIO | None = None,
        clock: Callable[[], float] | None = None,
        min_interval: float | None = None,
    ) -> None:
        self.label = str(label)
        self.total = max(int(total), 0)
        self.unit = str(unit)
        self.enabled = bool(enabled)
        self.stream = stream or sys.stderr
        self.clock = clock or time.monotonic
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.min_interval = float(
            min_interval if min_interval is not None else (0.5 if self.is_tty else 30.0)
        )
        self.started_at = self.clock()
        self.initial = 0
        self.completed = 0
        self.last_rendered_at = float("-inf")
        self.inline_active = False

    def message(self, message: str) -> None:
        """Print a flushed stage transition without losing future progress."""

        if not self.enabled:
            return
        self._end_inline()
        self.stream.write(f"[{self.label}] {message}\n")
        self.stream.flush()

    def start(self, completed: int = 0, *, status: str = "starting") -> None:
        self.initial = max(min(int(completed), self.total), 0)
        self.completed = self.initial
        self.started_at = self.clock()
        self.update(self.completed, metrics={"status": status}, force=True)

    def update(
        self,
        completed: int,
        *,
        metrics: Mapping[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        self.completed = max(min(int(completed), self.total), 0)
        now = self.clock()
        if (
            not force
            and self.completed < self.total
            and now - self.last_rendered_at < self.min_interval
        ):
            return
        line = self._render(now, metrics or {})
        if self.is_tty:
            self.stream.write("\r\x1b[2K" + line)
            self.inline_active = True
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
        self.last_rendered_at = now

    def finish(self, *, metrics: Mapping[str, Any] | None = None) -> None:
        payload = {"status": "done", **dict(metrics or {})}
        self.update(self.total, metrics=payload, force=True)
        self._end_inline()

    def _render(self, now: float, metrics: Mapping[str, Any]) -> str:
        elapsed = max(now - self.started_at, 0.0)
        completed_since_start = max(self.completed - self.initial, 0)
        remaining = max(self.total - self.completed, 0)
        rate = completed_since_start / elapsed if elapsed > 0 and completed_since_start else 0.0
        eta = remaining / rate if rate > 0 else None
        percent = (100.0 * self.completed / self.total) if self.total else 100.0
        pieces = [
            f"[{self.label}]",
            f"{self.completed}/{self.total} {self.unit}",
            f"{percent:5.1f}%",
        ]
        for key, value in metrics.items():
            if value is None:
                continue
            pieces.append(f"{key}={_metric_value(value)}")
        pieces.append(f"elapsed={_duration(elapsed)}")
        pieces.append(f"eta={_duration(eta) if eta is not None else '--:--'}")
        if rate > 0:
            pieces.append(f"rate={rate:.3g}/{self.unit}/s")
        return " | ".join(pieces)

    def _end_inline(self) -> None:
        if self.enabled and self.inline_active:
            self.stream.write("\n")
            self.stream.flush()
            self.inline_active = False


def _duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--"
    whole = max(int(seconds), 0)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _metric_value(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.5g}"
    text = str(value).replace("\n", " ")
    return text if len(text) <= 48 else text[:45] + "..."
