#!/usr/bin/env python3
"""Audit the generated anonymous release directory and ZIP archive."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_anonymous_release import DEFAULT_ARCHIVE, DEFAULT_OUTPUT, _audit_archive, audit_release


def main() -> int:
    errors = audit_release(DEFAULT_OUTPUT)
    if DEFAULT_ARCHIVE.is_file():
        errors.extend(_audit_archive(DEFAULT_ARCHIVE))
    if errors:
        print("\n".join(f"ERROR: {item}" for item in sorted(set(errors))))
        return 1
    print(f"Anonymous release audit passed: {DEFAULT_OUTPUT}")
    print(f"Anonymous release archive audit passed: {DEFAULT_ARCHIVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
