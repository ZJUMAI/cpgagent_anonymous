#!/usr/bin/env python3
"""Verify ucec_mri_roi runtime dependencies."""

from __future__ import annotations

import importlib
import sys


def main() -> int:
    for name in ("nibabel", "numpy", "PIL"):
        try:
            importlib.import_module(name)
        except ImportError:
            print(f"missing dependency: {name}", file=sys.stderr)
            return 1
    print("ucec_mri_roi env OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
