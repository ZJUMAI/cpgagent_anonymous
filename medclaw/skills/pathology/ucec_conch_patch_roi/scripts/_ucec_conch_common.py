#!/usr/bin/env python3
"""Shared paths and helpers for UCEC CONCH patch / prompt ROI pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

UCEC_SHARE_ROOT = Path("/share/endometrial_dataset/UCEC")
UCEC_PATCH_ROOT = Path("/data4/share/endometrial_dataset/UCEC/patches/patches")
CONCH_UCEC_ROOT = Path("/data4/share/CONCH_TCGA/UCEC/features")
CONCH_UCEC_WRITE_ROOT = Path(
    os.environ.get(
        "CONCH_UCEC_WRITE_ROOT",
        "/data4/tujiayong/share/CONCH_TCGA/UCEC/features",
    )
)
PROCESSED_UCEC_ROOT = Path(
    os.environ.get("PROCESSED_UCEC_ROOT", "/data4/tujiayong/processed/UCEC")
)
CONCH_DIR = Path("/data4/tujiayong/CONCH")
DEFAULT_CHECKPOINT = CONCH_DIR / "checkpoints/conch/pytorch_model.bin"


@dataclass(frozen=True)
class UcecConchPaths:
    conch_feat_dir: Path
    patch_dir: Path
    wsi_dir: Path
    out_root: Path


def default_paths() -> UcecConchPaths:
    return UcecConchPaths(
        conch_feat_dir=CONCH_UCEC_ROOT / "pt_files",
        patch_dir=UCEC_PATCH_ROOT,
        wsi_dir=UCEC_SHARE_ROOT,
        out_root=PROCESSED_UCEC_ROOT,
    )
