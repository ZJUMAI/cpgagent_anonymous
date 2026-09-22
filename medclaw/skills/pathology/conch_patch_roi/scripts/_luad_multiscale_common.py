#!/usr/bin/env python3
"""Shared paths and helpers for LUAD multiscale patch / CONCH pipeline."""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

TCGA_LUAD_ROOT = Path("/data4/share/TCGA/TCGA_LUAD")
TCGA_LUAD_WRITE_ROOT = Path(
    os.environ.get("TCGA_LUAD_WRITE_ROOT", "/data4/tujiayong/share/TCGA/TCGA_LUAD")
)
CONCH_LUAD_ROOT = Path("/data4/share/CONCH_TCGA/LUAD/features")
CONCH_LUAD_WRITE_ROOT = Path(
    os.environ.get(
        "CONCH_LUAD_WRITE_ROOT",
        "/data4/tujiayong/share/CONCH_TCGA/LUAD/features",
    )
)
CONCH_DIR = Path("/data4/tujiayong/CONCH")
DEFAULT_CHECKPOINT = CONCH_DIR / "checkpoints/conch/pytorch_model.bin"

VALID_SCALES = (256, 512, 1024)


@dataclass(frozen=True)
class ScalePaths:
    scale: int
    patch_dir: Path
    pt_dir: Path
    h5_dir: Path

    @property
    def skip_conch(self) -> bool:
        return self.scale == 1024


def scale_paths(
    scale: int,
    tcga_root: Path = TCGA_LUAD_ROOT,
    tcga_write: Path | None = None,
    conch_write: Path | None = None,
) -> ScalePaths:
    if scale not in VALID_SCALES:
        raise ValueError(f"scale must be one of {VALID_SCALES}, got {scale}")

    tw = tcga_write or TCGA_LUAD_WRITE_ROOT
    cw = conch_write or CONCH_LUAD_WRITE_ROOT

    if scale == 256:
        patch_dir = tcga_root / "patches/patches"
        pt_dir = CONCH_LUAD_ROOT / "pt_files"
        h5_dir = CONCH_LUAD_ROOT / "h5_files"
    else:
        patch_dir = tw / f"patches/patches_{scale}"
        pt_dir = cw / f"pt_files_{scale}"
        h5_dir = cw / f"h5_files_{scale}"

    return ScalePaths(scale=scale, patch_dir=patch_dir, pt_dir=pt_dir, h5_dir=h5_dir)


def stem_from_name(name: str) -> str:
    for suffix in (".svs", ".h5", ".pt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def slide_id_to_stem(slide_id: str) -> str:
    return stem_from_name(slide_id)


def case_id_from_stem(stem: str) -> str:
    m = re.match(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+)", stem)
    return m.group(1) if m else stem.split(".")[0]


def wsi_path_for_stem(wsi_dir: Path, stem: str) -> Path | None:
    matches = sorted(wsi_dir.glob(f"{stem}.svs"))
    if not matches:
        matches = sorted(wsi_dir.glob(f"{stem}*"))
    return matches[0] if matches else None


def wsi_path_for_stem_processed(processed_root: Path, stem: str) -> Path | None:
    case_id = case_id_from_stem(stem)
    return wsi_path_for_stem(processed_root / case_id / "pathology" / "wsi", stem)


def resolve_wsi_path(
    stem: str,
    wsi_dir: Path | None = None,
    processed_root: Path | None = None,
) -> Path | None:
    if wsi_dir is not None:
        hit = wsi_path_for_stem(wsi_dir, stem)
        if hit is not None:
            return hit
    if processed_root is not None:
        return wsi_path_for_stem_processed(processed_root, stem)
    return None


@dataclass
class SegParams:
    seg_level: int = 3
    sthresh: int = 8
    mthresh: int = 7
    close: int = 4
    use_otsu: bool = False
    a_t: float = 100.0
    a_h: float = 16.0
    max_n_holes: int = 8
    use_padding: bool = True
    contour_fn: str = "four_pt"


def read_process_list(path: Path) -> dict[str, SegParams]:
    """Return stem -> segmentation params from CLAM process_list CSV."""
    out: dict[str, SegParams] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide_id = row["slide_id"]
            stem = slide_id_to_stem(slide_id)
            out[stem] = SegParams(
                seg_level=int(float(row.get("seg_level", 3))),
                sthresh=int(float(row.get("sthresh", 8))),
                mthresh=int(float(row.get("mthresh", 7))),
                close=int(float(row.get("close", 4))),
                use_otsu=str(row.get("use_otsu", "False")).lower() == "true",
                a_t=float(row.get("a_t", 100)),
                a_h=float(row.get("a_h", 16)),
                max_n_holes=int(float(row.get("max_n_holes", 8))),
                use_padding=str(row.get("use_padding", "True")).lower() == "true",
                contour_fn=row.get("contour_fn", "four_pt") or "four_pt",
            )
    return out


def list_wsi_stems(wsi_dir: Path) -> list[str]:
    return sorted(stem_from_name(p.name) for p in wsi_dir.glob("*.svs"))
