#!/usr/bin/env python3
"""Verify LUAD multiscale patch coords and CONCH features."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import openslide
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _luad_multiscale_common import (  # noqa: E402
    TCGA_LUAD_ROOT,
    scale_paths,
    wsi_path_for_stem,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("luad_verify_scale")


def verify_slide(
    stem: str,
    coords_h5: Path,
    pt_path: Path | None,
    wsi_dir: Path,
    expect_size: int,
    sample_reads: int,
) -> dict:
    issues: list[str] = []

    if not coords_h5.exists():
        return {"stem": stem, "status": "missing_coords", "issues": ["coords h5 missing"]}

    try:
        with h5py.File(coords_h5, "r") as f:
            coords = f["coords"]
            n_coords = int(coords.shape[0])
            cattrs = dict(coords.attrs)
            fattrs = dict(f.attrs)
            patch_size = int(cattrs.get("patch_size", fattrs.get("patch_size", -1)))
            patch_level = int(cattrs.get("patch_level", fattrs.get("patch_level", -1)))
    except OSError as e:
        return {"stem": stem, "status": "corrupt", "issues": [str(e)]}

    if patch_size != expect_size:
        issues.append(f"patch_size={patch_size}, expected {expect_size}")
    if patch_level != 0:
        issues.append(f"patch_level={patch_level}, expected 0")

    n_feat = None
    if pt_path is not None:
        if not pt_path.exists():
            issues.append("pt missing")
        else:
            feat = torch.load(pt_path, map_location="cpu", weights_only=False)
            if not isinstance(feat, torch.Tensor):
                issues.append(f"pt type {type(feat)}")
            else:
                n_feat = int(feat.shape[0])
                if feat.shape[1] != 512:
                    issues.append(f"feature_dim={feat.shape[1]}, expected 512")
                if n_feat != n_coords:
                    issues.append(f"N mismatch coords={n_coords} pt={n_feat}")

    read_ok = 0
    if sample_reads > 0 and n_coords > 0:
        wsi_path = wsi_path_for_stem(wsi_dir, stem)
        if wsi_path is None:
            issues.append("wsi missing for sample read")
        else:
            idx = random.sample(range(n_coords), min(sample_reads, n_coords))
            slide = openslide.OpenSlide(str(wsi_path))
            try:
                with h5py.File(coords_h5, "r") as f:
                    coords_arr = f["coords"][:]
                    ps = int(f["coords"].attrs.get("patch_size", expect_size))
                    pl = int(f["coords"].attrs.get("patch_level", 0))
                for i in idx:
                    x, y = int(coords_arr[i, 0]), int(coords_arr[i, 1])
                    try:
                        region = slide.read_region((x, y), pl, (ps, ps))
                        if region.size[0] != ps or region.size[1] != ps:
                            issues.append(f"bad read size at idx={i}")
                        else:
                            read_ok += 1
                    except Exception as e:
                        issues.append(f"read failed idx={i}: {e}")
            finally:
                slide.close()

    status = "ok" if not issues else "fail"
    return {
        "stem": stem,
        "status": status,
        "n_coords": n_coords,
        "n_features": n_feat,
        "read_ok": read_ok,
        "issues": issues,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify LUAD multiscale outputs")
    ap.add_argument("--scale", type=int, choices=[256, 512, 1024], required=True)
    ap.add_argument("--tcga-root", type=Path, default=TCGA_LUAD_ROOT)
    ap.add_argument("--coords-dir", type=Path, default=None)
    ap.add_argument("--pt-dir", type=Path, default=None)
    ap.add_argument("--wsi-dir", type=Path, default=None)
    ap.add_argument("--expect-size", type=int, default=None)
    ap.add_argument("--skip-pt", action="store_true", help="Skip .pt checks (e.g. 1024)")
    ap.add_argument("--sample-reads", type=int, default=3)
    ap.add_argument("--stem", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = scale_paths(args.scale, args.tcga_root)
    coords_dir = args.coords_dir or paths.patch_dir
    pt_dir = None if args.skip_pt or paths.skip_conch else (args.pt_dir or paths.pt_dir)
    wsi_dir = args.wsi_dir or (args.tcga_root / "wsi")
    expect_size = args.expect_size or args.scale

    if args.stem:
        stems = [args.stem]
    else:
        stems = sorted(p.stem for p in coords_dir.glob("*.h5"))
        if args.limit > 0:
            stems = stems[: args.limit]

    logger.info(
        "verify scale=%s coords=%s pt=%s slides=%d",
        args.scale,
        coords_dir,
        pt_dir,
        len(stems),
    )

    ok = fail = missing = corrupt = 0
    for stem in stems:
        pt_path = (pt_dir / f"{stem}.pt") if pt_dir else None
        r = verify_slide(stem, coords_dir / f"{stem}.h5", pt_path, wsi_dir, expect_size, args.sample_reads)
        if r["status"] == "ok":
            ok += 1
        elif r["status"] == "missing_coords":
            missing += 1
            logger.warning("%s: missing coords", stem)
        elif r["status"] == "corrupt":
            corrupt += 1
            logger.warning("%s: corrupt coords (%s)", stem, "; ".join(r["issues"]))
        else:
            fail += 1
            logger.warning("%s: %s", stem, "; ".join(r["issues"]))

    logger.info("summary ok=%d fail=%d missing=%d corrupt=%d", ok, fail, missing, corrupt)
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
