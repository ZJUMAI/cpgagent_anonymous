#!/usr/bin/env python3
"""Create patch coordinate H5 files for LUAD at 256 / 512 / 1024 px (level 0).

Modes:
  segment         — CLAM-style tissue segmentation + grid (default for new scales)
  derive_from_256 — build 512/1024 coords from existing 256 H5 (faster, aligned grid)
  verify_only     — skip writing; only report existing outputs (256 default)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import openslide
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _luad_multiscale_common import (  # noqa: E402
    TCGA_LUAD_ROOT,
    SegParams,
    list_wsi_stems,
    read_process_list,
    scale_paths,
    slide_id_to_stem,
    wsi_path_for_stem,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("luad_create_patches")


def segment_tissue(
    slide: openslide.OpenSlide,
    params: SegParams,
) -> tuple[np.ndarray, float]:
    """Return binary tissue mask at seg_level and downsample factor to level 0."""
    seg_level = min(params.seg_level, slide.level_count - 1)
    w, h = slide.level_dimensions[seg_level]
    thumb = slide.read_region((0, 0), seg_level, (w, h)).convert("RGB")
    img = np.array(thumb)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = cv2.medianBlur(hsv[:, :, 1], params.mthresh)
    if params.use_otsu:
        _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(sat, params.sthresh, 255, cv2.THRESH_BINARY)

    if params.close > 0:
        kernel = np.ones((params.close, params.close), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    downsample = float(slide.level_downsamples[seg_level])
    return mask, downsample


def filter_contours(
    mask: np.ndarray,
    params: SegParams,
) -> list[np.ndarray]:
    """Keep tissue contours; fill small holes inside each contour."""
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    keep: list[np.ndarray] = []
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] != -1:
            continue
        area = cv2.contourArea(cnt)
        if area < params.a_t:
            continue

        hole_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(hole_mask, [cnt], -1, 255, thickness=-1)

        child = hierarchy[i][2]
        n_holes = 0
        while child != -1:
            hole_area = cv2.contourArea(contours[child])
            if hole_area >= params.a_h:
                cv2.drawContours(hole_mask, [contours[child]], -1, 0, thickness=-1)
            n_holes += 1
            child = hierarchy[child][0]
            if n_holes > params.max_n_holes:
                break

        keep.append(hole_mask)
    return keep


def patch_passes(
    tissue_masks: list[np.ndarray],
    x0: int,
    y0: int,
    patch_size: int,
    downsample: float,
    params: SegParams,
) -> bool:
    if not tissue_masks:
        return False

    sx = int(x0 / downsample)
    sy = int(y0 / downsample)
    ps = max(1, int(patch_size / downsample))

    h, w = tissue_masks[0].shape
    if sx < 0 or sy < 0 or sx + ps > w or sy + ps > h:
        return False

    if params.contour_fn == "four_pt":
        pts = [
            (sx, sy),
            (sx + ps - 1, sy),
            (sx, sy + ps - 1),
            (sx + ps - 1, sy + ps - 1),
        ]
        for px, py in pts:
            ok = any(m[py, px] > 0 for m in tissue_masks)
            if not ok:
                return False
        return True

    # center point fallback
    cx, cy = sx + ps // 2, sy + ps // 2
    return any(m[cy, cx] > 0 for m in tissue_masks)


def load_cached_mask(mask_path: Path) -> np.ndarray:
    """Load CLAM-style mask JPG (non-zero pixels = tissue)."""
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read mask: {mask_path}")
    return (img > 0).astype(np.uint8) * 255


def grid_coords_from_tissue(
    slide: openslide.OpenSlide,
    tissue: np.ndarray,
    downsample: float,
    patch_size: int,
    step: int,
    params: SegParams,
) -> np.ndarray:
    tissue_masks = filter_contours(tissue, params)
    w0, h0 = slide.dimensions
    coords: list[list[int]] = []
    for y in range(0, h0 - patch_size + 1, step):
        for x in range(0, w0 - patch_size + 1, step):
            if patch_passes(tissue_masks, x, y, patch_size, downsample, params):
                coords.append([x, y])
    if not coords:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(coords, dtype=np.int64)


def grid_coords_segment(
    slide: openslide.OpenSlide,
    patch_size: int,
    step: int,
    params: SegParams,
    mask_path: Path | None = None,
    skip_seg: bool = False,
) -> np.ndarray:
    seg_level = min(params.seg_level, slide.level_count - 1)
    downsample = float(slide.level_downsamples[seg_level])

    if skip_seg and mask_path is not None and mask_path.exists():
        tissue = load_cached_mask(mask_path)
    else:
        tissue, downsample = segment_tissue(slide, params)

    return grid_coords_from_tissue(slide, tissue, downsample, patch_size, step, params)


def derive_coords_from_256(
    coords256: np.ndarray,
    target_size: int,
    base_size: int = 256,
) -> np.ndarray:
    """Aggregate 512/1024 origins from CLAM 256 grid (supports padded offsets)."""
    if target_size % base_size != 0:
        raise ValueError("target_size must be a multiple of base_size")

    factor = target_size // base_size
    tile_set = {tuple(map(int, c)) for c in coords256}

    out: list[list[int]] = []
    for x, y in sorted(tile_set):
        ok = True
        for dy in range(factor):
            for dx in range(factor):
                if (x + dx * base_size, y + dy * base_size) not in tile_set:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            out.append([x, y])

    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(out, dtype=np.int64)


def write_coords_h5(path: Path, coords: np.ndarray, attrs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("coords", data=coords, compression="gzip")
        for k, v in attrs.items():
            ds.attrs[k] = v


def process_slide(
    stem: str,
    wsi_dir: Path,
    out_dir: Path,
    patch_size: int,
    step: int,
    mode: str,
    params: SegParams,
    coords256_dir: Path | None,
    skip_existing: bool,
    mask_dir: Path | None = None,
    skip_seg: bool = False,
) -> dict:
    out_h5 = out_dir / f"{stem}.h5"
    if skip_existing and out_h5.exists():
        with h5py.File(out_h5, "r") as f:
            n = int(f["coords"].shape[0])
        return {"stem": stem, "status": "skipped", "n_coords": n}

    if mode == "verify_only":
        if not out_h5.exists():
            return {"stem": stem, "status": "missing", "n_coords": 0}
        try:
            with h5py.File(out_h5, "r") as f:
                n = int(f["coords"].shape[0])
            return {"stem": stem, "status": "ok", "n_coords": n}
        except OSError as e:
            return {"stem": stem, "status": "corrupt", "n_coords": 0, "error": str(e)}

    if mode == "derive_from_256":
        src = coords256_dir / f"{stem}.h5"
        if not src.exists():
            return {"stem": stem, "status": "no_256", "n_coords": 0}
        try:
            with h5py.File(src, "r") as f:
                c256 = f["coords"][:]
        except OSError as e:
            return {"stem": stem, "status": "corrupt_256", "n_coords": 0, "error": str(e)}
        coords = derive_coords_from_256(c256, patch_size)
    else:
        wsi_path = wsi_path_for_stem(wsi_dir, stem)
        if wsi_path is None:
            return {"stem": stem, "status": "no_wsi", "n_coords": 0}
        mask_path = (mask_dir / f"{stem}.jpg") if mask_dir else None
        slide = openslide.OpenSlide(str(wsi_path))
        try:
            coords = grid_coords_segment(
                slide,
                patch_size,
                step,
                params,
                mask_path=mask_path,
                skip_seg=skip_seg,
            )
        finally:
            slide.close()

    attrs = {
        "patch_size": patch_size,
        "patch_level": 0,
        "step": step,
        "mode": mode,
        "source": "256_clam" if mode == "derive_from_256" else mode,
    }
    write_coords_h5(out_h5, coords, attrs)
    return {"stem": stem, "status": "written", "n_coords": int(coords.shape[0])}


def default_mode(scale: int) -> str:
    if scale == 256:
        return "verify_only"
    return "derive_from_256"


def main() -> int:
    ap = argparse.ArgumentParser(description="LUAD multiscale patch coords (H5)")
    ap.add_argument("--scale", type=int, choices=[256, 512, 1024], required=True)
    ap.add_argument("--tcga-root", type=Path, default=TCGA_LUAD_ROOT)
    ap.add_argument("--wsi-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--process-list",
        type=Path,
        default=TCGA_LUAD_ROOT / "patches/process_list_autogen.csv",
    )
    ap.add_argument(
        "--coords256-dir",
        type=Path,
        default=TCGA_LUAD_ROOT / "patches/patches",
        help="Source 256 H5 dir for derive_from_256 mode",
    )
    ap.add_argument(
        "--mode",
        choices=["segment", "derive_from_256", "verify_only"],
        default=None,
    )
    ap.add_argument("--step", type=int, default=None, help="Grid step (default=scale)")
    ap.add_argument("--stem", type=str, default="", help="Single slide stem")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--mask-dir",
        type=Path,
        default=TCGA_LUAD_ROOT / "patches/masks",
        help="CLAM tissue mask JPG dir (reuse with --skip-seg)",
    )
    ap.add_argument(
        "--skip-seg",
        action="store_true",
        help="Reuse masks/{stem}.jpg instead of re-segmenting",
    )
    args = ap.parse_args()

    paths = scale_paths(args.scale, args.tcga_root)
    wsi_dir = args.wsi_dir or (args.tcga_root / "wsi")
    out_dir = args.out_dir or paths.patch_dir
    step = args.step or args.scale
    mode = args.mode or default_mode(args.scale)

    seg_map = read_process_list(args.process_list) if args.process_list.exists() else {}

    if args.stem:
        stems = [args.stem]
    else:
        stems = list_wsi_stems(wsi_dir)
        if mode == "derive_from_256":
            stems = [
                slide_id_to_stem(p.name)
                for p in args.coords256_dir.glob("*.h5")
            ]
            stems = sorted(set(stems))
        if args.limit > 0:
            stems = stems[: args.limit]

    logger.info(
        "scale=%s mode=%s step=%s out=%s slides=%d",
        args.scale,
        mode,
        step,
        out_dir,
        len(stems),
    )

    results = []
    for stem in tqdm(stems, desc=f"patches_{args.scale}"):
        params = seg_map.get(stem, SegParams())
        results.append(
            process_slide(
                stem=stem,
                wsi_dir=wsi_dir,
                out_dir=out_dir,
                patch_size=args.scale,
                step=step,
                mode=mode,
                params=params,
                coords256_dir=args.coords256_dir,
                skip_existing=args.skip_existing,
                mask_dir=args.mask_dir,
                skip_seg=args.skip_seg,
            )
        )

    status_counts: dict[str, int] = {}
    total_coords = 0
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        total_coords += r["n_coords"]
        if r["status"] in ("corrupt", "missing") and mode == "verify_only":
            extra = f" ({r['error']})" if r.get("error") else ""
            logger.warning("%s: %s%s", r["stem"], r["status"], extra)

    logger.info("done: %s", status_counts)
    logger.info("total coords across slides: %d", total_coords)
    corrupt = status_counts.get("corrupt", 0)
    missing = status_counts.get("missing", 0)
    if mode == "verify_only" and corrupt:
        logger.warning("corrupt h5 files: %d (listed above)", corrupt)
    if mode == "verify_only" and missing:
        logger.warning("missing h5 files: %d", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
