#!/usr/bin/env python3
"""Extract CONCH patch features from WSI + coordinate H5 files."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import openslide
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _luad_multiscale_common import (  # noqa: E402
    CONCH_DIR,
    DEFAULT_CHECKPOINT,
    TCGA_LUAD_ROOT,
    scale_paths,
    wsi_path_for_stem,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("luad_extract_conch")


def read_patch_params(coords_h5: Path) -> tuple[int, int]:
    with h5py.File(coords_h5, "r") as f:
        c = f["coords"]
        patch_size = int(c.attrs.get("patch_size", 256))
        patch_level = int(c.attrs.get("patch_level", 0))
    return patch_size, patch_level


def load_conch_model(checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(CONCH_DIR))
    from conch.open_clip_custom import create_model_from_pretrained

    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path=str(checkpoint),
    )
    model = model.to(device).eval()
    return model, preprocess


@torch.no_grad()
def extract_slide(
    model,
    preprocess,
    wsi_path: Path,
    coords: np.ndarray,
    patch_size: int,
    patch_level: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    slide = openslide.OpenSlide(str(wsi_path))
    feats: list[torch.Tensor] = []
    try:
        for start in range(0, len(coords), batch_size):
            batch_coords = coords[start : start + batch_size]
            imgs = []
            for x, y in batch_coords:
                region = slide.read_region(
                    (int(x), int(y)),
                    patch_level,
                    (patch_size, patch_size),
                )
                imgs.append(preprocess(region.convert("RGB")))
            batch = torch.stack(imgs).to(device)
            emb = model.encode_image(batch, proj_contrast=False, normalize=False)
            feats.append(emb.cpu())
    finally:
        slide.close()
    if not feats:
        return torch.zeros((0, 512))
    return torch.cat(feats, dim=0)


def process_one(
    stem: str,
    coords_h5: Path,
    wsi_dir: Path,
    out_pt: Path,
    out_h5: Path | None,
    model,
    preprocess,
    device: torch.device,
    batch_size: int,
    skip_existing: bool,
) -> dict:
    if skip_existing and out_pt.exists():
        feat = torch.load(out_pt, map_location="cpu", weights_only=False)
        n = int(feat.shape[0]) if isinstance(feat, torch.Tensor) else 0
        return {"stem": stem, "status": "skipped", "n_features": n}

    wsi_path = wsi_path_for_stem(wsi_dir, stem)
    if wsi_path is None:
        return {"stem": stem, "status": "no_wsi", "n_features": 0}

    with h5py.File(coords_h5, "r") as f:
        coords = f["coords"][:]
    patch_size, patch_level = read_patch_params(coords_h5)

    if len(coords) == 0:
        feat = torch.zeros((0, 512))
    else:
        feat = extract_slide(
            model,
            preprocess,
            wsi_path,
            coords,
            patch_size,
            patch_level,
            device,
            batch_size,
        )

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feat, out_pt)

    if out_h5 is not None:
        out_h5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_h5, "w") as f:
            f.create_dataset("coords", data=coords, compression="gzip")
            f.create_dataset("features", data=feat.numpy(), compression="gzip")
            f["coords"].attrs["patch_size"] = patch_size
            f["coords"].attrs["patch_level"] = patch_level

    return {"stem": stem, "status": "written", "n_features": int(feat.shape[0])}


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract CONCH features for LUAD patches")
    ap.add_argument("--scale", type=int, choices=[256, 512, 1024], default=512)
    ap.add_argument("--tcga-root", type=Path, default=TCGA_LUAD_ROOT)
    ap.add_argument("--wsi-dir", type=Path, default=None)
    ap.add_argument("--coords-dir", type=Path, default=None)
    ap.add_argument("--out-pt-dir", type=Path, default=None)
    ap.add_argument("--out-h5-dir", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--stem", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-merged-h5", action="store_true")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        logger.error("checkpoint missing: %s", args.checkpoint)
        return 1

    paths = scale_paths(args.scale, args.tcga_root)
    wsi_dir = args.wsi_dir or (args.tcga_root / "wsi")
    coords_dir = args.coords_dir or paths.patch_dir
    out_pt_dir = args.out_pt_dir or paths.pt_dir
    out_h5_dir = None if args.no_merged_h5 else (args.out_h5_dir or paths.h5_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("device=%s scale=%s coords=%s", device, args.scale, coords_dir)

    model, preprocess = load_conch_model(args.checkpoint, device)

    if args.stem:
        stems = [args.stem]
    else:
        stems = sorted(p.stem for p in coords_dir.glob("*.h5"))
        if args.limit > 0:
            stems = stems[: args.limit]

    logger.info("slides: %d", len(stems))
    if not stems:
        logger.error("no coords H5 found in %s", coords_dir)
        return 1

    results = []
    for stem in tqdm(stems, desc=f"conch_{args.scale}"):
        coords_h5 = coords_dir / f"{stem}.h5"
        if not coords_h5.exists():
            results.append({"stem": stem, "status": "no_coords", "n_features": 0})
            continue
        results.append(
            process_one(
                stem=stem,
                coords_h5=coords_h5,
                wsi_dir=wsi_dir,
                out_pt=out_pt_dir / f"{stem}.pt",
                out_h5=(out_h5_dir / f"{stem}.h5") if out_h5_dir else None,
                model=model,
                preprocess=preprocess,
                device=device,
                batch_size=args.batch_size,
                skip_existing=args.skip_existing,
            )
        )

    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    logger.info("done: %s", status_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
