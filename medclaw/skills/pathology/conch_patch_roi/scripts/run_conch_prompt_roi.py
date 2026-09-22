#!/usr/bin/env python3
"""CONCH prompt scoring on precomputed patch features + WSI ROI crop."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import openslide
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("conch_prompt_roi")

CONCH_DIR = Path("/data4/tujiayong/CONCH")
DEFAULT_CHECKPOINT = CONCH_DIR / "checkpoints/conch/pytorch_model.bin"
DEFAULT_PROMPTS = Path(__file__).resolve().parent.parent / "prompts/luad_default.json"


@dataclass
class SlidePaths:
    stem: str
    conch_pt: Path
    coords_h5: Path
    wsi_path: Path | None


@dataclass
class SlideResult:
    stem: str
    status: str
    n_patches: int = 0
    n_roi: int = 0
    error: str = ""
    wsi_path: str = ""


def stem_from_pt(path: Path) -> str:
    return path.name[:-3] if path.suffix == ".pt" else path.stem


def case_id_from_stem(stem: str) -> str:
    m = re.match(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+)", stem)
    return m.group(1) if m else stem.split(".")[0]


def discover_slides(
    conch_feat_dir: Path,
    patch_dir: Path,
    wsi_dir: Path,
) -> list[SlidePaths]:
    slides: list[SlidePaths] = []
    for pt in sorted(conch_feat_dir.glob("*.pt")):
        stem = stem_from_pt(pt)
        coords_h5 = patch_dir / f"{stem}.h5"
        if not coords_h5.exists():
            continue
        wsi_matches = sorted(wsi_dir.glob(f"{stem}.svs"))
        if not wsi_matches:
            wsi_matches = sorted(wsi_dir.glob(f"{stem}*"))
        wsi_path = wsi_matches[0] if wsi_matches else None
        slides.append(
            SlidePaths(
                stem=stem,
                conch_pt=pt,
                coords_h5=coords_h5,
                wsi_path=wsi_path,
            )
        )
    return slides


def load_prompts(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data["prompts"] if isinstance(data, dict) else data
    for i, p in enumerate(prompts):
        if "id" not in p:
            p["id"] = f"prompt_{i:02d}"
        if "text" not in p:
            raise ValueError(f"prompt missing text: {p}")
    return prompts


def conch_tokenize(tokenizer, texts: list[str]) -> torch.Tensor:
    """CONCH tokenization; works with newer transformers (no batch_encode_plus)."""
    enc = tokenizer(
        texts,
        max_length=127,
        add_special_tokens=True,
        return_token_type_ids=False,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return F.pad(enc["input_ids"], (0, 1), value=tokenizer.pad_token_id)


def load_conch_text_encoder(checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(CONCH_DIR))
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

    model, _ = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path=str(checkpoint),
    )
    model = model.to(device).eval()
    tokenizer = get_tokenizer()
    return model, tokenizer


@torch.no_grad()
def encode_prompts(model, tokenizer, prompts: list[dict], device: torch.device):
    texts = [p["text"] for p in prompts]
    token_ids = conch_tokenize(tokenizer, texts).to(device)
    text_emb = model.encode_text(token_ids)
    text_emb = F.normalize(text_emb, dim=-1).cpu()
    return text_emb


def score_patches(patch_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    patch_emb = F.normalize(patch_emb.float(), dim=-1)
    return patch_emb @ text_emb.T


def read_patch_params(coords_h5: Path) -> tuple[int, int]:
    with h5py.File(coords_h5, "r") as f:
        c = f["coords"]
        patch_size = int(c.attrs.get("patch_size", 256))
        patch_level = int(c.attrs.get("patch_level", 0))
    return patch_size, patch_level


def crop_rois(
    wsi_path: Path,
    coords: np.ndarray,
    indices: np.ndarray,
    patch_size: int,
    patch_level: int,
    out_dir: Path,
    prompt_id: str,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    slide = openslide.OpenSlide(str(wsi_path))
    saved: list[str] = []
    try:
        for rank, idx in enumerate(indices):
            x, y = int(coords[idx, 0]), int(coords[idx, 1])
            region = slide.read_region((x, y), patch_level, (patch_size, patch_size))
            img = region.convert("RGB")
            fname = f"{prompt_id}_roi_{rank:02d}_idx{int(idx):05d}.png"
            fpath = out_dir / fname
            img.save(fpath)
            saved.append(str(fpath))
    finally:
        slide.close()
    return saved


def process_slide(
    sp: SlidePaths,
    prompts: list[dict],
    text_emb: torch.Tensor,
    out_root: Path,
    top_k: int,
    device: torch.device,
    skip_crop: bool,
) -> SlideResult:
    try:
        patch_emb = torch.load(sp.conch_pt, map_location="cpu", weights_only=False)
        if not isinstance(patch_emb, torch.Tensor):
            raise TypeError(f"expected Tensor in {sp.conch_pt}, got {type(patch_emb)}")

        with h5py.File(sp.coords_h5, "r") as f:
            coords = f["coords"][:]

        if patch_emb.shape[0] != coords.shape[0]:
            raise ValueError(
                f"N mismatch: features {patch_emb.shape[0]} vs coords {coords.shape[0]}"
            )

        scores = score_patches(patch_emb, text_emb).numpy()
        patch_size, patch_level = read_patch_params(sp.coords_h5)

        case_id = case_id_from_stem(sp.stem)
        slide_out = out_root / case_id / "pathology" / f"roi_{patch_size}" / sp.stem
        slide_out.mkdir(parents=True, exist_ok=True)

        prompt_results = []
        all_saved: list[str] = []
        for j, pr in enumerate(prompts):
            pid = pr["id"]
            sc = scores[:, j]
            order = np.argsort(-sc)[:top_k]
            top_scores = sc[order].tolist()
            top_coords = coords[order].tolist()

            roi_paths: list[str] = []
            if not skip_crop:
                if sp.wsi_path is None:
                    raise FileNotFoundError(f"no WSI for {sp.stem}")
                roi_paths = crop_rois(
                    sp.wsi_path,
                    coords,
                    order,
                    patch_size,
                    patch_level,
                    slide_out / "roi" / pid,
                    pid,
                )
                all_saved.extend(roi_paths)

            prompt_results.append(
                {
                    "id": pid,
                    "text": pr["text"],
                    "top_indices": order.tolist(),
                    "top_scores": top_scores,
                    "top_coords": top_coords,
                    "roi_paths": roi_paths,
                }
            )

        payload = {
            "stem": sp.stem,
            "case_id": case_id,
            "n_patches": int(patch_emb.shape[0]),
            "feature_dim": int(patch_emb.shape[1]),
            "patch_size": patch_size,
            "patch_level": patch_level,
            "top_k": top_k,
            "conch_pt": str(sp.conch_pt),
            "coords_h5": str(sp.coords_h5),
            "wsi_path": str(sp.wsi_path) if sp.wsi_path else None,
            "prompts": prompt_results,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        scores_path = slide_out / "prompt_scores.json"
        scores_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        packet = {
            "case_id": case_id,
            "stem": sp.stem,
            "modality": "pathology",
            "roi_packet_version": "1",
            "prompts": [
                {
                    "id": p["id"],
                    "text": p["text"],
                    "images": p["roi_paths"],
                    "top_scores": p["top_scores"],
                }
                for p in prompt_results
            ],
        }
        (slide_out / "roi_packet.json").write_text(
            json.dumps(packet, indent=2), encoding="utf-8"
        )

        return SlideResult(
            stem=sp.stem,
            status="ok",
            n_patches=int(patch_emb.shape[0]),
            n_roi=len(all_saved),
            wsi_path=str(sp.wsi_path) if sp.wsi_path else "",
        )
    except Exception as e:
        logger.exception("failed %s", sp.stem)
        return SlideResult(stem=sp.stem, status="error", error=str(e))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CONCH prompt ROI pipeline")
    p.add_argument(
        "--conch-feat-dir",
        type=Path,
        default=Path("/data4/share/CONCH_TCGA/LUAD/features/pt_files"),
    )
    p.add_argument(
        "--patch-dir",
        type=Path,
        default=Path("/data4/share/TCGA/TCGA_LUAD/patches/patches"),
    )
    p.add_argument(
        "--wsi-dir",
        type=Path,
        default=Path("/data4/share/TCGA/TCGA_LUAD/wsi"),
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("/data4/tujiayong/processed/LUNG"),
    )
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--stem", type=str, default="", help="single slide stem")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-crop", action="store_true")
    p.add_argument("--log-json", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        logger.error("CONCH checkpoint missing: %s", args.checkpoint)
        return 1

    prompts = load_prompts(args.prompts)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    logger.info("device=%s prompts=%d", device, len(prompts))

    model, tokenizer = load_conch_text_encoder(args.checkpoint, device)
    text_emb = encode_prompts(model, tokenizer, prompts, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    slides = discover_slides(args.conch_feat_dir, args.patch_dir, args.wsi_dir)
    if args.stem:
        slides = [s for s in slides if s.stem == args.stem]
    if args.limit > 0:
        slides = slides[: args.limit]

    logger.info("slides to process: %d", len(slides))
    if not slides:
        logger.error("no slides found")
        return 1

    results: list[SlideResult] = []
    for sp in tqdm(slides, desc="slides"):
        if sp.wsi_path is None and not args.skip_crop:
            results.append(
                SlideResult(
                    stem=sp.stem,
                    status="skip",
                    error="wsi not found",
                )
            )
            continue
        results.append(
            process_slide(
                sp,
                prompts,
                text_emb,
                args.out_root,
                args.top_k,
                device,
                args.skip_crop,
            )
        )

    ok = sum(1 for r in results if r.status == "ok")
    err = sum(1 for r in results if r.status == "error")
    skip = sum(1 for r in results if r.status == "skip")
    logger.info("done ok=%d error=%d skip=%d", ok, err, skip)

    log_path = args.log_json
    if log_path is None:
        log_dir = Path(
            os.environ.get(
                "MEDCLAW_PATHOLOGY_LOG_DIR",
                str(Path(__file__).resolve().parent.parent / "logs"),
            )
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"conch_prompt_roi_{ts}.json"

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "ok": ok,
        "error": err,
        "skip": skip,
        "results": [asdict(r) for r in results],
    }
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote log %s", log_path)
    return 0 if err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
