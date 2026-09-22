#!/usr/bin/env python3
"""CONCH prompt scoring + WSI ROI crop for UCEC pathology (mirror LUAD)."""

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

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from _ucec_conch_common import (  # noqa: E402
    CONCH_DIR,
    DEFAULT_CHECKPOINT,
    default_paths,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ucec_conch_prompt_roi")

DEFAULT_PROMPTS = SKILL_DIR / "prompts" / "ucec_default.json"


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
    match = re.match(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+)", stem)
    if match:
        return match.group(1)
    digits = re.search(r"(\d{8})", stem)
    if digits:
        return digits.group(1)
    return stem.split(".")[0]


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
        if not wsi_matches and wsi_dir.is_dir():
            case_id = case_id_from_stem(stem)
            case_wsi = wsi_dir / case_id
            if case_wsi.is_dir():
                wsi_matches = sorted(case_wsi.glob(f"{stem}*")) or sorted(
                    case_wsi.glob("*.svs")
                )
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
    for index, prompt in enumerate(prompts):
        if "id" not in prompt:
            prompt["id"] = f"prompt_{index:02d}"
        if "text" not in prompt:
            raise ValueError(f"prompt missing text: {prompt}")
    return prompts


def conch_tokenize(tokenizer, texts: list[str]) -> torch.Tensor:
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
    texts = [prompt["text"] for prompt in prompts]
    token_ids = conch_tokenize(tokenizer, texts).to(device)
    text_emb = model.encode_text(token_ids)
    text_emb = F.normalize(text_emb, dim=-1).cpu()
    return text_emb


def score_patches(patch_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    patch_emb = F.normalize(patch_emb.float(), dim=-1)
    return patch_emb @ text_emb.T


def read_patch_params(coords_h5: Path) -> tuple[int, int]:
    with h5py.File(coords_h5, "r") as handle:
        coords = handle["coords"]
        patch_size = int(coords.attrs.get("patch_size", 256))
        patch_level = int(coords.attrs.get("patch_level", 0))
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
    slide_paths: SlidePaths,
    prompts: list[dict],
    text_emb: torch.Tensor,
    out_root: Path,
    top_k: int,
    device: torch.device,
    skip_crop: bool,
) -> SlideResult:
    try:
        patch_emb = torch.load(
            slide_paths.conch_pt,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(patch_emb, torch.Tensor):
            raise TypeError(
                f"expected Tensor in {slide_paths.conch_pt}, got {type(patch_emb)}"
            )

        with h5py.File(slide_paths.coords_h5, "r") as handle:
            coords = handle["coords"][:]

        if patch_emb.shape[0] != coords.shape[0]:
            raise ValueError(
                f"N mismatch: features {patch_emb.shape[0]} vs coords {coords.shape[0]}"
            )

        scores = score_patches(patch_emb, text_emb).numpy()
        patch_size, patch_level = read_patch_params(slide_paths.coords_h5)

        case_id = case_id_from_stem(slide_paths.stem)
        slide_out = out_root / case_id / "pathology" / "conch_roi" / slide_paths.stem
        slide_out.mkdir(parents=True, exist_ok=True)

        prompt_results = []
        all_saved: list[str] = []
        for index, prompt in enumerate(prompts):
            prompt_id = prompt["id"]
            prompt_scores = scores[:, index]
            order = np.argsort(-prompt_scores)[:top_k]
            top_scores = prompt_scores[order].tolist()
            top_coords = coords[order].tolist()

            roi_paths: list[str] = []
            if not skip_crop:
                if slide_paths.wsi_path is None:
                    raise FileNotFoundError(f"no WSI for {slide_paths.stem}")
                roi_paths = crop_rois(
                    slide_paths.wsi_path,
                    coords,
                    order,
                    patch_size,
                    patch_level,
                    slide_out / "roi" / prompt_id,
                    prompt_id,
                )
                all_saved.extend(roi_paths)

            prompt_results.append(
                {
                    "id": prompt_id,
                    "text": prompt["text"],
                    "top_indices": order.tolist(),
                    "top_scores": top_scores,
                    "top_coords": top_coords,
                    "roi_paths": roi_paths,
                }
            )

        payload = {
            "stem": slide_paths.stem,
            "case_id": case_id,
            "n_patches": int(patch_emb.shape[0]),
            "feature_dim": int(patch_emb.shape[1]),
            "patch_size": patch_size,
            "patch_level": patch_level,
            "top_k": top_k,
            "conch_pt": str(slide_paths.conch_pt),
            "coords_h5": str(slide_paths.coords_h5),
            "wsi_path": str(slide_paths.wsi_path) if slide_paths.wsi_path else None,
            "prompts": prompt_results,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        scores_path = slide_out / "prompt_scores.json"
        scores_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        packet = {
            "case_id": case_id,
            "stem": slide_paths.stem,
            "modality": "pathology",
            "cancer_type": "UCEC",
            "roi_packet_version": "1",
            "prompts": [
                {
                    "id": item["id"],
                    "text": item["text"],
                    "images": item["roi_paths"],
                    "top_scores": item["top_scores"],
                }
                for item in prompt_results
            ],
        }
        (slide_out / "roi_packet.json").write_text(
            json.dumps(packet, indent=2),
            encoding="utf-8",
        )

        return SlideResult(
            stem=slide_paths.stem,
            status="ok",
            n_patches=int(patch_emb.shape[0]),
            n_roi=len(all_saved),
            wsi_path=str(slide_paths.wsi_path) if slide_paths.wsi_path else "",
        )
    except Exception as exc:
        logger.exception("failed %s", slide_paths.stem)
        return SlideResult(stem=slide_paths.stem, status="error", error=str(exc))


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser(description="UCEC CONCH prompt ROI pipeline")
    parser.add_argument("--conch-feat-dir", type=Path, default=defaults.conch_feat_dir)
    parser.add_argument("--patch-dir", type=Path, default=defaults.patch_dir)
    parser.add_argument("--wsi-dir", type=Path, default=defaults.wsi_dir)
    parser.add_argument("--out-root", type=Path, default=defaults.out_root)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--stem", type=str, default="", help="single slide stem")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-crop", action="store_true")
    parser.add_argument("--log-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        logger.error("CONCH checkpoint missing: %s", args.checkpoint)
        return 1

    prompts = load_prompts(args.prompts)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("device=%s prompts=%d", device, len(prompts))

    model, tokenizer = load_conch_text_encoder(args.checkpoint, device)
    text_emb = encode_prompts(model, tokenizer, prompts, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    slides = discover_slides(args.conch_feat_dir, args.patch_dir, args.wsi_dir)
    if args.stem:
        slides = [slide for slide in slides if slide.stem == args.stem]
    if args.limit > 0:
        slides = slides[: args.limit]

    logger.info("slides to process: %d", len(slides))
    if not slides:
        logger.error("no slides found")
        return 1

    results: list[SlideResult] = []
    for slide in tqdm(slides, desc="slides"):
        if slide.wsi_path is None and not args.skip_crop:
            results.append(
                SlideResult(
                    stem=slide.stem,
                    status="skip",
                    error="wsi not found",
                )
            )
            continue
        results.append(
            process_slide(
                slide,
                prompts,
                text_emb,
                args.out_root,
                args.top_k,
                device,
                args.skip_crop,
            )
        )

    ok = sum(1 for item in results if item.status == "ok")
    err = sum(1 for item in results if item.status == "error")
    skip = sum(1 for item in results if item.status == "skip")
    logger.info("done ok=%d error=%d skip=%d", ok, err, skip)

    log_path = args.log_json
    if log_path is None:
        log_dir = Path(
            os.environ.get(
                "MEDCLAW_PATHOLOGY_LOG_DIR",
                str(SKILL_DIR / "logs"),
            )
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"ucec_conch_prompt_roi_{timestamp}.json"

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "args": {key: str(value) for key, value in vars(args).items()},
        "ok": ok,
        "error": err,
        "skip": skip,
        "results": [asdict(item) for item in results],
    }
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote log %s", log_path)
    return 0 if err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
