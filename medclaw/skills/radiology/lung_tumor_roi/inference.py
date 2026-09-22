"""LungTumorMask inference for a preprocessed lung-window CT."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

from medclaw.skills.radiology.lung_tumor_roi.compat import (
    configure_model_cache,
    prepare_lungtumormask_import,
)


MODEL_CACHE = configure_model_cache()

import nibabel as nib
import numpy as np
import torch
from lungmask import LMInferer
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    SpatialCropd,
    Spacingd,
    ToTensord,
)


prepare_lungtumormask_import()

from lungtumormask.dataprocessing import (
    calculate_extremes,
    post_process,
)
from lungtumormask.mask import load_model


logger = logging.getLogger(__name__)

LUNG_WINDOW_WL = -600.0
LUNG_WINDOW_WW = 1500.0
MODEL_VERSION = "lungtumormask-1.3.1"
SEGMENTATION_THRESHOLD = 0.5
MORPHOLOGY_RADIUS = 3
LUNGTUMORMASK_WEIGHT_URL = (
    "https://github.com/VemundFredriksen/LungTumorMask/releases/download/"
    "0.0/dc_student.pth"
)
LUNGMASK_R231_WEIGHT_URL = (
    "https://github.com/JoHof/lungmask/releases/download/v0.0/"
    "unet_r231-d5d2fc3d.pth"
)
REQUIRED_MODEL_WEIGHTS = {
    "LungTumorMask": ("dc_student.pth", LUNGTUMORMASK_WEIGHT_URL),
    "lungmask R231": ("unet_r231-d5d2fc3d.pth", LUNGMASK_R231_WEIGHT_URL),
}


def denormalize_lung_window(
    volume: np.ndarray,
    *,
    wl: float = LUNG_WINDOW_WL,
    ww: float = LUNG_WINDOW_WW,
) -> np.ndarray:
    """Map a normalized lung-window volume back to an HU proxy."""

    low = wl - ww / 2.0
    high = wl + ww / 2.0
    return volume.astype(np.float32) * (high - low) + low


def resolve_device(preference: str | None = None) -> str:
    """Resolve cuda, auto, or cpu into the actual Torch device."""

    requested = (preference or os.environ.get("MEDCLAW_LUNG_TUMOR_DEVICE", "cuda")).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "MEDCLAW_LUNG_TUMOR_DEVICE must be one of: auto, cpu, cuda."
        )
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but Torch reports that CUDA is unavailable.")
    return requested


def segment_preprocessed_ct(
    ct_path: Path,
    *,
    device_preference: str | None = None,
    batch_size: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run LungTumorMask and return its raw mask plus model audit information."""

    ct_path = Path(ct_path).resolve()
    model_cache = configure_model_cache()
    _progress(f"model cache: {model_cache}")
    device = resolve_device(device_preference)
    _progress(f"resolved device: {device}")
    require_cached_model_weights(model_cache)

    _progress("loading LungTumorMask model weights")
    model = load_model()
    model = model.to(device)
    model.eval()
    _progress("hashing LungTumorMask model weights")
    model_hash = hash_model_state_dict(model)

    _progress("preprocessing CT and running lungmask R231")
    preprocess_dump = _preprocess_preprocessed_ct(
        ct_path,
        device=device,
        batch_size=batch_size,
    )
    logger.info("Running %s on %s with device=%s", MODEL_VERSION, ct_path.name, device)
    _progress("running LungTumorMask left and right lung inference")
    with torch.no_grad():
        left = (
            model(preprocess_dump["left_lung"].to(device))
            .squeeze(0)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )
        right = (
            model(preprocess_dump["right_lung"].to(device))
            .squeeze(0)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

    _progress("post-processing tumor mask")
    inferred = post_process(
        left,
        right,
        preprocess_dump,
        True,
        SEGMENTATION_THRESHOLD,
        MORPHOLOGY_RADIUS,
    )
    return np.asarray(inferred), {
        "model_name": "LungTumorMask",
        "model_version": MODEL_VERSION,
        "model_weights_sha256": model_hash,
        "device": device,
        "lung_filter": True,
        "threshold": SEGMENTATION_THRESHOLD,
        "morphology_radius": MORPHOLOGY_RADIUS,
        "batch_size": batch_size,
        "model_cache": str(model_cache),
    }


def required_model_weight_paths(cache_root: Path | None = None) -> dict[str, Path]:
    """Return the Torch hub checkpoint paths required for runtime inference."""

    if cache_root is not None:
        checkpoint_dir = Path(cache_root).expanduser().resolve() / "hub" / "checkpoints"
    else:
        checkpoint_dir = Path(torch.hub.get_dir()).expanduser() / "checkpoints"
    return {
        label: checkpoint_dir / filename
        for label, (filename, _) in REQUIRED_MODEL_WEIGHTS.items()
    }


def require_cached_model_weights(cache_root: Path | None = None) -> None:
    """Fail fast when model weights are missing and runtime downloads are disabled."""

    if _allow_model_download():
        _progress("runtime model downloads are enabled")
        return

    paths = required_model_weight_paths(cache_root)
    missing = [label for label, path in paths.items() if not path.is_file()]
    if not missing:
        return

    details = []
    for label in missing:
        filename, url = REQUIRED_MODEL_WEIGHTS[label]
        details.append(f"{label}: {filename} from {url}")
    expected_dir = next(iter(paths.values())).parent if paths else Path("<unknown>")
    raise RuntimeError(
        "Required lung tumor model weights are missing from the local Torch cache, "
        "and runtime downloads are disabled to avoid hanging on offline servers. "
        f"Copy the following files into {expected_dir}: "
        + "; ".join(details)
        + ". Or run scripts/prewarm_model.py on a machine with network access, "
        "then transfer the Torch cache. To deliberately allow runtime downloads, "
        "set MEDCLAW_LUNG_TUMOR_ALLOW_MODEL_DOWNLOAD=1."
    )


def _allow_model_download() -> bool:
    value = os.environ.get("MEDCLAW_LUNG_TUMOR_ALLOW_MODEL_DOWNLOAD", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _progress(message: str) -> None:
    """Append a stage message to an optional progress log and stderr."""

    logger.info(message)
    text = f"[lung_tumor_roi] {message}"
    print(text, file=sys.stderr, flush=True)
    path_value = os.environ.get("MEDCLAW_LUNG_TUMOR_PROGRESS_LOG")
    if not path_value:
        return
    try:
        path = Path(path_value).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
    except OSError:
        logger.exception("Could not write lung tumor ROI progress log")


def hash_model_state_dict(model: torch.nn.Module) -> str:
    """Hash loaded model weights without relying on a cache file location."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _preprocess_preprocessed_ct(
    image_path: Path,
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Build LungTumorMask inputs while preserving the preprocessed CT values."""

    scan_dict = {"image": str(image_path)}
    ref_nii = nib.load(str(image_path))
    preprocess_dump: dict[str, Any] = {
        "ref_nii": ref_nii,
        "org_shape": ref_nii.shape,
        "org_affine": np.asarray(ref_nii.affine),
        "pixdim": np.asarray(ref_nii.header.get_zooms()[:3], dtype=np.float64),
    }

    preprocessed = np.asarray(ref_nii.dataobj, dtype=np.float32)
    hu_proxy = denormalize_lung_window(preprocessed)
    _progress("initializing lungmask R231 model")
    lung_inferer = LMInferer(
        modelname="R231",
        force_cpu=device == "cpu",
        batch_size=batch_size,
    )
    _progress("applying lungmask R231 to CT volume")
    masked_lungs_zyx = lung_inferer.apply(np.swapaxes(hu_proxy, 0, 2))
    masked_lungs = np.swapaxes(masked_lungs_zyx, 0, 2)
    preprocess_dump["lungmask"] = masked_lungs

    right_extremes = calculate_extremes(masked_lungs, 1)
    left_extremes = calculate_extremes(masked_lungs, 2)
    _validate_lung_extremes(right_extremes, "right")
    _validate_lung_extremes(left_extremes, "left")
    right_lung = _process_lung_scan(scan_dict, right_extremes)
    left_lung = _process_lung_scan(scan_dict, left_extremes)

    preprocess_dump["right_extremes"] = right_extremes
    preprocess_dump["left_extremes"] = left_extremes
    preprocess_dump["affine"] = left_lung[1]
    preprocess_dump["right_lung"] = right_lung[0].unsqueeze(0)
    preprocess_dump["left_lung"] = left_lung[0].unsqueeze(0)
    return preprocess_dump


def _process_lung_scan(scan_dict: dict[str, str], extremes: tuple) -> tuple[Any, Any]:
    load_transformer = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            NormalizeIntensityd(keys=["image"]),
            SpatialCropd(
                keys=["image"],
                roi_start=(extremes[0][0], extremes[1][0], extremes[2][0]),
                roi_end=(extremes[0][1], extremes[1][1], extremes[2][1]),
            ),
            Spacingd(keys=["image"], pixdim=(1, 1, 1.5)),
        ]
    )
    processed = load_transformer(scan_dict)
    processed = Compose(
        [
            DivisiblePadd(keys=["image"], k=16, mode="constant"),
            ToTensord(keys=["image"]),
        ]
    )(processed)
    meta = processed["image"].meta if hasattr(processed["image"], "meta") else {}
    affine = meta.get("affine", meta.get("original_affine"))
    if affine is None:
        raise RuntimeError("MONAI preprocessing did not preserve the lung crop affine.")
    return processed["image"], affine


def _validate_lung_extremes(extremes: tuple, side: str) -> None:
    values = np.asarray(extremes, dtype=np.float64)
    if values.shape != (3, 2) or not np.isfinite(values).all():
        raise RuntimeError(f"lungmask did not produce a valid {side} lung bounding box.")
    if np.any(values[:, 0] < 0) or np.any(values[:, 1] <= values[:, 0]):
        raise RuntimeError(f"lungmask did not detect a usable {side} lung region.")
