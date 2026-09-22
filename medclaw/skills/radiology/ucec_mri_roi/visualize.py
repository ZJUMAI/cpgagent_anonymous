"""Axial slice rendering helpers for UCEC MRI ROI volumes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def axial_display_array(slice_xy: np.ndarray) -> np.ndarray:
    """Convert an x, y NIfTI slice to a deterministic top-down display array."""

    array = np.asarray(slice_xy)
    if array.ndim != 2:
        raise ValueError(f"Axial slice must be 2D, got shape {array.shape}.")
    return np.flipud(array.T)


def normalize_slice_for_display(slice_xy: np.ndarray) -> np.ndarray:
    """Map a scalar MRI slice to [0, 1] for PNG export."""

    array = np.asarray(slice_xy, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.0))
    if high <= low:
        high = low + 1.0
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    return scaled.astype(np.float32)


def center_z_index(shape: tuple[int, ...]) -> int:
    if len(shape) != 3:
        raise ValueError(f"ROI volume must be 3D, got shape {shape}.")
    return int(shape[2] // 2)


def save_grayscale_png(slice_xy: np.ndarray, output_path: Path) -> Path:
    """Save a normalized MRI slice as an 8-bit grayscale PNG."""

    display = axial_display_array(normalize_slice_for_display(slice_xy))
    pixels = np.rint(display * 255.0).astype(np.uint8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="L").save(output_path)
    return output_path


def save_overlay_png(
    volume_slice_xy: np.ndarray,
    mask_slice_xy: np.ndarray | None,
    output_path: Path,
) -> Path:
    """Save a full axial slice with optional red mask overlay."""

    gray = axial_display_array(normalize_slice_for_display(volume_slice_xy))
    pixels = np.rint(gray * 255.0).astype(np.uint8)
    rgb = np.repeat(pixels[:, :, None], 3, axis=2)
    if mask_slice_xy is not None:
        mask = axial_display_array(np.asarray(mask_slice_xy, dtype=bool))
        rgb[mask, 0] = 255
        rgb[mask, 1] = (rgb[mask, 1].astype(np.float32) * 0.35).astype(np.uint8)
        rgb[mask, 2] = (rgb[mask, 2].astype(np.float32) * 0.35).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path)
    return output_path
