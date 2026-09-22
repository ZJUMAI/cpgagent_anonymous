"""Axial slice rendering helpers for NPC MRI ROI volumes."""

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


def maximum_mask_area_z_index(mask: np.ndarray) -> int:
    """Return the axial index containing the largest ground-truth mask area."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3:
        raise ValueError(f"Tumor mask must be 3D, got shape {binary.shape}.")
    areas = binary.sum(axis=(0, 1))
    if not np.any(areas):
        raise ValueError("Tumor mask is empty.")
    return int(np.argmax(areas))


def mask_bbox_xyxy(mask_slice_xy: np.ndarray) -> tuple[int, int, int, int]:
    """Return the half-open x/y bounding box of a non-empty binary slice."""

    binary = np.asarray(mask_slice_xy, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"Mask slice must be 2D, got shape {binary.shape}.")
    coordinates = np.argwhere(binary)
    if not coordinates.size:
        raise ValueError("Selected mask slice is empty.")
    return (
        int(coordinates[:, 0].min()),
        int(coordinates[:, 1].min()),
        int(coordinates[:, 0].max()) + 1,
        int(coordinates[:, 1].max()) + 1,
    )


def expand_bbox_xyxy(
    bbox: tuple[int, int, int, int],
    shape_xy: tuple[int, int],
    margin_xy: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Expand a half-open bounding box and clamp it to the source image."""

    x0, y0, x1, y1 = bbox
    width, height = shape_xy
    margin_x, margin_y = margin_xy
    return (
        max(0, x0 - margin_x),
        max(0, y0 - margin_y),
        min(width, x1 + margin_x),
        min(height, y1 + margin_y),
    )


def crop_slice_xy(
    slice_xy: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop an x/y slice using a half-open x/y bounding box."""

    array = np.asarray(slice_xy)
    if array.ndim != 2:
        raise ValueError(f"Axial slice must be 2D, got shape {array.shape}.")
    x0, y0, x1, y1 = bbox
    return array[x0:x1, y0:y1]


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
    *,
    expanded_bbox_xyxy: tuple[int, int, int, int] | None = None,
) -> Path:
    """Save a full axial slice with red mask and optional yellow ROI box."""

    gray = axial_display_array(normalize_slice_for_display(volume_slice_xy))
    pixels = np.rint(gray * 255.0).astype(np.uint8)
    rgb = np.repeat(pixels[:, :, None], 3, axis=2)
    if mask_slice_xy is not None:
        mask = axial_display_array(np.asarray(mask_slice_xy, dtype=bool))
        rgb[mask, 0] = 255
        rgb[mask, 1] = (rgb[mask, 1].astype(np.float32) * 0.35).astype(np.uint8)
        rgb[mask, 2] = (rgb[mask, 2].astype(np.float32) * 0.35).astype(np.uint8)

    image = Image.fromarray(rgb, mode="RGB")
    if expanded_bbox_xyxy is not None:
        x0, y0, x1, y1 = expanded_bbox_xyxy
        height = volume_slice_xy.shape[1]
        display_box = (
            x0,
            height - y1,
            max(x0, x1 - 1),
            max(height - y1, height - y0 - 1),
        )
        ImageDraw.Draw(image).rectangle(display_box, outline=(255, 255, 0), width=2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path
