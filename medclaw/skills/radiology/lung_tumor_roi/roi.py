"""Pure tumor-mask selection and axial image rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


ROI_MARGIN_PX = 32


@dataclass(frozen=True)
class RoiSelection:
    """Largest-component and maximum-area axial ROI selection."""

    largest_mask: np.ndarray
    component_count: int
    component_voxels: int
    z_index: int
    area_pixels: int
    mask_bbox_xyxy: tuple[int, int, int, int]
    expanded_bbox_xyxy: tuple[int, int, int, int]


def select_largest_tumor_roi(
    mask: np.ndarray,
    *,
    margin_px: int = ROI_MARGIN_PX,
) -> RoiSelection | None:
    """Select the largest 26-connected 3D component and its largest axial slice."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3:
        raise ValueError(f"Tumor mask must be 3D, got shape {binary.shape}.")
    if margin_px < 0:
        raise ValueError("ROI margin must be non-negative.")
    if not binary.any():
        return None

    labels, component_count = ndimage.label(
        binary,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_label = int(np.argmax(counts))
    largest_mask = labels == largest_label
    component_voxels = int(counts[largest_label])

    areas = largest_mask.sum(axis=(0, 1))
    z_index = int(np.argmax(areas))
    area_pixels = int(areas[z_index])

    coordinates = np.argwhere(largest_mask[:, :, z_index])
    x0 = int(coordinates[:, 0].min())
    x1 = int(coordinates[:, 0].max()) + 1
    y0 = int(coordinates[:, 1].min())
    y1 = int(coordinates[:, 1].max()) + 1

    width, height = largest_mask.shape[:2]
    expanded = (
        max(0, x0 - margin_px),
        max(0, y0 - margin_px),
        min(width, x1 + margin_px),
        min(height, y1 + margin_px),
    )
    return RoiSelection(
        largest_mask=largest_mask,
        component_count=int(component_count),
        component_voxels=component_voxels,
        z_index=z_index,
        area_pixels=area_pixels,
        mask_bbox_xyxy=(x0, y0, x1, y1),
        expanded_bbox_xyxy=expanded,
    )


def crop_axial_roi(volume: np.ndarray, selection: RoiSelection) -> np.ndarray:
    """Crop the selected two-dimensional ROI from an x, y, z volume."""

    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"CT volume must be 3D, got shape {array.shape}.")
    x0, y0, x1, y1 = selection.expanded_bbox_xyxy
    return array[x0:x1, y0:y1, selection.z_index]


def axial_display_array(slice_xy: np.ndarray) -> np.ndarray:
    """Convert an x, y NIfTI slice to a deterministic top-down display array."""

    array = np.asarray(slice_xy)
    if array.ndim != 2:
        raise ValueError(f"Axial slice must be 2D, got shape {array.shape}.")
    return np.flipud(array.T)


def save_grayscale_png(slice_xy: np.ndarray, output_path: Path) -> Path:
    """Save a normalized CT slice as an 8-bit grayscale PNG."""

    display = axial_display_array(slice_xy)
    pixels = np.rint(np.clip(display, 0.0, 1.0) * 255.0).astype(np.uint8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="L").save(output_path)
    return output_path


def save_overlay_png(
    ct_slice_xy: np.ndarray,
    mask_slice_xy: np.ndarray,
    expanded_bbox_xyxy: tuple[int, int, int, int],
    output_path: Path,
) -> Path:
    """Save a full axial slice with red mask and yellow expanded ROI box."""

    gray = axial_display_array(ct_slice_xy)
    mask = axial_display_array(np.asarray(mask_slice_xy, dtype=bool))
    pixels = np.rint(np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = np.repeat(pixels[:, :, None], 3, axis=2)
    rgb[mask, 0] = 255
    rgb[mask, 1] = (rgb[mask, 1].astype(np.float32) * 0.35).astype(np.uint8)
    rgb[mask, 2] = (rgb[mask, 2].astype(np.float32) * 0.35).astype(np.uint8)

    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = expanded_bbox_xyxy
    height = ct_slice_xy.shape[1]
    display_box = (x0, height - y1, max(x0, x1 - 1), max(height - y1, height - y0 - 1))
    draw.rectangle(display_box, outline=(255, 255, 0), width=2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def selection_metadata(selection: RoiSelection) -> dict[str, Any]:
    """Return JSON-compatible ROI selection fields."""

    return {
        "component_count": selection.component_count,
        "largest_component_voxels": selection.component_voxels,
        "z_index": selection.z_index,
        "max_slice_area_pixels": selection.area_pixels,
        "mask_bbox_xyxy": list(selection.mask_bbox_xyxy),
        "expanded_bbox_xyxy": list(selection.expanded_bbox_xyxy),
        "display_transform": "flipud(transpose(slice_xy))",
    }
