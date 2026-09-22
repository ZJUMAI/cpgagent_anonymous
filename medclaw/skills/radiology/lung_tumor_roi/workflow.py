"""Validated lung tumor segmentation and ROI artifact workflow."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import nibabel as nib
import numpy as np

from medclaw.utils import DataIOError, read_yaml
from medclaw.skills.radiology.lung_tumor_roi.roi import (
    ROI_MARGIN_PX,
    crop_axial_roi,
    save_grayscale_png,
    save_overlay_png,
    select_largest_tumor_roi,
    selection_metadata,
)


EXPECTED_SPACING_MM = (1.0, 1.0, 1.0)
SPACING_TOLERANCE_MM = 0.05
VALUE_TOLERANCE = 0.05

Segmenter = Callable[[Path], tuple[np.ndarray, dict[str, Any]]]


@dataclass(frozen=True)
class LungTumorRoiResult:
    """Structured workflow result before conversion to the skill protocol."""

    tumor_detected: bool
    findings: dict[str, Any]
    artifact_files: tuple[tuple[str, str, Path], ...]
    warnings: tuple[str, ...]


def resolve_ct_path(
    case_id: str,
    *,
    ct_uri: str | None = None,
    cases_root: Path | None = None,
) -> Path:
    """Resolve an explicit CT URI or the unique preprocessed CT for a case."""

    if ct_uri:
        path = _path_from_local_uri(ct_uri)
        if not path.is_absolute():
            path = _default_project_root() / path
        if not path.is_file():
            raise FileNotFoundError(f"CT file does not exist: {path.resolve()}")
        _validate_ct_filename(path)
        return path.resolve()

    root = Path(
        cases_root
        or os.environ.get("MEDCLAW_CASES_ROOT")
        or _default_cases_root()
    ).resolve()
    case_dir = root / case_id
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")

    declared = _declared_ct_path(case_dir)
    if declared is not None:
        _validate_ct_filename(declared)
        if not declared.is_file():
            raise FileNotFoundError(
                f"Case metadata points to a CT file that does not exist: {declared}"
            )
        return declared.resolve()

    matches = _discover_preprocessed_cts(case_dir)
    if len(matches) != 1:
        raw_cts = _discover_raw_cts(case_dir)
        raw_detail = ""
        if raw_cts:
            raw_detail = (
                " Raw CT file(s) were found but cannot be used as preprocessed input: "
                + ", ".join(path.name for path in raw_cts)
                + ". Run the offline CT preprocessing step first."
            )
        raise ValueError(
            "Expected exactly one preprocessed CT under "
            f"{case_dir}, found {len(matches)}.{raw_detail}"
        )
    return matches[0].resolve()


def resolve_tumor_mask_path(
    ct_path: Path,
    *,
    tumor_mask_uri: str | None = None,
) -> Path | None:
    """Resolve an optional cached tumor mask aligned to a preprocessed CT."""

    ct_path = Path(ct_path).resolve()
    if tumor_mask_uri:
        path = _path_from_local_uri(tumor_mask_uri)
        if not path.is_absolute():
            path = _default_project_root() / path
        if not path.is_file():
            raise FileNotFoundError(f"Tumor mask file does not exist: {path.resolve()}")
        return path.resolve()

    stem = _strip_nifti_suffix(ct_path.name)
    case_stem = stem.removesuffix("_ct_preprocessed")
    masks_dir = ct_path.parent.parent / "masks"
    expected = masks_dir / f"{case_stem}_tumor_preprocessed.nii.gz"
    if expected.is_file():
        return expected.resolve()
    matches = sorted(
        path.resolve()
        for path in masks_dir.glob(f"{case_stem}_tumor*.nii.gz")
        if path.is_file()
    )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple cached tumor masks match CT {ct_path.name}: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0] if matches else None


def validate_preprocessed_ct(ct_path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load and validate a 1 mm lung-window normalized NIfTI CT."""

    ct_path = Path(ct_path).resolve()
    _validate_ct_filename(ct_path)
    image = nib.load(str(ct_path))
    if len(image.shape) != 3:
        raise ValueError(f"Preprocessed CT must be 3D, got shape {image.shape}.")

    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    if any(
        abs(actual - expected) > SPACING_TOLERANCE_MM
        for actual, expected in zip(spacing, EXPECTED_SPACING_MM)
    ):
        raise ValueError(
            f"Preprocessed CT spacing must be approximately 1 mm, got {spacing}."
        )

    volume = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError("Preprocessed CT contains non-finite values.")
    minimum = float(volume.min())
    maximum = float(volume.max())
    if minimum < -VALUE_TOLERANCE or maximum > 1.0 + VALUE_TOLERANCE:
        raise ValueError(
            "Preprocessed CT values must be lung-window normalized near [0, 1], "
            f"got min={minimum:.4f}, max={maximum:.4f}."
        )
    return image, volume


def run_lung_tumor_roi(
    ct_path: Path,
    output_dir: Path,
    *,
    segmenter: Segmenter | None = None,
    tumor_mask_path: Path | None = None,
    margin_px: int = ROI_MARGIN_PX,
) -> LungTumorRoiResult:
    """Run segmentation, select the largest tumor, and write audit artifacts."""

    ct_path = Path(ct_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ct_image, ct_volume = validate_preprocessed_ct(ct_path)

    if tumor_mask_path is not None:
        raw_mask, model_info = _load_cached_tumor_mask(tumor_mask_path, ct_image)
    else:
        if segmenter is None:
            from medclaw.skills.radiology.lung_tumor_roi.inference import (
                segment_preprocessed_ct,
            )

            segmenter = segment_preprocessed_ct
        raw_mask, model_info = segmenter(ct_path)
    raw_binary = np.asarray(raw_mask) > 0.5
    if raw_binary.shape != ct_volume.shape:
        raise ValueError(
            f"Tumor mask shape {raw_binary.shape} does not match CT shape {ct_volume.shape}."
        )

    stem = _strip_nifti_suffix(ct_path.name)
    raw_mask_path = output_dir / f"{stem}_tumor_mask_raw.nii.gz"
    metadata_path = output_dir / f"{stem}_roi_metadata.json"
    _save_mask_nifti(raw_binary, ct_image, raw_mask_path)

    artifacts: list[tuple[str, str, Path]] = [
        ("nifti", "raw_tumor_mask", raw_mask_path),
    ]
    selection = select_largest_tumor_roi(raw_binary, margin_px=margin_px)
    metadata: dict[str, Any] = {
        "ct_filename": ct_path.name,
        "ct_shape_xyz": list(ct_volume.shape),
        "ct_spacing_mm": [float(value) for value in ct_image.header.get_zooms()[:3]],
        "ct_affine": np.asarray(ct_image.affine).tolist(),
        "tumor_detected": selection is not None,
        "raw_positive_voxels": int(raw_binary.sum()),
        "roi_margin_px": margin_px,
        "model": dict(model_info),
    }

    if selection is None:
        warning = "LungTumorMask did not predict any tumor voxels; no ROI images were generated."
        metadata["selection"] = None
        _write_json(metadata_path, metadata)
        artifacts.append(("json", "roi_metadata", metadata_path))
        return LungTumorRoiResult(
            tumor_detected=False,
            findings={
                "summary": warning,
                "tumor_detected": False,
                "raw_positive_voxels": 0,
            },
            artifact_files=tuple(artifacts),
            warnings=(warning,),
        )

    largest_mask_path = output_dir / f"{stem}_tumor_mask_largest.nii.gz"
    roi_path = output_dir / f"{stem}_tumor_roi.png"
    full_slice_path = output_dir / f"{stem}_tumor_full_slice.png"
    overlay_path = output_dir / f"{stem}_tumor_overlay.png"

    _save_mask_nifti(selection.largest_mask, ct_image, largest_mask_path)
    z_index = selection.z_index
    ct_slice = ct_volume[:, :, z_index]
    mask_slice = selection.largest_mask[:, :, z_index]
    save_grayscale_png(crop_axial_roi(ct_volume, selection), roi_path)
    save_grayscale_png(ct_slice, full_slice_path)
    save_overlay_png(
        ct_slice,
        mask_slice,
        selection.expanded_bbox_xyxy,
        overlay_path,
    )

    center_x = (selection.mask_bbox_xyxy[0] + selection.mask_bbox_xyxy[2] - 1) / 2.0
    center_y = (selection.mask_bbox_xyxy[1] + selection.mask_bbox_xyxy[3] - 1) / 2.0
    center_world = nib.affines.apply_affine(
        ct_image.affine,
        [center_x, center_y, float(z_index)],
    )
    metadata["selection"] = {
        **selection_metadata(selection),
        "center_voxel_xyz": [center_x, center_y, z_index],
        "center_world_mm": [float(value) for value in center_world],
    }
    _write_json(metadata_path, metadata)

    artifacts.extend(
        [
            ("nifti", "largest_tumor_mask", largest_mask_path),
            ("image", "roi", roi_path),
            ("image", "full_slice", full_slice_path),
            ("image", "overlay", overlay_path),
            ("json", "roi_metadata", metadata_path),
        ]
    )
    return LungTumorRoiResult(
        tumor_detected=True,
        findings={
            "summary": (
                "LungTumorMask segmentation completed; the largest 3D tumor component "
                "and its maximum-area axial ROI were exported."
            ),
            "tumor_detected": True,
            "raw_positive_voxels": int(raw_binary.sum()),
            **selection_metadata(selection),
            "center_world_mm": [float(value) for value in center_world],
        },
        artifact_files=tuple(artifacts),
        warnings=(
            "Research model output; do not use as a clinical diagnosis.",
        ),
    )


def _save_mask_nifti(mask: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(
        np.asarray(mask, dtype=np.uint8),
        reference.affine,
        header,
    )
    nib.save(image, str(path))


def _load_cached_tumor_mask(
    mask_path: Path,
    ct_image: nib.Nifti1Image,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask_path = Path(mask_path).resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(f"Cached tumor mask does not exist: {mask_path}")
    mask_image = nib.load(str(mask_path))
    if mask_image.shape != ct_image.shape:
        raise ValueError(
            f"Cached tumor mask shape {mask_image.shape} does not match CT shape "
            f"{ct_image.shape}."
        )
    if not np.allclose(mask_image.affine, ct_image.affine, atol=1e-3):
        raise ValueError("Cached tumor mask affine does not match the preprocessed CT.")
    mask = np.asarray(mask_image.dataobj, dtype=np.float32)
    if not np.isfinite(mask).all():
        raise ValueError("Cached tumor mask contains non-finite values.")
    return mask, {
        "model_name": "cached_tumor_mask",
        "model_version": "precomputed",
        "model_weights_sha256": None,
        "device": "not_run",
        "source_mask_path": str(mask_path),
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _default_cases_root() -> Path:
    return _default_project_root() / "examples" / "cases"


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _declared_ct_path(case_dir: Path) -> Path | None:
    path = case_dir / "case.yaml"
    if not path.is_file():
        return None
    try:
        data = read_yaml(path)
    except DataIOError:
        return None
    if not isinstance(data, dict):
        return None
    case_data = data.get("data")
    if not isinstance(case_data, dict):
        return None
    uri = case_data.get("ct_preprocessed_uri")
    if not isinstance(uri, str) or not uri:
        return None
    declared = _path_from_local_uri(uri)
    if not declared.is_absolute():
        declared = _default_project_root() / declared
    return declared.resolve()


def _discover_preprocessed_cts(case_dir: Path) -> list[Path]:
    preferred = sorted(
        path
        for path in (case_dir / "radiology" / "nifti").glob("*_ct_preprocessed.nii.gz")
        if path.is_file()
    )
    if preferred:
        return preferred
    return sorted(
        path
        for path in case_dir.rglob("*_ct_preprocessed.nii.gz")
        if path.is_file()
    )


def _discover_raw_cts(case_dir: Path) -> list[Path]:
    nifti_root = case_dir / "radiology" / "nifti"
    return sorted(
        path
        for path in nifti_root.glob("*.nii.gz")
        if path.is_file() and not path.name.endswith("_ct_preprocessed.nii.gz")
    )


def _validate_ct_filename(path: Path) -> None:
    if not path.name.endswith("_ct_preprocessed.nii.gz"):
        raise ValueError(
            "CT input must be a preprocessed NIfTI named '*_ct_preprocessed.nii.gz'."
        )


def _path_from_local_uri(value: str) -> Path:
    parsed = urlparse(value)
    if parsed.scheme == "" or _looks_like_windows_drive_path(value, parsed.scheme):
        return Path(value).expanduser()
    if parsed.scheme != "file":
        raise ValueError("ct_uri must be a local path or file:// URI.")
    path_text = url2pathname(unquote(parsed.path))
    if parsed.netloc:
        path_text = f"//{parsed.netloc}{path_text}"
    return Path(path_text).expanduser()


def _looks_like_windows_drive_path(value: str, scheme: str) -> bool:
    return (
        len(scheme) == 1
        and len(value) >= 3
        and value[1] == ":"
        and value[2] in {"/", "\\"}
    )


def _strip_nifti_suffix(filename: str) -> str:
    return filename[:-7] if filename.endswith(".nii.gz") else Path(filename).stem
