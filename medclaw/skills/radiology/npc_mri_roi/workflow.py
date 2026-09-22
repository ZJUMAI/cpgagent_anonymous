"""NPC T1C review workflow driven by the ground-truth tumor mask."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

import nibabel as nib
import numpy as np

from medclaw.skills.radiology.npc_mri_roi.visualize import (
    crop_slice_xy,
    expand_bbox_xyxy,
    mask_bbox_xyxy,
    maximum_mask_area_z_index,
    save_grayscale_png,
    save_overlay_png,
)
from medclaw.utils import read_yaml, utc_now, write_json


SKILL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SKILL_DIR.parents[3]
DEFAULT_TIMEPOINT = "T0"
DEFAULT_ROI_KINDS = ("primary",)
DEFAULT_ROI_MARGIN_MM = 12.0
ALLOWED_ROI_KINDS = ("primary", "node")
MASK_SUFFIXES = {
    "primary": "primary_tumor.nii.gz",
    "node": "lymph_node.nii.gz",
}
AFFINE_ATOL = 1e-3


@dataclass(frozen=True)
class ImageMaskPaths:
    kind: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class NpcMriRoiResult:
    findings: dict[str, Any]
    artifact_files: tuple[tuple[str, str, Path], ...]
    warnings: tuple[str, ...]


def case_stem(case_id: str, timepoint: str = DEFAULT_TIMEPOINT) -> str:
    return f"{case_id}_{timepoint}"


def default_t1c_path(case_dir: Path, case_id: str, timepoint: str) -> Path:
    return (
        case_dir / "radiology" / "nifti" / f"{case_stem(case_id, timepoint)}_t1c.nii.gz"
    )


def default_mask_path(case_dir: Path, case_id: str, kind: str, timepoint: str) -> Path:
    return (
        case_dir
        / "radiology"
        / "masks"
        / f"{case_stem(case_id, timepoint)}_{MASK_SUFFIXES[kind]}"
    )


def resolve_case_dir(case_id: str, *, cases_root: Path | None = None) -> Path:
    root = Path(
        cases_root or os.environ.get("MEDCLAW_CASES_ROOT") or _default_cases_root()
    ).resolve()
    case_dir = root / case_id
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
    return case_dir


def resolve_mri_paths(
    case_id: str,
    *,
    timepoint: str = DEFAULT_TIMEPOINT,
    roi_kinds: tuple[str, ...] = DEFAULT_ROI_KINDS,
    t1c_uri: str | None = None,
    primary_mask_uri: str | None = None,
    node_mask_uri: str | None = None,
    cases_root: Path | None = None,
) -> list[ImageMaskPaths]:
    """Resolve the source T1C and ground-truth mask for each requested ROI kind."""

    case_dir = resolve_case_dir(case_id, cases_root=cases_root)
    case_yaml = _read_case_yaml(case_dir)
    manifest = _read_case_manifest(case_dir)
    image_path = _resolve_single_path(
        override=t1c_uri,
        declared=_declared_t1c_path(case_yaml),
        manifest_paths=_manifest_t1c_paths(manifest),
        default=default_t1c_path(case_dir, case_id, timepoint),
        case_dir=case_dir,
    )
    if not image_path.is_file():
        raise FileNotFoundError(
            f"T1C MRI file does not exist for case {case_id}: {image_path}"
        )

    declared_masks = _declared_mask_paths(case_yaml)
    mask_overrides = {
        "primary": primary_mask_uri,
        "node": node_mask_uri,
    }
    resolved: list[ImageMaskPaths] = []
    for kind in roi_kinds:
        key = str(kind).lower()
        if key not in ALLOWED_ROI_KINDS:
            raise ValueError(
                f"Unsupported NPC MRI ROI kind {kind!r}; expected primary or node."
            )
        mask_path = _resolve_single_path(
            override=mask_overrides.get(key),
            declared=declared_masks.get(key),
            manifest_paths=_manifest_mask_paths(manifest, key),
            default=default_mask_path(case_dir, case_id, key, timepoint),
            case_dir=case_dir,
        )
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Ground-truth {key} mask does not exist for case {case_id}: "
                f"{mask_path}"
            )
        resolved.append(
            ImageMaskPaths(
                kind=key,
                image_path=image_path.resolve(),
                mask_path=mask_path.resolve(),
            )
        )
    return resolved


def run_npc_mri_roi(
    case_id: str,
    output_dir: Path,
    *,
    timepoint: str = DEFAULT_TIMEPOINT,
    roi_kinds: tuple[str, ...] = DEFAULT_ROI_KINDS,
    t1c_uri: str | None = None,
    primary_mask_uri: str | None = None,
    node_mask_uri: str | None = None,
    roi_margin_mm: float = DEFAULT_ROI_MARGIN_MM,
) -> NpcMriRoiResult:
    """Select and crop T1C axial review images from ground-truth masks."""

    margin_mm = float(roi_margin_mm)
    if not math.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("roi_margin_mm must be a finite non-negative number.")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case_dir = resolve_case_dir(case_id)
    kind_paths = resolve_mri_paths(
        case_id,
        timepoint=timepoint,
        roi_kinds=roi_kinds,
        t1c_uri=t1c_uri,
        primary_mask_uri=primary_mask_uri,
        node_mask_uri=node_mask_uri,
    )

    artifacts: list[tuple[str, str, Path]] = []
    warnings: list[str] = []
    kind_meta: list[dict[str, Any]] = []

    for item in kind_paths:
        image = nib.load(str(item.image_path))
        mask_image = nib.load(str(item.mask_path))
        volume, mask = _validated_image_and_mask(image, mask_image, item.kind)
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])

        z_index = maximum_mask_area_z_index(mask)
        image_slice = volume[:, :, z_index]
        mask_slice = mask[:, :, z_index]
        mask_bbox = mask_bbox_xyxy(mask_slice)
        margin_xy = (
            int(math.ceil(margin_mm / spacing[0])),
            int(math.ceil(margin_mm / spacing[1])),
        )
        expanded_bbox = expand_bbox_xyxy(mask_bbox, volume.shape[:2], margin_xy)
        roi_slice = crop_slice_xy(image_slice, expanded_bbox)

        stem = case_stem(case_id, timepoint)
        prefix = f"{stem}_{item.kind}"
        roi_slice_path = output_dir / f"{prefix}_roi_slice.png"
        context_slice_path = output_dir / f"{prefix}_context_slice.png"
        overlay_path = output_dir / f"{prefix}_overlay.png"

        save_grayscale_png(roi_slice, roi_slice_path)
        save_grayscale_png(image_slice, context_slice_path)
        save_overlay_png(
            image_slice,
            mask_slice,
            overlay_path,
            expanded_bbox_xyxy=expanded_bbox,
        )

        artifacts.extend(
            [
                ("image", f"{item.kind}_roi_slice", roi_slice_path),
                ("image", f"{item.kind}_context_slice", context_slice_path),
                ("image", f"{item.kind}_overlay", overlay_path),
            ]
        )

        center_x = (mask_bbox[0] + mask_bbox[2] - 1) / 2.0
        center_y = (mask_bbox[1] + mask_bbox[3] - 1) / 2.0
        center_world = nib.affines.apply_affine(
            image.affine,
            [center_x, center_y, float(z_index)],
        )
        kind_meta.append(
            {
                "kind": item.kind,
                "source_image_path": str(item.image_path),
                "ground_truth_mask_path": str(item.mask_path),
                "source_shape_xyz": list(volume.shape),
                "source_spacing_mm": list(spacing),
                "source_axcodes": list(nib.aff2axcodes(image.affine)),
                "source_affine": np.asarray(image.affine).tolist(),
                "mask_positive_voxels": int(mask.sum()),
                "z_index": z_index,
                "max_slice_area_pixels": int(mask_slice.sum()),
                "mask_bbox_xyxy": list(mask_bbox),
                "expanded_bbox_xyxy": list(expanded_bbox),
                "roi_shape_xy": list(roi_slice.shape),
                "roi_margin_mm": margin_mm,
                "roi_margin_pixels_xy": list(margin_xy),
                "center_world_mm": [float(value) for value in center_world],
                "selection_source": "ground_truth_mask",
                "cached_roi_used": False,
            }
        )

    metadata = {
        "schema_version": "2.0",
        "created_at": utc_now(),
        "case_id": case_id,
        "case_dir": str(case_dir.resolve()),
        "cancer_type": "NPC",
        "timepoint": timepoint,
        "sequence": "t1c",
        "method": "ground_truth_mask_guided_npc_mri_review",
        "cached_roi_used": False,
        "roi_kinds": kind_meta,
        "warnings": warnings,
    }
    metadata_path = output_dir / "npc_mri_roi_metadata.json"
    write_json(metadata_path, metadata)
    artifacts.append(("json", "roi_metadata", metadata_path))

    roi_available = {item["kind"]: True for item in kind_meta}
    findings = {
        "summary": (
            f"Selected maximum-area T1C slices from ground-truth NPC masks for "
            f"case {case_id} ({', '.join(sorted(roi_available))})."
        ),
        "roi_available": roi_available,
        "roi_kinds_exported": sorted(roi_available),
        "timepoint": timepoint,
        "sequence": "t1c",
        "ground_truth_mask_used": True,
        "cached_roi_used": False,
    }
    return NpcMriRoiResult(
        findings=findings,
        artifact_files=tuple(artifacts),
        warnings=tuple(warnings),
    )


def _validated_image_and_mask(
    image: nib.spatialimages.SpatialImage,
    mask_image: nib.spatialimages.SpatialImage,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(image.shape) != 3:
        raise ValueError(f"T1C MRI must be 3D, got shape {image.shape}.")
    if len(mask_image.shape) != 3:
        raise ValueError(f"{kind} mask must be 3D, got shape {mask_image.shape}.")
    if image.shape != mask_image.shape:
        raise ValueError(
            f"{kind} mask shape {mask_image.shape} does not match T1C shape "
            f"{image.shape}."
        )
    if not np.allclose(image.affine, mask_image.affine, rtol=0.0, atol=AFFINE_ATOL):
        raise ValueError(
            f"{kind} mask affine does not match the T1C affine; refusing an "
            "unregistered ground-truth crop."
        )
    volume = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError("T1C MRI contains non-finite values.")
    raw_mask = np.asarray(mask_image.dataobj)
    if not np.isfinite(raw_mask).all():
        raise ValueError(f"{kind} mask contains non-finite values.")
    mask = raw_mask > 0
    if not mask.any():
        raise ValueError(f"Ground-truth {kind} mask is empty.")
    return volume, mask


def _default_cases_root() -> Path:
    return PROJECT_ROOT / "examples" / "cases"


def _read_case_yaml(case_dir: Path) -> Mapping[str, Any]:
    path = case_dir / "case.yaml"
    if not path.is_file():
        return {}
    data = read_yaml(path)
    if not isinstance(data, Mapping):
        raise ValueError(f"case.yaml must contain a mapping: {path}")
    return data


def _read_case_manifest(case_dir: Path) -> Mapping[str, Any]:
    path = case_dir / "case_manifest.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid case_manifest.json: {path}") from exc
    return data if isinstance(data, Mapping) else {}


def _radiology_mapping(case_yaml: Mapping[str, Any]) -> Mapping[str, Any]:
    data = case_yaml.get("data")
    if not isinstance(data, Mapping):
        return {}
    radiology = data.get("radiology")
    return radiology if isinstance(radiology, Mapping) else {}


def _declared_t1c_path(case_yaml: Mapping[str, Any]) -> Path | None:
    radiology = _radiology_mapping(case_yaml)
    for key in ("t1c_uri", "t1c", "mri_t1c_uri"):
        value = radiology.get(key)
        if isinstance(value, str) and value:
            return _path_from_local_uri(value)
    return None


def _declared_mask_paths(case_yaml: Mapping[str, Any]) -> dict[str, Path]:
    radiology = _radiology_mapping(case_yaml)
    result: dict[str, Path] = {}
    for kind, keys in (
        ("primary", ("primary_mask_uri", "primary_mask", "primary_tumor_mask_uri")),
        ("node", ("node_mask_uri", "node_mask", "lymph_node_mask_uri")),
    ):
        for key in keys:
            value = radiology.get(key)
            if isinstance(value, str) and value:
                result[kind] = _path_from_local_uri(value)
                break
    return result


def _manifest_file_list(manifest: Mapping[str, Any], key: str) -> list[Path]:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        return []
    values = files.get(key)
    if not isinstance(values, list):
        return []
    return [Path(value) for value in values if isinstance(value, str) and value]


def _manifest_t1c_paths(manifest: Mapping[str, Any]) -> list[Path]:
    paths = _manifest_file_list(manifest, "radiology_nifti")
    preferred = [path for path in paths if path.name.lower().endswith("_t1c.nii.gz")]
    return preferred or [
        path for path in paths if path.name.lower().endswith(".nii.gz")
    ]


def _manifest_mask_paths(manifest: Mapping[str, Any], kind: str) -> list[Path]:
    suffix = f"_{MASK_SUFFIXES[kind]}".lower()
    return [
        path
        for path in _manifest_file_list(manifest, "radiology_masks")
        if path.name.lower().endswith(suffix)
    ]


def _resolve_single_path(
    *,
    override: str | None,
    declared: Path | None,
    manifest_paths: list[Path],
    default: Path,
    case_dir: Path,
) -> Path:
    if override:
        return _resolve_relative_path(_path_from_local_uri(override), case_dir)
    if declared is not None:
        return _resolve_relative_path(declared, case_dir)
    existing_manifest_paths = [
        _resolve_relative_path(path, case_dir) for path in manifest_paths
    ]
    existing_manifest_paths = [
        path for path in existing_manifest_paths if path.is_file()
    ]
    if len(existing_manifest_paths) > 1:
        paths = ", ".join(str(path) for path in existing_manifest_paths)
        raise ValueError(f"Multiple matching MRI files were declared: {paths}")
    if existing_manifest_paths:
        return existing_manifest_paths[0]
    return default.resolve()


def _resolve_relative_path(path: Path, case_dir: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    case_relative = (case_dir / path).resolve()
    if case_relative.exists():
        return case_relative
    return (PROJECT_ROOT / path).resolve()


def _path_from_local_uri(value: str) -> Path:
    parsed = urlparse(value)
    if parsed.scheme == "" or _looks_like_windows_drive_path(value, parsed.scheme):
        return Path(value).expanduser()
    if parsed.scheme != "file":
        raise ValueError("URI must be a local path or file:// URI.")
    path_text = unquote(parsed.path)
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
