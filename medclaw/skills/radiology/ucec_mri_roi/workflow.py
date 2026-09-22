"""UCEC T2/DWI MRI ROI review workflow."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

import nibabel as nib
import numpy as np

from medclaw.skills.radiology.ucec_mri_roi.visualize import (
    center_z_index,
    save_grayscale_png,
    save_overlay_png,
)
from medclaw.utils import read_yaml, utc_now, write_json


SKILL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SKILL_DIR.parents[3]
DEFAULT_TIMEPOINT = "T0"
DEFAULT_MODALITIES = ("t2", "dwi")


@dataclass(frozen=True)
class ModalityPaths:
    modality: str
    roi_path: Path
    mask_path: Path | None


@dataclass(frozen=True)
class UcecMriRoiResult:
    findings: dict[str, Any]
    artifact_files: tuple[tuple[str, str, Path], ...]
    warnings: tuple[str, ...]


def case_stem(case_id: str, timepoint: str = DEFAULT_TIMEPOINT) -> str:
    return f"{case_id}_{timepoint}"


def default_roi_path(case_dir: Path, case_id: str, modality: str, timepoint: str) -> Path:
    stem = case_stem(case_id, timepoint)
    suffix = f"{modality}_roi.nii.gz"
    return case_dir / "radiology" / "roi" / f"{stem}_{suffix}"


def default_mask_path(case_dir: Path, case_id: str, modality: str, timepoint: str) -> Path:
    stem = case_stem(case_id, timepoint)
    return case_dir / "radiology" / "masks" / f"{stem}_{modality}_tumor.nii.gz"


def resolve_case_dir(case_id: str, *, cases_root: Path | None = None) -> Path:
    root = Path(
        cases_root
        or os.environ.get("MEDCLAW_CASES_ROOT")
        or _default_cases_root()
    ).resolve()
    case_dir = root / case_id
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
    return case_dir


def resolve_mri_paths(
    case_id: str,
    *,
    timepoint: str = DEFAULT_TIMEPOINT,
    modalities: tuple[str, ...] = DEFAULT_MODALITIES,
    t2_roi_uri: str | None = None,
    dwi_roi_uri: str | None = None,
    cases_root: Path | None = None,
) -> list[ModalityPaths]:
    """Resolve T2/DWI ROI (and optional mask) paths for one UCEC case."""

    case_dir = resolve_case_dir(case_id, cases_root=cases_root)
    case_yaml = _read_case_yaml(case_dir)
    manifest = _read_case_manifest(case_dir)
    overrides = {
        "t2": t2_roi_uri,
        "dwi": dwi_roi_uri,
    }
    declared = _declared_roi_paths(case_yaml, timepoint)

    resolved: list[ModalityPaths] = []
    for modality in modalities:
        key = modality.lower()
        if key not in {"t2", "dwi"}:
            raise ValueError(f"Unsupported MRI modality {modality!r}; expected t2 or dwi.")

        roi_path = _resolve_single_path(
            override=overrides.get(key),
            declared=declared.get(key),
            manifest_paths=_manifest_roi_paths(manifest, key),
            default=default_roi_path(case_dir, case_id, key, timepoint),
            case_dir=case_dir,
            modality=key,
            timepoint=timepoint,
        )
        if not roi_path.is_file():
            source_path = (
                case_dir
                / "radiology"
                / "nifti"
                / f"{case_stem(case_id, timepoint)}_mri_{key}.nii.gz"
            )
            source_detail = ""
            if source_path.is_file():
                source_detail = (
                    f" Source MRI exists at {source_path}, but it is not a tumor ROI; "
                    "run the offline ucec-mri-pipeline first."
                )
            raise FileNotFoundError(
                f"{key.upper()} ROI file does not exist for case {case_id}: "
                f"{roi_path}.{source_detail}"
            )

        mask_path = default_mask_path(case_dir, case_id, key, timepoint)
        if not mask_path.is_file():
            mask_path = None

        resolved.append(
            ModalityPaths(
                modality=key,
                roi_path=roi_path.resolve(),
                mask_path=mask_path.resolve() if mask_path else None,
            )
        )
    return resolved


def run_ucec_mri_roi(
    case_id: str,
    output_dir: Path,
    *,
    timepoint: str = DEFAULT_TIMEPOINT,
    modalities: tuple[str, ...] = DEFAULT_MODALITIES,
    t2_roi_uri: str | None = None,
    dwi_roi_uri: str | None = None,
) -> UcecMriRoiResult:
    """Load cached UCEC MRI ROIs and export review PNG artifacts."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case_dir = resolve_case_dir(case_id)
    modality_paths = resolve_mri_paths(
        case_id,
        timepoint=timepoint,
        modalities=modalities,
        t2_roi_uri=t2_roi_uri,
        dwi_roi_uri=dwi_roi_uri,
    )

    artifacts: list[tuple[str, str, Path]] = []
    warnings: list[str] = []
    modality_meta: list[dict[str, Any]] = []
    qc_summary = _read_qc_summary(case_dir, case_id, timepoint)

    for item in modality_paths:
        image = nib.load(str(item.roi_path))
        volume = np.asarray(image.dataobj, dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError(
                f"{item.modality.upper()} ROI must be 3D, got shape {volume.shape}."
            )

        z_index = center_z_index(volume.shape)
        roi_slice = volume[:, :, z_index]
        stem = case_stem(case_id, timepoint)
        prefix = f"{stem}_{item.modality}"

        roi_slice_path = output_dir / f"{prefix}_roi_slice.png"
        context_slice_path = output_dir / f"{prefix}_context_slice.png"
        overlay_path = output_dir / f"{prefix}_overlay.png"

        save_grayscale_png(roi_slice, roi_slice_path)
        save_grayscale_png(roi_slice, context_slice_path)

        mask_slice = None
        if item.mask_path is not None:
            mask_volume = np.asarray(nib.load(str(item.mask_path)).dataobj)
            if mask_volume.shape == volume.shape:
                mask_slice = mask_volume[:, :, z_index] > 0
            else:
                warnings.append(
                    f"{item.modality.upper()} mask shape {mask_volume.shape} "
                    f"does not match ROI shape {volume.shape}; overlay omitted."
                )
        else:
            warnings.append(
                f"No tumor mask found for {item.modality.upper()}; overlay uses intensity only."
            )

        save_overlay_png(roi_slice, mask_slice, overlay_path)

        artifacts.extend(
            [
                ("image", f"{item.modality}_roi_slice", roi_slice_path),
                ("image", f"{item.modality}_context_slice", context_slice_path),
                ("image", f"{item.modality}_overlay", overlay_path),
            ]
        )
        modality_meta.append(
            {
                "modality": item.modality,
                "roi_path": str(item.roi_path),
                "mask_path": str(item.mask_path) if item.mask_path else None,
                "shape_xyz": list(volume.shape),
                "spacing_mm": [float(v) for v in image.header.get_zooms()[:3]],
                "z_index": z_index,
            }
        )

    metadata = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "case_id": case_id,
        "case_dir": str(case_dir.resolve()),
        "timepoint": timepoint,
        "method": "cached_ucec_mri_roi_review",
        "modalities": modality_meta,
        "qc_summary": qc_summary,
        "warnings": warnings,
    }
    metadata_path = output_dir / "ucec_mri_roi_metadata.json"
    write_json(metadata_path, metadata)
    artifacts.append(("json", "roi_metadata", metadata_path))

    roi_available = {item["modality"]: True for item in modality_meta}
    findings = {
        "summary": (
            f"Exported UCEC MRI ROI review images for case {case_id} "
            f"({', '.join(sorted(roi_available))})."
        ),
        "roi_available": roi_available,
        "modalities_exported": sorted(roi_available),
        "timepoint": timepoint,
    }
    if qc_summary:
        for key in ("t2_pad_ratio", "dwi_pad_ratio"):
            if key in qc_summary:
                findings[key.replace("_pad_ratio", "_pad_ratio")] = qc_summary[key]

    return UcecMriRoiResult(
        findings=findings,
        artifact_files=tuple(artifacts),
        warnings=tuple(warnings),
    )


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


def _declared_roi_paths(
    case_yaml: Mapping[str, Any],
    timepoint: str,
) -> dict[str, Path]:
    data = case_yaml.get("data")
    if not isinstance(data, Mapping):
        return {}
    radiology = data.get("radiology")
    if not isinstance(radiology, Mapping):
        return {}

    result: dict[str, Path] = {}
    for modality, keys in (
        ("t2", ("t2_roi_uri", "t2_roi")),
        ("dwi", ("dwi_roi_uri", "dwi_roi")),
    ):
        for key in keys:
            value = radiology.get(key)
            if isinstance(value, str) and value:
                result[modality] = _path_from_local_uri(value)
                break
    _ = timepoint
    return result


def _manifest_roi_paths(manifest: Mapping[str, Any], modality: str) -> list[Path]:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        return []
    roi_list = files.get("radiology_roi")
    if not isinstance(roi_list, list):
        return []
    suffix = f"_{modality}_roi.nii.gz"
    return [
        Path(str(item))
        for item in roi_list
        if isinstance(item, str) and item.endswith(suffix)
    ]


def _resolve_single_path(
    *,
    override: str | None,
    declared: Path | None,
    manifest_paths: list[Path],
    default: Path,
    case_dir: Path,
    modality: str,
    timepoint: str,
) -> Path:
    if override:
        path = _path_from_local_uri(override)
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    candidates: list[Path] = []
    if declared is not None:
        if declared.is_absolute():
            candidates.append(declared)
        else:
            candidates.extend((case_dir / declared, PROJECT_ROOT / declared))
    for path in manifest_paths:
        candidates.append(path if path.is_absolute() else case_dir / path)
    candidates.append(default)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved

    roi_dir = case_dir / "radiology" / "roi"
    discovered = sorted(
        {
            path.resolve()
            for pattern in (
                f"*_{timepoint}_{modality}_roi.nii.gz",
                f"*_{modality}_roi.nii.gz",
            )
            for path in roi_dir.glob(pattern)
            if path.is_file() and path.resolve() not in seen
        }
    )
    if len(discovered) == 1:
        return discovered[0]
    if len(discovered) > 1:
        raise ValueError(
            f"Multiple {modality.upper()} ROI files were found under {roi_dir}: "
            + ", ".join(path.name for path in discovered)
        )
    return default.resolve()


def _read_qc_summary(case_dir: Path, case_id: str, timepoint: str) -> dict[str, Any]:
    qc_path = case_dir / "radiology" / "qc" / f"{case_stem(case_id, timepoint)}_radiology_qc.json"
    if not qc_path.is_file():
        return {}
    try:
        data = json.loads(qc_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


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
