from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from medclaw.skills.radiology.lung_tumor_roi.roi import (
    axial_display_array,
    crop_axial_roi,
    select_largest_tumor_roi,
)
from medclaw.skills.radiology.lung_tumor_roi.workflow import (
    resolve_ct_path,
    resolve_tumor_mask_path,
    run_lung_tumor_roi,
    validate_preprocessed_ct,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TCGA_CT = (
    PROJECT_ROOT
    / "examples"
    / "cases"
    / "TCGA-38-4626"
    / "radiology"
    / "nifti"
    / "TCGA-38-4626_T0_ct_preprocessed.nii.gz"
)


def save_ct(
    path: Path,
    volume: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.diag([*spacing, 1.0])
    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), str(path))
    return path


def fake_model_info() -> dict[str, object]:
    return {
        "model_name": "FakeLungTumorMask",
        "model_version": "test",
        "model_weights_sha256": "a" * 64,
        "device": "cpu",
        "lung_filter": True,
        "threshold": 0.5,
        "morphology_radius": 3,
    }


def test_selects_largest_26_connected_component_and_maximum_z_slice() -> None:
    mask = np.zeros((20, 20, 5), dtype=np.uint8)
    mask[1:3, 1:3, 0] = 1
    mask[2:4, 2:4, 1] = 1  # Diagonally touches the previous slice in 26-connectivity.
    mask[10:14, 10:15, 3] = 1
    mask[10:12, 10:12, 4] = 1

    selection = select_largest_tumor_roi(mask, margin_px=2)

    assert selection is not None
    assert selection.component_count == 2
    assert selection.component_voxels == 24
    assert selection.z_index == 3
    assert selection.area_pixels == 20
    assert selection.mask_bbox_xyxy == (10, 10, 14, 15)
    assert selection.expanded_bbox_xyxy == (8, 8, 16, 17)


def test_roi_expansion_clips_to_bounds_and_preserves_native_pixels() -> None:
    volume = np.arange(10 * 12 * 2, dtype=np.float32).reshape(10, 12, 2)
    mask = np.zeros_like(volume, dtype=np.uint8)
    mask[0:2, 9:12, 1] = 1

    selection = select_largest_tumor_roi(mask, margin_px=32)

    assert selection is not None
    assert selection.expanded_bbox_xyxy == (0, 0, 10, 12)
    np.testing.assert_array_equal(crop_axial_roi(volume, selection), volume[:, :, 1])


def test_roi_expansion_uses_exact_32_pixel_margin() -> None:
    mask = np.zeros((120, 140, 1), dtype=np.uint8)
    mask[50:55, 60:66, 0] = 1

    selection = select_largest_tumor_roi(mask)

    assert selection is not None
    assert selection.mask_bbox_xyxy == (50, 60, 55, 66)
    assert selection.expanded_bbox_xyxy == (18, 28, 87, 98)


def test_axial_display_orientation_is_deterministic() -> None:
    slice_xy = np.array([[1, 2, 3], [4, 5, 6]])

    np.testing.assert_array_equal(
        axial_display_array(slice_xy),
        np.array([[3, 6], [2, 5], [1, 4]]),
    )


def test_empty_mask_has_no_roi_selection() -> None:
    assert select_largest_tumor_roi(np.zeros((4, 4, 4), dtype=np.uint8)) is None


def test_resolve_ct_path_finds_unique_case_ct_and_rejects_ambiguous_case(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "CASE-1"
    expected = save_ct(
        case_dir / "CASE-1_T0_ct_preprocessed.nii.gz",
        np.zeros((2, 2, 2), dtype=np.float32),
    )

    assert resolve_ct_path("CASE-1", cases_root=tmp_path) == expected.resolve()

    save_ct(
        case_dir / "CASE-1_T1_ct_preprocessed.nii.gz",
        np.zeros((2, 2, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="Expected exactly one"):
        resolve_ct_path("CASE-1", cases_root=tmp_path)


def test_resolve_ct_path_supports_modality_subdirectories(tmp_path: Path) -> None:
    case_dir = tmp_path / "CASE-2"
    expected = save_ct(
        case_dir / "radiology" / "nifti" / "CASE-2_T0_ct_preprocessed.nii.gz",
        np.zeros((2, 2, 2), dtype=np.float32),
    )

    assert resolve_ct_path("CASE-2", cases_root=tmp_path) == expected.resolve()


def test_resolve_ct_path_rejects_raw_radiology_nifti(tmp_path: Path) -> None:
    case_dir = tmp_path / "CASE-2B"
    save_ct(
        case_dir / "radiology" / "nifti" / "CASE-2B_T0_ct.nii.gz",
        np.zeros((2, 2, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="offline CT preprocessing"):
        resolve_ct_path("CASE-2B", cases_root=tmp_path)


def test_resolve_ct_path_does_not_recursively_fallback_to_generic_nii_gz(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "CASE-2C"
    save_ct(
        case_dir / "radiology" / "nifti" / "nested" / "CASE-2C_T0_ct.nii.gz",
        np.zeros((2, 2, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="Expected exactly one"):
        resolve_ct_path("CASE-2C", cases_root=tmp_path)


def test_resolve_ct_path_uses_case_yaml_declared_uri(tmp_path: Path) -> None:
    case_dir = tmp_path / "CASE-3"
    expected = save_ct(
        case_dir / "radiology" / "nifti" / "CASE-3_T0_ct_preprocessed.nii.gz",
        np.zeros((2, 2, 2), dtype=np.float32),
    )
    save_ct(
        case_dir / "other" / "CASE-3_T1_ct_preprocessed.nii.gz",
        np.zeros((2, 2, 2), dtype=np.float32),
    )
    (case_dir / "case.yaml").write_text(
        "\n".join(
            [
                "case_id: CASE-3",
                "data:",
                f"  ct_preprocessed_uri: {expected.as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert resolve_ct_path("CASE-3", cases_root=tmp_path) == expected.resolve()


def test_real_case_ct_is_resolvable_without_copying() -> None:
    if not LOCAL_TCGA_CT.is_file():
        pytest.skip("Local TCGA-38-4626 CT data is not checked into Git.")

    path = resolve_ct_path(
        "TCGA-38-4626",
        cases_root=PROJECT_ROOT / "examples" / "cases",
    )

    assert path.name == "TCGA-38-4626_T0_ct_preprocessed.nii.gz"
    assert path.is_file()


def test_repository_relative_ct_uri_resolves_from_skill_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not LOCAL_TCGA_CT.is_file():
        pytest.skip("Local TCGA-38-4626 CT data is not checked into Git.")

    monkeypatch.chdir(tmp_path)

    path = resolve_ct_path(
        "TCGA-38-4626",
        ct_uri=(
            "examples/cases/TCGA-38-4626/radiology/nifti/"
            "TCGA-38-4626_T0_ct_preprocessed.nii.gz"
        ),
    )

    assert path == (
        PROJECT_ROOT
        / "examples"
        / "cases"
        / "TCGA-38-4626"
        / "radiology"
        / "nifti"
        / "TCGA-38-4626_T0_ct_preprocessed.nii.gz"
    ).resolve()


@pytest.mark.parametrize(
    ("volume", "spacing", "message"),
    [
        (np.zeros((3, 3, 3, 1), dtype=np.float32), (1.0, 1.0, 1.0), "must be 3D"),
        (np.zeros((3, 3, 3), dtype=np.float32), (1.0, 1.0, 1.2), "approximately 1 mm"),
        (np.full((3, 3, 3), np.nan, dtype=np.float32), (1.0, 1.0, 1.0), "non-finite"),
        (np.full((3, 3, 3), 1.2, dtype=np.float32), (1.0, 1.0, 1.0), "near \\[0, 1\\]"),
    ],
)
def test_preprocessed_ct_validation_rejects_invalid_inputs(
    tmp_path: Path,
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    message: str,
) -> None:
    path = save_ct(tmp_path / "bad_ct_preprocessed.nii.gz", volume, spacing=spacing)

    with pytest.raises(ValueError, match=message):
        validate_preprocessed_ct(path)


def test_workflow_writes_aligned_masks_images_and_metadata(tmp_path: Path) -> None:
    ct = np.linspace(0.0, 1.0, 16 * 18 * 4, dtype=np.float32).reshape(16, 18, 4)
    ct_path = save_ct(tmp_path / "CASE_T0_ct_preprocessed.nii.gz", ct)
    raw_mask = np.zeros_like(ct, dtype=np.uint8)
    raw_mask[1:3, 1:3, 0] = 1
    raw_mask[5:9, 7:12, 2] = 1

    result = run_lung_tumor_roi(
        ct_path,
        tmp_path / "out",
        segmenter=lambda _: (raw_mask, fake_model_info()),
        margin_px=3,
    )

    assert result.tumor_detected is True
    artifacts = {(kind, role): path for kind, role, path in result.artifact_files}
    assert set(artifacts) == {
        ("nifti", "raw_tumor_mask"),
        ("nifti", "largest_tumor_mask"),
        ("image", "roi"),
        ("image", "full_slice"),
        ("image", "overlay"),
        ("json", "roi_metadata"),
    }

    raw_image = nib.load(str(artifacts[("nifti", "raw_tumor_mask")]))
    largest_image = nib.load(str(artifacts[("nifti", "largest_tumor_mask")]))
    assert raw_image.shape == ct.shape
    assert largest_image.shape == ct.shape
    np.testing.assert_allclose(raw_image.affine, nib.load(str(ct_path)).affine)
    np.testing.assert_allclose(largest_image.affine, nib.load(str(ct_path)).affine)

    metadata = json.loads(artifacts[("json", "roi_metadata")].read_text(encoding="utf-8"))
    assert metadata["tumor_detected"] is True
    assert metadata["selection"]["z_index"] == 2
    assert metadata["selection"]["max_slice_area_pixels"] == 20
    assert metadata["selection"]["largest_component_voxels"] == 20
    assert metadata["selection"]["mask_bbox_xyxy"] == [5, 7, 9, 12]
    assert metadata["selection"]["expanded_bbox_xyxy"] == [2, 4, 12, 15]
    assert metadata["model"]["model_weights_sha256"] == "a" * 64

    with Image.open(artifacts[("image", "roi")]) as roi:
        assert roi.size == (10, 11)
    with Image.open(artifacts[("image", "full_slice")]) as full_slice:
        assert full_slice.size == (16, 18)
    with Image.open(artifacts[("image", "overlay")]) as overlay:
        assert overlay.size == (16, 18)


def test_workflow_uses_aligned_cached_mask_without_model_inference(tmp_path: Path) -> None:
    case_dir = tmp_path / "CASE-CACHED"
    ct = np.linspace(0.0, 1.0, 16 * 18 * 4, dtype=np.float32).reshape(16, 18, 4)
    ct_path = save_ct(
        case_dir / "radiology" / "nifti" / "CASE-CACHED_T0_ct_preprocessed.nii.gz",
        ct,
    )
    mask = np.zeros_like(ct, dtype=np.uint8)
    mask[5:9, 7:12, 2] = 1
    mask_path = save_ct(
        case_dir / "radiology" / "masks" / "CASE-CACHED_T0_tumor_preprocessed.nii.gz",
        mask,
    )

    resolved_mask = resolve_tumor_mask_path(ct_path)
    result = run_lung_tumor_roi(
        ct_path,
        tmp_path / "out-cached",
        tumor_mask_path=resolved_mask,
    )

    assert resolved_mask == mask_path.resolve()
    assert result.tumor_detected is True
    metadata = json.loads(
        next(path for kind, role, path in result.artifact_files if role == "roi_metadata").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["model"]["model_name"] == "cached_tumor_mask"
    assert metadata["model"]["device"] == "not_run"


def test_workflow_empty_mask_returns_only_raw_mask_and_metadata(tmp_path: Path) -> None:
    ct = np.zeros((8, 9, 3), dtype=np.float32)
    ct_path = save_ct(tmp_path / "CASE_T0_ct_preprocessed.nii.gz", ct)

    result = run_lung_tumor_roi(
        ct_path,
        tmp_path / "out",
        segmenter=lambda _: (np.zeros_like(ct), fake_model_info()),
    )

    assert result.tumor_detected is False
    assert result.findings["tumor_detected"] is False
    assert len(result.warnings) == 1
    assert [(kind, role) for kind, role, _ in result.artifact_files] == [
        ("nifti", "raw_tumor_mask"),
        ("json", "roi_metadata"),
    ]
    metadata = json.loads(result.artifact_files[-1][2].read_text(encoding="utf-8"))
    assert metadata["selection"] is None
    assert metadata["tumor_detected"] is False
