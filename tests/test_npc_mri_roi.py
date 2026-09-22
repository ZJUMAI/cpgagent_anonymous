from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from medclaw.registry.skill_registry import SkillRegistry
from medclaw.runtime.runtime_manager import RuntimeManager
from medclaw.runtime.skill_runner import SkillRunner
from medclaw.skills.radiology.npc_mri_roi.workflow import (
    resolve_mri_paths,
    run_npc_mri_roi,
)
from medclaw.stores.artifact_store import ArtifactStore
from medclaw.stores.audit_log import AuditLog


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def save_nifti(path: Path, volume: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), str(path))


def write_npc_mri_case(
    tmp_path: Path,
    *,
    include_node: bool = True,
    mask_affine: np.ndarray | None = None,
) -> Path:
    cases_root = tmp_path / "cases"
    case_id = "NPC-DEMO"
    case_dir = cases_root / case_id
    stem = f"{case_id}_T0"
    t1c = case_dir / "radiology" / "nifti" / f"{stem}_t1c.nii.gz"
    primary_mask = case_dir / "radiology" / "masks" / f"{stem}_primary_tumor.nii.gz"
    node_mask = case_dir / "radiology" / "masks" / f"{stem}_lymph_node.nii.gz"

    affine = np.diag([-0.5, 0.5, 3.0, 1.0])
    volume = np.linspace(
        0.0,
        1.0,
        24 * 30 * 8,
        dtype=np.float32,
    ).reshape(24, 30, 8)
    save_nifti(t1c, volume, affine)

    mask = np.zeros_like(volume)
    mask[11:14, 13:16, 2] = 1.0
    mask[10:15, 12:19, 5] = 1.0
    save_nifti(primary_mask, mask, mask_affine if mask_affine is not None else affine)
    if include_node:
        node = np.zeros_like(volume)
        node[3:7, 20:25, 6] = 1.0
        save_nifti(node_mask, node, affine)

    # A deliberately incorrect cached ROI must never be used by the v0.2 workflow.
    bad_cached_roi = case_dir / "radiology" / "roi" / f"{stem}_primary.nii.gz"
    save_nifti(bad_cached_roi, np.zeros((96, 96, 96), dtype=np.float32), np.eye(4))

    (case_dir / "case.yaml").write_text(
        "\n".join(
            [
                f"case_id: {case_id}",
                "data:",
                "  radiology:",
                f"    t1c_uri: {t1c.as_posix()}",
                f"    primary_mask_uri: {primary_mask.as_posix()}",
                *(
                    [f"    node_mask_uri: {node_mask.as_posix()}"]
                    if include_node
                    else []
                ),
                # Retain a legacy declaration to prove that it is ignored.
                f"    primary_roi_uri: {bad_cached_roi.as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return cases_root


def test_resolve_mri_paths_uses_source_image_and_ground_truth_mask(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_root = write_npc_mri_case(tmp_path, include_node=False)
    monkeypatch.setenv("MEDCLAW_CASES_ROOT", str(cases_root))

    paths = resolve_mri_paths("NPC-DEMO")

    assert [item.kind for item in paths] == ["primary"]
    assert paths[0].image_path.name == "NPC-DEMO_T0_t1c.nii.gz"
    assert paths[0].mask_path.name == "NPC-DEMO_T0_primary_tumor.nii.gz"
    assert "radiology\\roi" not in str(paths[0].image_path)


def test_npc_mri_roi_uses_mask_max_area_slice_and_real_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_root = write_npc_mri_case(tmp_path)
    monkeypatch.setenv("MEDCLAW_CASES_ROOT", str(cases_root))

    result = run_npc_mri_roi(
        "NPC-DEMO",
        tmp_path / "out",
        roi_kinds=("primary", "node"),
        roi_margin_mm=1.0,
    )

    assert result.findings["roi_available"] == {"primary": True, "node": True}
    assert result.findings["ground_truth_mask_used"] is True
    assert result.findings["cached_roi_used"] is False
    roles = [role for _, role, _ in result.artifact_files]
    assert "primary_roi_slice" in roles
    assert "primary_context_slice" in roles
    assert "node_overlay" in roles

    metadata = json.loads(
        (tmp_path / "out" / "npc_mri_roi_metadata.json").read_text(encoding="utf-8")
    )
    primary = next(item for item in metadata["roi_kinds"] if item["kind"] == "primary")
    assert metadata["schema_version"] == "2.0"
    assert metadata["cached_roi_used"] is False
    assert primary["selection_source"] == "ground_truth_mask"
    assert primary["z_index"] == 5
    assert primary["mask_bbox_xyxy"] == [10, 12, 15, 19]
    assert primary["expanded_bbox_xyxy"] == [8, 10, 17, 21]
    assert primary["roi_shape_xy"] == [9, 11]

    with Image.open(tmp_path / "out" / "NPC-DEMO_T0_primary_roi_slice.png") as roi:
        assert roi.size == (9, 11)
    with Image.open(
        tmp_path / "out" / "NPC-DEMO_T0_primary_context_slice.png"
    ) as context:
        assert context.size == (24, 30)


def test_npc_mri_roi_rejects_unregistered_mask(tmp_path: Path, monkeypatch) -> None:
    bad_affine = np.diag([-0.5, 0.5, 3.0, 1.0])
    bad_affine[1, 3] = 20.0
    cases_root = write_npc_mri_case(
        tmp_path,
        include_node=False,
        mask_affine=bad_affine,
    )
    monkeypatch.setenv("MEDCLAW_CASES_ROOT", str(cases_root))

    with pytest.raises(ValueError, match="mask affine does not match"):
        run_npc_mri_roi("NPC-DEMO", tmp_path / "out")


def test_npc_mri_roi_runtime_registers_image_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_root = write_npc_mri_case(tmp_path, include_node=False)
    monkeypatch.setenv("MEDCLAW_CASES_ROOT", str(cases_root))

    registry = SkillRegistry(PROJECT_ROOT / "medclaw" / "skills")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    runtime = RuntimeManager(
        registry,
        SkillRunner(),
        artifact_store,
        AuditLog(tmp_path / "runs"),
        tmp_path / "runs",
    )

    result = runtime.invoke(
        "radiology.npc_mri_roi",
        {"case_id": "NPC-DEMO", "roi_kinds": ["primary"]},
    )

    assert result["status"] == "success"
    assert result["findings"]["ground_truth_mask_used"] is True
    image_artifacts = [
        artifact for artifact in result["artifacts"] if artifact["type"] == "image"
    ]
    assert len(image_artifacts) == 3
    for artifact in image_artifacts:
        assert artifact_store.resolve_uri(artifact["uri"]).is_file()
