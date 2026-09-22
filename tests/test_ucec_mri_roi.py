from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from medclaw.registry.skill_registry import SkillRegistry
from medclaw.runtime.runtime_manager import RuntimeManager
from medclaw.runtime.skill_runner import SkillRunner
from medclaw.skills.radiology.ucec_mri_roi.workflow import (
    resolve_mri_paths,
    run_ucec_mri_roi,
)
from medclaw.stores.artifact_store import ArtifactStore
from medclaw.stores.audit_log import AuditLog


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def save_roi(path: Path, volume: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume.astype(np.float32), np.eye(4)), str(path))


def write_ucec_mri_case(tmp_path: Path) -> Path:
    cases_root = tmp_path / "cases"
    case_id = "03850333"
    case_dir = cases_root / case_id
    stem = f"{case_id}_T0"
    t2_roi = case_dir / "radiology" / "roi" / f"{stem}_t2_roi.nii.gz"
    dwi_roi = case_dir / "radiology" / "roi" / f"{stem}_dwi_roi.nii.gz"
    t2_mask = case_dir / "radiology" / "masks" / f"{stem}_t2_tumor.nii.gz"

    volume = np.linspace(0.0, 1.0, 96 * 96 * 96, dtype=np.float32).reshape(96, 96, 96)
    save_roi(t2_roi, volume)
    save_roi(dwi_roi, volume * 0.8)

    mask = np.zeros((96, 96, 96), dtype=np.float32)
    mask[40:50, 40:50, 45:55] = 1.0
    save_roi(t2_mask, mask)

    (case_dir / "case.yaml").write_text(
        "\n".join(
            [
                f"case_id: {case_id}",
                "data:",
                "  radiology:",
                f"    t2_roi_uri: {t2_roi.as_posix()}",
                f"    dwi_roi_uri: {dwi_roi.as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return cases_root


def test_resolve_mri_paths_uses_layout_defaults(tmp_path: Path, monkeypatch) -> None:
    cases_root = write_ucec_mri_case(tmp_path)
    monkeypatch.setenv("MEDCLAW_CASES_ROOT", str(cases_root))

    paths = resolve_mri_paths("03850333")
    assert [item.modality for item in paths] == ["t2", "dwi"]
    assert paths[0].roi_path.name == "03850333_T0_t2_roi.nii.gz"
    assert paths[1].roi_path.name == "03850333_T0_dwi_roi.nii.gz"


def test_resolve_mri_paths_falls_back_from_stale_declared_path_to_local_roi(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_root = write_ucec_mri_case(tmp_path)
    monkeypatch.setenv("MEDCLAW_CASES_ROOT", str(cases_root))
    roi_dir = cases_root / "03850333" / "radiology" / "roi"
    original = roi_dir / "03850333_T0_t2_roi.nii.gz"
    discovered = roi_dir / "LOCAL_T0_t2_roi.nii.gz"
    original.rename(discovered)

    paths = resolve_mri_paths("03850333", modalities=("t2",))

    assert paths[0].roi_path == discovered.resolve()


def test_ucec_mri_roi_exports_png_artifacts(tmp_path: Path, monkeypatch) -> None:
    cases_root = write_ucec_mri_case(tmp_path)
    monkeypatch.setenv("MEDCLAW_CASES_ROOT", str(cases_root))

    result = run_ucec_mri_roi("03850333", tmp_path / "out", modalities=("t2", "dwi"))

    assert result.findings["roi_available"] == {"t2": True, "dwi": True}
    roles = [role for _, role, _ in result.artifact_files]
    assert "t2_roi_slice" in roles
    assert "dwi_overlay" in roles
    assert "roi_metadata" in roles
    for _, _, path in result.artifact_files:
        assert path.is_file()

    metadata = json.loads(
        (tmp_path / "out" / "ucec_mri_roi_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["case_id"] == "03850333"
    assert len(metadata["modalities"]) == 2


def test_ucec_mri_roi_runtime_registers_image_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_root = write_ucec_mri_case(tmp_path)
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
        "radiology.ucec_mri_roi",
        {"case_id": "03850333", "modalities": ["t2"]},
    )

    assert result["status"] == "success"
    image_artifacts = [
        artifact for artifact in result["artifacts"] if artifact["type"] == "image"
    ]
    assert len(image_artifacts) == 3
    for artifact in image_artifacts:
        assert artifact_store.resolve_uri(artifact["uri"]).is_file()
