from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from medclaw.registry.skill_registry import SkillRegistry
from medclaw.runtime.runtime_manager import RuntimeManager
from medclaw.runtime.skill_runner import SkillRunner
from medclaw.skills.pathology.conch_patch_roi.workflow import run_conch_patch_roi
from medclaw.stores.artifact_store import ArtifactStore
from medclaw.stores.audit_log import AuditLog


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_conch_patch_roi_selects_cached_prompt_rois(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_root = write_cached_conch_case(tmp_path)
    monkeypatch.setenv("MEDCLAW_CASES_ROOT", str(cases_root))

    result = run_conch_patch_roi(
        "CASE-ROI",
        tmp_path / "out",
        top_k_per_prompt=1,
    )

    assert result.findings["roi_selected"] is True
    assert result.findings["patches_selected"] == 2
    assert result.findings["prompt_ids"] == ["tumor", "tumor_nests"]

    roles = [role for _, role, _ in result.artifact_files]
    assert roles == [
        "wsi_overview",
        "patch_contact_sheet",
        "patch_roi",
        "patch_roi",
        "roi_metadata",
    ]
    for _, _, path in result.artifact_files:
        assert path.is_file()

    metadata_path = tmp_path / "out" / "conch_patch_roi_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["case_id"] == "CASE-ROI"
    assert metadata["slide_stem"] == "SLIDE-1"
    assert len(metadata["selected_patches"]) == 2
    assert metadata["selected_patches"][0]["prompt_id"] == "tumor"


def test_conch_patch_roi_runtime_registers_image_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_root = write_cached_conch_case(tmp_path)
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
        "pathology.conch_patch_roi",
        {"case_id": "CASE-ROI", "top_k_per_prompt": 1},
    )

    assert result["status"] == "success"
    assert result["findings"]["patches_selected"] == 2
    image_artifacts = [
        artifact for artifact in result["artifacts"] if artifact["type"] == "image"
    ]
    assert [artifact["role"] for artifact in image_artifacts] == [
        "wsi_overview",
        "patch_contact_sheet",
        "patch_roi",
        "patch_roi",
    ]
    for artifact in image_artifacts:
        assert artifact_store.resolve_uri(artifact["uri"]).is_file()


def write_cached_conch_case(tmp_path: Path) -> Path:
    cases_root = tmp_path / "cases"
    case_dir = cases_root / "CASE-ROI"
    slide_dir = case_dir / "pathology" / "roi_256" / "SLIDE-1"
    roi_dir = slide_dir / "roi"
    (roi_dir / "tumor").mkdir(parents=True)
    (roi_dir / "tumor_nests").mkdir(parents=True)
    wsi_dir = case_dir / "pathology" / "wsi"
    wsi_dir.mkdir(parents=True)
    write_png(wsi_dir / "SLIDE-1.png", (230, 230, 230), size=(96, 64))

    write_png(roi_dir / "tumor" / "tumor_roi_00_idx00005.png", (180, 40, 40))
    write_png(
        roi_dir / "tumor_nests" / "tumor_nests_roi_00_idx00009.png",
        (40, 90, 180),
    )
    write_png(roi_dir / "tumor" / "tumor_roi_01_idx00006.png", (100, 40, 40))
    write_png(
        roi_dir / "tumor_nests" / "tumor_nests_roi_01_idx00010.png",
        (40, 40, 100),
    )

    prompt_scores = {
        "case_id": "CASE-ROI",
        "stem": "SLIDE-1",
        "n_patches": 11,
        "feature_dim": 512,
        "patch_size": 256,
        "patch_level": 0,
        "prompts": [
            {
                "id": "tumor",
                "text": "lung adenocarcinoma tumor tissue",
                "top_indices": [5, 6],
                "top_scores": [0.91, 0.75],
                "top_coords": [[100, 200], [300, 400]],
                "roi_paths": ["/not/local/tumor_roi_00_idx00005.png"],
            },
            {
                "id": "tumor_nests",
                "text": "malignant tumor nests in lung tissue",
                "top_indices": [9, 10],
                "top_scores": [0.88, 0.71],
                "top_coords": [[500, 600], [700, 800]],
            },
        ],
    }
    (slide_dir / "prompt_scores.json").write_text(
        json.dumps(prompt_scores, indent=2),
        encoding="utf-8",
    )
    (slide_dir / "roi_packet.json").write_text(
        json.dumps({"case_id": "CASE-ROI", "stem": "SLIDE-1"}),
        encoding="utf-8",
    )
    return cases_root


def write_png(
    path: Path,
    color: tuple[int, int, int],
    *,
    size: tuple[int, int] = (32, 32),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
