from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from medclaw.registry.skill_registry import SkillRegistry
from medclaw.runtime.runtime_manager import RuntimeManager
from medclaw.runtime.skill_runner import SkillRunner
from medclaw.skills.pathology.ucec_conch_patch_roi.workflow import run_ucec_conch_patch_roi
from medclaw.stores.artifact_store import ArtifactStore
from medclaw.stores.audit_log import AuditLog


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ucec_conch_patch_roi_selects_cached_prompt_rois(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_root = write_cached_ucec_conch_case(tmp_path)
    monkeypatch.setenv("MEDCLAW_CASES_ROOT", str(cases_root))

    result = run_ucec_conch_patch_roi(
        "03850333",
        tmp_path / "out",
        top_k_per_prompt=1,
    )

    assert result.findings["roi_selected"] is True
    assert result.findings["patches_selected"] == 2
    assert result.findings["prompt_ids"] == ["endometrioid_tumor", "tumor_glands"]

    roles = [role for _, role, _ in result.artifact_files]
    assert roles == [
        "patch_contact_sheet",
        "patch_roi",
        "patch_roi",
        "roi_metadata",
    ]
    for _, _, path in result.artifact_files:
        assert path.is_file()

    metadata_path = tmp_path / "out" / "ucec_conch_patch_roi_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["case_id"] == "03850333"
    assert metadata["cancer_type"] == "UCEC"
    assert metadata["slide_stem"] == "03850333-SLIDE-1"
    assert len(metadata["selected_patches"]) == 2


def test_ucec_conch_patch_roi_runtime_registers_image_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_root = write_cached_ucec_conch_case(tmp_path)
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
        "pathology.ucec_conch_patch_roi",
        {"case_id": "03850333", "top_k_per_prompt": 1},
    )

    assert result["status"] == "success"
    assert result["findings"]["patches_selected"] == 2
    image_artifacts = [
        artifact for artifact in result["artifacts"] if artifact["type"] == "image"
    ]
    assert [artifact["role"] for artifact in image_artifacts] == [
        "patch_contact_sheet",
        "patch_roi",
        "patch_roi",
    ]
    for artifact in image_artifacts:
        assert artifact_store.resolve_uri(artifact["uri"]).is_file()


def test_ucec_conch_patch_roi_discovers_roi_256_after_stale_configured_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_root = write_cached_ucec_conch_case(tmp_path)
    monkeypatch.setenv("MEDCLAW_CASES_ROOT", str(cases_root))
    pathology_dir = cases_root / "03850333" / "pathology"
    (pathology_dir / "conch_roi").rename(pathology_dir / "roi_256")

    result = run_ucec_conch_patch_roi(
        "03850333",
        tmp_path / "out-roi-256",
        top_k_per_prompt=1,
    )

    assert result.findings["roi_selected"] is True
    assert result.findings["patches_selected"] == 2


def write_cached_ucec_conch_case(tmp_path: Path) -> Path:
    cases_root = tmp_path / "cases"
    case_id = "03850333"
    case_dir = cases_root / case_id
    slide_dir = case_dir / "pathology" / "conch_roi" / "03850333-SLIDE-1"
    roi_dir = slide_dir / "roi"
    (roi_dir / "endometrioid_tumor").mkdir(parents=True)
    (roi_dir / "tumor_glands").mkdir(parents=True)
    (case_dir / "pathology" / "wsi").mkdir(parents=True)

    write_png(
        roi_dir / "endometrioid_tumor" / "endometrioid_tumor_roi_00_idx00005.png",
        (180, 40, 40),
    )
    write_png(
        roi_dir / "tumor_glands" / "tumor_glands_roi_00_idx00009.png",
        (40, 90, 180),
    )

    prompt_scores = {
        "case_id": case_id,
        "stem": "03850333-SLIDE-1",
        "n_patches": 11,
        "feature_dim": 512,
        "patch_size": 256,
        "patch_level": 0,
        "prompts": [
            {
                "id": "endometrioid_tumor",
                "text": "endometrioid adenocarcinoma tumor tissue",
                "top_indices": [5],
                "top_scores": [0.91],
                "top_coords": [[100, 200]],
            },
            {
                "id": "tumor_glands",
                "text": "malignant glandular structures in endometrium",
                "top_indices": [9],
                "top_scores": [0.88],
                "top_coords": [[500, 600]],
            },
        ],
    }
    (slide_dir / "prompt_scores.json").write_text(
        json.dumps(prompt_scores, indent=2),
        encoding="utf-8",
    )
    (slide_dir / "roi_packet.json").write_text(
        json.dumps({"case_id": case_id, "stem": "03850333-SLIDE-1"}),
        encoding="utf-8",
    )
    (case_dir / "case.yaml").write_text(
        "\n".join(
            [
                f"case_id: {case_id}",
                "data:",
                "  pathology:",
                f"    conch_roi_dir: {slide_dir.parent.as_posix()}",
                f"    wsi_dir: {(case_dir / 'pathology' / 'wsi').as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return cases_root


def write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)
