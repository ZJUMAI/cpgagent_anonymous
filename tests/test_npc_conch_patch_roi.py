from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, PngImagePlugin

from medclaw.registry.skill_registry import SkillRegistry
from medclaw.runtime.runtime_manager import RuntimeManager
from medclaw.runtime.skill_runner import SkillRunner
from medclaw.skills.pathology.npc_conch_patch_roi.workflow import run_npc_conch_patch_roi
from medclaw.stores.artifact_store import ArtifactStore
from medclaw.stores.audit_log import AuditLog


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_npc_conch_patch_roi_selects_cached_prompt_rois(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_root = write_cached_npc_conch_case(tmp_path)
    monkeypatch.setenv("MEDCLAW_CASES_ROOT", str(cases_root))

    result = run_npc_conch_patch_roi(
        "NPC-DEMO",
        tmp_path / "out",
        top_k_per_prompt=1,
    )

    assert result.findings["roi_selected"] is True
    assert result.findings["patches_selected"] == 2
    assert result.findings["prompt_ids"] == ["undifferentiated_npc", "nested_tumor"]

    roles = [role for _, role, _ in result.artifact_files]
    assert roles == [
        "patch_contact_sheet",
        "patch_roi",
        "patch_roi",
        "roi_metadata",
    ]
    for _, _, path in result.artifact_files:
        assert path.is_file()

    metadata_path = tmp_path / "out" / "npc_conch_patch_roi_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["case_id"] == "NPC-DEMO"
    assert metadata["cancer_type"] == "NPC"
    assert metadata["slide_stem"] == "NPC-DEMO-SLIDE-1"
    assert len(metadata["selected_patches"]) == 2


def test_npc_conch_patch_roi_runtime_registers_image_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_root = write_cached_npc_conch_case(tmp_path)
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
        "pathology.npc_conch_patch_roi",
        {"case_id": "NPC-DEMO", "top_k_per_prompt": 1},
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


def test_npc_conch_patch_roi_falls_back_when_requested_prompt_ids_do_not_match(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_root = write_cached_npc_conch_case(tmp_path)
    monkeypatch.setenv("MEDCLAW_CASES_ROOT", str(cases_root))
    slide_dir = (
        cases_root / "NPC-DEMO" / "pathology" / "roi_256" / "NPC-DEMO-SLIDE-1"
    )
    scores_path = slide_dir / "prompt_scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    replacements = {
        "undifferentiated_npc": "prompts_00",
        "nested_tumor": "prompts_01",
    }
    for prompt in scores["prompts"][:2]:
        old_id = prompt["id"]
        new_id = replacements[old_id]
        old_dir = slide_dir / "roi" / old_id
        new_dir = slide_dir / "roi" / new_id
        old_dir.rename(new_dir)
        old_png = next(new_dir.glob("*.png"))
        old_png.rename(new_dir / old_png.name.replace(old_id, new_id))
        prompt["id"] = new_id
    scores_path.write_text(json.dumps(scores), encoding="utf-8")

    result = run_npc_conch_patch_roi(
        "NPC-DEMO",
        tmp_path / "out-fallback",
        prompt_ids=("undifferentiated_npc", "nested_tumor"),
        top_k_per_prompt=1,
    )

    assert result.findings["prompt_ids"] == ["prompts_00", "prompts_01"]
    assert result.findings["patches_selected"] == 2
    assert any("fell back to available prompt IDs" in item for item in result.warnings)


def test_npc_conch_patch_roi_ignores_oversized_png_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_root = write_cached_npc_conch_case(tmp_path)
    monkeypatch.setenv("MEDCLAW_CASES_ROOT", str(cases_root))
    source = next(
        (
            cases_root
            / "NPC-DEMO"
            / "pathology"
            / "roi_256"
            / "NPC-DEMO-SLIDE-1"
            / "roi"
            / "undifferentiated_npc"
        ).glob("*.png")
    )
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("generator_payload", "x" * (2 * 1024 * 1024), zip=True)
    Image.new("RGB", (32, 32), (180, 40, 40)).save(
        source,
        pnginfo=pnginfo,
        icc_profile=b"x" * (2 * 1024 * 1024),
    )

    result = run_npc_conch_patch_roi(
        "NPC-DEMO",
        tmp_path / "out-large-text",
        prompt_ids=("undifferentiated_npc",),
        top_k_per_prompt=1,
    )

    assert result.findings["patches_selected"] == 1
    with Image.open(next(path for kind, role, path in result.artifact_files if role == "patch_roi")) as image:
        assert image.size == (32, 32)


def write_cached_npc_conch_case(tmp_path: Path) -> Path:
    cases_root = tmp_path / "cases"
    case_id = "NPC-DEMO"
    case_dir = cases_root / case_id
    slide_dir = case_dir / "pathology" / "roi_256" / "NPC-DEMO-SLIDE-1"
    roi_dir = slide_dir / "roi"
    (roi_dir / "undifferentiated_npc").mkdir(parents=True)
    (roi_dir / "nested_tumor").mkdir(parents=True)
    (case_dir / "pathology" / "wsi").mkdir(parents=True)

    write_png(
        roi_dir / "undifferentiated_npc" / "undifferentiated_npc_roi_00_idx00005.png",
        (180, 40, 40),
    )
    write_png(
        roi_dir / "nested_tumor" / "nested_tumor_roi_00_idx00009.png",
        (40, 90, 180),
    )

    prompt_scores = {
        "case_id": case_id,
        "stem": "NPC-DEMO-SLIDE-1",
        "n_patches": 11,
        "feature_dim": 512,
        "patch_size": 256,
        "patch_level": 0,
        "prompts": [
            {
                "id": "undifferentiated_npc",
                "text": "undifferentiated nasopharyngeal carcinoma tumor tissue",
                "top_indices": [5],
                "top_scores": [0.91],
                "top_coords": [[100, 200]],
            },
            {
                "id": "nested_tumor",
                "text": "nested nasopharyngeal carcinoma tumor nests",
                "top_indices": [9],
                "top_scores": [0.88],
                "top_coords": [[500, 600]],
            },
            {
                "id": "lymphoepithelial",
                "text": "lymphoepithelial carcinoma with mixed lymphoid stroma",
                "top_indices": [2],
                "top_scores": [0.40],
                "top_coords": [[10, 20]],
            },
        ],
    }
    (slide_dir / "prompt_scores.json").write_text(
        json.dumps(prompt_scores, indent=2),
        encoding="utf-8",
    )
    (slide_dir / "roi_packet.json").write_text(
        json.dumps({"case_id": case_id, "stem": "NPC-DEMO-SLIDE-1"}),
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
