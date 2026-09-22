from pathlib import Path

import pytest

from medclaw.registry.skill_registry import SkillManifestError, SkillRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_registry_discovers_expected_skills() -> None:
    registry = SkillRegistry(PROJECT_ROOT / "medclaw" / "skills")

    names = {card["name"] for card in registry.list_tools_for_agent()}
    assert len(registry) == 7
    assert names == {
        "guideline.retrieve",
        "pathology.conch_patch_roi",
        "pathology.npc_conch_patch_roi",
        "pathology.ucec_conch_patch_roi",
        "radiology.lung_tumor_roi",
        "radiology.npc_mri_roi",
        "radiology.ucec_mri_roi",
    }

    guideline = registry.get_skill("guideline.retrieve")
    assert guideline.runtime == {"entrypoint": "python run.py"}
    assert guideline.resources["timeout_sec"] >= 600
    assert guideline.input_schema["required"] == ["case_id"]
    assert guideline.input_schema["properties"]["cancer_type"]["type"] == "string"
    assert guideline.input_schema["properties"]["retrieval_mode"]["enum"] == [
        "auto",
        "vector",
        "lexical",
    ]
    assert guideline.input_schema["properties"]["rerank_mode"]["enum"] == [
        "auto",
        "llm",
        "none",
    ]
    assert guideline.input_schema["properties"]["rerank_candidate_count"]["maximum"] == 24
    assert guideline.input_schema["properties"]["chunk_mode"]["enum"] == ["page", "chapter"]
    assert guideline.input_schema["properties"]["chunk_mode"]["default"] == "chapter"
    assert guideline.input_schema["properties"]["force_rebuild_embeddings"]["type"] == "boolean"

    lung = registry.get_skill("radiology.lung_tumor_roi")
    assert lung.runtime == {"entrypoint": "python run.py"}
    assert lung.resources["gpu"] == "optional"
    assert lung.input_schema["required"] == ["case_id"]
    assert lung.input_schema["properties"]["ct_uri"]["type"] == "string"
    assert lung.input_schema["properties"]["tumor_mask_uri"]["type"] == "string"

    ucec_mri = registry.get_skill("radiology.ucec_mri_roi")
    assert ucec_mri.runtime == {"entrypoint": "python run.py"}
    assert ucec_mri.resources["gpu"] == "optional"
    assert ucec_mri.input_schema["required"] == ["case_id"]
    assert "modalities" in ucec_mri.input_schema["properties"]

    luad_path = registry.get_skill("pathology.conch_patch_roi")
    assert luad_path.runtime == {"entrypoint": "python run.py"}
    assert luad_path.input_schema["required"] == ["case_id"]
    assert luad_path.input_schema["properties"]["top_k_per_prompt"]["maximum"] == 10

    ucec_path = registry.get_skill("pathology.ucec_conch_patch_roi")
    assert ucec_path.runtime == {"entrypoint": "python run.py"}
    assert ucec_path.input_schema["required"] == ["case_id"]
    assert ucec_path.input_schema["properties"]["top_k_per_prompt"]["maximum"] == 10

    npc_path = registry.get_skill("pathology.npc_conch_patch_roi")
    assert npc_path.runtime == {"entrypoint": "python run.py"}
    assert npc_path.input_schema["required"] == ["case_id"]
    assert npc_path.input_schema["properties"]["top_k_per_prompt"]["maximum"] == 10

    npc_mri = registry.get_skill("radiology.npc_mri_roi")
    assert npc_mri.runtime == {"entrypoint": "python run.py"}
    assert npc_mri.input_schema["required"] == ["case_id"]
    assert npc_mri.input_schema["properties"]["roi_kinds"]["items"]["enum"] == [
        "primary",
        "node",
    ]
    assert npc_mri.input_schema["properties"]["roi_kinds"]["default"] == ["primary"]


@pytest.mark.parametrize(
    "runtime_yaml",
    ["{}", "{entrypoint: ''}", "{entrypoint: []}"],
)
def test_registry_rejects_missing_or_invalid_entrypoint(
    tmp_path: Path,
    runtime_yaml: str,
) -> None:
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        "\n".join(
            [
                "name: demo.skill",
                "version: '1.0'",
                "description: Demo skill",
                "input_schema: {}",
                "output_schema: {}",
                f"runtime: {runtime_yaml}",
                "resources: {}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SkillManifestError, match="entrypoint"):
        SkillRegistry(tmp_path)
