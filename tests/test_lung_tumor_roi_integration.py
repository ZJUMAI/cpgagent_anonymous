from __future__ import annotations

import os
from pathlib import Path

import pytest

from examples.run_lung_tumor_roi_demo import run_demo
from medclaw.registry.skill_registry import SkillRegistry
from medclaw.runtime.runtime_manager import RuntimeManager
from medclaw.runtime.skill_runner import SkillRunner
from medclaw.stores.artifact_store import ArtifactStore
from medclaw.stores.audit_log import AuditLog


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    os.environ.get("MEDCLAW_RUN_RADIOLOGY_INTEGRATION") != "1",
    reason="Set MEDCLAW_RUN_RADIOLOGY_INTEGRATION=1 to run the real model.",
)
def test_real_tcga_case_generates_complete_roi_audit_set(tmp_path: Path) -> None:
    registry = SkillRegistry(PROJECT_ROOT / "medclaw" / "skills")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    runtime = RuntimeManager(
        registry,
        SkillRunner(),
        artifacts,
        AuditLog(tmp_path / "runs"),
        tmp_path / "runs",
    )

    result = runtime.invoke(
        "radiology.lung_tumor_roi",
        {"case_id": "TCGA-38-4626"},
    )

    assert result["status"] == "success"
    assert result["findings"]["tumor_detected"] is True
    roles = {artifact["role"] for artifact in result["artifacts"]}
    assert roles == {
        "raw_tumor_mask",
        "largest_tumor_mask",
        "roi",
        "full_slice",
        "overlay",
        "roi_metadata",
    }
    assert all(
        artifacts.resolve_uri(artifact["uri"]).is_file()
        for artifact in result["artifacts"]
    )
    assert all(
        isinstance(artifact.get("sha256"), str)
        and len(artifact["sha256"]) == 64
        and artifact.get("size_bytes", 0) > 0
        for artifact in result["artifacts"]
    )


@pytest.mark.skipif(
    os.environ.get("MEDCLAW_RUN_QWEN_INTEGRATION") != "1"
    or not os.environ.get("DASHSCOPE_API_KEY"),
    reason="Set MEDCLAW_RUN_QWEN_INTEGRATION=1 and DASHSCOPE_API_KEY to call Qwen.",
)
def test_qwen_auto_selects_real_roi_skill_and_answers_after_images(tmp_path: Path) -> None:
    demo = run_demo(
        project_root=PROJECT_ROOT,
        runs_root=tmp_path / "runs",
        artifacts_root=tmp_path / "artifacts",
    )

    assert demo["answer"].strip()
    assert demo["tool_result"]["status"] == "success"
    assert demo["artifact_paths"]
    assert Path(demo["conversation_log_path"]).is_file()
