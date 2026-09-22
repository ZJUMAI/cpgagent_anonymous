from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from medclaw.registry.skill_registry import SkillRegistry
from medclaw.runtime.runtime_manager import RuntimeManager
from medclaw.runtime.skill_runner import SkillRunner
from medclaw.stores.artifact_store import ArtifactStore
from medclaw.stores.audit_log import AuditLog
from medclaw.skills.guideline.retrieve import run as guideline_retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_runtime(
    tmp_path: Path,
    *,
    skills_root: Path | None = None,
) -> tuple[RuntimeManager, ArtifactStore, AuditLog]:
    registry = SkillRegistry(skills_root or PROJECT_ROOT / "medclaw" / "skills")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    audit_log = AuditLog(tmp_path / "runs")
    runtime = RuntimeManager(
        registry,
        SkillRunner(),
        artifact_store,
        audit_log,
        tmp_path / "runs",
    )
    return runtime, artifact_store, audit_log


def test_runtime_registers_artifacts_and_audits_success(tmp_path: Path) -> None:
    skills_root = write_dummy_skill(tmp_path)
    runtime, artifact_store, audit_log = build_runtime(tmp_path, skills_root=skills_root)
    arguments = {"case_id": "TCGA-38-4626", "value": "roi"}

    result = runtime.invoke("test.echo_artifact", arguments)

    assert result["status"] == "success"
    output_path = tmp_path / "runs" / arguments["case_id"] / result["call_id"] / "output.json"
    assert output_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "success"

    assert len(result["artifacts"]) == 1
    artifact_path = artifact_store.resolve_uri(result["artifacts"][0]["uri"])
    assert artifact_path.name == "artifact.json"
    assert artifact_path.is_file()
    assert result["artifacts"][0]["size_bytes"] == artifact_path.stat().st_size
    assert len(result["artifacts"][0]["sha256"]) == 64

    audit_path = audit_log.path_for_case(arguments["case_id"])
    records = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["status"] == "success"
    assert "env_id" not in result["provenance"]


def test_invalid_lung_tumor_arguments_return_audited_failure(tmp_path: Path) -> None:
    runtime, _, audit_log = build_runtime(tmp_path)

    result = runtime.invoke(
        "radiology.lung_tumor_roi",
        {
            "case_id": "TCGA-38-4626",
            "unexpected": "value",
        },
    )

    assert result["status"] == "failed"
    assert "validation failed" in result["findings"]["error"]
    assert audit_log.path_for_case("TCGA-38-4626").is_file()


def test_invalid_conch_patch_arguments_return_audited_failure(tmp_path: Path) -> None:
    runtime, _, audit_log = build_runtime(tmp_path)

    result = runtime.invoke(
        "pathology.conch_patch_roi",
        {
            "case_id": "TCGA-38-4626",
            "top_k_per_prompt": 99,
        },
    )

    assert result["status"] == "failed"
    assert "validation failed" in result["findings"]["error"]
    assert audit_log.path_for_case("TCGA-38-4626").is_file()


def test_guideline_retrieve_returns_release_nsclc_snippets_and_artifact(tmp_path: Path) -> None:
    runtime, artifact_store, audit_log = build_runtime(tmp_path)

    result = runtime.invoke(
        "guideline.retrieve",
        {
            "case_id": "TCGA-38-4626",
            "query": "NCCN 2010 非小细胞肺癌 诊断 分期 治疗",
            "cancer_type": "nsclc",
            "guideline_family": "nccn",
            "guideline_version": "2010",
            "max_snippets": 3,
            "retrieval_mode": "lexical",
            "rerank_mode": "none",
        },
    )

    assert result["status"] == "success"
    assert result["findings"]["retrieved"] is True
    assert result["findings"]["retrieval_method"] == "lexical"
    assert result["findings"]["chunk_mode"] == "chapter"
    assert 1 <= len(result["findings"]["snippets"]) <= 3
    assert result["findings"]["snippets"][0]["source_id"] == "nccn_nsclc_2010"
    assert result["artifacts"][0]["type"] == "json"
    artifact_path = artifact_store.resolve_uri(result["artifacts"][0]["uri"])
    assert artifact_path.name == "guideline_retrieval.json"
    assert artifact_path.is_file()
    assert audit_log.path_for_case("TCGA-38-4626").is_file()


def test_skill_runner_uses_current_python(
    tmp_path: Path,
) -> None:
    registry = SkillRegistry(PROJECT_ROOT / "medclaw" / "skills")
    skill = registry.get_skill("guideline.retrieve")
    runner = SkillRunner()

    command = runner._build_command(
        skill,
        input_path=tmp_path / "input.json",
        output_path=tmp_path / "output.json",
    )

    assert command[0] == sys.executable


def test_skill_runner_captures_nonzero_exit_and_streams(tmp_path: Path) -> None:
    skills_root = write_dummy_skill(tmp_path)
    skill = SkillRegistry(skills_root).get_skill("test.echo_artifact")
    (skill.skill_dir / "fail.py").write_text(
        "import sys\nprint('out')\nprint('err', file=sys.stderr)\nraise SystemExit(7)\n",
        encoding="utf-8",
    )
    skill = replace(skill, runtime={"entrypoint": "python fail.py"})
    work_dir = tmp_path / "fail-work"
    work_dir.mkdir()

    result = SkillRunner().run(
        skill,
        work_dir=work_dir,
        input_path=work_dir / "input.json",
        output_path=work_dir / "output.json",
    )

    assert result.status == "failed"
    assert result.returncode == 7
    assert result.stdout.strip() == "out"
    assert result.stderr.strip() == "err"


def test_skill_runner_reports_timeout(tmp_path: Path) -> None:
    skills_root = write_dummy_skill(tmp_path)
    skill = SkillRegistry(skills_root).get_skill("test.echo_artifact")
    (skill.skill_dir / "slow.py").write_text(
        "import time\ntime.sleep(10)\n",
        encoding="utf-8",
    )
    skill = replace(skill, runtime={"entrypoint": "python slow.py"})
    work_dir = tmp_path / "slow-work"
    work_dir.mkdir()

    result = SkillRunner().run(
        skill,
        work_dir=work_dir,
        input_path=work_dir / "input.json",
        output_path=work_dir / "output.json",
        timeout_sec=0.01,
    )

    assert result.status == "failed"
    assert result.returncode == 124
    assert result.timed_out is True
    assert "timed out" in result.stderr


def test_skill_runner_reports_startup_failure(tmp_path: Path) -> None:
    skills_root = write_dummy_skill(tmp_path)
    skill = SkillRegistry(skills_root).get_skill("test.echo_artifact")
    missing_work_dir = tmp_path / "missing-work"

    result = SkillRunner().run(
        skill,
        work_dir=missing_work_dir,
        input_path=missing_work_dir / "input.json",
        output_path=missing_work_dir / "output.json",
    )

    assert result.status == "failed"
    assert result.returncode is None
    assert "start" in result.stderr.lower()


def test_guideline_retrieve_chapter_mode_reuses_planner_chunks(tmp_path: Path) -> None:
    runtime, artifact_store, _audit_log = build_runtime(tmp_path)

    result = runtime.invoke(
        "guideline.retrieve",
        {
            "case_id": "TCGA-38-4626",
            "query": "NCCN 2010 NSCLC stage workup treatment",
            "cancer_type": "nsclc",
            "guideline_family": "nccn",
            "guideline_version": "2010",
            "chunk_mode": "chapter",
            "max_snippets": 1,
            "retrieval_mode": "lexical",
            "rerank_mode": "none",
        },
    )

    assert result["status"] == "success"
    assert result["findings"]["chunk_mode"] == "chapter"
    assert len(result["findings"]["snippets"]) == 1
    snippet = result["findings"]["snippets"][0]
    assert snippet["chunk_mode"] == "chapter"
    assert snippet["chunk_id"].startswith("NSCLC_2010")
    assert snippet["guideline_memory_id"] == snippet["chunk_id"]
    assert "source_span_ids" in snippet
    artifact_path = artifact_store.resolve_uri(result["artifacts"][0]["uri"])
    audit = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert audit["chunk_mode"] == "chapter"
    assert audit["snippets"][0]["chunk_mode"] == "chapter"


def test_guideline_retrieve_selects_nccn_2010_from_multiversion_corpus(
    tmp_path: Path,
) -> None:
    runtime, _artifact_store, _audit_log = build_runtime(tmp_path)

    result = runtime.invoke(
        "guideline.retrieve",
        {
            "case_id": "TCGA-44-2655",
            "query": "NCCN 2010 NSCLC stage IA surgery follow-up",
            "cancer_type": "nsclc",
            "guideline_family": "auto",
            "guideline_version": "auto",
            "max_snippets": 3,
            "retrieval_mode": "lexical",
            "rerank_mode": "none",
        },
    )

    assert result["status"] == "success"
    assert result["findings"]["selected_guideline_family"] == "nccn"
    assert result["findings"]["selected_guideline_version"] == "2010"
    assert result["findings"]["snippets"]
    assert {
        snippet["source_id"] for snippet in result["findings"]["snippets"]
    } == {"nccn_nsclc_2010"}


def test_guideline_retrieve_recognizes_release_ucec_and_npc_documents() -> None:
    documents = {
        document.path.name: (document.source_id, document.cancer_type)
        for document in guideline_retrieve._load_documents(
            guideline_retrieve._guideline_dir()
        )
    }

    assert documents == {
        "NSCLC_2010.md": ("nccn_nsclc_2010", "nsclc"),
        "SCLC_2010.md": ("nccn_sclc_2010", "sclc"),
        "CSCO子宫内膜癌2023.md": ("csco_ucec_2023", "ucec"),
        "CSCO鼻咽癌2022.md": ("csco_npc_2022", "npc"),
    }
    assert guideline_retrieve._select_cancer_type("auto", "子宫内膜癌 2023") == "ucec"
    assert guideline_retrieve._select_cancer_type("auto", "鼻咽癌 2022") == "npc"


def test_guideline_retrieve_builds_and_reuses_embedding_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeEmbeddings:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            dimensions = int(kwargs["dimensions"])
            vectors = []
            for text in kwargs["input"]:
                assert 0 < len(text) <= guideline_retrieve.EMBEDDING_SEGMENT_MAX_CHARS
                seed = sum(ord(char) for char in text) or 1
                vectors.append(
                    [
                        ((seed + index * 17) % 101) / 100.0
                        for index in range(dimensions)
                    ]
                )
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=vector) for vector in vectors]
            )

    fake_embeddings = FakeEmbeddings()
    fake_client = SimpleNamespace(embeddings=fake_embeddings)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setenv(
        "MEDCLAW_GUIDELINE_EMBEDDING_CACHE_DIR",
        str(tmp_path / "embedding_cache"),
    )
    monkeypatch.setenv("MEDCLAW_GUIDELINE_EMBEDDING_DIMENSIONS", "8")
    monkeypatch.setattr(
        guideline_retrieve,
        "_create_embedding_client",
        lambda config: fake_client,
    )

    arguments = {
        "case_id": "TCGA-38-4626",
        "query": "非小细胞肺癌 EGFR ALK 分子检测",
        "cancer_type": "nsclc",
        "max_snippets": 2,
        "retrieval_mode": "vector",
        "rerank_mode": "none",
    }
    first = guideline_retrieve.run(arguments, output_dir=tmp_path / "first")

    assert first["status"] == "success"
    assert first["findings"]["retrieval_method"] in {"faiss", "numpy_vector"}
    assert first["findings"]["embedding_cache_status"] == "built"
    assert first["findings"]["query_embedding_cache_status"] == "built"
    assert fake_embeddings.calls > 0
    calls_after_first = fake_embeddings.calls

    monkeypatch.delenv("DASHSCOPE_API_KEY")
    monkeypatch.setattr(
        guideline_retrieve,
        "_create_embedding_client",
        lambda config: (_ for _ in ()).throw(AssertionError("embedding API should not be called")),
    )
    second = guideline_retrieve.run(arguments, output_dir=tmp_path / "second")

    assert second["status"] == "success"
    assert second["findings"]["embedding_cache_status"] == "reused"
    assert second["findings"]["query_embedding_cache_status"] == "reused"
    assert fake_embeddings.calls == calls_after_first


def test_guideline_retrieve_uses_core_model_to_rerank_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeChatCompletions:
        def create(self, **kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            candidates = payload["candidates"]
            selected = list(reversed(candidates[:3]))
            return SimpleNamespace(
                model="fake-core-qwen",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "items": [
                                        {
                                            "chunk_id": candidate["chunk_id"],
                                            "relevance_score": 100 - index,
                                            "reason": "fake rerank",
                                        }
                                        for index, candidate in enumerate(selected)
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        )
                    )
                ],
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeChatCompletions())
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setenv("MEDCLAW_GUIDELINE_RERANK_PROVIDER", "qwen")
    monkeypatch.setenv("MEDCLAW_QWEN_MODEL", "fake-core-qwen")
    monkeypatch.setattr(
        guideline_retrieve,
        "_create_rerank_client",
        lambda config: fake_client,
    )

    result = guideline_retrieve.run(
        {
            "case_id": "TCGA-38-4626",
            "query": "非小细胞肺癌 分子检测 EGFR ALK KRAS PD-L1 治疗推荐",
            "cancer_type": "nsclc",
            "max_snippets": 2,
            "retrieval_mode": "lexical",
            "rerank_mode": "llm",
            "rerank_candidate_count": 6,
        },
        output_dir=tmp_path / "rerank",
    )

    assert result["status"] == "success"
    assert result["findings"]["rerank_status"] == "llm_reranked"
    assert result["findings"]["rerank_model"] == "fake-core-qwen"
    assert len(result["findings"]["snippets"]) == 2
    assert result["findings"]["snippets"][0]["retrieval_rank"] == 3
    assert result["findings"]["snippets"][0]["rerank_reason"] == "fake rerank"
    audit = json.loads((tmp_path / "rerank" / "guideline_retrieval.json").read_text(encoding="utf-8"))
    assert len(audit["pre_rerank_snippets"]) == 6
    assert audit["rerank_status"] == "llm_reranked"


def test_guideline_rerank_defaults_to_local_vllm(monkeypatch) -> None:
    monkeypatch.delenv("MEDCLAW_GUIDELINE_RERANK_PROVIDER", raising=False)
    monkeypatch.delenv("MEDCLAW_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("MEDCLAW_LOCAL_API_KEY", "local-test-key")

    config = guideline_retrieve._rerank_config_from_env()

    assert config.provider == "local_openai"
    assert config.api_key == "local-test-key"
    assert config.base_url == "http://127.0.0.1:8087/v1"
    assert config.model == "qwen3.6-27b"
    assert config.enable_thinking is False


def write_dummy_skill(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "skills" / "test" / "echo_artifact"
    skill_dir.mkdir(parents=True)
    (skill_dir / "requirements.txt").write_text("", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text("# test.echo_artifact\n", encoding="utf-8")
    (skill_dir / "skill.yaml").write_text(
        """
name: test.echo_artifact
version: 0.1.0
description: Emit a tiny JSON artifact for runtime tests.
runtime:
  entrypoint: python run.py
resources:
  cpu: 1
  memory_gb: 1
  timeout_sec: 30
input_schema:
  type: object
  additionalProperties: false
  required:
    - case_id
    - value
  properties:
    case_id:
      type: string
    value:
      type: string
output_schema:
  type: object
  required:
    - status
    - findings
    - artifacts
    - provenance
    - warnings
  properties:
    status:
      enum:
        - success
        - failed
    findings:
      type: object
    artifacts:
      type: array
    provenance:
      type: object
    warnings:
      type: array
visibility:
  expose_to_agent: false
  agent_card: SKILL.md
""".lstrip(),
        encoding="utf-8",
    )
    (skill_dir / "run.py").write_text(
        """
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    artifact = Path(args.output).with_name("artifact.json")
    artifact.write_text(
        json.dumps({"value": payload["arguments"]["value"]}, sort_keys=True),
        encoding="utf-8",
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "status": "success",
                "findings": {"summary": "dummy artifact emitted"},
                "artifacts": [
                    {"type": "json", "role": "dummy", "path": artifact.name}
                ],
                "provenance": {},
                "warnings": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""".lstrip(),
        encoding="utf-8",
    )
    return tmp_path / "skills"
