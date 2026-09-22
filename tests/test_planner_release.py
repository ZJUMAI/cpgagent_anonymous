from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from guideline_planner import cli as planner_cli
from guideline_planner.artifacts import (
    ArtifactBindingError,
    build_memory_store_fingerprint,
    sha256_path,
)
from guideline_planner.planner import _fuse_latent_topk_memories
from guideline_planner.release import (
    build_planner_release_manifest,
    resolve_planner_release,
)
from guideline_planner.release_scope import (
    PLANNER_V2_LUNG_ENDOMETRIAL_NPC_RELEASE_SCOPE,
    PLANNER_V2_LUNG_ENDOMETRIAL_RELEASE_SCOPE,
    release_scope_hash,
)
from medclaw_benchmark import cli as benchmark_cli
from medclaw_benchmark.patient_state import initialize_patient_state


def test_release_resolves_modes_and_enforces_action_scope(tmp_path: Path) -> None:
    artifacts = _write_bound_artifacts(tmp_path)
    result = build_planner_release_manifest(
        tmp_path / "release",
        release_id="planner-v2-test",
        dataset_dir=artifacts["dataset"],
        memory_encoder_dir=artifacts["encoder"],
        memory_dir=artifacts["memory"],
        baseline_decoder_dir=artifacts["baseline"],
        runtime_decoder_dir=artifacts["runtime"],
        routing_config=artifacts["routing_config"],
        routing_checkpoint=artifacts["routing_checkpoint"],
        default_mode="daa_full",
    )

    daa = resolve_planner_release(result["manifest_path"])
    baseline = resolve_planner_release(result["manifest_path"], mode="latent_topk")

    assert daa.mode == "daa_full"
    assert daa.decoder_artifact_dir == artifacts["runtime"].resolve()
    assert daa.routing_checkpoint == artifacts["routing_checkpoint"].resolve()
    assert baseline.decoder_artifact_dir == artifacts["baseline"].resolve()
    assert baseline.routing_checkpoint is None
    assert baseline.top_k == 4
    assert baseline.supports_action("lung", "nsclc")
    assert baseline.supports_action("endometrial", "endometrial_carcinoma")
    with pytest.raises(ArtifactBindingError, match="deferred"):
        baseline.require_supported_action("lung", "sclc")
    with pytest.raises(ArtifactBindingError, match="deferred"):
        baseline.require_supported_action("nasopharyngeal", "npc")


def test_release_fails_after_bound_artifact_changes(tmp_path: Path) -> None:
    artifacts = _write_bound_artifacts(tmp_path)
    result = build_planner_release_manifest(
        tmp_path / "release",
        release_id="planner-v2-test",
        dataset_dir=artifacts["dataset"],
        memory_encoder_dir=artifacts["encoder"],
        memory_dir=artifacts["memory"],
        baseline_decoder_dir=artifacts["baseline"],
    )
    (artifacts["dataset"] / "unexpected.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactBindingError, match="dataset.*hash mismatch"):
        resolve_planner_release(result["manifest_path"])


def test_release_debug_override_is_explicit_and_auditable(tmp_path: Path) -> None:
    artifacts = _write_bound_artifacts(tmp_path)
    result = build_planner_release_manifest(
        tmp_path / "release",
        release_id="planner-v2-test",
        dataset_dir=artifacts["dataset"],
        memory_encoder_dir=artifacts["encoder"],
        memory_dir=artifacts["memory"],
        baseline_decoder_dir=artifacts["baseline"],
    )

    release = resolve_planner_release(
        result["manifest_path"],
        overrides={"top_k": 2, "memory_dir": artifacts["memory"]},
    )

    assert release.top_k == 2
    assert release.public_config()["immutable_release"] is False
    assert release.debug_overrides == ("memory_dir", "top_k")


def test_latent_topk_fuses_all_memories_with_normalized_weights() -> None:
    memories = [
        {
            "guideline_memory_id": f"memory-{index}",
            "score": score,
            "memory_slots": np.full((2, 3), index + 1, dtype=np.float32),
            "metadata": {
                "h1_title": f"section {index}",
                "source_rule_ids": [f"rule-{index}"],
                "source_span_ids": [f"span-{index}"],
            },
        }
        for index, score in enumerate((4.0, 3.0, 2.0, 1.0))
    ]

    enriched, slots, activations, diagnostics = _fuse_latent_topk_memories(memories)

    assert slots.shape == (2, 3)
    assert len(enriched) == len(activations) == 4
    assert sum(item.weight for item in activations) == pytest.approx(1.0)
    assert all(item.selected for item in activations)
    assert diagnostics["memory_ids"] == [f"memory-{index}" for index in range(4)]
    expected = sum((index + 1) * item.weight for index, item in enumerate(activations))
    assert slots == pytest.approx(np.full((2, 3), expected, dtype=np.float32))
    for index, activation in enumerate(activations):
        assert enriched[index]["metadata"]["routing_gate_weight"] == pytest.approx(
            activation.weight
        )
    assert diagnostics["slot_count"] == 2
    assert diagnostics["strategy"] == "normalized_weighted_slot_fusion"


@pytest.mark.parametrize(
    ("cancer_type", "guideline_id", "version", "decision_date", "subtype"),
    [
        ("NSCLC-LUAD", "NSCLC_2010", "2010", "2010-12-31", "nsclc"),
        (
            "UCEC",
            "CSCO子宫内膜癌2023",
            "2023",
            "2023-12-31",
            "ucec",
        ),
        (
            "NPC",
            "CSCO鼻咽癌2022",
            "2022",
            "2022-12-31",
            "npc",
        ),
    ],
)
def test_patient_state_uses_release_aligned_guideline_defaults(
    tmp_path: Path,
    cancer_type: str,
    guideline_id: str,
    version: str,
    decision_date: str,
    subtype: str,
) -> None:
    case_dir = tmp_path / cancer_type
    case_dir.mkdir()
    (case_dir / "case_manifest.json").write_text(
        json.dumps({"case_id": "case", "cancer_type": cancer_type}),
        encoding="utf-8",
    )

    state = initialize_patient_state(
        case_dir,
        {"case_id": "case", "visible_information": {}},
    )

    assert state["disease_subtype"] == subtype
    assert state["decision_date"] == decision_date
    assert state["guideline_context"] == {
        "decision_date": decision_date,
        "guidelines": [{"guideline_id": guideline_id, "version": version}],
    }


def test_three_cancer_release_enables_npc_with_versioned_default(
    tmp_path: Path,
) -> None:
    artifacts = _write_bound_artifacts(
        tmp_path,
        release_scope=PLANNER_V2_LUNG_ENDOMETRIAL_NPC_RELEASE_SCOPE,
    )
    result = build_planner_release_manifest(
        tmp_path / "release",
        release_id="planner-v2-three-cancer-test",
        dataset_dir=artifacts["dataset"],
        memory_encoder_dir=artifacts["encoder"],
        memory_dir=artifacts["memory"],
        baseline_decoder_dir=artifacts["baseline"],
        release_scope=PLANNER_V2_LUNG_ENDOMETRIAL_NPC_RELEASE_SCOPE,
    )

    release = resolve_planner_release(result["manifest_path"])

    assert release.supports_action("nasopharyngeal", "npc")
    assert release.guideline_default("nasopharyngeal", "npc") == {
        "guideline_id": "CSCO鼻咽癌2022",
        "version": "2022",
        "decision_date": "2022-12-31",
    }
    with pytest.raises(ArtifactBindingError, match="deferred"):
        release.require_supported_action("lung", "sclc")


def test_patient_state_uses_safe_hidden_case_metadata_for_ambiguous_lung_name(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "LUNG" / "TCGA-78-7540"
    case_dir.mkdir(parents=True)
    (case_dir / "hidden_state.json").write_text(
        json.dumps(
            {
                "case_id": "TCGA-78-7540",
                "cancer_type": "NSCLC-LUAD",
                "project_id": "TCGA-LUAD",
                "initial_prompt": {"chief_problem": "hidden report detail"},
            }
        ),
        encoding="utf-8",
    )

    state = initialize_patient_state(
        case_dir,
        {
            "case_id": "TCGA-78-7540",
            "phase": "diagnosis_phase",
            "visible_information": {
                "chief_problem": "Bronchioloalveolar carcinoma, mucinous"
            },
        },
    )

    assert state["cancer_type"] == "NSCLC-LUAD"
    assert state["cancer_family"] == "lung"
    assert state["disease_subtype"] == "nsclc"
    assert "hidden report detail" not in json.dumps(state)


def test_patient_state_rejects_incompatible_explicit_guideline(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "case_manifest.json").write_text(
        json.dumps(
            {
                "case_id": "case",
                "cancer_type": "nsclc",
                "guideline_context": {
                    "decision_date": "2025-12-31",
                    "guidelines": [
                        {
                            "guideline_id": "2025_CSCO_NSCLC_Guideline",
                            "version": "2025",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incompatible"):
        initialize_patient_state(
            case_dir,
            {"case_id": "case", "visible_information": {}},
        )


def test_patient_state_rejects_deferred_sclc(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "case_manifest.json").write_text(
        json.dumps({"case_id": "case", "cancer_type": "SCLC"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not supported"):
        initialize_patient_state(
            case_dir,
            {"case_id": "case", "visible_information": {}},
        )


def test_medclaw_cli_release_does_not_inject_raw_artifact_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)
            self.run_dir = tmp_path / "run"

        def run(self) -> object:
            self.run_dir.mkdir()
            return type("Result", (), {"run_dir": self.run_dir})()

    monkeypatch.setattr(benchmark_cli, "DualAgentBenchmarkRunner", FakeRunner)
    release_dir = tmp_path / "release"

    assert benchmark_cli.main(
        [
            "dual-agent-run",
            "--case-dir",
            str(tmp_path / "case"),
            "--planner-release-dir",
            str(release_dir),
        ]
    ) == 0
    assert seen["planner_release_dir"] == release_dir
    assert "planner_memory_dir" not in seen
    assert "planner_decoder_artifact_dir" not in seen
    assert "planner_top_k" not in seen


def test_root_guidelines_are_included_in_package_data() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"knowledge/guidelines/*.md"' in pyproject
    guideline_names = {
        path.name
        for path in (project_root / "medclaw" / "knowledge" / "guidelines").glob(
            "*.md"
        )
    }
    assert guideline_names == {
        "NSCLC_2010.md",
        "SCLC_2010.md",
        "CSCO子宫内膜癌2023.md",
        "CSCO鼻咽癌2022.md",
    }


def test_model_snapshot_lock_is_reused_without_network_or_overwrite(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot-commit"
    snapshot.mkdir()
    lock = tmp_path / "model_snapshot.json"
    payload = {
        "schema_version": "model_snapshot_lock.v2",
        "model_name": "Qwen/Qwen3.5-9B",
        "requested_revision": None,
        "resolved_revision": "0123456789abcdef",
        "snapshot_path": str(snapshot),
    }
    original = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    lock.write_text(original, encoding="utf-8")

    assert planner_cli.main(
        [
            "resolve-model-snapshot",
            "--model-name",
            "Qwen/Qwen3.5-9B",
            "--output",
            str(lock),
        ]
    ) == 0
    assert lock.read_text(encoding="utf-8") == original

    with pytest.raises(RuntimeError, match="different revision"):
        planner_cli.main(
            [
                "resolve-model-snapshot",
                "--model-name",
                "Qwen/Qwen3.5-9B",
                "--revision",
                "different",
                "--output",
                str(lock),
            ]
        )


def test_predict_warn_only_quality_gates_returns_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "guideline_planner.fixed_prediction.predict_planner_v2_fixed_test",
        lambda **_: {"reports": {"latent_topk_planner": {"accepted": False}}},
    )

    assert planner_cli.main(["predict-planner-v2"]) == 2
    assert (
        planner_cli.main(
            ["predict-planner-v2", "--warn-only-quality-gates"]
        )
        == 0
    )


def test_package_warn_only_records_failed_quality_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = tmp_path / "summary.json"
    evaluation.write_text(
        json.dumps(
            {
                "reports": {
                    "latent_topk_planner": {"accepted": False},
                    "daa_full_planner": {"accepted": False},
                    "daa_full_router": {"accepted": False},
                },
                "comparison": {"accepted_as_default": False},
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_build(*_: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"manifest_path": str(tmp_path / "release" / "planner_release.json")}

    monkeypatch.setattr(
        "guideline_planner.release.build_planner_release_manifest",
        fake_build,
    )

    assert (
        planner_cli.main(
            [
                "package-planner-release",
                "--evaluation-summary",
                str(evaluation),
                "--warn-only-quality-gates",
            ]
        )
        == 0
    )
    assert captured["default_mode"] == "daa_full"
    assert captured["quality_gate_report"] == {
        "enforcement": "warn_only",
        "latent_topk_passed": False,
        "daa_full_passed": False,
        "evaluation_summary_hash": sha256_path(evaluation),
    }


def _write_bound_artifacts(
    root: Path,
    *,
    release_scope: dict[str, object] = PLANNER_V2_LUNG_ENDOMETRIAL_RELEASE_SCOPE,
) -> dict[str, Path]:
    paths = {
        "dataset": root / "dataset",
        "encoder": root / "encoder",
        "memory": root / "memory",
        "baseline": root / "baseline",
        "runtime": root / "runtime",
        "snapshot": root / "snapshot-0123456789abcdef",
    }
    for path in paths.values():
        path.mkdir()
    (paths["dataset"] / "test.jsonl").write_text("{}\n", encoding="utf-8")
    scope_hash = release_scope_hash(release_scope)
    trajectory_dataset_hash = "4" * 64
    _write_json(
        paths["dataset"] / "manifest.json",
        {
            "ready": True,
            "dataset_hash": trajectory_dataset_hash,
            "release_scope_hash": scope_hash,
        },
    )
    _write_json(
        paths["dataset"] / "admission_report.json",
        {"ready": True, "errors": [], "release_scope_hash": scope_hash},
    )
    hashes = {
        "adapter": "a" * 64,
        "tokenizer": "b" * 64,
        "projection": "c" * 64,
        "baseline_adapter": "d" * 64,
        "baseline_bridge": "e" * 64,
        "runtime_adapter": "f" * 64,
        "runtime_bridge": "1" * 64,
    }
    encoder_meta = {
        "format_version": 2,
        "artifact_role": "memory_encoder",
        "base_model_name": "Qwen/Qwen3.5-9B",
        "base_model_revision": "0123456789abcdef",
        "base_model_snapshot_path": str(paths["snapshot"].resolve()),
        "memory_encoder_adapter_hash": hashes["adapter"],
        "tokenizer_hash": hashes["tokenizer"],
        "retrieval_projection_hash": hashes["projection"],
    }
    _write_json(paths["encoder"] / "memory_encoder_meta.json", encoder_meta)
    memory_meta = {
        "format_version": 2,
        "artifact_role": "memory_store",
        "base_model": "Qwen/Qwen3.5-9B",
        "base_model_name": "Qwen/Qwen3.5-9B",
        "base_model_revision": "0123456789abcdef",
        "base_model_snapshot_path": str(paths["snapshot"].resolve()),
        "tokenizer_hash": hashes["tokenizer"],
        "memory_encoder_adapter_hash": hashes["adapter"],
        "retrieval_projection_hash": hashes["projection"],
        "chunk_corpus_hash": "2" * 64,
        "memory_tokens": 64,
        "slot_hidden_size": 128,
        "schema_hash": "3" * 64,
        "trained": True,
    }
    memory_meta["memory_store_fingerprint"] = build_memory_store_fingerprint(
        memory_meta
    )
    _write_json(paths["memory"] / "memory_store_meta.json", memory_meta)
    for name, adapter_hash, bridge_hash in (
        ("baseline", hashes["baseline_adapter"], hashes["baseline_bridge"]),
        ("runtime", hashes["runtime_adapter"], hashes["runtime_bridge"]),
    ):
        _write_json(
            paths[name] / "planner_decoder_meta.json",
            {
                "format_version": 2,
                "artifact_role": "planner_decoder",
                "base_model_revision": "0123456789abcdef",
                "base_model_snapshot_path": str(paths["snapshot"].resolve()),
                "trajectory_dataset_hash": trajectory_dataset_hash,
                "memory_store_fingerprint": memory_meta[
                    "memory_store_fingerprint"
                ],
                "memory_encoder_adapter_hash": hashes["adapter"],
                "planner_decoder_adapter_hash": adapter_hash,
                "memory_to_decoder_bridge_hash": bridge_hash,
            },
        )
    routing_config = root / "routing.yaml"
    routing_config.write_text("latent_guideline_routing:\n  enabled: true\n", encoding="utf-8")
    routing_checkpoint = root / "routing_checkpoint.pt"
    routing_checkpoint.write_bytes(b"routing-test")
    return {
        **paths,
        "routing_config": routing_config,
        "routing_checkpoint": routing_checkpoint,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
