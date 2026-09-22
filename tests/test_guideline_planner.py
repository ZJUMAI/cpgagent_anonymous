from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import guideline_planner.memory as memory_module
from guideline_planner.chunking import chunk_guidelines
from guideline_planner.artifacts import sha256_path
from guideline_planner.dataset import build_training_data
from guideline_planner.io_utils import read_jsonl
from guideline_planner.medclaw_adapter import to_medclaw_plan
from guideline_planner.memory import extract_memory_slots
from guideline_planner.modeling import (
    PlannerModelBundle,
    PlannerTrainingConfig,
    _align_projection_heads_to_model,
    _encode_target_with_eos,
    _patch_deepspeed_plugin_config,
    _resolve_generation_max_new_tokens,
    _split_train_validation_records,
    add_planner_special_tokens,
    enable_non_reentrant_gradient_checkpointing,
    encode_prompt_and_full_target,
    generate_with_memory_slots,
)
from guideline_planner.planner import (
    LatentGuidelinePlanner,
    LatentPlannerError,
    _memory_metadata_summary,
    _normalize_planner_output,
    _parse_planner_json,
    _routing_memories_for_prompt,
    build_planner_query,
)
from guideline_planner.retrieval import retrieve_latent_guideline_memory


def test_chunk_guidelines_uses_h1_not_page_boundaries(tmp_path: Path) -> None:
    guideline_dir = tmp_path / "guidelines"
    guideline_dir.mkdir()
    (guideline_dir / "2025_CSCO_NSCLC_Guideline.md").write_text(
        "\n".join(
            [
                "# 提取自: source.pdf",
                "## 第 1 页",
                "版权信息",
                "# 指南要点",
                "## 第 2 页",
                "NSCLC 诊断和治疗推荐。",
                "## 第 3 页",
                "继续同一个一级标题，不应该切成新 chunk。",
                "# 一、诊断原则",
                "## 第 4 页",
                "影像、病理、分子检测和分期诊断原则。",
            ]
        ),
        encoding="utf-8",
    )

    chunks = chunk_guidelines(guideline_dir)

    assert [chunk.h1_title for chunk in chunks] == [
        "提取自: source.pdf",
        "指南要点",
        "一、诊断原则",
    ]
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[1].page_start == 1
    assert chunks[1].page_end == 3
    assert "继续同一个一级标题" in chunks[1].text
    assert chunks[2].cancer_type == "nsclc"
    assert chunks[2].version == "2025"


def test_chunk_guidelines_supports_english_nccn_clinical_headings(
    tmp_path: Path,
) -> None:
    guideline_dir = tmp_path / "guidelines" / "nccn"
    guideline_dir.mkdir(parents=True)
    (guideline_dir / "NCCN NSCLC 2010.md").write_text(
        "\n".join(
            [
                "# NCCN Non-Small Cell Lung Cancer Panel Members",
                "Panel roster and disclosures.",
                "# Initial Evaluation and Clinical Stages (NSCLC-1)",
                "Diagnostic evaluation, staging, and treatment recommendations.",
                "# Surveillance (NSCL-12)",
                "Surveillance after definitive therapy.",
                "# Principles of Surgical Therapy (NSCLC-B)",
                "Principles and recommendations for surgical treatment.",
                "# References",
                "Treatment reference list.",
            ]
        ),
        encoding="utf-8",
    )

    chunks = chunk_guidelines(tmp_path / "guidelines")

    assert [chunk.h1_title for chunk in chunks] == [
        "NCCN Non-Small Cell Lung Cancer Panel Members",
        "Initial Evaluation and Clinical Stages (NSCLC-1)",
        "Surveillance (NSCL-12)",
        "Principles of Surgical Therapy (NSCLC-B)",
        "References",
    ]
    assert all(chunk.version == "2010" for chunk in chunks)
    assert all(chunk.cancer_type == "nsclc" for chunk in chunks)
    assert all(chunk.language == "en" for chunk in chunks)
    # All H1 chunks stay in Memory Encoder scope, while non-clinical panel
    # metadata is intentionally excluded from Planner action-rule supervision.
    assert chunks[0].source_rule_ids == []
    assert chunks[1].source_rule_ids


def test_extract_memory_uses_h1_title_file_names_and_metadata(tmp_path: Path) -> None:
    chunks = [
        {
            "guideline_id": "2025_CSCO_NSCLC_Guideline",
            "version": "2025",
            "cancer_type": "nsclc",
            "chapter": "一、诊断原则",
            "section": "一、诊断原则",
            "h1_title": "一、诊断原则",
            "source_span_id": "nsclc::h1::0001",
            "source_rule_ids": ["r1"],
            "source_span_ids": ["nsclc::h1::0001"],
            "page_start": 1,
            "page_end": 2,
            "text": "影像、病理、分子检测和分期诊断原则。",
        }
    ]

    result = extract_memory_slots(chunks, tmp_path / "memory_store", memory_tokens=4, mock=True)
    metadata = read_jsonl(Path(result["metadata_path"]))

    assert result["slot_count"] == 1
    latent_path = Path(metadata[0]["latent_vector_path"])
    embedding_path = Path(metadata[0]["retrieval_embedding_path"])
    assert latent_path.name == "一、诊断原则.pt"
    assert embedding_path.name == "一、诊断原则.npy"
    memory_dir = Path(result["memory_dir"])
    assert (memory_dir / latent_path).is_file()
    assert (memory_dir / embedding_path).is_file()
    meta = json.loads((memory_dir / "memory_store_meta.json").read_text(encoding="utf-8"))
    assert meta["trained"] is False
    assert meta["memory_tokens"] == 4
    assert meta["slot_hidden_size"] == 128
    assert metadata[0]["h1_title"] == "一、诊断原则"


def test_build_training_data_writes_clean_memory_task_families(tmp_path: Path) -> None:
    chunks = [
        _chunk("m1", "nsclc", "诊断原则", "病理和影像诊断推荐。"),
        _chunk("m2", "nsclc", "治疗原则", "根据分期和分子检测治疗推荐。"),
        _chunk("m3", "sclc", "随访", "治疗后随访和复查推荐。"),
    ]

    manifest = build_training_data(chunks, tmp_path / "train")
    mixed = read_jsonl(tmp_path / "train" / "mixed_train.jsonl")

    assert manifest["task_counts"] == {
        "AE": 3,
        "RETRIEVE": 3,
        "CONTINUE": 3,
    }
    assert {record["task"] for record in mixed} == {"AE", "RETRIEVE", "CONTINUE"}
    assert not (tmp_path / "train" / "plan.jsonl").exists()
    retrieval = [record for record in mixed if record["task"] == "RETRIEVE"][0]
    assert retrieval["strong_positive"]
    assert "hard_negative" in retrieval
    assert "easy_negative" in retrieval


def test_extract_memory_with_training_run_writes_trained_meta(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "train_run"
    snapshot_dir = tmp_path / "snapshots" / "qwen-fixed"
    snapshot_dir.mkdir(parents=True)
    adapter_dir = run_dir / "memory_encoder_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "tokenizer").mkdir()
    (run_dir / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "retrieval_projection.pt").write_bytes(b"fake")
    (run_dir / "training_config.json").write_text(
        json.dumps(
            {
                "model_name": "fake-qwen",
                "model_revision": "abc123",
                "model_snapshot_path": str(snapshot_dir),
                "model_load_path": str(snapshot_dir),
                "memory_tokens": 5,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "memory_encoder_meta.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "artifact_role": "memory_encoder",
                "memory_encoder_adapter_hash": sha256_path(adapter_dir),
                "tokenizer_hash": sha256_path(run_dir / "tokenizer"),
                "retrieval_projection_hash": sha256_path(run_dir / "retrieval_projection.pt"),
            }
        ),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def fake_load_bundle(**kwargs):
        seen.update(kwargs)
        return {"bundle_kwargs": kwargs}

    monkeypatch.setattr(memory_module, "load_planner_model_bundle", fake_load_bundle)
    monkeypatch.setattr(
        memory_module,
        "encode_memory_slots_with_bundle",
        lambda bundle, text, memory_tokens, max_encoder_tokens: np.ones(
            (memory_tokens, 7),
            dtype="float32",
        ),
    )
    monkeypatch.setattr(
        memory_module,
        "projected_slot_embedding_with_bundle",
        lambda bundle, slots: np.arange(slots.shape[-1], dtype="float32"),
    )

    result = extract_memory_slots(
        [_chunk("nsclc_diag", "nsclc", "诊断原则", "NSCLC 诊断推荐。")],
        tmp_path / "memory_store",
        training_run_dir=run_dir,
    )

    memory_dir = Path(result["memory_dir"])
    meta = json.loads((memory_dir / "memory_store_meta.json").read_text(encoding="utf-8"))
    assert result["trained"] is True
    assert meta["trained"] is True
    assert meta["base_model"] == str(snapshot_dir)
    assert meta["base_model_name"] == "fake-qwen"
    assert meta["base_model_revision"] == "abc123"
    assert meta["base_model_snapshot_path"] == str(snapshot_dir)
    assert meta["memory_tokens"] == 5
    assert meta["slot_hidden_size"] == 7
    assert meta["memory_encoder_adapter_path"] == "artifacts/memory_encoder_adapter"
    exported_adapter = memory_dir / meta["memory_encoder_adapter_path"]
    assert exported_adapter.is_dir()
    assert meta["memory_encoder_adapter_hash"] == sha256_path(exported_adapter)
    assert meta["memory_encoder_adapter_hash"] == sha256_path(adapter_dir)
    assert meta["tokenizer_path"] == "artifacts/tokenizer"
    assert meta["retrieval_projection_path"] == "artifacts/retrieval_projection.pt"
    assert seen["model_name"] == str(snapshot_dir)
    assert np.load(memory_dir / "memory" / "guide_nsclc" / "诊断原则.npy").shape == (7,)


def test_extract_memory_requires_snapshot_for_training_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "old_train_run"
    run_dir.mkdir()
    (run_dir / "training_config.json").write_text(
        json.dumps({"model_name": "floating-qwen", "memory_tokens": 5}),
        encoding="utf-8",
    )

    try:
        extract_memory_slots(
            [_chunk("nsclc_diag", "nsclc", "诊断原则", "NSCLC 诊断推荐。")],
            tmp_path / "memory_store",
            training_run_dir=run_dir,
        )
    except RuntimeError as exc:
        assert "fixed model snapshot" in str(exc)
    else:
        raise AssertionError("old training run without snapshot should fail")


def test_retrieve_filters_metadata_then_ranks(tmp_path: Path) -> None:
    chunks = [
        _chunk("nsclc_diag", "nsclc", "诊断原则", "NSCLC 影像 病理 分子检测 推荐。"),
        _chunk("sclc_follow", "sclc", "随访", "SCLC 随访 复查 推荐。"),
    ]
    extract_memory_slots(chunks, tmp_path / "memory_store", memory_tokens=3, mock=True)

    results = retrieve_latent_guideline_memory(
        "NSCLC 诊断 分子检测",
        5,
        memory_dir=tmp_path / "memory_store",
        filters={"cancer_type": "nsclc"},
        load_slots=False,
    )

    assert len(results) == 1
    assert results[0]["metadata"]["cancer_type"] == "nsclc"
    assert results[0]["guideline_memory_id"] == "nsclc_diag"


def test_retrieve_accepts_legacy_root_prefixed_relative_paths(tmp_path: Path) -> None:
    chunks = [
        _chunk("nsclc_diag", "nsclc", "诊断原则", "NSCLC 影像 病理 分子检测 推荐。")
    ]
    memory_dir = tmp_path / "memory_store"
    extract_memory_slots(chunks, memory_dir, memory_tokens=3, mock=True)
    metadata_path = memory_dir / "slot_metadata.jsonl"
    records = read_jsonl(metadata_path)
    for record in records:
        record["latent_vector_path"] = str(memory_dir / record["latent_vector_path"])
        record["retrieval_embedding_path"] = str(
            memory_dir / record["retrieval_embedding_path"]
        )
    metadata_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    results = retrieve_latent_guideline_memory(
        "NSCLC 诊断",
        1,
        memory_dir=memory_dir,
        filters={"cancer_type": "nsclc"},
        load_slots=False,
    )

    assert results[0]["guideline_memory_id"] == "nsclc_diag"


def test_mock_memory_store_cannot_run_latent_planner(tmp_path: Path) -> None:
    chunks = [
        _chunk("nsclc_diag", "nsclc", "诊断原则", "NSCLC 影像 病理 分子检测 推荐。")
    ]
    extract_memory_slots(chunks, tmp_path / "memory_store", memory_tokens=3, mock=True)

    try:
        LatentGuidelinePlanner(memory_dir=tmp_path / "memory_store", decoder=_FakeLatentDecoder())
    except LatentPlannerError as exc:
        assert "trained=false" in str(exc)
    else:
        raise AssertionError("mock memory store should not run latent planner")


def test_planner_next_step_uses_latent_decoder_slots_and_medclaw_adapter(tmp_path: Path) -> None:
    chunks = [
        _chunk("nsclc_diag", "nsclc", "诊断原则", "NSCLC 影像 病理 分子检测 推荐。")
    ]
    result = extract_memory_slots(chunks, tmp_path / "memory_store", memory_tokens=3, mock=True)
    _mark_mock_store_as_trained(Path(result["memory_dir"]))
    decoder = _FakeLatentDecoder()
    planner = LatentGuidelinePlanner(
        memory_dir=tmp_path / "memory_store",
        decoder=decoder,
        max_new_tokens=77,
    )

    output = planner.next_step(_patient_state_v2(), trajectory_plan=[])
    medclaw_plan = to_medclaw_plan(output)

    assert output["current_phase"] == "diagnostic_workup"
    assert "pathology" in output["missing_information"]
    assert "pathology.read_report" in output["actions"][0]["required_skills"]
    assert decoder.seen_slots is not None
    assert decoder.seen_slots.shape == (3, 128)
    assert decoder.seen_max_new_tokens == 77
    assert "[PLAN]" in decoder.seen_prompt
    assert decoder.seen_prompt.endswith("\n")
    assert planner.last_attempt["generation_attempts"][0]["label"] == "raw"
    generation_info = planner.last_attempt["generation_attempts"][0][
        "decoder_generation"
    ]
    assert generation_info["prompt_token_count"] == 42
    assert planner.last_attempt["raw_text_token_count"] == len(
        planner.last_attempt["raw_text"]
    )
    assert planner.last_attempt["planner_output_token_count"] > 0
    assert (
        planner.last_attempt["planner_output_text_metrics"]["token_count_method"]
        == "decoder_tokenizer"
    )
    assert medclaw_plan["planner"] == "latent_guideline_memory_v2"
    assert medclaw_plan["suggested_skills"] == output["actions"][0]["required_skills"]


def test_planner_retrieval_filters_target_cancer_and_version(tmp_path: Path) -> None:
    chunks = [
        _chunk(
            "nsclc_diag",
            "nsclc",
            "diagnosis",
            "NSCLC imaging pathology molecular recommendation.",
        ),
        _chunk(
            "breast_diag",
            "breast",
            "diagnosis",
            "Breast cancer diagnosis recommendation.",
        ),
    ]
    result = extract_memory_slots(chunks, tmp_path / "memory_store", memory_tokens=3, mock=True)
    _mark_mock_store_as_trained(Path(result["memory_dir"]))
    decoder = _FakeLatentDecoder()
    planner = LatentGuidelinePlanner(memory_dir=tmp_path / "memory_store", decoder=decoder)

    output = planner.next_step(
        _patient_state_v2(known_diagnosis="Adenocarcinoma, NOS"),
        trajectory_plan=[],
    )

    assert output["actions"][0]["provenance"][0]["memory_id"] == "nsclc_diag"
    assert "nsclc_diag" in decoder.seen_prompt
    assert "breast_diag" not in decoder.seen_prompt


def test_planner_repairs_non_json_decoder_output(tmp_path: Path) -> None:
    chunks = [
        _chunk("nsclc_diag", "nsclc", "diagnosis", "NSCLC diagnostic recommendation.")
    ]
    result = extract_memory_slots(chunks, tmp_path / "memory_store", memory_tokens=3, mock=True)
    _mark_mock_store_as_trained(Path(result["memory_dir"]))
    decoder = _RepairingLatentDecoder()
    planner = LatentGuidelinePlanner(memory_dir=tmp_path / "memory_store", decoder=decoder)

    output = planner.next_step(
        _patient_state_v2(known_diagnosis="Adenocarcinoma, NOS"),
        trajectory_plan=[],
    )

    assert output["actions"][0]["objective"] == "Collect missing pathology evidence."
    assert planner.last_attempt["status"] == "repaired"
    assert "not json" in planner.last_attempt["raw_text"]
    assert "repair_latent_guideline_planner_json" in planner.last_attempt["repair_prompt_preview"]
    assert decoder.generate_calls == 2


def test_planner_v2_rejects_free_text_mode(tmp_path: Path) -> None:
    chunks = [
        _chunk("nsclc_diag", "nsclc", "diagnosis", "NSCLC diagnostic recommendation.")
    ]
    result = extract_memory_slots(
        chunks,
        tmp_path / "memory_store",
        memory_tokens=3,
        mock=True,
    )
    _mark_mock_store_as_trained(Path(result["memory_dir"]))
    with pytest.raises(ValueError, match="strict JSON"):
        LatentGuidelinePlanner(
            memory_dir=tmp_path / "memory_store",
            decoder=_FakeLatentDecoder(),
            output_mode="free_text",
        )


def test_planner_strict_repairs_after_second_bad_output(tmp_path: Path) -> None:
    chunks = [
        _chunk("nsclc_diag", "nsclc", "diagnosis", "NSCLC diagnostic recommendation.")
    ]
    result = extract_memory_slots(chunks, tmp_path / "memory_store", memory_tokens=3, mock=True)
    _mark_mock_store_as_trained(Path(result["memory_dir"]))
    decoder = _StrictRepairingLatentDecoder()
    planner = LatentGuidelinePlanner(
        memory_dir=tmp_path / "memory_store",
        decoder=decoder,
        max_new_tokens=512,
    )

    output = planner.next_step(_patient_state_v2(), trajectory_plan=[])

    assert output["actions"][0]["objective"] == "Run guideline-aligned evidence collection."
    assert planner.last_attempt["status"] == "strict_repaired"
    assert "strict_planner_json_repair" in planner.last_attempt["strict_repair_prompt_preview"]
    assert decoder.generate_calls == 3
    assert decoder.max_new_tokens_seen[-1] == 1536


def test_planner_error_carries_decoder_debug_details(tmp_path: Path) -> None:
    chunks = [
        _chunk("nsclc_diag", "nsclc", "diagnosis", "NSCLC diagnostic recommendation.")
    ]
    result = extract_memory_slots(chunks, tmp_path / "memory_store", memory_tokens=3, mock=True)
    _mark_mock_store_as_trained(Path(result["memory_dir"]))
    planner = LatentGuidelinePlanner(
        memory_dir=tmp_path / "memory_store",
        decoder=_AlwaysBadLatentDecoder(),
    )

    try:
        planner.next_step(_patient_state_v2(), trajectory_plan=[])
    except LatentPlannerError as exc:
        assert "after two format-repair retries" in str(exc)
        assert exc.details["status"] == "failed"
        assert exc.details["raw_text"] == "not json"
        assert exc.details["repair_raw_text"] == "still not json"
        assert exc.details["strict_repair_raw_text"] == "still not json"
        assert exc.details["retrieved_memories"][0]["guideline_memory_id"] == "nsclc_diag"
    else:
        raise AssertionError("bad decoder should fail with debug details")


def test_planner_v2_rejects_v1_alias_output() -> None:
    payload = {
        "blocked_pathways": [],
        "current_phase": "diagnosis_phase",
        "guideline_memory": "nsclc_diag",
        "missing_evidence": ["pathology", "radiology"],
        "next_step": "Collect missing evidence.",
        "reason": "The diagnostic evidence is incomplete.",
        "required_skills": ["pathology.read_report"],
        "supporting_rule_ids": ["r1"],
    }

    with pytest.raises(LatentPlannerError, match="planner_action.v2"):
        _normalize_planner_output(payload)


def test_planner_runtime_normalizer_does_not_apply_v2_validator() -> None:
    payload = _planner_action_v2()
    payload["current_phase"] = "non_taxonomy_runtime_phase"
    payload["actions"][0]["provenance"][0]["memory_id"] = "inactive_memory"
    output = _normalize_planner_output(payload)

    assert output["schema_version"] == "planner_action.v2"
    assert output["current_phase"] == "non_taxonomy_runtime_phase"
    assert output["actions"][0]["provenance"][0]["memory_id"] == "inactive_memory"


def test_planner_v2_rejects_incomplete_v1_output() -> None:
    payload = {
        "blocked_pathways": [],
        "current_phase": "diagnosis_phase",
        "guideline_memory": "nsclc_diag",
        "missing_evidence": ["pathology"],
        "next_step": "Collect missing evidence.",
        "required_skills": [],
        "supporting_rule_ids": [],
    }

    with pytest.raises(LatentPlannerError, match="planner_action.v2"):
        _normalize_planner_output(payload)


def test_planner_query_expands_lung_adenocarcinoma_terms() -> None:
    query = build_planner_query(
        {
            "cancer_type": "LUNG",
            "known_diagnosis": "Adenocarcinoma, NOS",
            "current_phase": "diagnosis_phase",
        }
    )

    assert "NSCLC" in query
    assert "non-small cell lung cancer" in query
    assert "肺腺癌" in query


def test_routing_prompt_memories_expose_only_active_memories() -> None:
    def memory(memory_id: str):
        return SimpleNamespace(
            memory_id=memory_id,
            guideline_name="guide",
            guideline_version="2025",
            cancer_type="nsclc",
            section_title=memory_id,
            section_path=(memory_id,),
            language="zh",
            source_rule_ids=(f"{memory_id}:r1",),
            source_chunk_ids=(memory_id,),
            page_start=1,
            page_end=2,
            metadata={"guideline_id": "guide"},
        )

    def candidate(memory_id: str):
        return SimpleNamespace(
            memory_id=memory_id,
            memory=memory(memory_id),
            seed_score=0.5,
            combined_score=0.5,
        )

    active = candidate("active")
    inactive_seed = candidate("inactive_seed")
    routing = SimpleNamespace(
        active_memories=[SimpleNamespace(memory_id="active", weight=1.0)],
        merged_candidates=[active, inactive_seed],
    )

    memories = _routing_memories_for_prompt(routing)

    assert [item["guideline_memory_id"] for item in memories] == ["active"]
    assert "inactive_seed" not in json.dumps(memories)


def test_auto_generation_limit_uses_remaining_context() -> None:
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096))
    tokenizer = SimpleNamespace(model_max_length=8192)

    assert _resolve_generation_max_new_tokens(model, tokenizer, 1500, "auto") == 2596
    assert _resolve_generation_max_new_tokens(model, tokenizer, 1500, None) == 2596
    assert _resolve_generation_max_new_tokens(model, tokenizer, 1500, 0) == 2596
    assert _resolve_generation_max_new_tokens(model, tokenizer, 1500, 768) == 768


def test_latent_generation_preserves_first_token_eos_for_auditing() -> None:
    torch = pytest.importorskip("torch")

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 0
        model_max_length = 4096

        def __call__(self, text, **kwargs):
            length = max(min(len(text), 32), 1)
            return Batch(input_ids=torch.ones((1, length), dtype=torch.long))

        def decode(self, token_ids, *, skip_special_tokens):
            values = [int(value) for value in token_ids]
            if values == [0]:
                return "" if skip_special_tokens else "<eos>"
            return "unexpected"

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.config = SimpleNamespace(max_position_embeddings=4096)
            self.generate_kwargs = []

        def get_input_embeddings(self):
            return self.embedding

        def generate(self, **kwargs):
            self.generate_kwargs.append(kwargs)
            return torch.tensor([[0]], dtype=torch.long)

    model = Model()
    bundle = PlannerModelBundle(
        torch=torch,
        tokenizer=Tokenizer(),
        model=model,
        device="cpu",
        hidden_size=4,
        model_name="fake",
    )
    diagnostics = {}

    output = generate_with_memory_slots(
        bundle,
        np.zeros((3, 4), dtype="float32"),
        "[PLAN]\n{}",
        max_new_tokens=128,
        generation_diagnostics=diagnostics,
    )

    assert output == ""
    assert diagnostics["initial_generation"]["first_token_is_eos"] is True
    assert len(model.generate_kwargs) == 1


def test_causal_target_reserves_final_token_for_eos() -> None:
    class TargetTokenizer:
        eos_token_id = 99

        def encode(self, text: str, **kwargs) -> list[int]:
            assert text == "planner target"
            assert kwargs["add_special_tokens"] is False
            assert kwargs["truncation"] is True
            assert kwargs["max_length"] == 3
            return [10, 11, 12, 13]

    tokenizer = TargetTokenizer()

    assert _encode_target_with_eos(
        tokenizer,
        "planner target",
        max_length=4,
    ) == [10, 11, 12, 99]
    assert _encode_target_with_eos(tokenizer, "", max_length=1) == [99]


def test_planner_target_is_never_silently_truncated() -> None:
    class Batch(dict):
        def to(self, device):
            return self

    class FakeTensor:
        def __init__(self, length: int) -> None:
            self.shape = (1, length)

    class TargetTokenizer:
        eos_token_id = 99

        def encode(self, text: str, **kwargs) -> list[int]:
            assert kwargs["add_special_tokens"] is False
            assert kwargs["truncation"] is False
            return list(range(len(text)))

        def __call__(self, text: str, **kwargs):
            length = min(len(text), kwargs["max_length"])
            return Batch(input_ids=FakeTensor(length))

    tokenizer = TargetTokenizer()
    prompt, target, audit = encode_prompt_and_full_target(
        tokenizer,
        "p" * 20,
        "t" * 8,
        max_decoder_tokens=16,
    )
    assert prompt["input_ids"].shape[1] == 7
    assert target == [*range(8), 99]
    assert audit == {
        "max_decoder_tokens": 16,
        "target_tokens_with_eos": 9,
        "prompt_token_budget": 7,
        "prompt_tokens_original": 20,
        "prompt_tokens_used": 7,
        "prompt_truncated": True,
    }
    with pytest.raises(ValueError, match="will not be silently truncated"):
        encode_prompt_and_full_target(
            tokenizer,
            "prompt",
            "t" * 16,
            max_decoder_tokens=16,
        )


def test_non_reentrant_gradient_checkpointing_is_explicit() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.kwargs = None
            self.inputs_enabled = False

        def gradient_checkpointing_enable(self, **kwargs) -> None:
            self.kwargs = kwargs

        def enable_input_require_grads(self) -> None:
            self.inputs_enabled = True

    model = FakeModel()
    result = enable_non_reentrant_gradient_checkpointing(model)

    assert model.kwargs == {
        "gradient_checkpointing_kwargs": {"use_reentrant": False}
    }
    assert model.inputs_enabled is True
    assert result == {
        "enabled": True,
        "use_reentrant": False,
        "input_require_grads_enabled": True,
    }


def test_memory_metadata_summary_limits_rule_ids() -> None:
    summary = _memory_metadata_summary(
        {
            "guideline_memory_id": "memory_1",
            "metadata": {
                "source_rule_ids": [f"rule_{index}" for index in range(10)],
                "source_span_ids": [f"span_{index}" for index in range(8)],
            },
        }
    )

    assert summary["source_rule_ids"] == [f"rule_{index}" for index in range(5)]
    assert summary["source_rule_id_count"] == 10
    assert summary["source_span_ids"] == [f"span_{index}" for index in range(5)]
    assert summary["source_span_id_count"] == 8


def test_special_token_setup_resizes_to_saved_training_tokenizer_length() -> None:
    tokenizer = _FakeTokenizer(vocab_size=248082, missing_special_tokens=False)
    model = _FakeModel(vocab_size=248320)

    add_planner_special_tokens(tokenizer, model)

    assert tokenizer.added_tokens == []
    assert model.resized_to == 248082


def test_special_token_setup_resizes_after_adding_missing_tokens() -> None:
    tokenizer = _FakeTokenizer(vocab_size=100, missing_special_tokens=True)
    model = _FakeModel(vocab_size=100)

    add_planner_special_tokens(tokenizer, model)

    assert tokenizer.added_tokens
    assert model.resized_to == len(tokenizer)


def test_projection_heads_align_to_model_embedding_dtype_and_device() -> None:
    model = _FakeModel(vocab_size=128, dtype="bf16")
    model.guideline_query_projection = _FakeProjectionHead()
    model.guideline_slot_projection = _FakeProjectionHead()

    _align_projection_heads_to_model(model, _FakeTorch(), "cuda:3")

    assert model.guideline_query_projection.seen_device == "cuda:3"
    assert model.guideline_query_projection.seen_dtype == "bf16"
    assert model.guideline_slot_projection.seen_device == "cuda:3"
    assert model.guideline_slot_projection.seen_dtype == "bf16"


def test_deepspeed_plugin_patch_sets_micro_batch_size(tmp_path: Path) -> None:
    class DummyPlugin:
        def __init__(self) -> None:
            self.deepspeed_config = {}

    plugin = DummyPlugin()
    config = PlannerTrainingConfig(
        train_data_path=str(tmp_path / "train.jsonl"),
        output_dir=str(tmp_path / "out"),
        train_micro_batch_size_per_gpu=2,
        gradient_accumulation_steps=3,
    )

    _patch_deepspeed_plugin_config(plugin, config)

    assert plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == 2
    assert plugin.deepspeed_config["gradient_accumulation_steps"] == 3


def test_training_validation_split_keeps_task_mix() -> None:
    records = [
        {"task": task, "id": f"{task}_{index}"}
        for task in ("AE", "RETRIEVE", "CONTINUE", "PLAN")
        for index in range(10)
    ]

    train_records, validation_records = _split_train_validation_records(records, 0.1)

    assert len(train_records) == 36
    assert len(validation_records) == 4
    assert {record["task"] for record in validation_records} == {
        "AE",
        "RETRIEVE",
        "CONTINUE",
        "PLAN",
    }
    assert {record["id"] for record in train_records}.isdisjoint(
        {record["id"] for record in validation_records}
    )


def _chunk(memory_id: str, cancer_type: str, title: str, text: str) -> dict[str, object]:
    return {
        "guideline_id": f"guide_{cancer_type}",
        "version": "2025",
        "cancer_type": cancer_type,
        "chapter": title,
        "section": title,
        "h1_title": title,
        "source_span_id": memory_id,
        "source_rule_ids": [f"{memory_id}:rule:001"],
        "source_span_ids": [memory_id],
        "page_start": 1,
        "page_end": 1,
        "text": text,
    }


def _patient_state_v2(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "schema_version": "patient_state.v2",
        "case_id": "case-1",
        "cancer_family": "lung",
        "disease_subtype": "nsclc",
        "current_phase": "diagnostic_workup",
        "decision_date": "2025-01-01",
        "guideline_context": {
            "decision_date": "2025-01-01",
            "guidelines": [{"guideline_id": "guide_nsclc", "version": "2025"}],
        },
        "known_diagnosis": None,
        "known_stage": None,
        "known_biomarkers": {},
        "risk_stratification": {},
        "available_modalities": ["clinical"],
        "completed_skills": [],
        "completed_actions": [],
        "treatment_history": [],
        "current_treatment_line": 0,
        "evidence_ledger": [],
        "last_transition": None,
        "pending_actions": [],
        "blocked_actions": [],
        "unresolved_information": ["pathology"],
    }
    state.update(overrides)
    return state


def _planner_action_v2(
    *,
    objective: str = "调用病理和影像工具补齐诊断证据。",
    required_skills: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "planner_action.v2",
        "current_phase": "diagnostic_workup",
        "proposed_phase": None,
        "missing_information": ["pathology"],
        "actions": [
            {
                "objective": objective,
                "action_type": "evidence_gathering",
                "required_skills": required_skills or ["pathology.read_report"],
                "preconditions": [],
                "expected_state_delta": ["pathology_summary"],
                "provenance": [
                    {
                        "memory_id": "nsclc_diag",
                        "rule_ids": ["nsclc_diag:rule:001"],
                        "source_spans": ["nsclc_diag"],
                        "guideline_id": "guide_nsclc",
                        "version": "2025",
                    }
                ],
            }
        ],
        "blocked_actions": [],
        "should_stop": False,
        "reason": "根据 active guideline memory 规划下一步。",
    }


class _FakeLatentDecoder:
    hidden_size = 128

    def __init__(self) -> None:
        self.seen_slots: np.ndarray | None = None
        self.seen_prompt = ""
        self.seen_max_new_tokens = 0
        self.last_generation_info: dict[str, object] = {}

    def encode_query(self, query: str, *, embedding_dim: int | None = None) -> np.ndarray:
        dim = int(embedding_dim or self.hidden_size)
        vector = np.ones(dim, dtype="float32")
        vector /= np.linalg.norm(vector)
        return vector

    def score(self, query_embedding: np.ndarray, slot_embedding: np.ndarray) -> float:
        return float(np.dot(query_embedding, slot_embedding))

    def count_text_tokens(self, text: str) -> int:
        return len(text)

    def generate_plan(
        self,
        memory_slots: np.ndarray,
        prompt: str,
        *,
        max_new_tokens: int = 512,
    ) -> str:
        self.seen_slots = np.asarray(memory_slots)
        self.seen_prompt = prompt
        self.seen_max_new_tokens = max_new_tokens
        self.last_generation_info = {
            "prompt_token_count": 42,
            "initial_generation": {"first_token_is_eos": False},
        }
        return json.dumps(
            _planner_action_v2(),
            ensure_ascii=False,
        )


class _RepairingLatentDecoder(_FakeLatentDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.generate_calls = 0

    def generate_plan(
        self,
        memory_slots: np.ndarray,
        prompt: str,
        *,
        max_new_tokens: int = 512,
    ) -> str:
        self.generate_calls += 1
        self.seen_slots = np.asarray(memory_slots)
        self.seen_prompt = prompt
        self.seen_max_new_tokens = max_new_tokens
        if self.generate_calls == 1:
            return "not json"
        return json.dumps(
            _planner_action_v2(objective="Collect missing pathology evidence."),
            ensure_ascii=False,
        )


class _StrictRepairingLatentDecoder(_FakeLatentDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.generate_calls = 0
        self.max_new_tokens_seen: list[int] = []

    def generate_plan(
        self,
        memory_slots: np.ndarray,
        prompt: str,
        *,
        max_new_tokens: int = 512,
    ) -> str:
        self.generate_calls += 1
        self.max_new_tokens_seen.append(max_new_tokens)
        if self.generate_calls < 3:
            return "not json"
        return json.dumps(
            _planner_action_v2(
                objective="Run guideline-aligned evidence collection.",
                required_skills=["guideline.retrieve"],
            ),
            ensure_ascii=False,
        )


class _AlwaysBadLatentDecoder(_FakeLatentDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.generate_calls = 0

    def generate_plan(
        self,
        memory_slots: np.ndarray,
        prompt: str,
        *,
        max_new_tokens: int = 512,
    ) -> str:
        self.generate_calls += 1
        return "not json" if self.generate_calls == 1 else "still not json"


class _LongFreeTextLatentDecoder(_FakeLatentDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.response = "Detailed planner guidance. " * 30

    def generate_plan(
        self,
        memory_slots: np.ndarray,
        prompt: str,
        *,
        max_new_tokens: int = 512,
    ) -> str:
        return self.response


class _FakeTorch:
    pass


class _FakeProjectionHead:
    def __init__(self) -> None:
        self.seen_device: str | None = None
        self.seen_dtype: str | None = None

    def to(self, *, device=None, dtype=None) -> None:
        self.seen_device = device
        self.seen_dtype = dtype


class _FakeWeight:
    def __init__(self, rows: int, dtype: str = "float32") -> None:
        self.shape = (rows, 16)
        self.dtype = dtype


class _FakeEmbeddings:
    def __init__(self, rows: int, dtype: str = "float32") -> None:
        self.weight = _FakeWeight(rows, dtype=dtype)


class _FakeModel:
    def __init__(self, vocab_size: int, dtype: str = "float32") -> None:
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.resized_to: int | None = None

    def get_input_embeddings(self) -> _FakeEmbeddings:
        return _FakeEmbeddings(self.vocab_size, dtype=self.dtype)

    def resize_token_embeddings(self, size: int) -> None:
        self.vocab_size = int(size)
        self.resized_to = int(size)


class _FakeTokenizer:
    def __init__(self, vocab_size: int, *, missing_special_tokens: bool) -> None:
        self.vocab_size = vocab_size
        self.missing_special_tokens = missing_special_tokens
        self.added_tokens: list[str] = []

    def get_vocab(self) -> dict[str, int]:
        vocab = {f"tok_{index}": index for index in range(self.vocab_size)}
        if not self.missing_special_tokens:
            from guideline_planner.constants import SPECIAL_TOKENS

            for offset, token in enumerate(SPECIAL_TOKENS):
                vocab[token] = self.vocab_size - len(SPECIAL_TOKENS) + offset
        return vocab

    def add_special_tokens(self, payload: dict[str, list[str]]) -> None:
        tokens = list(payload.get("additional_special_tokens") or [])
        self.added_tokens.extend(tokens)
        self.vocab_size += len(tokens)

    def __len__(self) -> int:
        return self.vocab_size


def _mark_mock_store_as_trained(memory_dir: Path) -> None:
    meta_path = memory_dir / "memory_store_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "trained": True,
            "mock": False,
            "base_model": "fake-qwen",
            "base_model_revision": "fake-revision",
            "memory_encoder_adapter_path": "fake-adapter",
            "memory_encoder_adapter_hash": "fake-adapter-hash",
            "tokenizer_path": "fake-tokenizer",
            "tokenizer_hash": "fake-tokenizer-hash",
            "retrieval_projection_path": "fake-retrieval-projection.pt",
            "retrieval_projection_hash": "fake-projection-hash",
            "memory_store_fingerprint": "fake-store-fingerprint",
            "slot_hidden_size": 128,
        }
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
