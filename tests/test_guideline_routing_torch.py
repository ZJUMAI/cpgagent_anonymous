from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from guideline_planner.expert_competition import ExpertCompetitionModule
from guideline_planner.io_utils import read_jsonl
from guideline_planner.latent_memory_retriever import (
    GuidelineMemoryBank,
    LatentMemoryRetriever,
    filter_memories,
)
from guideline_planner.memory import extract_memory_slots
from guideline_planner.memory_expert import MemoryExpert
from guideline_planner.memory_fusion import MemoryFusion, slot_aligned_weighted_sum
from guideline_planner.planner import (
    LatentGuidelinePlanner,
    LatentPlannerError,
    normalize_planner_ablation,
)
from guideline_planner.routing_config import LatentGuidelineRoutingConfig
from guideline_planner.routing_pipeline import (
    LatentGuidelineRoutingPipeline,
    deterministic_random_memory_ids,
)
from guideline_planner.routing_losses import gate_margin_loss, provenance_multilabel_loss
from guideline_planner.routing_types import (
    GuidelineMemory,
    MemoryActivation,
    MemoryCandidate,
)
from guideline_planner.soft_memory_gate import SoftMemoryGate


def test_retrieval_coarse_filter_and_cosine_sorting(tmp_path: Path) -> None:
    memories = [
        _memory("lung_best", "nsclc", [1.0, 0.0, 0.0, 0.0]),
        _memory("lung_other", "sclc", [0.2, 0.8, 0.0, 0.0]),
        _memory("breast", "breast", [1.0, 0.0, 0.0, 0.0]),
    ]
    bank = GuidelineMemoryBank(memories, root=tmp_path)
    retriever = LatentMemoryRetriever(4, 4)

    results = retriever.search(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        bank,
        top_k=4,
        metadata_filter={"cancer_type": "LUNG", "language": "zh"},
    )

    assert [item.memory_id for item in results] == ["lung_best", "lung_other"]
    assert [
        item.memory_id
        for item in filter_memories(memories, {"cancer_type": "NSCLC"})
    ] == ["lung_best"]
    generic = _memory("lung_generic", "lung", [0.5, 0.5, 0.0, 0.0])
    assert {
        item.memory_id
        for item in filter_memories(
            [*memories, generic],
            {
                "cancer_type": {
                    "family": "lung",
                    "subtype": "nsclc",
                    "allow_generic": True,
                }
            },
        )
    } == {"lung_best", "lung_generic"}
    with pytest.raises(ValueError, match="coarse"):
        filter_memories(memories, {"stage": "IV"})


def test_gate_supports_single_and_batch_and_zeroes_inactive_weights() -> None:
    candidates = [
        MemoryCandidate(_memory("m1", "nsclc", [1, 0, 0, 0]), seed_score=2.0),
        MemoryCandidate(_memory("m2", "nsclc", [0, 1, 0, 0]), seed_score=1.0),
        MemoryCandidate(_memory("m3", "nsclc", [0, 0, 1, 0]), seed_score=0.0),
    ]
    gate = SoftMemoryGate(4, 4, routing_dim=8)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0])

    single = gate(state, candidates, top_k=2)
    batched = gate(
        torch.stack([state, state]),
        [candidates, candidates],
        top_k=2,
    )

    assert len(single.selected_activations) == 2
    assert sum(item.weight for item in single.activations) == pytest.approx(1.0)
    assert sum(item.weight == 0.0 for item in single.activations) == 1
    assert len(batched) == 2
    assert all(len(item.selected_activations) == 2 for item in batched)


def test_gate_micro_overfit_separates_positive_from_hard_negative() -> None:
    torch.manual_seed(17)
    candidates = [
        MemoryCandidate(_memory("positive", "nsclc", [1, 0, 0, 0])),
        MemoryCandidate(_memory("hard-negative", "nsclc", [0, 1, 0, 0])),
    ]
    gate = SoftMemoryGate(
        4,
        4,
        routing_dim=8,
        use_seed_score=False,
    )
    optimizer = torch.optim.Adam(gate.parameters(), lr=0.05)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0])
    positive_mask = torch.tensor([True, False])
    hard_negative_mask = torch.tensor([False, True])
    initial = gate(state, candidates, top_k=2).probabilities[0].item()
    for _ in range(80):
        output = gate(state, candidates, top_k=2)
        loss = provenance_multilabel_loss(output.logits, positive_mask)
        loss = loss + gate_margin_loss(
            output.logits,
            positive_mask,
            hard_negative_mask,
            margin=0.1,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final = gate(state, candidates, top_k=2)

    assert final.probabilities[0].item() > initial
    assert final.probabilities[0].item() >= 0.90
    assert (final.logits[0] - final.logits[1]).item() >= 0.10


def test_weighted_fusion_concatenates_memory_slots_without_position_sum() -> None:
    memories = {
        "m1": _memory("m1", "nsclc", [1, 0, 0, 0], slot_value=1.0),
        "m2": _memory("m2", "nsclc", [0, 1, 0, 0], slot_value=2.0),
    }
    fusion = MemoryFusion(4, 4, add_memory_identity_embedding=False)
    result = fusion(
        [MemoryActivation("m1", 0.25), MemoryActivation("m2", 0.75)],
        memories,
    )

    assert result.slots.shape == (6, 4)
    assert result.slot_slices == {"m1": (0, 3), "m2": (3, 6)}
    assert torch.allclose(result.slots[:3], torch.full((3, 4), 0.25))
    assert torch.allclose(result.slots[3:], torch.full((3, 4), 1.5))
    assert result.segment_ids.tolist() == [0, 0, 0, 1, 1, 1]


def test_slot_aligned_weighted_sum_keeps_fixed_prefix_length() -> None:
    memories = {
        "m1": _memory("m1", "nsclc", [1, 0, 0, 0], slot_value=1.0),
        "m2": _memory("m2", "nsclc", [0, 1, 0, 0], slot_value=3.0),
    }
    result = slot_aligned_weighted_sum(
        [MemoryActivation("m1", 0.25), MemoryActivation("m2", 0.75)],
        memories,
    )

    assert result.slots.shape == (3, 4)
    assert torch.allclose(result.slots, torch.full((3, 4), 2.5))
    assert result.diagnostics["strategy"] == "slot_aligned_weighted_sum"


def test_planner_a1_zero_memory_and_a2_latent_topk_keep_64_style_slots(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory_store"
    extract_memory_slots(
        [_chunk(f"m{index}", "nsclc", f"Section {index}") for index in range(1, 6)],
        memory_dir,
        memory_tokens=3,
        mock=True,
    )
    _mark_store_trained(memory_dir)
    routing = {
        "enabled": True,
        "profile": "dynamic_anchor",
        "retrieval": {"seed_top_k": 5},
        "gate": {"active_top_k": 4, "routing_dim": 16},
        "memory_fusion": {
            "dynamic_anchor_attention": {"num_heads": 8, "max_num_memories": 4}
        },
        "runtime": {"allow_untrained": True},
    }

    no_memory_decoder = _FakeRoutingDecoder()
    no_memory = LatentGuidelinePlanner(
        memory_dir=memory_dir,
        decoder=no_memory_decoder,
        routing_config=routing,
        planner_ablation="no_memory",
    )
    no_memory_result = no_memory.plan(_patient_state(), trajectory_history=[])
    assert no_memory_decoder.seen_slots.shape == (3, 128)
    assert np.count_nonzero(no_memory_decoder.seen_slots) == 0
    assert no_memory_result.active_memories == []
    assert '"guideline_memory_candidates": []' in no_memory_decoder.seen_prompt

    latent_decoder = _FakeRoutingDecoder()
    latent = LatentGuidelinePlanner(
        memory_dir=memory_dir,
        decoder=latent_decoder,
        top_k=4,
        routing_config=routing,
        planner_ablation="latent_topk",
    )
    latent_result = latent.plan(_patient_state(), trajectory_history=[])
    assert latent_decoder.seen_slots.shape == (3, 128)
    assert len(latent_result.active_memories) == 4
    assert latent_result.diagnostics["planner_ablation"] == "latent_topk"


def test_routed_a3_a4_a6_runtime_ablations_are_controlled(tmp_path: Path) -> None:
    memories = [
        _memory(
            f"m{index}",
            "nsclc" if index <= 4 else "breast",
            [1.0 if offset == (index % 4) else 0.0 for offset in range(4)],
            slot_value=float(index),
        )
        for index in range(1, 9)
    ]
    bank = GuidelineMemoryBank(memories, root=tmp_path)

    def encode_query(text: str, *, embedding_dim: int | None = None) -> np.ndarray:
        vector = np.ones(int(embedding_dim or 4), dtype="float32")
        return vector / np.linalg.norm(vector)

    config = LatentGuidelineRoutingConfig.from_value(
        {
            "enabled": True,
            "retrieval": {
                "seed_top_k": 8,
                "filter_cancer_type": False,
                "filter_guideline_id": False,
                "filter_version": False,
                "filter_language": False,
            },
            "gate": {"active_top_k": 4, "routing_dim": 8},
            "memory_fusion": {
                "dynamic_anchor_attention": {
                    "num_heads": 2,
                    "max_num_memories": 4,
                }
            },
            "runtime": {"allow_untrained": True},
        }
    )
    pipeline = LatentGuidelineRoutingPipeline(
        memory_bank=bank,
        query_encoder=encode_query,
        config=config,
        device="cpu",
    )
    pipeline.eval()
    route = pipeline(_patient_state())
    original_ids = [item.memory_id for item in route.active_memories]

    a3 = pipeline.apply_runtime_ablation(route, "router_topk_no_daa", seed=17)
    assert a3.decoder_prefix.shape == (3, 4)
    assert [item.memory_id for item in a3.active_memories] == original_ids
    assert a3.fused_memory.diagnostics["strategy"] == "slot_aligned_weighted_sum"

    a4 = pipeline.apply_runtime_ablation(route, "daa_uniform_gate", seed=17)
    assert [item.memory_id for item in a4.active_memories] == original_ids
    assert [item.weight for item in a4.active_memories] == pytest.approx([0.25] * 4)
    assert a4.decoder_prefix.shape == (3, 4)

    a6_first = pipeline.apply_runtime_ablation(route, "random_memory_global", seed=17)
    a6_second = pipeline.apply_runtime_ablation(route, "random_memory_global", seed=17)
    random_ids = [item.memory_id for item in a6_first.active_memories]
    assert random_ids == [item.memory_id for item in a6_second.active_memories]
    assert len(random_ids) == 4
    assert not set(random_ids).intersection(original_ids)
    assert [item.weight for item in a6_first.active_memories] == pytest.approx([0.25] * 4)
    assert a6_first.decoder_prefix.shape == (3, 4)


def test_random_memory_selector_and_ablation_names_are_deterministic() -> None:
    first = deterministic_random_memory_ids(
        [f"m{index}" for index in range(10)],
        excluded_ids=["m0", "m1"],
        count=4,
        seed=17,
        case_id="case-1",
        current_time=2,
    )
    second = deterministic_random_memory_ids(
        [f"m{index}" for index in reversed(range(10))],
        excluded_ids=["m1", "m0"],
        count=4,
        seed=17,
        case_id="case-1",
        current_time=2,
    )
    assert first == second
    assert not set(first[0]).intersection({"m0", "m1"})
    assert normalize_planner_ablation("router-topk-no-daa") == "router_topk_no_daa"
    with pytest.raises(ValueError, match="Unknown Planner ablation"):
        normalize_planner_ablation("pathway")


def test_memory_expert_and_competition_output_ranges() -> None:
    expert = MemoryExpert(4, 4, 4, num_proposal_tokens=4)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0])
    first = expert(state, torch.ones(3, 4), torch.tensor([1.0, 0, 0, 0]), memory_id="m1")
    second = expert(state, torch.full((3, 4), 2.0), torch.tensor([0, 1.0, 0, 0]), memory_id="m2")
    competition = ExpertCompetitionModule(4, 4, 4, expert_top_k=2)
    output = competition(
        state,
        [first, second],
        [MemoryActivation("m1", 0.6, gate_score=1.0), MemoryActivation("m2", 0.4, gate_score=0.5)],
    )

    assert first.proposal_tokens.shape == (4, 4)
    assert 0.0 <= float(first.confidence) <= 1.0
    assert 0.0 <= float(first.progress_score) <= 1.0
    assert output.aggregated_proposals.shape == (8, 4)
    assert float(output.expert_weights.sum()) == pytest.approx(1.0)


def test_full_cpu_routing_forward_and_planner_compatibility(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_store"
    extract_memory_slots(
        [
            _chunk("m1", "nsclc", "Diagnosis"),
            _chunk("m2", "nsclc", "Treatment"),
            _chunk("m3", "breast", "Breast treatment"),
        ],
        memory_dir,
        memory_tokens=3,
        mock=True,
    )
    _mark_store_trained(memory_dir)
    decoder = _FakeRoutingDecoder()
    config = {
        "enabled": True,
        "profile": "dynamic_anchor",
        "retrieval": {"seed_top_k": 4},
        "gate": {"active_top_k": 2, "routing_dim": 16},
        "runtime": {"allow_untrained": True, "mode": "test"},
    }
    planner = LatentGuidelinePlanner(
        memory_dir=memory_dir,
        decoder=decoder,
        routing_config=config,
    )

    result = planner.plan(
        _patient_state(known_diagnosis="adenocarcinoma"),
        trajectory_history=[],
    )

    assert result.action["schema_version"] == "planner_action.v2"
    assert result.action["actions"][0]["provenance"][0]["memory_id"] == "m1"
    assert len(result.active_memories) == 2
    assert decoder.seen_slots.shape == (3, 128)
    assert not hasattr(planner.routing_pipeline, "memory_expert")
    assert not hasattr(planner.routing_pipeline, "expert_competition")
    assert result.diagnostics["gate_diagnostics"]["active_count"] == 2
    assert len(result.diagnostics["candidate_activations"]) == 2

    planner.plan(
        _patient_state(),
        trajectory_history=[{"round_index": 1, "tool_skills": ["clinical.read_summary"]}],
    )
    assert '"previous_activation"' not in decoder.seen_prompt
    assert "m1" in decoder.seen_prompt or "m2" in decoder.seen_prompt


def test_weighted_and_top1_pipeline_baselines_keep_expected_lengths(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory_store"
    extract_memory_slots(
        [_chunk("m1", "nsclc", "Diagnosis"), _chunk("m2", "nsclc", "Treatment")],
        memory_dir,
        memory_tokens=3,
        mock=True,
    )
    _mark_store_trained(memory_dir)
    state = _patient_state()

    weighted_decoder = _FakeRoutingDecoder()
    weighted = LatentGuidelinePlanner(
        memory_dir=memory_dir,
        decoder=weighted_decoder,
        routing_config={
            "enabled": True,
            "profile": "latent_topk_concatenation",
            "gate": {"active_top_k": 2, "routing_dim": 16},
            "runtime": {"allow_untrained": True},
        },
    )
    weighted.plan(state, trajectory_history=[])
    assert weighted_decoder.seen_slots.shape == (6, 128)

    top1_decoder = _FakeRoutingDecoder()
    top1 = LatentGuidelinePlanner(
        memory_dir=memory_dir,
        decoder=top1_decoder,
        routing_config={
            "enabled": True,
            "profile": "latent_top1",
            "gate": {"routing_dim": 16},
            "runtime": {"allow_untrained": True},
        },
    )
    top1.plan(state, trajectory_history=[])
    assert top1_decoder.seen_slots.shape == (3, 128)


def test_formal_routing_requires_trained_checkpoint(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_store"
    extract_memory_slots([_chunk("m1", "nsclc", "Diagnosis")], memory_dir, memory_tokens=3, mock=True)
    _mark_store_trained(memory_dir)

    with pytest.raises(LatentPlannerError, match="no trained routing checkpoint"):
        LatentGuidelinePlanner(
            memory_dir=memory_dir,
            decoder=_FakeRoutingDecoder(),
            routing_config={"enabled": True},
        )


def test_routing_checkpoint_strictly_binds_memory_store(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_store"
    extract_memory_slots(
        [_chunk("m1", "nsclc", "Diagnosis"), _chunk("m2", "nsclc", "Treatment")],
        memory_dir,
        memory_tokens=3,
        mock=True,
    )
    bank = GuidelineMemoryBank.from_directory(memory_dir)
    config = LatentGuidelineRoutingConfig.from_value(
        {
            "enabled": True,
            "gate": {"routing_dim": 16},
            "runtime": {"allow_untrained": True},
        }
    )

    def encode_query(text: str, *, embedding_dim: int | None = None) -> np.ndarray:
        vector = np.ones(int(embedding_dim or 128), dtype="float32")
        return vector / np.linalg.norm(vector)

    pipeline = LatentGuidelineRoutingPipeline(
        memory_bank=bank,
        query_encoder=encode_query,
        config=config,
        device="cpu",
    )
    checkpoint = pipeline.checkpoint_payload(
        trained=True,
        created_at="2026-01-01T00:00:00Z",
    )
    checkpoint_path = tmp_path / "routing_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    legacy_checkpoint = dict(checkpoint)
    legacy_checkpoint.pop("daa_trained", None)
    legacy_path = tmp_path / "legacy_expert_checkpoint.pt"
    torch.save(legacy_checkpoint, legacy_path)
    formal_config = LatentGuidelineRoutingConfig.from_value(
        {
            "enabled": True,
            "gate": {"routing_dim": 16},
            "runtime": {"allow_untrained": False},
        }
    )
    formal = LatentGuidelineRoutingPipeline(
        memory_bank=GuidelineMemoryBank.from_directory(memory_dir),
        query_encoder=encode_query,
        config=formal_config,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="not trained with Dynamic Anchor Attention"):
        formal.load_checkpoint(legacy_path)

    restored = LatentGuidelineRoutingPipeline(
        memory_bank=GuidelineMemoryBank.from_directory(memory_dir),
        query_encoder=encode_query,
        config=config,
        device="cpu",
    )
    restored.load_checkpoint(checkpoint_path)

    mismatched_config = LatentGuidelineRoutingConfig.from_value(
        {
            "enabled": True,
            "gate": {"active_top_k": 1, "routing_dim": 16},
            "runtime": {"allow_untrained": True},
        }
    )
    mismatched = LatentGuidelineRoutingPipeline(
        memory_bank=GuidelineMemoryBank.from_directory(memory_dir),
        query_encoder=encode_query,
        config=mismatched_config,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="prefix configuration mismatch"):
        mismatched.load_checkpoint(checkpoint_path)

    metadata = read_jsonl(memory_dir / "slot_metadata.jsonl")
    embedding_path = memory_dir / str(metadata[0]["retrieval_embedding_path"])
    embedding = np.load(embedding_path)
    np.save(embedding_path, embedding + 0.01)
    changed = LatentGuidelineRoutingPipeline(
        memory_bank=GuidelineMemoryBank.from_directory(memory_dir),
        query_encoder=encode_query,
        config=config,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        changed.load_checkpoint(checkpoint_path)


def _memory(
    memory_id: str,
    cancer_type: str,
    key: list[float],
    *,
    slot_value: float = 1.0,
) -> GuidelineMemory:
    return GuidelineMemory(
        memory_id=memory_id,
        guideline_name="CSCO",
        guideline_version="2025",
        cancer_type=cancer_type,
        section_title=memory_id,
        section_path=[memory_id],
        page_start=1,
        page_end=2,
        language="zh",
        source_chunk_ids=[memory_id],
        slots=torch.full((3, 4), slot_value),
        retrieval_key=torch.tensor(key, dtype=torch.float32),
        source_rule_ids=[f"{memory_id}:rule:1"],
    )


def _chunk(memory_id: str, cancer_type: str, title: str) -> dict[str, object]:
    return {
        "guideline_id": f"guide_{cancer_type}",
        "version": "2025",
        "cancer_type": cancer_type,
        "chapter": title,
        "section": title,
        "h1_title": title,
        "source_span_id": memory_id,
        "source_rule_ids": [f"{memory_id}:rule:1"],
        "source_span_ids": [memory_id],
        "page_start": 1,
        "page_end": 2,
        "language": "zh",
        "text": f"{title} guideline content",
    }


def _mark_store_trained(memory_dir: Path) -> None:
    meta_path = memory_dir / "memory_store_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "trained": True,
            "mock": False,
            "base_model": "fake-qwen",
            "memory_encoder_adapter_path": "fake-adapter",
            "memory_encoder_adapter_hash": "fake-adapter-hash",
            "tokenizer_path": "fake-tokenizer",
            "tokenizer_hash": "fake-tokenizer-hash",
            "retrieval_projection_path": "fake-projection",
            "retrieval_projection_hash": "fake-projection-hash",
        }
    )
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


class _FakeRoutingDecoder:
    hidden_size = 128
    device = "cpu"

    def __init__(self) -> None:
        self.seen_slots: np.ndarray | None = None
        self.seen_prompt = ""

    def encode_query(self, query: str, *, embedding_dim: int | None = None) -> np.ndarray:
        vector = np.ones(int(embedding_dim or 128), dtype="float32")
        return vector / np.linalg.norm(vector)

    def generate_plan(
        self,
        memory_slots: np.ndarray,
        prompt: str,
        *,
        max_new_tokens: int | str | None = 512,
    ) -> str:
        self.seen_slots = np.asarray(memory_slots)
        self.seen_prompt = prompt
        memory_id = "m1" if '"guideline_memory_id": "m1"' in prompt else "m2"
        return json.dumps(
            {
                "schema_version": "planner_action.v2",
                "current_phase": "diagnostic_workup",
                "proposed_phase": None,
                "missing_information": ["pathology"],
                "actions": [
                    {
                        "action_id": "collect-pathology",
                        "objective": "Collect pathology evidence.",
                        "action_type": "evidence_gathering",
                        "required_skills": ["pathology.read_report"],
                        "preconditions": [],
                        "expected_state_delta": ["known_diagnosis"],
                        "provenance": [
                            {
                                "memory_id": memory_id,
                                "rule_ids": [f"{memory_id}:rule:1"],
                                "source_spans": [memory_id],
                                "guideline_id": "guide_nsclc",
                                "version": "2025",
                            }
                        ],
                    }
                ],
                "blocked_actions": [],
                "should_stop": False,
                "reason": "The state still lacks pathology evidence.",
            }
        )

    def count_text_tokens(self, text: str) -> int:
        return len(text)


def _patient_state(**overrides: object) -> dict[str, object]:
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
