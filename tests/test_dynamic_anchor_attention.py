from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from guideline_planner.dynamic_anchor_attention import (
    DynamicAnchorAttentionResampler,
)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_daa_keeps_fixed_slot_length_for_variable_k(count: int) -> None:
    module = _module()
    memories = torch.randn(2, count, 3, 8)
    gates = torch.rand(2, count) + 0.1
    result = module(memories, torch.randn(2, 4), gates)

    assert result["resampled_slots"].shape == (2, 3, 8)
    assert result["memory_attention_mass"].shape == (2, count)
    assert torch.allclose(
        result["memory_attention_mass"].sum(dim=-1),
        torch.ones(2),
        atol=1e-5,
    )


def test_daa_selects_anchor_per_batch_and_masks_padding() -> None:
    module = _module(max_num_memories=4)
    memories = torch.randn(2, 4, 3, 8)
    gates = torch.tensor([[0.1, 0.7, 0.2, 99.0], [0.6, 0.1, 0.3, 99.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)
    result = module(memories, torch.randn(2, 4), gates, mask)

    assert result["anchor_indices"].tolist() == [1, 0]
    assert torch.equal(result["effective_memory_mask"], mask)
    assert torch.all(result["memory_attention_mass"][:, 3] == 0)


def test_routing_bias_controls_memory_attention_mass() -> None:
    module = _module(
        use_patient_state_condition=False,
        use_rank_embedding=False,
    )
    with torch.no_grad():
        module.cross_attention.q_proj.weight.zero_()
        module.cross_attention.q_proj.bias.zero_()
        module.cross_attention.k_proj.weight.zero_()
        module.cross_attention.k_proj.bias.zero_()
    result = module(
        torch.randn(1, 2, 3, 8),
        torch.zeros(1, 4),
        torch.tensor([[0.8, 0.2]]),
    )

    mass = result["memory_attention_mass"][0]
    assert mass[0] == pytest.approx(0.8, abs=1e-5)
    assert mass[1] == pytest.approx(0.2, abs=1e-5)


def test_daa_without_state_condition_is_state_invariant() -> None:
    module = _module(use_patient_state_condition=False)
    module.eval()
    memories = torch.randn(1, 2, 3, 8)
    gates = torch.tensor([[0.6, 0.4]])

    first = module(memories, torch.zeros(1, 4), gates)["resampled_slots"]
    second = module(memories, torch.ones(1, 4), gates)["resampled_slots"]

    assert torch.allclose(first, second)


def test_single_memory_bypass_returns_exact_anchor() -> None:
    module = _module(bypass_single_memory=True)
    memory = torch.randn(2, 1, 3, 8)
    result = module(memory, torch.randn(2, 4), torch.ones(2, 1))

    assert torch.equal(result["resampled_slots"], memory[:, 0])


@pytest.mark.parametrize(
    ("gates", "mask", "message"),
    [
        (torch.tensor([[float("nan")]]), None, "NaN"),
        (torch.tensor([[-0.1]]), None, "non-negative"),
        (torch.tensor([[1.0]]), torch.tensor([[0]], dtype=torch.bool), "masked"),
        (torch.tensor([[0.0]]), None, "sum to zero"),
    ],
)
def test_daa_rejects_invalid_gate_inputs(
    gates: torch.Tensor,
    mask: torch.Tensor | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _module()(torch.randn(1, 1, 3, 8), torch.randn(1, 4), gates, mask)


def test_memory_dropout_preserves_anchor_and_gradients() -> None:
    torch.manual_seed(3)
    module = _module(memory_dropout=0.9)
    module.train()
    memories = torch.randn(2, 4, 3, 8, requires_grad=True)
    result = module(
        memories,
        torch.randn(2, 4),
        torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]]),
    )
    result["resampled_slots"].sum().backward()

    assert result["effective_memory_mask"][0, 0]
    assert result["effective_memory_mask"][1, 2]
    assert result["effective_memory_mask"].any(dim=-1).all()
    assert memories.grad is not None
    assert torch.isfinite(memories.grad).all()
    assert module.cross_attention.q_proj.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_daa_cuda_mixed_precision(dtype: torch.dtype) -> None:
    module = _module().cuda().to(dtype=dtype)
    result = module(
        torch.randn(1, 3, 3, 8, device="cuda", dtype=dtype),
        torch.randn(1, 4, device="cuda", dtype=dtype),
        torch.tensor([[0.5, 0.3, 0.2]], device="cuda", dtype=dtype),
    )
    assert result["resampled_slots"].dtype == dtype
    assert torch.isfinite(result["resampled_slots"]).all()


def _module(**overrides: object) -> DynamicAnchorAttentionResampler:
    values = {
        "hidden_size": 8,
        "num_heads": 2,
        "num_emb": 3,
        "patient_state_dim": 4,
        "memory_key_dim": 4,
        "max_num_memories": 4,
        "ffn_multiplier": 2,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "use_patient_state_condition": True,
        "use_gate_attention_bias": True,
        "use_rank_embedding": True,
        "use_memory_identity_embedding": False,
        "memory_dropout": 0.0,
        "return_attention_weights": True,
    }
    values.update(overrides)
    return DynamicAnchorAttentionResampler(**values)
