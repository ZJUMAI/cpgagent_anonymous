"""Weighted concatenation of independently encoded guideline memories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from guideline_planner.routing_types import (
    FusedMemory,
    GuidelineMemory,
    MemoryActivation,
)


class MemoryFusion(torch.nn.Module):
    """Fuse active chapters without aligning unrelated slot positions."""

    def __init__(
        self,
        retrieval_dim: int,
        hidden_size: int,
        *,
        add_memory_identity_embedding: bool = True,
        max_active_memories: int = 32,
    ) -> None:
        super().__init__()
        self.retrieval_dim = int(retrieval_dim)
        self.hidden_size = int(hidden_size)
        self.add_memory_identity_embedding = bool(add_memory_identity_embedding)
        self.identity_projection = torch.nn.Linear(
            self.retrieval_dim,
            self.hidden_size,
            bias=False,
        )
        if self.retrieval_dim == self.hidden_size:
            torch.nn.init.eye_(self.identity_projection.weight)
        else:
            torch.nn.init.xavier_uniform_(self.identity_projection.weight)
        self.rank_embedding = torch.nn.Embedding(
            max(int(max_active_memories), 1),
            self.hidden_size,
        )
        torch.nn.init.normal_(self.rank_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        active_memories: Sequence[MemoryActivation],
        memory_lookup: Mapping[str, GuidelineMemory],
        *,
        weights: torch.Tensor | None = None,
    ) -> FusedMemory:
        if not active_memories:
            raise ValueError("Memory fusion requires at least one active memory.")
        device = self.identity_projection.weight.device
        dtype = self.identity_projection.weight.dtype
        if weights is None:
            weights = torch.tensor(
                [item.weight for item in active_memories],
                dtype=dtype,
                device=device,
            )
        else:
            weights = weights.to(device=device, dtype=dtype)
        if weights.ndim != 1 or len(weights) != len(active_memories):
            raise ValueError("Fusion weights must contain one scalar per active memory.")
        weights = weights / weights.sum().clamp_min(1e-12)

        blocks: list[torch.Tensor] = []
        segment_blocks: list[torch.Tensor] = []
        slot_slices: dict[str, tuple[int, int]] = {}
        memory_ids: list[str] = []
        offset = 0
        for rank, activation in enumerate(active_memories):
            memory = memory_lookup.get(activation.memory_id)
            if memory is None:
                raise KeyError(f"Unknown active guideline memory: {activation.memory_id}")
            slots = memory.slots.to(device=device, dtype=dtype)
            if slots.ndim != 2 or int(slots.shape[-1]) != self.hidden_size:
                raise RuntimeError(
                    f"Memory {memory.memory_id} slot shape {tuple(slots.shape)} does "
                    f"not match hidden size {self.hidden_size}."
                )
            block = weights[rank] * slots
            if self.add_memory_identity_embedding:
                key = memory.retrieval_key.to(device=device, dtype=dtype)
                identity = self.identity_projection(key)
                rank_id = torch.tensor(
                    min(rank, self.rank_embedding.num_embeddings - 1),
                    dtype=torch.long,
                    device=device,
                )
                block = block + identity.unsqueeze(0) + self.rank_embedding(rank_id).unsqueeze(0)
            blocks.append(block)
            length = int(block.shape[0])
            slot_slices[memory.memory_id] = (offset, offset + length)
            segment_blocks.append(
                torch.full((length,), rank, dtype=torch.long, device=device)
            )
            memory_ids.append(memory.memory_id)
            offset += length
        fused = torch.cat(blocks, dim=0)
        segments = torch.cat(segment_blocks, dim=0)
        return FusedMemory(
            slots=fused,
            attention_mask=torch.ones(fused.shape[0], dtype=torch.long, device=device),
            memory_ids=memory_ids,
            slot_slices=slot_slices,
            segment_ids=segments,
            diagnostics={
                "strategy": "weighted_concatenation",
                "active_memory_count": len(memory_ids),
                "slot_count": int(fused.shape[0]),
                "hidden_size": int(fused.shape[1]),
                "memory_ids": memory_ids,
                "identity_embedding": self.add_memory_identity_embedding,
            },
        )


def fuse_active_memories(
    active_memories: Sequence[MemoryActivation],
    memory_lookup: Mapping[str, GuidelineMemory],
    *,
    weights: torch.Tensor | None = None,
    fusion_module: MemoryFusion | None = None,
) -> FusedMemory:
    """Functional entrypoint used by tests and simple integrations."""

    if not active_memories:
        raise ValueError("Memory fusion requires at least one active memory.")
    first = memory_lookup[active_memories[0].memory_id]
    module = fusion_module or MemoryFusion(
        int(first.retrieval_key.shape[-1]),
        int(first.slots.shape[-1]),
    )
    return module(active_memories, memory_lookup, weights=weights)


def slot_aligned_weighted_sum(
    active_memories: Sequence[MemoryActivation],
    memory_lookup: Mapping[str, GuidelineMemory],
    *,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> FusedMemory:
    """Fuse equal-shape memories without increasing the decoder prefix length.

    This is the controlled Router-without-DAA ablation: Router selection and
    weights remain intact, while the learned DAA resampler is replaced by a
    deterministic slot-aligned weighted sum.
    """

    if not active_memories:
        raise ValueError("Slot-aligned fusion requires at least one active memory.")
    blocks: list[torch.Tensor] = []
    expected_shape: tuple[int, int] | None = None
    for activation in active_memories:
        memory = memory_lookup.get(activation.memory_id)
        if memory is None:
            raise KeyError(f"Unknown active guideline memory: {activation.memory_id}")
        slots = memory.slots.to(device=device, dtype=dtype)
        if slots.ndim != 2:
            raise RuntimeError(
                f"Memory {memory.memory_id} must be rank-2, got {tuple(slots.shape)}."
            )
        shape = (int(slots.shape[0]), int(slots.shape[1]))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise RuntimeError(
                "Slot-aligned fusion requires identical memory shapes: "
                f"expected {expected_shape}, got {shape} for {memory.memory_id}."
            )
        blocks.append(slots)
    if weights is None:
        weights = torch.tensor(
            [item.weight for item in active_memories],
            dtype=dtype,
            device=device,
        )
    else:
        weights = weights.to(device=device, dtype=dtype)
    if weights.shape != (len(active_memories),):
        raise ValueError("Slot-aligned weights must align with active memories.")
    weights = weights / weights.sum().clamp_min(1e-12)
    stacked = torch.stack(blocks, dim=0)
    slots = (stacked * weights.reshape(-1, 1, 1)).sum(dim=0)
    memory_ids = [item.memory_id for item in active_memories]
    return FusedMemory(
        slots=slots,
        attention_mask=torch.ones(slots.shape[0], dtype=torch.long, device=device),
        memory_ids=memory_ids,
        slot_slices={},
        segment_ids=torch.full(
            (slots.shape[0],), -1, dtype=torch.long, device=device
        ),
        diagnostics={
            "strategy": "slot_aligned_weighted_sum",
            "active_memory_count": len(memory_ids),
            "memory_ids": memory_ids,
            "normalized_weights": [
                float(item.detach().float().cpu()) for item in weights
            ],
            "slot_count": int(slots.shape[0]),
            "hidden_size": int(slots.shape[1]),
        },
    )


@dataclass
class PackedActiveMemories:
    """Padded active-memory tensors consumed by fixed-length resamplers."""

    memories: torch.Tensor  # [B,K,N,H]
    gate_weights: torch.Tensor  # [B,K]
    memory_mask: torch.Tensor  # [B,K]
    memory_ids: list[list[str]]
    memory_keys: torch.Tensor  # [B,K,retrieval_dim]


def pack_active_memories(
    activation_batches: Sequence[Sequence[MemoryActivation]],
    memory_lookup: Mapping[str, GuidelineMemory],
    *,
    weight_batches: Sequence[torch.Tensor] | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PackedActiveMemories:
    """Pack variable-K activation lists without duplicating padding memories."""

    batches = [list(items) for items in activation_batches]
    if not batches or any(not items for items in batches):
        raise ValueError("Every active-memory batch item must contain at least one memory.")
    max_count = max(len(items) for items in batches)
    first = memory_lookup[batches[0][0].memory_id]
    num_emb, hidden_size = (int(value) for value in first.slots.shape)
    retrieval_dim = int(first.retrieval_key.shape[-1])
    memories = torch.zeros(
        (len(batches), max_count, num_emb, hidden_size),
        dtype=dtype,
        device=device,
    )
    keys = torch.zeros(
        (len(batches), max_count, retrieval_dim),
        dtype=dtype,
        device=device,
    )
    gates = torch.zeros((len(batches), max_count), dtype=dtype, device=device)
    mask = torch.zeros((len(batches), max_count), dtype=torch.bool, device=device)
    memory_ids: list[list[str]] = []
    if weight_batches is not None and len(weight_batches) != len(batches):
        raise ValueError("weight_batches must align with activation_batches.")
    for batch_index, activations in enumerate(batches):
        ids: list[str] = []
        weights = (
            weight_batches[batch_index].to(device=device, dtype=dtype)
            if weight_batches is not None
            else torch.tensor(
                [item.weight for item in activations],
                device=device,
                dtype=dtype,
            )
        )
        if weights.shape != (len(activations),):
            raise ValueError(
                "Each weight batch must contain one value per active memory, "
                f"got {tuple(weights.shape)} for K={len(activations)}."
            )
        for memory_index, activation in enumerate(activations):
            memory = memory_lookup.get(activation.memory_id)
            if memory is None:
                raise KeyError(f"Unknown active guideline memory: {activation.memory_id}")
            if tuple(memory.slots.shape) != (num_emb, hidden_size):
                raise RuntimeError(
                    f"Memory {memory.memory_id} has slot shape {tuple(memory.slots.shape)}; "
                    f"expected {(num_emb, hidden_size)}."
                )
            if tuple(memory.retrieval_key.shape) != (retrieval_dim,):
                raise RuntimeError(
                    f"Memory {memory.memory_id} has retrieval key shape "
                    f"{tuple(memory.retrieval_key.shape)}; expected {(retrieval_dim,)}."
                )
            memories[batch_index, memory_index] = memory.slots.to(device=device, dtype=dtype)
            keys[batch_index, memory_index] = memory.retrieval_key.to(
                device=device,
                dtype=dtype,
            )
            gates[batch_index, memory_index] = weights[memory_index]
            mask[batch_index, memory_index] = True
            ids.append(memory.memory_id)
        memory_ids.append(ids)
    return PackedActiveMemories(
        memories=memories,
        gate_weights=gates,
        memory_mask=mask,
        memory_ids=memory_ids,
        memory_keys=keys,
    )


def top1_memory(
    active_memories: Sequence[MemoryActivation],
    memory_lookup: Mapping[str, GuidelineMemory],
    *,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> FusedMemory:
    """Return the highest-weight original memory without changing slot length."""

    if not active_memories:
        raise ValueError("Top-1 fusion requires at least one active memory.")
    if weights is None:
        index = max(range(len(active_memories)), key=lambda i: active_memories[i].weight)
    else:
        if weights.shape != (len(active_memories),):
            raise ValueError("Top-1 weights must align with active memories.")
        index = int(weights.argmax().item())
    memory = memory_lookup[active_memories[index].memory_id]
    slots = memory.slots.to(device=device, dtype=dtype)
    return FusedMemory(
        slots=slots,
        attention_mask=torch.ones(slots.shape[0], dtype=torch.long, device=device),
        memory_ids=[memory.memory_id],
        slot_slices={memory.memory_id: (0, int(slots.shape[0]))},
        segment_ids=torch.zeros(slots.shape[0], dtype=torch.long, device=device),
        diagnostics={
            "strategy": "top1_only",
            "active_memory_count": len(active_memories),
            "selected_memory_id": memory.memory_id,
            "slot_count": int(slots.shape[0]),
            "hidden_size": int(slots.shape[1]),
        },
    )
