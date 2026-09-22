"""Shared memory-as-expert proposal generation."""

from __future__ import annotations

from typing import Any

import torch

from guideline_planner.routing_types import ExpertOutput


class MemoryExpert(torch.nn.Module):
    """Generate latent proposals from one active guideline chapter."""

    def __init__(
        self,
        state_dim: int,
        hidden_size: int,
        memory_key_dim: int,
        *,
        num_proposal_tokens: int = 4,
        num_attention_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.memory_key_dim = int(memory_key_dim)
        self.num_proposal_tokens = int(num_proposal_tokens)
        heads = num_attention_heads or _attention_heads(self.hidden_size)
        self.proposal_queries = torch.nn.Parameter(
            torch.empty(self.num_proposal_tokens, self.hidden_size)
        )
        torch.nn.init.normal_(self.proposal_queries, mean=0.0, std=0.02)
        self.state_projection = torch.nn.Linear(self.state_dim, self.hidden_size)
        self.key_projection = torch.nn.Linear(self.memory_key_dim, self.hidden_size)
        self.cross_attention = torch.nn.MultiheadAttention(
            self.hidden_size,
            heads,
            batch_first=True,
        )
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size)
        self.confidence_head = torch.nn.Linear(self.hidden_size, 1)
        self.progress_head = torch.nn.Linear(self.hidden_size, 1)

    def forward(
        self,
        state_repr: torch.Tensor,
        memory_slots: torch.Tensor,
        memory_key: torch.Tensor,
        *,
        memory_id: str = "",
    ) -> ExpertOutput:
        device = self.proposal_queries.device
        dtype = self.proposal_queries.dtype
        state = state_repr.to(device=device, dtype=dtype)
        slots = memory_slots.to(device=device, dtype=dtype)
        key = memory_key.to(device=device, dtype=dtype)
        if state.ndim != 1 or int(state.shape[0]) != self.state_dim:
            raise ValueError(f"Expected state [{self.state_dim}], got {tuple(state.shape)}")
        if slots.ndim != 2 or int(slots.shape[-1]) != self.hidden_size:
            raise ValueError(
                f"Expected memory slots [M,{self.hidden_size}], got {tuple(slots.shape)}"
            )
        if key.ndim != 1 or int(key.shape[0]) != self.memory_key_dim:
            raise ValueError(
                f"Expected memory key [{self.memory_key_dim}], got {tuple(key.shape)}"
            )
        state_token = self.state_projection(state)
        key_token = self.key_projection(key)
        queries = self.proposal_queries + state_token.unsqueeze(0) + key_token.unsqueeze(0)
        attended, _ = self.cross_attention(
            queries.unsqueeze(0),
            slots.unsqueeze(0),
            slots.unsqueeze(0),
            need_weights=False,
        )
        proposal_tokens = self.layer_norm(attended.squeeze(0) + queries)
        pooled = proposal_tokens.mean(dim=0)
        confidence = torch.sigmoid(self.confidence_head(pooled)).squeeze(-1)
        progress_score = torch.sigmoid(self.progress_head(pooled)).squeeze(-1)
        return ExpertOutput(
            memory_id=memory_id,
            proposal_tokens=proposal_tokens,
            confidence=confidence,
            progress_score=progress_score,
            memory_key=key,
        )


def _attention_heads(hidden_size: int, maximum: int = 8) -> int:
    for heads in range(min(maximum, hidden_size), 0, -1):
        if hidden_size % heads == 0:
            return heads
    return 1
