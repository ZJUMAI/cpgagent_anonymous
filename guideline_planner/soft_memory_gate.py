"""Sparse state-conditioned gate for latent guideline memories."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from guideline_planner.routing_types import (
    GateOutput,
    MemoryActivation,
    MemoryCandidate,
)


class SoftMemoryGate(torch.nn.Module):
    """Select a differentiable Top-K mixture over variable memory candidates."""

    def __init__(
        self,
        state_dim: int,
        memory_key_dim: int,
        *,
        routing_dim: int = 256,
        temperature: float = 1.0,
        use_seed_score: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.memory_key_dim = int(memory_key_dim)
        self.routing_dim = int(routing_dim)
        self.temperature = float(temperature)
        self.use_seed_score = bool(use_seed_score)
        self.state_projection = torch.nn.Linear(self.state_dim, self.routing_dim)
        self.memory_projection = torch.nn.Linear(self.memory_key_dim, self.routing_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.routing_dim * 3 + 1, self.routing_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.routing_dim, 1),
        )
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        state_repr: torch.Tensor,
        candidates: Sequence[MemoryCandidate] | Sequence[Sequence[MemoryCandidate]],
        top_k: int = 2,
    ) -> GateOutput | list[GateOutput]:
        if state_repr.ndim == 1:
            return self._forward_one(
                state_repr,
                list(candidates),  # type: ignore[arg-type]
                top_k,
            )
        if state_repr.ndim != 2:
            raise ValueError(f"state_repr must be [D] or [B,D], got {tuple(state_repr.shape)}")
        candidate_batches = list(candidates)  # type: ignore[arg-type]
        if len(candidate_batches) != int(state_repr.shape[0]):
            raise ValueError("Candidate batch size must match state_repr batch size.")
        return [
            self._forward_one(state_repr[index], list(batch), top_k)
            for index, batch in enumerate(candidate_batches)
        ]

    def _forward_one(
        self,
        state_repr: torch.Tensor,
        candidates: list[MemoryCandidate],
        top_k: int,
    ) -> GateOutput:
        if not candidates:
            raise ValueError("SoftMemoryGate requires at least one memory candidate.")
        device = self.state_projection.weight.device
        dtype = self.state_projection.weight.dtype
        state = state_repr.to(device=device, dtype=dtype)
        if state.ndim != 1 or int(state.shape[0]) != self.state_dim:
            raise ValueError(
                f"Expected state shape ({self.state_dim},), got {tuple(state.shape)}."
            )
        state_feature = self.state_projection(state)
        keys = torch.stack(
            [candidate.memory.retrieval_key.to(device=device, dtype=dtype) for candidate in candidates],
            dim=0,
        )
        memory_features = self.memory_projection(keys)
        repeated_state = state_feature.unsqueeze(0).expand(len(candidates), -1)
        scalar_features = torch.tensor(
            [
                [_score(candidate.seed_score, self.use_seed_score)]
                for candidate in candidates
            ],
            dtype=dtype,
            device=device,
        )
        features = torch.cat(
            [
                repeated_state,
                memory_features,
                repeated_state * memory_features,
                scalar_features,
            ],
            dim=-1,
        )
        learned_logits = self.mlp(features).squeeze(-1)
        prior_logits = torch.zeros_like(learned_logits)
        if self.use_seed_score:
            prior_logits = prior_logits + scalar_features[:, 0]
        logits = learned_logits + prior_logits
        selected_count = min(max(int(top_k), 1), len(candidates))
        selected_logits, selected_indices_tensor = torch.topk(logits, selected_count)
        temperature = max(self.temperature, 1e-6)
        selected_weights = torch.softmax(selected_logits / temperature, dim=0)
        probabilities = torch.zeros_like(logits).scatter(
            0,
            selected_indices_tensor,
            selected_weights,
        )
        selected_indices = [int(index) for index in selected_indices_tensor.tolist()]
        selected_set = set(selected_indices)
        activations: list[MemoryActivation] = []
        for index, candidate in enumerate(candidates):
            weight = float(probabilities[index].detach().float().cpu())
            activations.append(
                MemoryActivation(
                    memory_id=candidate.memory_id,
                    weight=weight,
                    seed_score=candidate.seed_score,
                    gate_score=float(logits[index].detach().float().cpu()),
                    selected=index in selected_set,
                    section_title=candidate.memory.section_title,
                    source_pages=candidate.memory.source_pages,
                )
            )
        selected_activations = [activations[index] for index in selected_indices]
        entropy = -(
            selected_weights.clamp_min(1e-12)
            * selected_weights.clamp_min(1e-12).log()
        ).sum()
        normalized_entropy = (
            entropy / math.log(selected_count) if selected_count > 1 else entropy * 0.0
        )
        return GateOutput(
            activations=activations,
            selected_activations=selected_activations,
            logits=logits,
            probabilities=probabilities,
            selected_indices=selected_indices,
            selected_weights=selected_weights,
            diagnostics={
                "candidate_count": len(candidates),
                "active_count": selected_count,
                "selected_memory_ids": [item.memory_id for item in selected_activations],
                "weight_sum": float(selected_weights.detach().sum().cpu()),
                "entropy": float(entropy.detach().float().cpu()),
                "normalized_entropy": float(
                    normalized_entropy.detach().float().cpu()
                ),
                "temperature": temperature,
                "use_seed_score": self.use_seed_score,
            },
        )


def uniform_gate_output(
    candidates: Sequence[MemoryCandidate],
    *,
    top_k: int,
    device: str | torch.device = "cpu",
) -> GateOutput:
    """Build an equal-weight Top-K output for top-k concatenation ablations."""

    if not candidates:
        raise ValueError("Uniform gate requires at least one candidate.")
    count = min(max(int(top_k), 1), len(candidates))
    selected_indices = list(range(count))
    weights = torch.full((count,), 1.0 / count, dtype=torch.float32, device=device)
    probabilities = torch.zeros(len(candidates), dtype=torch.float32, device=device)
    probabilities[:count] = weights
    logits = torch.tensor(
        [float(candidate.combined_score or candidate.seed_score or 0.0) for candidate in candidates],
        dtype=torch.float32,
        device=device,
    )
    activations = [
        MemoryActivation(
            memory_id=candidate.memory_id,
            weight=(1.0 / count) if index < count else 0.0,
            seed_score=candidate.seed_score,
            gate_score=float(logits[index].cpu()),
            selected=index < count,
            section_title=candidate.memory.section_title,
            source_pages=candidate.memory.source_pages,
        )
        for index, candidate in enumerate(candidates)
    ]
    return GateOutput(
        activations=activations,
        selected_activations=activations[:count],
        logits=logits,
        probabilities=probabilities,
        selected_indices=selected_indices,
        selected_weights=weights,
        diagnostics={
            "candidate_count": len(candidates),
            "active_count": count,
            "selected_memory_ids": [item.memory_id for item in activations[:count]],
            "weight_sum": 1.0,
            "entropy": math.log(count) if count > 0 else 0.0,
            "mode": "uniform_top_k",
        },
    )


def _score(value: float | None, enabled: bool) -> float:
    return float(value or 0.0) if enabled else 0.0
