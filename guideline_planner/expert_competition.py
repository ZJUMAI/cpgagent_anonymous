"""Competition and optional pairwise relations among memory experts."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from guideline_planner.memory_expert import _attention_heads
from guideline_planner.routing_types import (
    CompetitionOutput,
    ExpertOutput,
    MemoryActivation,
)


class ExpertRelationHead(torch.nn.Module):
    """Optional complement/redundant/conflict classifier interface."""

    LABELS = ("complement", "redundant", "conflict")

    def __init__(self, state_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.state_projection = torch.nn.Linear(state_dim, hidden_size)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 4, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, len(self.LABELS)),
        )

    def forward(
        self,
        expert_a: torch.Tensor,
        expert_b: torch.Tensor,
        state_repr: torch.Tensor,
    ) -> torch.Tensor:
        a = expert_a.mean(dim=0)
        b = expert_b.mean(dim=0)
        state = self.state_projection(state_repr.to(self.state_projection.weight.dtype))
        return self.classifier(torch.cat([a, b, a * b, state], dim=-1))


class ExpertCompetitionModule(torch.nn.Module):
    """Combine the strongest latent proposals while preserving expert identity."""

    def __init__(
        self,
        state_dim: int,
        hidden_size: int,
        memory_key_dim: int,
        *,
        expert_top_k: int = 2,
        confidence_weight: float = 0.5,
        progress_weight: float = 0.5,
        transformer_layers: int = 1,
        relation_head_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.expert_top_k = int(expert_top_k)
        self.confidence_weight = float(confidence_weight)
        self.progress_weight = float(progress_weight)
        self.identity_projection = torch.nn.Linear(memory_key_dim, hidden_size, bias=False)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=_attention_heads(hidden_size),
            dim_feedforward=max(hidden_size * 2, 64),
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.competition_transformer = torch.nn.TransformerEncoder(
            layer,
            num_layers=max(int(transformer_layers), 1),
        )
        self.relation_head = (
            ExpertRelationHead(state_dim, hidden_size)
            if relation_head_enabled
            else None
        )

    def forward(
        self,
        state_repr: torch.Tensor,
        expert_outputs: Sequence[ExpertOutput],
        activations: Sequence[MemoryActivation],
    ) -> CompetitionOutput:
        if not expert_outputs:
            raise ValueError("Expert competition requires at least one expert output.")
        activation_by_id = {item.memory_id: item for item in activations}
        logits: list[torch.Tensor] = []
        for output in expert_outputs:
            activation = activation_by_id.get(output.memory_id)
            if activation is None:
                raise KeyError(f"Missing activation for expert {output.memory_id}")
            gate_logit = activation.gate_score
            if gate_logit is None:
                gate_logit = math.log(max(activation.weight, 1e-12))
            gate_tensor = output.confidence.new_tensor(float(gate_logit))
            logits.append(
                gate_tensor
                + self.confidence_weight * torch.log(output.confidence.clamp_min(1e-12))
                + self.progress_weight * output.progress_score
            )
        all_logits = torch.stack(logits)
        count = min(max(self.expert_top_k, 1), len(expert_outputs))
        selected_logits, selected_indices_tensor = torch.topk(all_logits, count)
        expert_weights = torch.softmax(selected_logits, dim=0)
        selected_indices = [int(index) for index in selected_indices_tensor.tolist()]
        selected = [expert_outputs[index] for index in selected_indices]
        blocks: list[torch.Tensor] = []
        for rank, output in enumerate(selected):
            block = expert_weights[rank] * output.proposal_tokens
            if output.memory_key is not None:
                identity = self.identity_projection(
                    output.memory_key.to(
                        device=self.identity_projection.weight.device,
                        dtype=self.identity_projection.weight.dtype,
                    )
                )
                block = block + identity.unsqueeze(0)
            blocks.append(block)
        concatenated = torch.cat(blocks, dim=0)
        aggregated = self.competition_transformer(concatenated.unsqueeze(0)).squeeze(0)
        pair_relations = self._pair_relations(state_repr, selected)
        return CompetitionOutput(
            aggregated_proposals=aggregated,
            expert_weights=expert_weights,
            selected_memory_ids=[output.memory_id for output in selected],
            expert_pair_relations=pair_relations,
            diagnostics={
                "expert_count": len(expert_outputs),
                "selected_expert_count": count,
                "selected_memory_ids": [output.memory_id for output in selected],
                "expert_weights": [
                    float(value) for value in expert_weights.detach().float().cpu().tolist()
                ],
                "confidence_weight": self.confidence_weight,
                "progress_weight": self.progress_weight,
                "relation_head_enabled": self.relation_head is not None,
            },
        )

    def _pair_relations(
        self,
        state_repr: torch.Tensor,
        outputs: Sequence[ExpertOutput],
    ) -> torch.Tensor | None:
        if self.relation_head is None or len(outputs) < 2:
            return None
        relations = []
        for left in range(len(outputs)):
            for right in range(left + 1, len(outputs)):
                relations.append(
                    self.relation_head(
                        outputs[left].proposal_tokens,
                        outputs[right].proposal_tokens,
                        state_repr.to(self.relation_head.state_projection.weight.device),
                    )
                )
        return torch.stack(relations, dim=0) if relations else None
