"""Losses for training state-conditioned latent guideline routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass
class RoutingLossBreakdown:
    total: torch.Tensor
    action: torch.Tensor
    retrieval: torch.Tensor
    utility: torch.Tensor
    provenance: torch.Tensor
    gate_margin: torch.Tensor
    sparse: torch.Tensor
    identity: torch.Tensor
    distillation: torch.Tensor
    attention_entropy: torch.Tensor
    anchor: torch.Tensor
    diagnostics: dict[str, float] = field(default_factory=dict)

    def detached_metrics(self) -> dict[str, float]:
        metrics = {
            name: float(getattr(self, name).detach().float().cpu())
            for name in (
                "total",
                "action",
                "retrieval",
                "utility",
                "provenance",
                "gate_margin",
                "sparse",
                "identity",
                "distillation",
                "attention_entropy",
                "anchor",
            )
        }
        metrics.update({key: float(value) for key, value in self.diagnostics.items()})
        return metrics


def retrieval_contrastive_loss(
    query_embedding: torch.Tensor,
    slot_embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    temperature: float = 0.07,
    positive_weights: torch.Tensor | None = None,
    candidate_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-positive InfoNCE over one state and candidate memories."""

    query = F.normalize(query_embedding.reshape(1, -1), dim=-1)
    slots = F.normalize(slot_embeddings, dim=-1)
    logits = torch.matmul(slots, query.transpose(0, 1)).squeeze(-1) / max(
        float(temperature),
        1e-6,
    )
    if candidate_weights is not None:
        weights = candidate_weights.to(
            device=logits.device,
            dtype=logits.dtype,
        ).reshape(-1)
        if weights.numel() != logits.numel():
            raise ValueError("Candidate weights must align with retrieval candidates.")
        logits = logits + weights.clamp_min(1e-6).log()
    mask = positive_mask.to(device=logits.device, dtype=torch.bool)
    if mask.numel() != logits.numel() or not mask.any():
        return logits.sum() * 0.0
    log_probs = logits - torch.logsumexp(logits, dim=0)
    selected = log_probs[mask]
    if positive_weights is None:
        return -selected.mean()
    weights = positive_weights.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    if weights.numel() != logits.numel():
        raise ValueError("Positive weights must align with retrieval candidates.")
    selected_weights = weights[mask].clamp_min(0.0)
    if selected_weights.sum() <= 0:
        return -selected.mean()
    selected_weights = selected_weights / selected_weights.sum()
    return -(selected * selected_weights).sum()


def gate_utility_loss(
    gate_distribution: torch.Tensor,
    reward_full: torch.Tensor,
    rewards_without: torch.Tensor,
    *,
    utility_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distill leave-one-memory-out reward deltas into gate probabilities."""

    utility = reward_full.reshape(1) - rewards_without.reshape(-1)
    target = torch.softmax(utility / max(float(utility_temperature), 1e-6), dim=0)
    predicted = gate_distribution.reshape(-1).clamp_min(1e-12)
    predicted = predicted / predicted.sum().clamp_min(1e-12)
    if predicted.numel() != target.numel():
        raise ValueError("Gate distribution and utility targets must have equal length.")
    return F.kl_div(predicted.log(), target.detach(), reduction="batchmean"), utility


def sparse_gate_loss(gate_distribution: torch.Tensor) -> torch.Tensor:
    alpha = gate_distribution.reshape(-1)
    return torch.sum(alpha * (1.0 - alpha))


def gate_margin_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    hard_negative_mask: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    """Require the best grounded positive to beat the best hard negative."""

    values = logits.reshape(-1)
    positives = positive_mask.to(device=values.device, dtype=torch.bool).reshape(-1)
    negatives = hard_negative_mask.to(device=values.device, dtype=torch.bool).reshape(-1)
    if not positives.any() or not negatives.any():
        return values.sum() * 0.0
    return F.relu(float(margin) + values[negatives].max() - values[positives].max())


def provenance_multilabel_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    targets = positive_mask.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    values = logits.reshape(-1)
    if values.numel() != targets.numel() or not targets.any():
        return values.sum() * 0.0
    return F.binary_cross_entropy_with_logits(values, targets)


def combine_routing_losses(
    *,
    action: torch.Tensor,
    retrieval: torch.Tensor | None = None,
    utility: torch.Tensor | None = None,
    provenance: torch.Tensor | None = None,
    gate_margin: torch.Tensor | None = None,
    sparse: torch.Tensor | None = None,
    identity: torch.Tensor | None = None,
    distillation: torch.Tensor | None = None,
    attention_entropy: torch.Tensor | None = None,
    anchor: torch.Tensor | None = None,
    weights: Mapping[str, float] | None = None,
    diagnostics: Mapping[str, float] | None = None,
) -> RoutingLossBreakdown:
    values = {
        "action": action,
        "retrieval": retrieval if retrieval is not None else action.new_zeros(()),
        "utility": utility if utility is not None else action.new_zeros(()),
        "provenance": provenance if provenance is not None else action.new_zeros(()),
        "gate_margin": gate_margin if gate_margin is not None else action.new_zeros(()),
        "sparse": sparse if sparse is not None else action.new_zeros(()),
        "identity": identity if identity is not None else action.new_zeros(()),
        "distillation": (
            distillation if distillation is not None else action.new_zeros(())
        ),
        "attention_entropy": (
            attention_entropy
            if attention_entropy is not None
            else action.new_zeros(())
        ),
        "anchor": anchor if anchor is not None else action.new_zeros(()),
    }
    selected = {
        "action": 1.0,
        "retrieval": 0.3,
        "utility": 0.0,
        "provenance": 0.2,
        "gate_margin": 0.1,
        "sparse": 0.01,
        "identity": 0.0,
        "distillation": 0.0,
        "attention_entropy": 0.0,
        "anchor": 0.0,
    }
    if weights:
        selected.update({str(key): float(value) for key, value in weights.items()})
    total = sum(selected[name] * value for name, value in values.items())
    return RoutingLossBreakdown(
        total=total,
        diagnostics={str(key): float(value) for key, value in (diagnostics or {}).items()},
        **values,
    )
