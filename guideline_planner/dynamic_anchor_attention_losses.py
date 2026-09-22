"""Training objectives for Dynamic Anchor Attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def identity_loss(
    resampled_slots: torch.Tensor,
    target_slots: torch.Tensor,
    *,
    cosine_weight: float = 0.1,
) -> torch.Tensor:
    """Keep K=1 DAA output in the original latent-memory distribution."""

    if resampled_slots.shape != target_slots.shape:
        raise ValueError(
            "Identity loss requires aligned slots, got "
            f"{tuple(resampled_slots.shape)} and {tuple(target_slots.shape)}."
        )
    mse = F.mse_loss(resampled_slots, target_slots)
    cosine = 1.0 - F.cosine_similarity(
        resampled_slots.float(),
        target_slots.float(),
        dim=-1,
    ).mean()
    return mse + float(cosine_weight) * cosine


def planner_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    token_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Token-level KL from an original-memory teacher to a DAA student."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Planner distillation logits must align, got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}."
        )
    scale = max(float(temperature), 1e-6)
    student = F.log_softmax(student_logits.float() / scale, dim=-1)
    teacher = F.softmax(teacher_logits.detach().float() / scale, dim=-1)
    loss = F.kl_div(student, teacher, reduction="none").sum(dim=-1) * (scale**2)
    if token_mask is None:
        return loss.mean()
    mask = token_mask.to(device=loss.device, dtype=loss.dtype)
    if mask.shape != loss.shape:
        raise ValueError(
            f"Distillation token mask must be {tuple(loss.shape)}, got {tuple(mask.shape)}."
        )
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def attention_entropy_regularization(
    memory_attention_mass: torch.Tensor | None,
    *,
    eps: float = 1e-8,
) -> torch.Tensor | None:
    """Return mean entropy over memory-level attention mass."""

    if memory_attention_mass is None:
        return None
    probabilities = memory_attention_mass / memory_attention_mass.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    return -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1).mean()


def anchor_preservation_loss(
    resampled_slots: torch.Tensor,
    anchor_slots: torch.Tensor,
) -> torch.Tensor:
    """Apply a low-weight cosine constraint to preserve the anchor manifold."""

    if resampled_slots.shape != anchor_slots.shape:
        raise ValueError(
            "Anchor preservation requires aligned slots, got "
            f"{tuple(resampled_slots.shape)} and {tuple(anchor_slots.shape)}."
        )
    return 1.0 - F.cosine_similarity(
        resampled_slots.float(),
        anchor_slots.float(),
        dim=-1,
    ).mean()
