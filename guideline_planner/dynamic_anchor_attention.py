"""Dynamic Anchor Attention for fixed-length latent guideline fusion."""

from __future__ import annotations

from typing import Any

import torch


class MultiHeadCrossAttentionWithRoutingBias(torch.nn.Module):
    """Cross-attention with an additive, batch-specific routing prior.

    Query has shape ``[B,Q,H]`` and key/value has shape ``[B,S,H]``.
    Routing bias and key padding mask both have shape ``[B,S]``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}."
            )
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = torch.nn.Dropout(float(dropout))

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        *,
        routing_bias: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if query.ndim != 3 or key_value.ndim != 3:
            raise ValueError(
                "Cross-attention expects query [B,Q,H] and key_value [B,S,H], "
                f"got {tuple(query.shape)} and {tuple(key_value.shape)}."
            )
        batch, query_length, hidden = query.shape
        kv_batch, key_length, kv_hidden = key_value.shape
        if batch != kv_batch or hidden != self.hidden_size or kv_hidden != self.hidden_size:
            raise ValueError(
                "Cross-attention shape mismatch: "
                f"query={tuple(query.shape)}, key_value={tuple(key_value.shape)}, "
                f"hidden_size={self.hidden_size}."
            )
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key_value))
        v = self._split_heads(self.v_proj(key_value))

        # The score/softmax path stays in fp32 for stable fp16/bf16 masking.
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        if routing_bias is not None:
            if routing_bias.shape != (batch, key_length):
                raise ValueError(
                    f"routing_bias must be {(batch, key_length)}, got "
                    f"{tuple(routing_bias.shape)}."
                )
            scores = scores + routing_bias.float()[:, None, None, :]
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, key_length):
                raise ValueError(
                    f"key_padding_mask must be {(batch, key_length)}, got "
                    f"{tuple(key_padding_mask.shape)}."
                )
            scores = scores.masked_fill(
                ~key_padding_mask.bool()[:, None, None, :],
                torch.finfo(scores.dtype).min,
            )
        attention = torch.softmax(scores, dim=-1)
        if not torch.isfinite(attention).all():
            raise RuntimeError(
                "Dynamic Anchor Attention produced non-finite attention weights; "
                f"query={tuple(query.shape)}, key_value={tuple(key_value.shape)}."
            )
        dropped_attention = self.attention_dropout(attention).to(dtype=v.dtype)
        context = torch.matmul(dropped_attention, v)
        context = context.transpose(1, 2).contiguous().view(
            batch,
            query_length,
            self.hidden_size,
        )
        output = self.out_proj(context)
        return output, attention if need_weights else None

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)


class DynamicAnchorAttentionResampler(torch.nn.Module):
    """Fuse ``[B,K,N,H]`` memories into fixed ``[B,N,H]`` latent slots."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_emb: int,
        *,
        patient_state_dim: int | None = None,
        memory_key_dim: int | None = None,
        max_num_memories: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_patient_state_condition: bool = True,
        use_gate_attention_bias: bool = True,
        use_rank_embedding: bool = True,
        use_memory_identity_embedding: bool = False,
        bypass_single_memory: bool = False,
        memory_dropout: float = 0.0,
        min_memories: int = 1,
        max_memories: int | None = None,
        return_attention_weights: bool = False,
        log_memory_attention_mass: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_emb <= 0 or max_num_memories <= 0:
            raise ValueError("num_emb and max_num_memories must be positive.")
        if not 0.0 <= memory_dropout < 1.0:
            raise ValueError("memory_dropout must be in [0, 1).")
        selected_max = int(max_memories or max_num_memories)
        if not 1 <= int(min_memories) <= selected_max <= int(max_num_memories):
            raise ValueError(
                "Expected 1 <= min_memories <= max_memories <= max_num_memories."
            )
        self.hidden_size = int(hidden_size)
        self.num_emb = int(num_emb)
        self.patient_state_dim = int(patient_state_dim or hidden_size)
        self.memory_key_dim = int(memory_key_dim or hidden_size)
        self.max_num_memories = int(max_num_memories)
        self.use_patient_state_condition = bool(use_patient_state_condition)
        self.use_gate_attention_bias = bool(use_gate_attention_bias)
        self.use_rank_embedding = bool(use_rank_embedding)
        self.use_memory_identity_embedding = bool(use_memory_identity_embedding)
        self.bypass_single_memory = bool(bypass_single_memory)
        self.memory_dropout = float(memory_dropout)
        self.min_memories = int(min_memories)
        self.max_memories = selected_max
        self.return_attention_weights = bool(return_attention_weights)
        self.log_memory_attention_mass = bool(log_memory_attention_mass)
        self.eps = float(eps)

        self.state_projection = torch.nn.Linear(self.patient_state_dim, self.hidden_size)
        self.query_norm = torch.nn.LayerNorm(self.hidden_size)
        self.rank_embedding = torch.nn.Embedding(
            self.max_num_memories,
            self.hidden_size,
        )
        self.memory_key_projection = torch.nn.Linear(
            self.memory_key_dim,
            self.hidden_size,
            bias=False,
        )
        self.cross_attention = MultiHeadCrossAttentionWithRoutingBias(
            self.hidden_size,
            int(num_heads),
            dropout=attention_dropout,
        )
        ffn_hidden = self.hidden_size * max(int(ffn_multiplier), 1)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, ffn_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(ffn_hidden, self.hidden_size),
        )
        self.dropout = torch.nn.Dropout(float(dropout))
        self.norm1 = torch.nn.LayerNorm(self.hidden_size)
        self.norm2 = torch.nn.LayerNorm(self.hidden_size)
        torch.nn.init.normal_(self.rank_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        memories: torch.Tensor,
        patient_state_repr: torch.Tensor,
        gate_weights: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        memory_ids: list[list[str]] | None = None,
        memory_keys: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if memories.ndim != 4:
            raise ValueError(f"memories must be [B,K,N,H], got {tuple(memories.shape)}.")
        batch, count, num_emb, hidden = memories.shape
        if count > self.max_num_memories:
            raise ValueError(
                f"Input K={count} exceeds max_num_memories={self.max_num_memories}."
            )
        if num_emb != self.num_emb or hidden != self.hidden_size:
            raise ValueError(
                "DAA memory shape mismatch: "
                f"input={tuple(memories.shape)}, expected N={self.num_emb}, "
                f"H={self.hidden_size}."
            )
        if patient_state_repr.ndim == 1:
            patient_state_repr = patient_state_repr.unsqueeze(0)
        if patient_state_repr.shape != (batch, self.patient_state_dim):
            raise ValueError(
                f"patient_state_repr must be {(batch, self.patient_state_dim)}, got "
                f"{tuple(patient_state_repr.shape)}."
            )
        if gate_weights.shape != (batch, count):
            raise ValueError(
                f"gate_weights must be {(batch, count)}, got {tuple(gate_weights.shape)}."
            )
        device = memories.device
        dtype = memories.dtype
        gates = gate_weights.to(device=device, dtype=dtype)
        if not torch.isfinite(gates).all():
            raise ValueError("gate_weights contains NaN or infinite values.")
        if memory_mask is None:
            mask = torch.ones((batch, count), dtype=torch.bool, device=device)
        else:
            if memory_mask.shape != (batch, count):
                raise ValueError(
                    f"memory_mask must be {(batch, count)}, got {tuple(memory_mask.shape)}."
                )
            mask = memory_mask.to(device=device, dtype=torch.bool)
        invalid_rows = (~mask).all(dim=1)
        if invalid_rows.any():
            rows = invalid_rows.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"All guideline memories are masked for batch rows {rows}.")
        if (gates.masked_select(mask) < 0).any():
            raise ValueError("Valid gate_weights must be non-negative.")
        gates = gates.masked_fill(~mask, 0.0)
        sums = gates.sum(dim=-1, keepdim=True)
        if (sums <= 0).any():
            rows = (sums.squeeze(-1) <= 0).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"Valid gate_weights sum to zero for batch rows {rows}.")
        normalized_gates = gates / sums
        masked_for_anchor = normalized_gates.masked_fill(~mask, -1.0)
        anchor_indices = masked_for_anchor.argmax(dim=-1)
        batch_indices = torch.arange(batch, device=device)
        anchor = memories[batch_indices, anchor_indices]

        effective_mask, effective_gates = self._apply_memory_dropout(
            mask,
            normalized_gates,
            anchor_indices,
        )
        memory_features = memories
        if self.use_rank_embedding:
            rank_ids = _gate_rank_ids(normalized_gates, mask)
            rank_features = self.rank_embedding(rank_ids).to(dtype=dtype)
            memory_features = memory_features + rank_features.unsqueeze(2)
        if self.use_memory_identity_embedding:
            if memory_keys is None:
                raise ValueError(
                    "memory_keys is required when use_memory_identity_embedding=true."
                )
            if memory_keys.shape != (batch, count, self.memory_key_dim):
                raise ValueError(
                    "memory_keys shape mismatch: "
                    f"got {tuple(memory_keys.shape)}, expected "
                    f"{(batch, count, self.memory_key_dim)}."
                )
            identity = self.memory_key_projection(
                memory_keys.to(device=device, dtype=self.memory_key_projection.weight.dtype)
            ).to(dtype=dtype)
            memory_features = memory_features + identity.unsqueeze(2)

        state = self.state_projection(
            patient_state_repr.to(
                device=device,
                dtype=self.state_projection.weight.dtype,
            )
        ).to(dtype=dtype)
        query = anchor + state.unsqueeze(1) if self.use_patient_state_condition else anchor
        query = self.query_norm(query)
        key_value = memory_features.reshape(batch, count * num_emb, hidden)
        slot_mask = effective_mask.repeat_interleave(num_emb, dim=-1)
        slot_gates = effective_gates.repeat_interleave(num_emb, dim=-1)
        routing_bias = (
            torch.log(slot_gates.clamp_min(self.eps))
            if self.use_gate_attention_bias
            else torch.zeros_like(slot_gates)
        )
        attended, attention = self.cross_attention(
            query,
            key_value,
            routing_bias=routing_bias,
            key_padding_mask=slot_mask,
            need_weights=(self.return_attention_weights or self.log_memory_attention_mass),
        )
        hidden_slots = self.norm1(anchor + self.dropout(attended))
        output = self.norm2(hidden_slots + self.dropout(self.ffn(hidden_slots)))
        valid_counts = effective_mask.sum(dim=-1)
        if self.bypass_single_memory:
            output = torch.where((valid_counts == 1)[:, None, None], anchor, output)
        if output.shape != (batch, self.num_emb, self.hidden_size):
            raise AssertionError(
                f"DAA output shape changed unexpectedly: {tuple(output.shape)}."
            )

        attention_mass = None
        if attention is not None:
            attention_mass = attention.mean(dim=(1, 2)).reshape(batch, count, num_emb).sum(-1)
            attention_mass = attention_mass.masked_fill(~effective_mask, 0.0)
        anchor_gate_weights = effective_gates[batch_indices, anchor_indices]
        return {
            "resampled_slots": output,
            "anchor_slots": anchor,
            "anchor_indices": anchor_indices,
            "anchor_gate_weights": anchor_gate_weights,
            "normalized_gate_weights": normalized_gates,
            "effective_gate_weights": effective_gates,
            "effective_memory_mask": effective_mask,
            "memory_attention_mass": attention_mass,
            "attention_weights": attention if self.return_attention_weights else None,
            "diagnostics": {
                "memory_fusion_strategy": "dynamic_anchor_attention",
                "input_shape": list(memories.shape),
                "input_memory_count": count,
                "input_num_emb": num_emb,
                "input_total_slots": count * num_emb,
                "output_shape": list(output.shape),
                "output_num_emb": self.num_emb,
                "memory_ids": memory_ids,
                "use_patient_state_condition": self.use_patient_state_condition,
                "use_gate_attention_bias": self.use_gate_attention_bias,
                "use_rank_embedding": self.use_rank_embedding,
                "use_memory_identity_embedding": self.use_memory_identity_embedding,
                "memory_dropout": self.memory_dropout if self.training else 0.0,
                "min_memories": self.min_memories,
                "max_memories": self.max_memories,
                "bypass_single_memory": self.bypass_single_memory,
            },
        }

    def _apply_memory_dropout(
        self,
        memory_mask: torch.Tensor,
        gate_weights: torch.Tensor,
        anchor_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.memory_dropout <= 0:
            return memory_mask, gate_weights
        keep = torch.rand_like(gate_weights) >= self.memory_dropout
        keep = keep & memory_mask
        batch_indices = torch.arange(memory_mask.shape[0], device=memory_mask.device)
        keep[batch_indices, anchor_indices] = True
        for row in range(memory_mask.shape[0]):
            valid_indices = memory_mask[row].nonzero(as_tuple=False).flatten()
            target_max = min(self.max_memories, int(valid_indices.numel()))
            if int(keep[row].sum()) > target_max:
                ranked = torch.argsort(
                    gate_weights[row].masked_fill(~memory_mask[row], -1.0),
                    descending=True,
                )
                limited = torch.zeros_like(keep[row])
                limited[ranked[:target_max]] = True
                keep[row] = limited & memory_mask[row]
                keep[row, anchor_indices[row]] = True
            target_min = min(self.min_memories, int(valid_indices.numel()))
            if int(keep[row].sum()) < target_min:
                ranked = torch.argsort(
                    gate_weights[row].masked_fill(~memory_mask[row], -1.0),
                    descending=True,
                )
                for index in ranked.tolist():
                    if memory_mask[row, index]:
                        keep[row, index] = True
                    if int(keep[row].sum()) >= target_min:
                        break
        dropped = gate_weights.masked_fill(~keep, 0.0)
        dropped = dropped / dropped.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return keep, dropped


def _gate_rank_ids(gate_weights: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
    """Return per-row descending gate ranks while preserving input order."""

    masked = gate_weights.masked_fill(~memory_mask, -1.0)
    order = torch.argsort(masked, dim=-1, descending=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(order.shape[1], device=order.device).expand_as(order)
    ranks.scatter_(1, order, rank_values)
    return ranks
