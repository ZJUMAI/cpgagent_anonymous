"""End-to-end patient-state-conditioned latent guideline routing."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from guideline_planner.dynamic_anchor_attention import DynamicAnchorAttentionResampler
from guideline_planner.latent_memory_retriever import (
    GuidelineMemoryBank,
    LatentMemoryRetriever,
)
from guideline_planner.memory_fusion import (
    MemoryFusion,
    pack_active_memories,
    slot_aligned_weighted_sum,
    top1_memory,
)
from guideline_planner.routing_config import LatentGuidelineRoutingConfig
from guideline_planner.routing_types import (
    FusedMemory,
    GateOutput,
    MemoryActivation,
    MemoryCandidate,
    PatientState,
    RoutingResult,
)
from guideline_planner.soft_memory_gate import SoftMemoryGate, uniform_gate_output
from guideline_planner.state_encoder import PatientStateEncoder


ROUTING_CHECKPOINT_VERSION = "latent_guideline_routing.v3"


class LatentGuidelineRoutingPipeline(torch.nn.Module):
    """Compose current-state retrieval, sparse gating, and latent fusion."""

    def __init__(
        self,
        *,
        memory_bank: GuidelineMemoryBank,
        query_encoder: Any,
        config: LatentGuidelineRoutingConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if not config.enabled:
            raise ValueError("LatentGuidelineRoutingPipeline requires enabled=true.")
        self.memory_bank = memory_bank
        self.config = config
        self.state_dim = int(memory_bank.retrieval_dim)
        self.retrieval_dim = int(memory_bank.retrieval_dim)
        self.hidden_size = int(memory_bank.slot_hidden_size)
        self.state_encoder = PatientStateEncoder(
            query_encoder,
            self.state_dim,
            device=device,
        )
        self.retriever = LatentMemoryRetriever(self.state_dim, self.retrieval_dim)
        self.gate = SoftMemoryGate(
            self.state_dim,
            self.retrieval_dim,
            routing_dim=config.gate.routing_dim,
            temperature=config.gate.temperature,
            use_seed_score=config.gate.use_seed_score,
        )
        strategy = config.memory_fusion.strategy.strip().lower()
        if strategy not in {
            "dynamic_anchor_attention",
            "weighted_concatenation",
            "top1_only",
        }:
            raise ValueError(f"Unsupported memory fusion strategy: {strategy!r}.")
        self.fusion_strategy = strategy
        self.memory_fusion = None
        self.dynamic_anchor_attention = None
        if strategy == "weighted_concatenation":
            self.memory_fusion = MemoryFusion(
                self.retrieval_dim,
                self.hidden_size,
                add_memory_identity_embedding=(
                    config.memory_fusion.add_memory_identity_embedding
                ),
                max_active_memories=config.memory_fusion.max_active_memories,
            )
        elif strategy == "dynamic_anchor_attention":
            daa = config.memory_fusion.dynamic_anchor_attention
            if not daa.enabled:
                raise ValueError(
                    "memory_fusion.strategy='dynamic_anchor_attention' requires "
                    "dynamic_anchor_attention.enabled=true."
                )
            if daa.anchor_strategy != "highest_gate_weight":
                raise ValueError(
                    "DAA currently supports anchor_strategy='highest_gate_weight' only."
                )
            output_num_emb = config.memory_fusion.output_num_emb
            if output_num_emb is not None and int(output_num_emb) != int(
                memory_bank.memory_tokens
            ):
                raise ValueError(
                    "memory_fusion.output_num_emb must match the memory store: "
                    f"configured={output_num_emb}, store={memory_bank.memory_tokens}."
                )
            if config.gate.active_top_k > daa.max_num_memories:
                raise ValueError(
                    "gate.active_top_k exceeds DAA max_num_memories: "
                    f"{config.gate.active_top_k} > {daa.max_num_memories}."
                )
            self.dynamic_anchor_attention = DynamicAnchorAttentionResampler(
                self.hidden_size,
                daa.num_heads,
                int(memory_bank.memory_tokens),
                patient_state_dim=self.state_dim,
                memory_key_dim=self.retrieval_dim,
                max_num_memories=daa.max_num_memories,
                ffn_multiplier=daa.ffn_multiplier,
                dropout=daa.dropout,
                attention_dropout=daa.attention_dropout,
                use_patient_state_condition=daa.use_patient_state_condition,
                use_gate_attention_bias=daa.use_gate_attention_bias,
                use_rank_embedding=daa.use_rank_embedding,
                use_memory_identity_embedding=daa.use_memory_identity_embedding,
                bypass_single_memory=daa.bypass_single_memory,
                memory_dropout=(daa.memory_dropout if daa.train_with_variable_k else 0.0),
                min_memories=daa.min_memories,
                max_memories=daa.max_memories,
                return_attention_weights=daa.return_attention_weights,
                log_memory_attention_mass=daa.log_memory_attention_mass,
                eps=daa.eps,
            )
        if not config.gate.enabled:
            _set_requires_grad(self.gate, False)
        self.checkpoint_meta: dict[str, Any] = {}
        self.to(device)

    def forward(
        self,
        patient_state: PatientState | Mapping[str, Any],
        trajectory_history: Sequence[Any] | None = None,
        *,
        candidate_memory_ids: Sequence[str] | None = None,
    ) -> RoutingResult:
        state = (
            patient_state
            if isinstance(patient_state, PatientState)
            else PatientState.from_mapping(
                patient_state,
                trajectory_history=trajectory_history,
            )
        )
        state_repr = self.state_encoder(state, list(trajectory_history or []))
        metadata_filter = self._metadata_filter(state)
        seed_candidates = self.retriever.search(
            state_repr,
            self.memory_bank,
            self.config.retrieval.seed_top_k,
            metadata_filter,
        )
        if not seed_candidates:
            raise RuntimeError(
                "No guideline memory matched the configured coarse metadata filters: "
                f"{metadata_filter}."
            )
        merged = list(seed_candidates[: self.config.retrieval.max_candidates])
        if candidate_memory_ids:
            existing = {candidate.memory_id for candidate in merged}
            projected_state = self.retriever.state_projection(state_repr)
            for memory_id in candidate_memory_ids:
                if memory_id in existing:
                    continue
                memory = self.memory_bank.lookup.get(str(memory_id))
                if memory is None:
                    raise RuntimeError(
                        f"Training candidate memory does not exist: {memory_id}"
                    )
                score = torch.nn.functional.cosine_similarity(
                    projected_state.reshape(1, -1),
                    memory.retrieval_key.to(projected_state).reshape(1, -1),
                ).reshape(())
                merged.append(
                    MemoryCandidate(
                        memory=memory,
                        seed_score=float(score.detach().cpu()),
                        combined_score=float(score.detach().cpu()),
                    )
                )
                existing.add(str(memory_id))
        if self.config.gate.enabled:
            gate_output = self.gate(
                state_repr,
                merged,
                self.config.gate.active_top_k,
            )
            if isinstance(gate_output, list):
                raise AssertionError("Single-state routing returned a batched gate output.")
        else:
            gate_output = uniform_gate_output(
                merged,
                top_k=self.config.gate.active_top_k,
                device=state_repr.device,
            )
        fused, prefix = self.compose_prefix(
            state_repr,
            gate_output.selected_activations,
            weights=gate_output.selected_weights,
        )
        return RoutingResult(
            patient_state=state,
            state_repr=state_repr,
            seed_candidates=seed_candidates,
            merged_candidates=merged,
            gate_output=gate_output,
            fused_memory=fused,
            expert_outputs=[],
            competition_output=None,
            decoder_prefix=prefix,
            diagnostics={
                "profile": self.config.profile,
                "memory_store_fingerprint": self.memory_bank.fingerprint,
                "metadata_filter": metadata_filter,
                "gate_enabled": self.config.gate.enabled,
                "memory_fusion_strategy": self.fusion_strategy,
                "checkpoint": dict(self.checkpoint_meta),
            },
        )

    def compose_prefix(
        self,
        state_repr: torch.Tensor,
        activations: Sequence[MemoryActivation],
        *,
        weights: torch.Tensor | None = None,
    ) -> tuple[FusedMemory, torch.Tensor]:
        """Build a decoder prefix, also used by utility counterfactuals."""
        if not activations:
            raise ValueError("Prefix composition requires at least one active memory.")
        if self.fusion_strategy == "weighted_concatenation":
            if self.memory_fusion is None:
                raise AssertionError("Weighted fusion module was not initialized.")
            fused = self.memory_fusion(
                activations,
                self.memory_bank.lookup,
                weights=weights,
            )
            return fused, fused.slots
        if self.fusion_strategy == "top1_only":
            first = self.memory_bank.lookup[activations[0].memory_id]
            fused = top1_memory(
                activations,
                self.memory_bank.lookup,
                weights=weights,
                device=state_repr.device,
                dtype=first.slots.dtype,
            )
            return fused, fused.slots
        if self.dynamic_anchor_attention is None:
            raise AssertionError("DAA module was not initialized.")
        parameter = next(self.dynamic_anchor_attention.parameters())
        packed = pack_active_memories(
            [activations],
            self.memory_bank.lookup,
            weight_batches=([weights] if weights is not None else None),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        daa_output = self.dynamic_anchor_attention(
            packed.memories,
            state_repr.unsqueeze(0),
            packed.gate_weights,
            packed.memory_mask,
            packed.memory_ids,
            packed.memory_keys,
        )
        slots = daa_output["resampled_slots"].squeeze(0)
        anchor_index = int(daa_output["anchor_indices"][0].detach().cpu())
        anchor_id = packed.memory_ids[0][anchor_index]
        normalized_gates = daa_output["normalized_gate_weights"][0]
        effective_gates = daa_output["effective_gate_weights"][0]
        attention_mass_tensor = daa_output["memory_attention_mass"]
        attention_mass = (
            attention_mass_tensor[0] if attention_mass_tensor is not None else None
        )
        diagnostics = dict(daa_output["diagnostics"])
        diagnostics.update(
            {
                "strategy": "dynamic_anchor_attention",
                "anchor_memory_id": anchor_id,
                "anchor_index": anchor_index,
                "anchor_gate_weight": float(
                    daa_output["anchor_gate_weights"][0].detach().float().cpu()
                ),
                "active_memories": [
                    {
                        "memory_id": memory_id,
                        "gate_weight": float(normalized_gates[index].detach().float().cpu()),
                        "effective_gate_weight": float(
                            effective_gates[index].detach().float().cpu()
                        ),
                    }
                    for index, memory_id in enumerate(packed.memory_ids[0])
                ],
                "memory_attention_mass": (
                    {
                        memory_id: float(attention_mass[index].detach().float().cpu())
                        for index, memory_id in enumerate(packed.memory_ids[0])
                    }
                    if attention_mass is not None
                    else None
                ),
            }
        )
        fused = FusedMemory(
            slots=slots,
            attention_mask=torch.ones(
                slots.shape[0],
                dtype=torch.long,
                device=slots.device,
            ),
            memory_ids=list(packed.memory_ids[0]),
            slot_slices={},
            segment_ids=torch.full(
                (slots.shape[0],),
                -1,
                dtype=torch.long,
                device=slots.device,
            ),
            diagnostics=diagnostics,
            memory_attention_mass=attention_mass,
            attention_weights=(
                daa_output["attention_weights"][0]
                if daa_output["attention_weights"] is not None
                else None
            ),
            anchor_slots=daa_output["anchor_slots"][0],
        )
        return fused, slots

    def apply_runtime_ablation(
        self,
        route: RoutingResult,
        ablation: str,
        *,
        seed: int = 17,
    ) -> RoutingResult:
        """Apply an inference-only ablation after normal checkpointed routing."""

        name = str(ablation).strip().lower()
        if name in {"", "none", "daa_full"}:
            return route
        original_ids = [item.memory_id for item in route.active_memories]
        original_weights = [float(item.weight) for item in route.active_memories]
        common = {
            "planner_ablation": name,
            "ablation_seed": int(seed),
            "original_active_memory_ids": original_ids,
            "original_gate_weights": original_weights,
        }
        if name == "router_topk_no_daa":
            if not route.active_memories:
                raise RuntimeError("A3 requires at least one Router-selected memory.")
            first = self.memory_bank.lookup[route.active_memories[0].memory_id]
            fused = slot_aligned_weighted_sum(
                route.active_memories,
                self.memory_bank.lookup,
                weights=route.gate_output.selected_weights,
                device=route.state_repr.device,
                dtype=first.slots.dtype,
            )
            diagnostics = {
                **dict(route.diagnostics),
                **common,
                "selected_memory_policy": "learned_router_topk",
                "effective_active_memory_ids": list(original_ids),
                "effective_gate_weights": list(original_weights),
                "daa_bypassed": True,
            }
            fused.diagnostics.update(diagnostics)
            return replace(
                route,
                fused_memory=fused,
                decoder_prefix=fused.slots,
                diagnostics=diagnostics,
            )
        if name == "daa_uniform_gate":
            count = len(route.active_memories)
            if count < 1:
                raise RuntimeError("A4 requires at least one Router-selected memory.")
            weights = torch.full_like(
                route.gate_output.selected_weights,
                1.0 / float(count),
            )
            activations = [
                replace(item, weight=1.0 / float(count), selected=True)
                for item in route.active_memories
            ]
            gate_output = _replace_selected_gate(
                route.gate_output,
                activations,
                weights,
                mode="daa_uniform_gate",
            )
            fused, prefix = self.compose_prefix(
                route.state_repr,
                activations,
                weights=weights,
            )
            effective_weights = [1.0 / float(count)] * count
            diagnostics = {
                **dict(route.diagnostics),
                **common,
                "selected_memory_policy": "learned_router_topk_uniform_weights",
                "effective_active_memory_ids": list(original_ids),
                "effective_gate_weights": effective_weights,
                "daa_bypassed": False,
            }
            fused.diagnostics.update(diagnostics)
            return replace(
                route,
                gate_output=gate_output,
                fused_memory=fused,
                decoder_prefix=prefix,
                diagnostics=diagnostics,
            )
        if name == "random_memory_global":
            random_count = int(self.config.gate.active_top_k)
            random_ids, derived_seed = deterministic_random_memory_ids(
                sorted(self.memory_bank.lookup),
                excluded_ids=original_ids,
                count=random_count,
                seed=seed,
                case_id=route.patient_state.case_id,
                current_time=route.patient_state.current_time,
            )
            count = len(random_ids)
            weights = torch.full(
                (count,),
                1.0 / float(count),
                dtype=route.gate_output.selected_weights.dtype,
                device=route.gate_output.selected_weights.device,
            )
            candidates = [
                MemoryCandidate(memory=self.memory_bank.lookup[memory_id])
                for memory_id in random_ids
            ]
            activations = [
                MemoryActivation(
                    memory_id=memory_id,
                    weight=1.0 / float(count),
                    selected=True,
                    section_title=self.memory_bank.lookup[memory_id].section_title,
                    source_pages=self.memory_bank.lookup[memory_id].source_pages,
                )
                for memory_id in random_ids
            ]
            gate_output = GateOutput(
                activations=list(activations),
                selected_activations=list(activations),
                logits=torch.zeros(count, dtype=weights.dtype, device=weights.device),
                probabilities=weights.clone(),
                selected_indices=list(range(count)),
                selected_weights=weights,
                diagnostics={
                    "candidate_count": count,
                    "active_count": count,
                    "selected_memory_ids": list(random_ids),
                    "weight_sum": 1.0,
                    "mode": "random_memory_global_uniform",
                    "derived_seed": derived_seed,
                    "entropy": math.log(count) if count > 1 else 0.0,
                    "normalized_entropy": 1.0 if count > 1 else 0.0,
                },
            )
            fused, prefix = self.compose_prefix(
                route.state_repr,
                activations,
                weights=weights,
            )
            effective_weights = [1.0 / float(count)] * count
            diagnostics = {
                **dict(route.diagnostics),
                **common,
                "selected_memory_policy": "global_random_excluding_normal_active",
                "effective_active_memory_ids": list(random_ids),
                "effective_gate_weights": effective_weights,
                "random_memory_derived_seed": derived_seed,
                "daa_bypassed": False,
            }
            fused.diagnostics.update(diagnostics)
            return replace(
                route,
                merged_candidates=candidates,
                gate_output=gate_output,
                fused_memory=fused,
                decoder_prefix=prefix,
                diagnostics=diagnostics,
            )
        raise ValueError(f"Unsupported routed Planner ablation: {ablation!r}.")

    def load_checkpoint(self, path: str | Path, *, strict: bool = True) -> dict[str, Any]:
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Routing checkpoint must be a mapping: {checkpoint_path}")
        if payload.get("format_version") != ROUTING_CHECKPOINT_VERSION:
            raise RuntimeError(
                f"Unsupported routing checkpoint version: {payload.get('format_version')!r}"
            )
        if not payload.get("trained") and not self.config.runtime.allow_untrained:
            raise RuntimeError("Routing checkpoint is not marked as trained.")
        if (
            self.fusion_strategy == "dynamic_anchor_attention"
            and not payload.get("daa_trained")
            and not self.config.runtime.allow_untrained
        ):
            raise RuntimeError(
                "Routing checkpoint was not trained with Dynamic Anchor Attention. "
                "Re-run train-router with the DAA routing config."
            )
        expected_fingerprint = payload.get("memory_store_fingerprint")
        if expected_fingerprint != self.memory_bank.fingerprint:
            raise RuntimeError(
                "Routing checkpoint memory-store fingerprint mismatch: "
                f"checkpoint={expected_fingerprint}, current={self.memory_bank.fingerprint}."
            )
        for key, actual in (
            ("state_dim", self.state_dim),
            ("retrieval_dim", self.retrieval_dim),
            ("hidden_size", self.hidden_size),
            ("memory_tokens", self.memory_bank.memory_tokens),
        ):
            if int(payload.get(key, -1)) != int(actual):
                raise RuntimeError(
                    f"Routing checkpoint {key}={payload.get(key)} does not match {actual}."
                )
        checkpoint_config = payload.get("routing_config")
        if not isinstance(checkpoint_config, Mapping):
            raise RuntimeError("Routing checkpoint is missing routing_config.")
        checkpoint_prefix = _prefix_config_signature(checkpoint_config)
        runtime_prefix = _prefix_config_signature(self.config.to_dict())
        if checkpoint_prefix != runtime_prefix:
            raise RuntimeError(
                "Routing checkpoint prefix configuration mismatch: "
                f"checkpoint={checkpoint_prefix}, runtime={runtime_prefix}. "
                "Use the same fusion and DAA configuration used during train-router."
            )
        for checkpoint_key, store_key in (
            ("memory_store_lineage_fingerprint", "memory_store_fingerprint"),
            ("memory_encoder_adapter_hash", "memory_encoder_adapter_hash"),
        ):
            expected = payload.get(checkpoint_key)
            actual = self.memory_bank.store_meta.get(store_key)
            if expected in (None, "") or actual in (None, "") or str(expected) != str(actual):
                raise RuntimeError(
                    "Routing checkpoint model provenance mismatch for "
                    f"{checkpoint_key}: checkpoint={expected!r}, memory_store={actual!r}."
                )
        state_dict = payload.get("routing_state_dict")
        if not isinstance(state_dict, Mapping):
            raise RuntimeError("Routing checkpoint is missing routing_state_dict.")
        self.load_state_dict(state_dict, strict=strict)
        self.checkpoint_meta = {
            "path": str(checkpoint_path),
            "format_version": payload.get("format_version"),
            "trained": bool(payload.get("trained")),
            "daa_trained": bool(payload.get("daa_trained")),
            "created_at": payload.get("created_at"),
            "created_from_run": payload.get("created_from_run"),
            "memory_store_fingerprint": expected_fingerprint,
            "memory_store_lineage_fingerprint": payload.get(
                "memory_store_lineage_fingerprint"
            ),
            "memory_encoder_adapter_hash": payload.get(
                "memory_encoder_adapter_hash"
            ),
            "planner_decoder_adapter_hash": payload.get(
                "planner_decoder_adapter_hash"
            ),
            "memory_to_decoder_bridge_hash": payload.get(
                "memory_to_decoder_bridge_hash"
            ),
            "base_model": payload.get("base_model"),
            "base_model_snapshot_path": payload.get("base_model_snapshot_path"),
            "tokenizer_path": payload.get("tokenizer_path"),
            "retrieval_projection_path": payload.get("retrieval_projection_path"),
            "memory_tokens": payload.get("memory_tokens"),
        }
        return dict(self.checkpoint_meta)

    def checkpoint_payload(
        self,
        *,
        trained: bool,
        created_at: str,
        created_from_run: str | None = None,
        training_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "format_version": ROUTING_CHECKPOINT_VERSION,
            "trained": bool(trained),
            "created_at": created_at,
            "created_from_run": created_from_run,
            "memory_store_fingerprint": self.memory_bank.fingerprint,
            "memory_store_lineage_fingerprint": self.memory_bank.store_meta.get(
                "memory_store_fingerprint"
            ),
            "memory_encoder_adapter_hash": self.memory_bank.store_meta.get(
                "memory_encoder_adapter_hash"
            ),
            "memory_ids": sorted(self.memory_bank.lookup),
            "state_dim": self.state_dim,
            "retrieval_dim": self.retrieval_dim,
            "hidden_size": self.hidden_size,
            "memory_tokens": self.memory_bank.memory_tokens,
            "daa_trained": bool(
                trained and self.fusion_strategy == "dynamic_anchor_attention"
            ),
            "routing_config": self.config.to_dict(),
            "routing_state_dict": self.state_dict(),
            "training_summary": dict(training_summary or {}),
        }

    def _metadata_filter(self, state: PatientState) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if self.config.retrieval.filter_cancer_type and state.cancer_type:
            subtype = str(state.raw_state.get("disease_subtype") or "").strip()
            filters["cancer_type"] = (
                {
                    "family": state.cancer_type,
                    "subtype": subtype,
                    "allow_generic": True,
                }
                if subtype and subtype != "unknown"
                else state.cancer_type
            )
        context = state.raw_state.get("guideline_context")
        guidelines = (
            context.get("guidelines", []) if isinstance(context, Mapping) else []
        )
        versions = sorted(
            {
                str(item.get("version") or "")
                for item in guidelines
                if isinstance(item, Mapping) and str(item.get("version") or "")
            }
        )
        guideline_ids = sorted(
            {
                str(item.get("guideline_id") or "")
                for item in guidelines
                if isinstance(item, Mapping) and str(item.get("guideline_id") or "")
            }
        )
        if self.config.retrieval.filter_guideline_id and guideline_ids:
            filters["guideline_id"] = guideline_ids
        if self.config.retrieval.filter_version and versions:
            filters["version"] = versions
        elif self.config.retrieval.filter_version and state.guideline_version:
            filters["version"] = state.guideline_version
        if self.config.retrieval.filter_language and state.language:
            filters["language"] = state.language
        return filters


def _set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = value


def deterministic_random_memory_ids(
    memory_ids: Sequence[str],
    *,
    excluded_ids: Sequence[str],
    count: int,
    seed: int,
    case_id: str,
    current_time: str | int,
) -> tuple[list[str], int]:
    """Select a reproducible global-memory negative set for A6."""

    excluded = {str(item) for item in excluded_ids}
    pool = sorted({str(item) for item in memory_ids if str(item) not in excluded})
    if count < 1:
        raise ValueError("Random-memory ablation requires a positive memory count.")
    if len(pool) < count:
        raise ValueError(
            f"Random-memory pool has {len(pool)} entries after exclusions; need {count}."
        )
    material = f"{int(seed)}|{case_id}|{current_time}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(derived_seed).sample(pool, count), derived_seed


def _replace_selected_gate(
    gate: GateOutput,
    selected: Sequence[MemoryActivation],
    weights: torch.Tensor,
    *,
    mode: str,
) -> GateOutput:
    by_id = {item.memory_id: item for item in selected}
    activations = [
        by_id.get(item.memory_id, replace(item, weight=0.0, selected=False))
        for item in gate.activations
    ]
    probabilities = torch.zeros_like(gate.probabilities)
    for offset, index in enumerate(gate.selected_indices):
        probabilities[index] = weights[offset]
    diagnostics = {
        **dict(gate.diagnostics),
        "mode": mode,
        "original_normalized_entropy": gate.diagnostics.get(
            "normalized_entropy"
        ),
        "entropy": math.log(len(selected)) if len(selected) > 1 else 0.0,
        "normalized_entropy": 1.0 if len(selected) > 1 else 0.0,
        "original_selected_weights": [
            float(item.detach().float().cpu()) for item in gate.selected_weights
        ],
        "effective_selected_weights": [
            float(item.detach().float().cpu()) for item in weights
        ],
    }
    return replace(
        gate,
        activations=activations,
        selected_activations=list(selected),
        probabilities=probabilities,
        selected_weights=weights,
        diagnostics=diagnostics,
    )


def _prefix_config_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    gate = config.get("gate", {})
    fusion = config.get("memory_fusion", {})
    if not isinstance(gate, Mapping):
        gate = {}
    if not isinstance(fusion, Mapping):
        fusion = {}
    daa = fusion.get("dynamic_anchor_attention", {})
    if not isinstance(daa, Mapping):
        daa = {}
    return {
        "gate_enabled": bool(gate.get("enabled", True)),
        "active_top_k": int(gate.get("active_top_k", 0)),
        "fusion_strategy": str(fusion.get("strategy") or ""),
        "output_num_emb": fusion.get("output_num_emb"),
        "add_memory_identity_embedding": bool(
            fusion.get("add_memory_identity_embedding", True)
        ),
        "dynamic_anchor_attention": {
            str(key): value for key, value in sorted(daa.items())
        },
    }
