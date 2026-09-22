"""Planner interface built on top of decoded latent guideline memory."""

from __future__ import annotations

import json
import ast
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from guideline_planner.constants import PLANNER_OUTPUT_FIELDS, TASK_TOKENS
from guideline_planner.latent_decoder import LatentMemoryQueryEncoder, LatentPlannerDecoder
from guideline_planner.memory import load_memory_store_meta
from guideline_planner.schemas_v2 import V2SchemaError, validate_patient_state_v2
from guideline_planner.retrieval import retrieve_latent_guideline_memory
from guideline_planner.release import ResolvedPlannerRelease, resolve_planner_release
from guideline_planner.routing_config import LatentGuidelineRoutingConfig
from guideline_planner.routing_types import (
    MemoryActivation,
    PatientState,
    PlannerStepResult,
)


PLANNER_ABLATION_MODES = (
    "none",
    "no_memory",
    "latent_topk",
    "router_topk_no_daa",
    "daa_uniform_gate",
    "random_memory_global",
)
PLANNER_ABLATION_IDS = {
    "none": "A5",
    "no_memory": "A1",
    "latent_topk": "A2",
    "router_topk_no_daa": "A3",
    "daa_uniform_gate": "A4",
    "random_memory_global": "A6",
}


class LatentPlannerError(RuntimeError):
    """Raised when latent guideline-memory planning cannot produce usable output."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


class LatentGuidelinePlanner:
    """Plan the next step by decoding trained latent guideline-memory slots."""

    def __init__(
        self,
        *,
        memory_dir: str | Path,
        top_k: int = 3,
        device: str | None = None,
        query_encoder_device: str | None = "cpu",
        max_new_tokens: int | str | None = 512,
        model_dtype: str = "auto",
        output_mode: str = "json",
        decoder: Any | None = None,
        query_encoder: Any | None = None,
        decoder_artifact_dir: str | Path | None = None,
        routing_config: LatentGuidelineRoutingConfig | Mapping[str, Any] | str | Path | None = None,
        routing_checkpoint: str | Path | None = None,
        release: ResolvedPlannerRelease | None = None,
        planner_ablation: str = "none",
        planner_ablation_seed: int = 17,
    ) -> None:
        self.memory_dir = Path(memory_dir)
        self.top_k = int(top_k)
        self.device = device or "auto"
        self.query_encoder_device = query_encoder_device or "cpu"
        self.max_new_tokens = _normalize_generation_limit(max_new_tokens)
        self.model_dtype = model_dtype
        self.output_mode = _normalize_output_mode(output_mode)
        self.planner_ablation = normalize_planner_ablation(planner_ablation)
        self.planner_ablation_seed = int(planner_ablation_seed)
        self.release = release
        if (
            self.planner_ablation != "none"
            and self.release is not None
            and self.release.mode != "daa_full"
        ):
            raise LatentPlannerError(
                "Controlled A1-A6 ablations require mode='daa_full' so every "
                "variant uses the A5 runtime decoder."
            )
        self.meta = load_memory_store_meta(self.memory_dir)
        self._validate_meta()
        self.routing_config = LatentGuidelineRoutingConfig.from_value(routing_config)
        resolved_routing_checkpoint = (
            routing_checkpoint or self.routing_config.runtime.checkpoint_path
        )
        if decoder is None and decoder_artifact_dir is None:
            raise LatentPlannerError(
                "Planner V2 requires decoder_artifact_dir; the Memory Encoder "
                "adapter cannot be reused implicitly as the Planner Decoder."
            )
        self.decoder_artifact_dir = (
            str(Path(decoder_artifact_dir)) if decoder_artifact_dir is not None else None
        )
        self.decoder = decoder or LatentPlannerDecoder.from_artifacts(
            self.memory_dir,
            decoder_artifact_dir,
            device=device,
            model_dtype=model_dtype,
        )
        self.query_encoder = query_encoder or (
            self.decoder
            if decoder is not None and hasattr(self.decoder, "encode_query")
            else LatentMemoryQueryEncoder.from_memory_store(
                self.memory_dir,
                self.meta,
                device=self.query_encoder_device,
                model_dtype=model_dtype,
            )
        )
        self._validate_decoder_shape()
        self.last_attempt: dict[str, Any] = {}
        self.last_step_result: PlannerStepResult | None = None
        self.last_routing_result: Any | None = None
        self.routing_pipeline: Any | None = None
        if self.routing_config.enabled:
            self._initialize_routing(
                routing_checkpoint=resolved_routing_checkpoint,
            )

    @classmethod
    def from_release(
        cls,
        release_dir_or_manifest: str | Path,
        *,
        mode: str | None = None,
        validate_hashes: bool = True,
        debug_overrides: Mapping[str, Any] | None = None,
        device: str | None = None,
        query_encoder_device: str | None = "cpu",
        max_new_tokens: int | str | None = 512,
        model_dtype: str = "auto",
        output_mode: str = "json",
        planner_ablation: str = "none",
        planner_ablation_seed: int = 17,
    ) -> "LatentGuidelinePlanner":
        """Construct a Planner from one immutable ``planner_release.v2``."""

        release = resolve_planner_release(
            release_dir_or_manifest,
            mode=mode,
            validate_hashes=validate_hashes,
            overrides=debug_overrides,
        )
        return cls(
            memory_dir=release.memory_dir,
            top_k=release.top_k,
            device=device,
            query_encoder_device=query_encoder_device,
            max_new_tokens=max_new_tokens,
            model_dtype=model_dtype,
            output_mode=output_mode,
            decoder_artifact_dir=release.decoder_artifact_dir,
            routing_config=release.routing_config,
            routing_checkpoint=release.routing_checkpoint,
            release=release,
            planner_ablation=planner_ablation,
            planner_ablation_seed=planner_ablation_seed,
        )

    def next_step(
        self,
        patient_state: Mapping[str, Any],
        trajectory_plan: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return one parsed Planner action object."""

        return self.plan(
            patient_state,
            trajectory_history=trajectory_plan,
        ).action

    def plan(
        self,
        patient_state: PatientState | Mapping[str, Any],
        trajectory_history: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    ) -> PlannerStepResult:
        """Generate an action and its complete latent-memory activation trace."""

        state_mapping = (
            patient_state.to_mapping()
            if isinstance(patient_state, PatientState)
            else dict(patient_state)
        )
        try:
            state_mapping = validate_patient_state_v2(state_mapping)
        except V2SchemaError as exc:
            raise LatentPlannerError(str(exc)) from exc
        if self.release is not None:
            try:
                self.release.require_supported_action(
                    str(state_mapping.get("cancer_family") or ""),
                    str(state_mapping.get("disease_subtype") or ""),
                )
            except Exception as exc:
                raise LatentPlannerError(str(exc)) from exc
        history = trajectory_history
        if self.planner_ablation == "no_memory":
            return self._plan_without_memory(state_mapping, history)
        if self.planner_ablation == "latent_topk":
            return self._plan_with_latent_topk(state_mapping, history)
        if self.routing_pipeline is not None:
            import torch

            self.routing_pipeline.eval()
            with torch.no_grad():
                routing = self.routing_pipeline(
                    state_mapping,
                    _history_sequence(history),
                )
                routing = self.routing_pipeline.apply_runtime_ablation(
                    routing,
                    self.planner_ablation,
                    seed=self.planner_ablation_seed,
                )
            self.last_routing_result = routing
            memories = _routing_memories_for_prompt(routing)
            prefix_slots = routing.decoder_prefix.detach().cpu().float().numpy()
            query = build_planner_query(state_mapping, history)
            routing_audit = routing.audit_dict()
            for key in (
                "planner_ablation",
                "ablation_seed",
                "selected_memory_policy",
                "original_active_memory_ids",
                "effective_active_memory_ids",
                "original_gate_weights",
                "effective_gate_weights",
                "random_memory_derived_seed",
                "daa_bypassed",
            ):
                if key in routing.diagnostics:
                    routing_audit[key] = routing.diagnostics[key]
            routing_audit["ablation_id"] = PLANNER_ABLATION_IDS.get(
                self.planner_ablation
            )
            routing_audit["decoder_role"] = "runtime"
            action = self._decode_action(
                state_mapping,
                history,
                memories,
                prefix_slots,
                query=query,
                routing_audit=routing_audit,
            )
            result = PlannerStepResult(
                action=action,
                current_objective=_planner_current_objective(action),
                expected_state_change=_expected_state_changes(action),
                active_memories=list(routing.active_memories),
                expert_outputs=[],
                diagnostics=routing_audit,
            )
            self.last_step_result = result
            return result

        if self.planner_ablation != "none":
            raise LatentPlannerError(
                f"Planner ablation {self.planner_ablation!r} requires a daa_full "
                "release with a trained routing checkpoint."
            )
        return self._plan_with_latent_topk(state_mapping, history)

    def _plan_with_latent_topk(
        self,
        state_mapping: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    ) -> PlannerStepResult:
        query = build_planner_query(state_mapping, history)
        self.last_routing_result = None
        memories = retrieve_latent_guideline_memory(
            query,
            self.top_k,
            memory_dir=self.memory_dir,
            filters=_state_memory_filters(state_mapping),
            selector=self.query_encoder,
            load_slots=True,
        )
        if not memories:
            raise LatentPlannerError(
                "No guideline memory slots are available for latent retrieval."
            )
        fused_memories, prefix_slots, activations, fusion_diagnostics = (
            _fuse_latent_topk_memories(memories)
        )
        action = self._decode_action(
            state_mapping,
            history,
            fused_memories,
            prefix_slots,
            query=query,
            routing_audit=None,
        )
        result = PlannerStepResult(
            action=action,
            current_objective=_planner_current_objective(action),
            expected_state_change=_expected_state_changes(action),
            active_memories=activations,
            expert_outputs=[],
            diagnostics={
                "routing_enabled": False,
                "mode": "latent_topk",
                "ablation_id": PLANNER_ABLATION_IDS.get(self.planner_ablation),
                "planner_ablation": self.planner_ablation,
                "ablation_seed": self.planner_ablation_seed,
                "decoder_role": "runtime" if self.release is not None else "direct",
                "selected_memory_policy": "encoder_latent_topk",
                "original_active_memory_ids": [
                    item.memory_id for item in activations
                ],
                "effective_active_memory_ids": [
                    item.memory_id for item in activations
                ],
                "original_gate_weights": [item.weight for item in activations],
                "effective_gate_weights": [item.weight for item in activations],
                "fusion_diagnostics": fusion_diagnostics,
                "retrieved_memories": [
                    _memory_metadata_summary(memory) for memory in fused_memories
                ],
            },
        )
        self.last_step_result = result
        return result

    def _plan_without_memory(
        self,
        state_mapping: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    ) -> PlannerStepResult:
        query = build_planner_query(state_mapping, history)
        self.last_routing_result = None
        slot_count = int(self.meta["memory_tokens"])
        hidden_size = int(self.meta["slot_hidden_size"])
        prefix_slots = np.zeros((slot_count, hidden_size), dtype=np.float32)
        diagnostics = {
            "routing_enabled": False,
            "mode": "no_memory",
            "ablation_id": "A1",
            "planner_ablation": "no_memory",
            "ablation_seed": self.planner_ablation_seed,
            "decoder_role": "runtime" if self.release is not None else "direct",
            "selected_memory_policy": "zero_memory_prefix",
            "original_active_memory_ids": [],
            "effective_active_memory_ids": [],
            "original_gate_weights": [],
            "effective_gate_weights": [],
            "decoder_prefix_shape": [slot_count, hidden_size],
        }
        action = self._decode_action(
            state_mapping,
            history,
            [],
            prefix_slots,
            query=query,
            routing_audit=diagnostics,
        )
        result = PlannerStepResult(
            action=action,
            current_objective=_planner_current_objective(action),
            expected_state_change=_expected_state_changes(action),
            active_memories=[],
            expert_outputs=[],
            diagnostics=diagnostics,
        )
        self.last_step_result = result
        return result

    def _decode_action(
        self,
        patient_state: Mapping[str, Any],
        trajectory_plan: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
        memories: list[dict[str, Any]],
        prefix_slots: Any,
        *,
        query: str,
        routing_audit: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        best = memories[0] if memories else {}
        prompt = _decoder_prompt(
            patient_state,
            trajectory_plan,
            memories,
            routing_audit,
            output_mode=self.output_mode,
        )
        self.last_attempt = {
            "query": query,
            "output_mode": self.output_mode,
            "selected_guideline_memory_id": best.get("guideline_memory_id"),
            "retrieved_memories": [_memory_metadata_summary(memory) for memory in memories],
            "prompt_preview": _truncate_text(prompt, 8000),
            "max_new_tokens": self.max_new_tokens,
            "routing": dict(routing_audit or {}),
        }
        raw_text = self.decoder.generate_plan(
            prefix_slots,
            prompt,
            max_new_tokens=self.max_new_tokens,
        )
        self.last_attempt["raw_text"] = _truncate_text(raw_text, 8000)
        _record_generation_attempt(self.last_attempt, self.decoder, "raw", raw_text)
        try:
            payload = _parse_planner_json(raw_text)
            normalized = _normalize_planner_output(payload)
            self.last_attempt["status"] = "parsed"
            _record_planner_output_metrics(
                self.last_attempt,
                self.decoder,
                normalized,
                source_generation_label="raw",
            )
            return normalized
        except LatentPlannerError as first_error:
            self.last_attempt["parse_error"] = str(first_error)

        repair_prompt = _decoder_repair_prompt(
            patient_state,
            memories,
            raw_text,
            self.last_attempt["parse_error"],
            routing_audit,
        )
        self.last_attempt["repair_prompt_preview"] = _truncate_text(repair_prompt, 8000)
        repair_raw_text = self.decoder.generate_plan(
            prefix_slots,
            repair_prompt,
            max_new_tokens=self.max_new_tokens,
        )
        self.last_attempt["repair_raw_text"] = _truncate_text(repair_raw_text, 8000)
        _record_generation_attempt(
            self.last_attempt,
            self.decoder,
            "repair",
            repair_raw_text,
        )
        try:
            payload = _parse_planner_json(repair_raw_text)
            normalized = _normalize_planner_output(payload)
            self.last_attempt["status"] = "repaired"
            _record_planner_output_metrics(
                self.last_attempt,
                self.decoder,
                normalized,
                source_generation_label="repair",
            )
            return normalized
        except LatentPlannerError as repair_error:
            self.last_attempt["repair_error"] = str(repair_error)

        strict_repair_prompt = _decoder_strict_repair_prompt(
            patient_state,
            memories,
            raw_text,
            repair_raw_text,
            self.last_attempt["repair_error"],
        )
        self.last_attempt["strict_repair_prompt_preview"] = _truncate_text(
            strict_repair_prompt,
            8000,
        )
        strict_repair_raw_text = self.decoder.generate_plan(
            prefix_slots,
            strict_repair_prompt,
            max_new_tokens=_strict_repair_generation_limit(self.max_new_tokens),
        )
        self.last_attempt["strict_repair_raw_text"] = _truncate_text(
            strict_repair_raw_text,
            12000,
        )
        _record_generation_attempt(
            self.last_attempt,
            self.decoder,
            "strict_repair",
            strict_repair_raw_text,
        )
        try:
            payload = _parse_planner_json(strict_repair_raw_text)
            normalized = _normalize_planner_output(payload)
            self.last_attempt["status"] = "strict_repaired"
            _record_planner_output_metrics(
                self.last_attempt,
                self.decoder,
                normalized,
                source_generation_label="strict_repair",
            )
            return normalized
        except LatentPlannerError as strict_repair_error:
            self.last_attempt["status"] = "failed"
            self.last_attempt["strict_repair_error"] = str(strict_repair_error)
            raise LatentPlannerError(
                "Planner decoder did not return a JSON object with the required "
                "planner fields after two format-repair retries.",
                details=self.last_attempt,
            ) from strict_repair_error

    def public_config(self) -> dict[str, Any]:
        """Return audit-safe planner configuration without loading raw slots."""

        routing_enabled = self.routing_pipeline is not None
        return {
            "planning_mode": (
                "latent_decoder_state_conditioned_routing"
                if routing_enabled
                else "latent_decoder"
            ),
            "memory_dir": str(self.memory_dir),
            "decoder_artifact_dir": self.decoder_artifact_dir,
            "top_k": self.top_k,
            "device": self.device,
            "query_encoder_device": self.query_encoder_device,
            "max_new_tokens": self.max_new_tokens,
            "output_mode": self.output_mode,
            "routing": {
                "enabled": routing_enabled,
                "config": self.routing_config.to_dict(),
                "checkpoint": dict(
                    getattr(self.routing_pipeline, "checkpoint_meta", {}) or {}
                ),
            },
            "ablation": {
                "ablation_id": PLANNER_ABLATION_IDS.get(self.planner_ablation),
                "planner_ablation": self.planner_ablation,
                "ablation_seed": self.planner_ablation_seed,
                "decoder_role": (
                    "runtime"
                    if self.release is not None and self.release.mode == "daa_full"
                    else "baseline_or_direct"
                ),
            },
            "release": (
                self.release.public_config() if self.release is not None else None
            ),
            "memory_store_meta": {
                key: self.meta.get(key)
                for key in (
                    "base_model",
                    "memory_encoder_adapter_path",
                    "memory_encoder_adapter_hash",
                    "tokenizer_path",
                    "tokenizer_hash",
                    "retrieval_projection_path",
                    "retrieval_projection_hash",
                    "memory_store_fingerprint",
                    "memory_tokens",
                    "slot_hidden_size",
                    "created_from_run",
                    "trained",
                )
            },
        }

    def _initialize_routing(
        self,
        *,
        routing_checkpoint: str | Path | None,
    ) -> None:
        from guideline_planner.latent_memory_retriever import GuidelineMemoryBank
        from guideline_planner.routing_pipeline import LatentGuidelineRoutingPipeline

        config = self.routing_config
        bank = GuidelineMemoryBank.from_directory(self.memory_dir)
        pipeline = LatentGuidelineRoutingPipeline(
            memory_bank=bank,
            query_encoder=self.query_encoder.encode_query,
            config=config,
            device=getattr(self.decoder, "device", "cpu"),
        )
        checkpoint = routing_checkpoint or config.runtime.checkpoint_path
        if checkpoint:
            pipeline.load_checkpoint(checkpoint)
            self._validate_routing_decoder_binding(pipeline.checkpoint_meta)
        elif not config.runtime.allow_untrained:
            raise LatentPlannerError(
                "Latent routing is enabled but no trained routing checkpoint was "
                "provided. Set runtime.checkpoint_path/--planner-routing-checkpoint, "
                "or use allow_untrained only for the CPU smoke test."
            )
        else:
            pipeline_device = str(next(pipeline.parameters()).device)
            if not pipeline_device.startswith("cpu"):
                raise LatentPlannerError(
                    "Untrained routing modules are permitted only for the CPU "
                    f"smoke test, but the pipeline is on {pipeline_device}."
                )
            pipeline.checkpoint_meta = {
                "trained": False,
                "mode": "untrained_smoke_only",
                "memory_store_fingerprint": bank.fingerprint,
            }
        self.routing_pipeline = pipeline

    def _validate_routing_decoder_binding(
        self,
        checkpoint_meta: Mapping[str, Any],
    ) -> None:
        expected = {
            "memory_store_lineage_fingerprint": self.meta.get("memory_store_fingerprint"),
            "planner_decoder_adapter_hash": getattr(self.decoder, "artifact_meta", {}).get(
                "planner_decoder_adapter_hash"
            ),
            "memory_to_decoder_bridge_hash": getattr(self.decoder, "artifact_meta", {}).get(
                "memory_to_decoder_bridge_hash"
            ),
            "memory_encoder_adapter_hash": self.meta.get("memory_encoder_adapter_hash"),
        }
        for key, actual in expected.items():
            recorded = checkpoint_meta.get(key)
            if actual in (None, "") or recorded in (None, "") or str(actual) != str(recorded):
                raise LatentPlannerError(
                    f"Routing checkpoint artifact binding mismatch for {key}: "
                    f"runtime={actual!r}, checkpoint={recorded!r}."
                )

    def _validate_meta(self) -> None:
        if not self.meta.get("trained"):
            raise LatentPlannerError(
                "This memory store is not a trained V2 Memory Encoder store "
                "(memory_store_meta.json has trained=false). Re-run extract-memory "
                "with --training-run-dir."
            )
        for field in (
            "base_model",
            "base_model_revision",
            "memory_encoder_adapter_path",
            "memory_encoder_adapter_hash",
            "tokenizer_path",
            "tokenizer_hash",
            "retrieval_projection_path",
            "retrieval_projection_hash",
            "memory_store_fingerprint",
            "memory_tokens",
            "slot_hidden_size",
        ):
            if self.meta.get(field) in (None, ""):
                raise LatentPlannerError(
                    f"memory_store_meta.json is missing required field {field!r}."
                )

    def _validate_decoder_shape(self) -> None:
        decoder_hidden = getattr(self.decoder, "hidden_size", None)
        expected_hidden = int(self.meta.get("slot_hidden_size") or 0)
        if decoder_hidden is not None and expected_hidden and int(decoder_hidden) != expected_hidden:
            raise LatentPlannerError(
                f"Decoder hidden size {decoder_hidden} does not match memory store "
                f"slot_hidden_size {expected_hidden}."
            )


def _state_memory_filters(patient_state: Mapping[str, Any]) -> dict[str, Any]:
    subtype = str(patient_state.get("disease_subtype") or "").strip().lower()
    family = str(patient_state.get("cancer_family") or "").strip().lower()
    cancer_type: Any = family
    if subtype not in {"", "unknown"}:
        cancer_type = {
            "family": family,
            "subtype": subtype,
            "allow_generic": True,
        }
    guidelines = patient_state.get("guideline_context", {}).get("guidelines", [])
    versions = {
        str(item.get("version") or "")
        for item in guidelines
        if isinstance(item, Mapping) and str(item.get("version") or "")
    }
    guideline_ids = {
        str(item.get("guideline_id") or "")
        for item in guidelines
        if isinstance(item, Mapping) and str(item.get("guideline_id") or "")
    }
    filters: dict[str, Any] = {"cancer_type": cancer_type}
    if versions:
        filters["version"] = sorted(versions)
    if guideline_ids:
        filters["guideline_id"] = sorted(guideline_ids)
    return filters


def _routing_memories_for_prompt(routing: Any) -> list[dict[str, Any]]:
    active_ids = [item.memory_id for item in routing.active_memories]
    by_id = {item.memory_id: item for item in routing.merged_candidates}
    # planner_action.v2 may cite only memories that received a non-zero gate
    # activation for this decision.
    missing = [memory_id for memory_id in active_ids if memory_id not in by_id]
    if missing:
        raise LatentPlannerError(
            "Routing returned active memories absent from merged candidates: "
            + ", ".join(missing)
        )
    result: list[dict[str, Any]] = []
    activation_by_id = {item.memory_id: item for item in routing.active_memories}
    for memory_id in active_ids:
        candidate = by_id[memory_id]
        activation = activation_by_id[memory_id]
        metadata = dict(candidate.memory.metadata)
        metadata.update(
            {
                "guideline_memory_id": memory_id,
                "guideline_name": candidate.memory.guideline_name,
                "version": candidate.memory.guideline_version,
                "cancer_type": candidate.memory.cancer_type,
                "h1_title": candidate.memory.section_title,
                "section_path": list(candidate.memory.section_path),
                "language": candidate.memory.language,
                "source_rule_ids": list(candidate.memory.source_rule_ids),
                "source_span_ids": list(candidate.memory.source_chunk_ids),
                "page_start": candidate.memory.page_start,
                "page_end": candidate.memory.page_end,
                "routing_seed_score": candidate.seed_score,
                "routing_gate_weight": activation.weight,
                "routing_memory_role": "current_state_retrieval",
            }
        )
        result.append(
            {
                "score": activation.weight,
                "guideline_memory_id": memory_id,
                "metadata": metadata,
            }
        )
    return result


def _fuse_latent_topk_memories(
    memories: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, list[MemoryActivation], dict[str, Any]]:
    """Fuse every retrieved top-K memory into the trained fixed slot count.

    Decoder SFT sees one slot-aligned weighted/mean prefix. Keeping the same
    shape at inference avoids the former 64-slot training versus 256-slot
    latent_topk mismatch while all top-K memories remain active provenance.
    """

    if not memories:
        raise LatentPlannerError("latent_topk fusion requires at least one memory.")
    scores: list[float] = []
    arrays: list[np.ndarray] = []
    hidden_size: int | None = None
    for index, memory in enumerate(memories):
        score = _optional_float(memory.get("score"))
        scores.append(score if score is not None and math.isfinite(score) else 0.0)
        if "memory_slots" not in memory:
            raise LatentPlannerError(
                f"Retrieved memory at rank {index} has no latent memory_slots."
            )
        slots = np.asarray(memory["memory_slots"], dtype=np.float32)
        if slots.ndim != 2 or slots.shape[0] < 1 or slots.shape[1] < 1:
            raise LatentPlannerError(
                f"Retrieved memory at rank {index} has invalid slot shape {slots.shape}."
            )
        if hidden_size is None:
            hidden_size = int(slots.shape[1])
        elif int(slots.shape[1]) != hidden_size:
            raise LatentPlannerError(
                "Retrieved memories cannot be fused because their hidden sizes differ."
            )
        arrays.append(slots)
    shifted = np.asarray(scores, dtype=np.float64)
    shifted -= float(shifted.max())
    weights = np.exp(shifted)
    weights /= float(weights.sum())

    enriched: list[dict[str, Any]] = []
    activations: list[MemoryActivation] = []
    weighted_blocks: list[np.ndarray] = []
    for rank, (source, slots, weight) in enumerate(zip(memories, arrays, weights)):
        memory = dict(source)
        metadata_value = memory.get("metadata")
        metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
        metadata.update(
            {
                "routing_seed_score": scores[rank],
                "routing_gate_weight": float(weight),
                "routing_memory_role": "latent_topk_retrieval",
            }
        )
        memory["metadata"] = metadata
        memory["fusion_weight"] = float(weight)
        memory_id = str(memory.get("guideline_memory_id") or "")
        if not memory_id:
            raise LatentPlannerError(f"Retrieved memory at rank {rank} has no memory ID.")
        if any(item.memory_id == memory_id for item in activations):
            raise LatentPlannerError(f"Duplicate memory ID in latent_topk: {memory_id}")
        block = slots * np.float32(weight)
        weighted_blocks.append(block)
        enriched.append(memory)
        activations.append(
            MemoryActivation(
                memory_id=memory_id,
                weight=float(weight),
                seed_score=scores[rank],
                gate_score=scores[rank],
                selected=True,
                section_title=_optional_nonempty(
                    metadata.get("h1_title") or metadata.get("section")
                ),
                source_pages=_source_pages(metadata),
            )
        )
    slot_shapes = {tuple(block.shape) for block in weighted_blocks}
    if len(slot_shapes) != 1:
        raise LatentPlannerError(
            "Retrieved memories require identical slot shapes for slot-level fusion."
        )
    prefix = np.stack(weighted_blocks, axis=0).sum(axis=0)
    diagnostics = {
        "strategy": "normalized_weighted_slot_fusion",
        "active_memory_count": len(enriched),
        "memory_ids": [item.memory_id for item in activations],
        "normalized_weights": [float(value) for value in weights],
        "slot_count": int(prefix.shape[0]),
        "hidden_size": int(prefix.shape[1]),
    }
    return enriched, prefix, activations, diagnostics


def _history_sequence(
    value: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    return list(value)


def _expected_state_changes(action: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for item in action.get("actions", []):
        if not isinstance(item, Mapping):
            continue
        for change in item.get("expected_state_delta", []):
            if str(change).strip() and str(change) not in result:
                result.append(str(change))
    return result[:12]


def _planner_current_objective(action: Mapping[str, Any]) -> str | None:
    actions = action.get("actions")
    if not isinstance(actions, list) or not actions or not isinstance(actions[0], Mapping):
        return None
    return _optional_nonempty(actions[0].get("objective"))


def _source_pages(metadata: Mapping[str, Any]) -> str | None:
    start = metadata.get("page_start")
    end = metadata.get("page_end")
    if start is None and end is None:
        return None
    if start == end or end is None:
        return str(start)
    if start is None:
        return str(end)
    return f"{start}-{end}"


def _optional_nonempty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_generation_limit(value: int | str | None) -> int | str:
    if value is None:
        return "auto"
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"auto", "none", "unlimited"}:
            return "auto"
        parsed = int(stripped)
        return "auto" if parsed <= 0 else parsed
    parsed = int(value)
    return "auto" if parsed <= 0 else parsed


def _normalize_output_mode(value: str | None) -> str:
    normalized = str(value or "json").strip().lower().replace("-", "_")
    aliases = {"json": "json", "structured": "json", "strict_json": "json"}
    if normalized not in aliases:
        raise ValueError(
            "Planner V2 output_mode must be strict JSON; free-text/V1 output is not accepted."
        )
    return aliases[normalized]


def normalize_planner_ablation(value: str | None) -> str:
    normalized = str(value or "none").strip().lower().replace("-", "_")
    if normalized not in PLANNER_ABLATION_MODES:
        raise ValueError(
            "Unknown Planner ablation. Expected one of: "
            + ", ".join(PLANNER_ABLATION_MODES)
            + f"; got {value!r}."
        )
    return normalized


def _strict_repair_generation_limit(value: int | str | None) -> int | str:
    if value in (None, "auto"):
        return "auto"
    return max(int(value), 1536)


def _record_generation_attempt(
    attempt: dict[str, Any],
    decoder: Any,
    label: str,
    text: str,
) -> None:
    metrics = _decoder_text_metrics(decoder, text)
    metrics["label"] = label
    decoder_generation = getattr(decoder, "last_generation_info", None)
    if isinstance(decoder_generation, Mapping) and decoder_generation:
        metrics["decoder_generation"] = dict(decoder_generation)
    attempt[f"{label}_text_metrics"] = metrics
    attempt[f"{label}_text_token_count"] = metrics.get("token_count")
    generations = attempt.setdefault("generation_attempts", [])
    if isinstance(generations, list):
        generations.append(metrics)


def _record_planner_output_metrics(
    attempt: dict[str, Any],
    decoder: Any,
    planner_output: Mapping[str, Any],
    *,
    source_generation_label: str,
) -> None:
    output_text = json.dumps(
        dict(planner_output),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    metrics = _decoder_text_metrics(decoder, output_text)
    metrics["source_generation_label"] = source_generation_label
    attempt["planner_output_text_metrics"] = metrics
    attempt["planner_output_token_count"] = metrics.get("token_count")


def _decoder_text_metrics(decoder: Any, text: str) -> dict[str, Any]:
    token_count: int | None = None
    token_count_method = "unavailable"
    counter = getattr(decoder, "count_text_tokens", None)
    if callable(counter):
        try:
            token_count = int(counter(text))
            token_count_method = "decoder_tokenizer"
        except Exception as exc:  # pragma: no cover - audit fallback only
            token_count_method = f"decoder_tokenizer_failed:{type(exc).__name__}"
    return {
        "token_count": token_count,
        "token_count_method": token_count_method,
        "char_count": len(text),
        "byte_count": len(text.encode("utf-8")),
        "line_count": text.count("\n") + 1 if text else 0,
    }


def build_planner_query(
    patient_state: Mapping[str, Any],
    trajectory_plan: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
) -> str:
    cancer_type = str(
        patient_state.get("cancer_family") or patient_state.get("cancer_type") or ""
    )
    diagnosis = str(patient_state.get("diagnosis") or patient_state.get("known_diagnosis") or "")
    parts = [
        cancer_type,
        diagnosis,
        _expanded_cancer_query_terms(cancer_type, diagnosis),
        str(patient_state.get("stage") or patient_state.get("known_stage") or ""),
        str(patient_state.get("current_phase") or ""),
        str(patient_state.get("disease_subtype") or ""),
        str(patient_state.get("decision_date") or ""),
        " ".join(map(str, patient_state.get("symptoms") or [])),
        " ".join(map(str, (patient_state.get("known_biomarkers") or {}).keys())),
        json.dumps(
            patient_state.get("risk_stratification") or {},
            ensure_ascii=False,
            sort_keys=True,
        ),
        " ".join(map(str, patient_state.get("completed_skills") or [])),
        " ".join(map(str, patient_state.get("unresolved_information") or [])),
        json.dumps(
            patient_state.get("guideline_context") or {},
            ensure_ascii=False,
            sort_keys=True,
        ),
        json.dumps(patient_state.get("treatment_history") or [], ensure_ascii=False, sort_keys=True),
        json.dumps(patient_state.get("last_transition") or {}, ensure_ascii=False, sort_keys=True),
    ]
    if trajectory_plan:
        parts.append(json.dumps(trajectory_plan, ensure_ascii=False, sort_keys=True))
    return " ".join(part for part in parts if part).strip() or "肿瘤 诊疗 路径 规划"


def _expanded_cancer_query_terms(cancer_type: str, diagnosis: str) -> str:
    haystack = f"{cancer_type} {diagnosis}".lower()
    terms: list[str] = []
    if "lung" in haystack or "luad" in haystack or "lusc" in haystack or "肺" in haystack:
        terms.extend(["lung cancer", "肺癌", "CSCO lung cancer"])
    if (
        "adenocarcinoma" in haystack
        or "luad" in haystack
        or "腺癌" in haystack
        or "nsclc" in haystack
        or "non-small" in haystack
        or "非小细胞" in haystack
    ):
        terms.extend(["NSCLC", "non-small cell lung cancer", "非小细胞肺癌", "肺腺癌"])
    if "small cell" in haystack or "sclc" in haystack or "小细胞" in haystack:
        terms.extend(["SCLC", "small cell lung cancer", "小细胞肺癌"])
    if "breast" in haystack or "乳腺" in haystack:
        terms.extend(["breast cancer", "乳腺癌"])
    if "lymphoma" in haystack or "淋巴瘤" in haystack:
        terms.extend(["lymphoma", "淋巴瘤"])
    return " ".join(dict.fromkeys(terms))


def _decoder_prompt(
    patient_state: Mapping[str, Any],
    trajectory_plan: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    memories: list[dict[str, Any]],
    routing_audit: Mapping[str, Any] | None = None,
    *,
    output_mode: str = "json",
) -> str:
    payload = _compact_patient_state_for_prompt(patient_state)
    payload["guideline_memory_candidates"] = [
        _memory_prompt_summary(memory) for memory in memories
    ]
    trajectory_tail = _compact_trajectory_for_prompt(trajectory_plan)
    if trajectory_tail:
        payload["trajectory_tail"] = trajectory_tail
    _normalize_output_mode(output_mode)
    payload["planner_command"] = (
        "Plan up to three ordered next actions without executing tools. Return one "
        "planner_action.v2 JSON object only. Every action must cite an active memory, "
        "rule IDs, and source spans. Skills are reusable and may be called again for "
        "a different report, section, question, or cross-check; make the new objective "
        "and expected state delta explicit. current_phase must echo the patient state."
    )
    payload["required_output_fields"] = list(PLANNER_OUTPUT_FIELDS)
    payload["planner_action_v2_shape"] = {
        "schema_version": "planner_action.v2",
        "current_phase": "diagnostic_workup",
        "proposed_phase": None,
        "missing_information": [],
        "actions": [
            {
                "objective": "one concrete next objective",
                "action_type": "evidence_gathering",
                "required_skills": [],
                "preconditions": [],
                "expected_state_delta": [],
                "provenance": [
                    {
                        "memory_id": "active memory ID",
                        "rule_ids": ["rule ID"],
                        "source_spans": ["source span ID"],
                        "guideline_id": "target guideline ID",
                        "version": "target guideline version",
                    }
                ],
            }
        ],
        "blocked_actions": [],
        "should_stop": False,
        "reason": "brief state-conditioned rationale",
    }
    return (
        TASK_TOKENS["PLAN"]
        + "\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        + "\n"
    )


def _decoder_repair_prompt(
    patient_state: Mapping[str, Any],
    memories: list[dict[str, Any]],
    raw_text: str,
    parse_error: str,
    routing_audit: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "task": "repair_latent_guideline_planner_json",
        "parse_error": parse_error,
        "previous_decoder_output": _truncate_text(raw_text, 1200),
        "patient_state": _compact_patient_state_for_prompt(patient_state),
        "guideline_memory_candidates": [
            _memory_prompt_summary(memory) for memory in memories
        ],
        "required_output_schema": list(PLANNER_OUTPUT_FIELDS),
        "instruction": (
            "Return one corrected planner_action.v2 JSON object only. Keep all V2 "
            "fields, no prose or invented evidence. Every action must cite one of the "
            "active guideline memory candidates with rule IDs and source spans."
        ),
    }
    return (
        TASK_TOKENS["PLAN"]
        + "\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        + "\n"
    )


def _decoder_strict_repair_prompt(
    patient_state: Mapping[str, Any],
    memories: list[dict[str, Any]],
    raw_text: str,
    repair_raw_text: str,
    parse_error: str,
) -> str:
    metadata = [_memory_metadata_summary(memory) for memory in memories]
    selected = metadata[0] if metadata else {}
    payload = {
        "task": "strict_planner_json_repair",
        "parse_error": parse_error,
        "patient_state_minimal": _minimal_patient_state(patient_state),
        "selected_guideline_memory": selected,
        "retrieved_guideline_memory_ids": [
            item.get("guideline_memory_id") for item in metadata
        ],
        "previous_outputs": {
            "first": _truncate_text(raw_text, 500),
            "repair": _truncate_text(repair_raw_text, 500),
        },
        "instruction": (
            "Return exactly one compact planner_action.v2 JSON object. No markdown or "
            "prose. Use [] or null where the V2 schema permits it."
        ),
        "schema": {
            "schema_version": "planner_action.v2",
            "current_phase": patient_state.get("current_phase"),
            "proposed_phase": None,
            "missing_information": [],
            "actions": [
                {
                    "objective": "one concrete next objective",
                    "action_type": "evidence_gathering",
                    "required_skills": [],
                    "preconditions": [],
                    "expected_state_delta": [],
                    "provenance": [
                        {
                            "memory_id": selected.get("guideline_memory_id"),
                            "rule_ids": list(selected.get("source_rule_ids") or [])[:1],
                            "source_spans": list(selected.get("source_span_ids") or [])[:1],
                            "guideline_id": selected.get("guideline_id"),
                            "version": selected.get("version"),
                        }
                    ],
                }
            ],
            "blocked_actions": [],
            "should_stop": False,
            "reason": "brief state-conditioned guideline rationale",
        },
    }
    return (
        TASK_TOKENS["PLAN"]
        + "\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        + "\n"
    )


def _minimal_patient_state(patient_state: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "case_id",
        "cancer_family",
        "disease_subtype",
        "cancer_type",
        "current_phase",
        "decision_date",
        "guideline_context",
        "known_diagnosis",
        "diagnosis",
        "known_stage",
        "stage",
        "known_biomarkers",
        "risk_stratification",
        "completed_skills",
        "completed_actions",
        "treatment_history",
        "current_treatment_line",
        "evidence_ledger",
        "last_transition",
        "missing_information",
        "blocked_pathways",
        "blocked_actions",
        "available_modalities",
    )
    return {
        key: patient_state.get(key)
        for key in keys
        if patient_state.get(key) not in (None, "", [], {})
    }


def _compact_patient_state_for_prompt(
    patient_state: Mapping[str, Any],
) -> dict[str, Any]:
    keys = (
        "schema_version",
        "case_id",
        "cancer_family",
        "disease_subtype",
        "cancer_type",
        "current_phase",
        "decision_date",
        "guideline_context",
        "current_decision_stage",
        "known_diagnosis",
        "diagnosis",
        "known_stage",
        "stage",
        "known_biomarkers",
        "confirmed_evidence",
        "completed_skills",
        "completed_actions",
        "treatment_history",
        "current_treatment_line",
        "evidence_ledger",
        "last_transition",
        "pending_actions",
        "missing_information",
        "unresolved_information",
        "blocked_pathways",
        "available_modalities",
    )
    return {
        key: _bounded_prompt_value(patient_state.get(key))
        for key in keys
        if patient_state.get(key) not in (None, "", [], {})
    }


def _memory_prompt_summary(memory: Mapping[str, Any]) -> dict[str, Any]:
    summary = _memory_metadata_summary(memory)
    result = {
        "guideline_memory_id": summary.get("guideline_memory_id"),
        "guideline_id": summary.get("guideline_id"),
        "version": summary.get("version"),
        "title": summary.get("h1_title") or summary.get("section"),
        "role": summary.get("routing_memory_role"),
        "weight": summary.get("routing_gate_weight") or summary.get("score"),
        "pages": _source_pages(summary),
        "source_rule_ids": list(summary.get("source_rule_ids") or [])[:5],
        "source_span_ids": list(summary.get("source_span_ids") or [])[:5],
    }
    return {key: value for key, value in result.items() if value not in (None, "", [])}


def _compact_trajectory_for_prompt(
    trajectory_plan: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    history = _history_sequence(trajectory_plan)
    result: list[dict[str, Any]] = []
    for item in history[-2:]:
        action = item.get("planner_output") if isinstance(item, Mapping) else None
        source = action if isinstance(action, Mapping) else item
        compact = {
            key: _bounded_prompt_value(source.get(key))
            for key in (
                "schema_version",
                "current_phase",
                "proposed_phase",
                "actions",
                "should_stop",
            )
            if source.get(key) not in (None, "", [], {})
        }
        if compact:
            result.append(compact)
    return result


def _bounded_prompt_value(value: Any, *, limit: int = 12) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _bounded_prompt_value(item, limit=limit)
            for key, item in list(value.items())[:limit]
        }
    if isinstance(value, (list, tuple)):
        return [_bounded_prompt_value(item, limit=limit) for item in value[:limit]]
    if isinstance(value, str):
        return _truncate_text(value, 500)
    return value


def _memory_metadata_summary(memory: Mapping[str, Any]) -> dict[str, Any]:
    metadata = memory.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    source_rule_ids = _string_list_for_metadata(metadata.get("source_rule_ids"))
    source_span_ids = _string_list_for_metadata(metadata.get("source_span_ids"))
    return {
        "score": memory.get("score"),
        "guideline_memory_id": memory.get("guideline_memory_id"),
        "guideline_id": metadata.get("guideline_id"),
        "version": metadata.get("version"),
        "cancer_type": metadata.get("cancer_type"),
        "chapter": metadata.get("chapter"),
        "section": metadata.get("section"),
        "h1_title": metadata.get("h1_title"),
        "source_rule_ids": source_rule_ids[:5],
        "source_rule_id_count": len(source_rule_ids),
        "source_span_ids": source_span_ids[:5],
        "source_span_id_count": len(source_span_ids),
        "page_start": metadata.get("page_start"),
        "page_end": metadata.get("page_end"),
        "routing_memory_role": metadata.get("routing_memory_role"),
        "routing_seed_score": metadata.get("routing_seed_score"),
        "routing_gate_weight": metadata.get("routing_gate_weight"),
    }


def _parse_planner_json(raw_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            candidate = _unwrap_planner_payload(payload)
            if isinstance(candidate, dict) and _has_planner_fields(candidate):
                return candidate
    except json.JSONDecodeError:
        pass
    payload = _parse_python_literal_dict(raw_text)
    if isinstance(payload, dict):
        candidate = _unwrap_planner_payload(payload)
        if isinstance(candidate, dict) and _has_planner_fields(candidate):
            return candidate
    for candidate in _extract_json_objects(raw_text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            payload = _parse_python_literal_dict(candidate)
        if isinstance(payload, dict):
            unwrapped = _unwrap_planner_payload(payload)
            if isinstance(unwrapped, dict) and _has_planner_fields(unwrapped):
                return dict(unwrapped)
    partial = _parse_required_fields_from_partial_json(raw_text)
    if partial is not None:
        return partial
    raise LatentPlannerError(
        "Planner decoder did not return a JSON object with the required planner fields."
    )


def _parse_python_literal_dict(text: str) -> dict[str, Any] | None:
    try:
        payload = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return None
    return payload if isinstance(payload, dict) else None


def _unwrap_planner_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if _has_planner_fields(payload):
        return payload
    for key in (
        "planner_output",
        "planner_next_step",
        "next_step_plan",
        "plan",
        "output",
        "result",
    ):
        value = payload.get(key)
        if isinstance(value, Mapping) and _has_planner_fields(value):
            return value
    return payload


def _has_planner_fields(payload: Mapping[str, Any]) -> bool:
    return all(field in payload for field in PLANNER_OUTPUT_FIELDS)


def _parse_required_fields_from_partial_json(raw_text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    payload: dict[str, Any] = {}
    for field in PLANNER_OUTPUT_FIELDS:
        value = _decode_json_value_for_key(raw_text, field, decoder)
        if value is None:
            return None
        payload[field] = value
    return payload


def _decode_json_value_for_key(
    raw_text: str,
    field: str,
    decoder: json.JSONDecoder,
) -> Any:
    marker = json.dumps(field, ensure_ascii=False)
    search_from = 0
    while True:
        key_index = raw_text.find(marker, search_from)
        if key_index < 0:
            return None
        colon_index = raw_text.find(":", key_index + len(marker))
        if colon_index < 0:
            return None
        value_start = colon_index + 1
        while value_start < len(raw_text) and raw_text[value_start].isspace():
            value_start += 1
        try:
            value, _ = decoder.raw_decode(raw_text, value_start)
            return value
        except json.JSONDecodeError:
            search_from = key_index + len(marker)


def _extract_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    search_from = 0
    while search_from < len(text):
        try:
            item, end = _extract_json_object(text, search_from=search_from)
        except LatentPlannerError:
            break
        objects.append(item)
        search_from = max(end, search_from + 1)
    return objects


def _extract_json_object(text: str, *, search_from: int = 0) -> tuple[str, int]:
    start = text.find("{", search_from)
    if start < 0:
        raise LatentPlannerError("Planner decoder did not return a JSON object.")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1], index + 1
    raise LatentPlannerError("Planner decoder returned incomplete JSON.")


def _normalize_planner_output(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [field for field in PLANNER_OUTPUT_FIELDS if field not in payload]
    if missing:
        raise LatentPlannerError(
            "planner_action.v2 decoder output is missing required fields: "
            + ", ".join(missing)
        )
    # Runtime intentionally performs no planner_action.v2 schema, phase,
    # precondition, guideline-version, or provenance validation. Those checks
    # remain available to offline dataset audit and evaluation only.
    return {field: payload[field] for field in PLANNER_OUTPUT_FIELDS}


def _string_list_for_metadata(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value is None:
        return []
    return [str(value)]


def _truncate_text(text: Any, max_chars: int) -> str:
    value = str(text)
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"...[truncated {len(value) - max_chars} chars]"
