"""Training data and trainer for latent guideline routing modules."""

from __future__ import annotations

import gc
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from guideline_planner.artifacts import sha256_path
from guideline_planner.dynamic_anchor_attention_losses import (
    anchor_preservation_loss,
    attention_entropy_regularization,
    identity_loss,
    planner_distillation_loss,
)
from guideline_planner.io_utils import read_jsonl, write_json
from guideline_planner.latent_decoder import (
    LatentMemoryQueryEncoder,
    LatentPlannerDecoder,
)
from guideline_planner.latent_memory_retriever import GuidelineMemoryBank
from guideline_planner.memory import load_memory_store_meta
from guideline_planner.modeling import (
    _get_input_embeddings,
    enable_non_reentrant_gradient_checkpointing,
    encode_prompt_and_full_target,
)
from guideline_planner.planner import (
    _decoder_prompt,
    _parse_planner_json,
    _routing_memories_for_prompt,
)
from guideline_planner.progress import ProgressReporter
from guideline_planner.routing_config import LatentGuidelineRoutingConfig
from guideline_planner.routing_losses import (
    combine_routing_losses,
    gate_margin_loss,
    gate_utility_loss,
    provenance_multilabel_loss,
    retrieval_contrastive_loss,
    sparse_gate_loss,
)
from guideline_planner.routing_pipeline import LatentGuidelineRoutingPipeline
from guideline_planner.routing_types import PatientState
from guideline_planner.sampling import (
    PlannerTrajectorySampler,
    balanced_validation_records,
)
from guideline_planner.schemas_v2 import validate_planner_action_v2
from guideline_planner.state_encoder import serialize_patient_state
from guideline_planner.training_checkpoint import (
    load_latest_training_checkpoint,
    refuse_completed_output,
    save_training_checkpoint,
    training_config_hash,
)


@dataclass
class RoutingTrainerConfig:
    train_data_path: str
    memory_dir: str
    decoder_artifact_dir: str
    output_dir: str
    routing_config_path: str
    device: str = "cuda"
    model_dtype: str = "bf16"
    max_steps: int = 1000
    learning_rate: float = 1e-4
    max_decoder_tokens: int = 2048
    gradient_accumulation_steps: int = 1
    eval_steps: int = 100
    save_steps: int = 500
    keep_last_checkpoints: int = 2
    auto_resume: bool = True
    seed: int = 17
    generation_eval_steps: int = 500
    generation_eval_examples: int = 8
    generation_max_new_tokens: int = 1024
    training_memory_strategy: str = "non_reentrant_v1"


class _FrozenQueryCache:
    def __init__(self, values: Mapping[str, Any]) -> None:
        self.values = dict(values)

    def encode_query(self, query: str, *, embedding_dim: int | None = None) -> Any:
        if query not in self.values:
            raise RuntimeError(
                "Router attempted to encode a state absent from the frozen query cache."
            )
        value = self.values[query]
        if embedding_dim is not None and int(value.shape[-1]) != int(embedding_dim):
            raise RuntimeError("Frozen query-cache dimension mismatch.")
        return value


class LatentRoutingTrainer:
    """Train routing modules while freezing the exact latent decoder by default."""

    def __init__(self, config: RoutingTrainerConfig) -> None:
        self.config = config

    def train(self) -> dict[str, Any]:
        torch.manual_seed(self.config.seed)
        if self.config.training_memory_strategy != "non_reentrant_v1":
            raise ValueError(
                "Router only supports training_memory_strategy="
                "'non_reentrant_v1'."
            )
        train_records, validation_records, routing_manifest = _load_routing_splits(
            self.config.train_data_path
        )
        _validate_routing_records([*train_records, *validation_records])
        if not train_records:
            raise ValueError("Router V2 has no train-split records.")
        routing_config = LatentGuidelineRoutingConfig.from_value(
            self.config.routing_config_path
        )
        if not routing_config.enabled:
            raise ValueError("Routing training requires enabled=true in its config.")
        retrieval_steps = max(int(routing_config.training.retrieval_pretrain_steps), 0)
        warmup_steps = (
            max(int(routing_config.training.identity_warmup_steps), 0)
            if routing_config.memory_fusion.strategy == "dynamic_anchor_attention"
            else 0
        )
        distillation_steps = (
            max(int(routing_config.training.distillation_steps), 0)
            if routing_config.memory_fusion.strategy == "dynamic_anchor_attention"
            else 0
        )
        joint_steps = max(
            int(round(self.config.max_steps * routing_config.training.joint_calibration_fraction)),
            1,
        )
        joint_start = self.config.max_steps - joint_steps
        if self.config.max_steps <= retrieval_steps + warmup_steps + distillation_steps or joint_start <= retrieval_steps + warmup_steps + distillation_steps:
            raise ValueError(
                "train-router needs retrieval pretraining, K=1 identity warm-up, "
                "variable-K DAA, and a final joint calibration window; "
                f"max_steps={self.config.max_steps}."
            )
        memory_meta = load_memory_store_meta(self.config.memory_dir)
        bank = GuidelineMemoryBank.from_directory(self.config.memory_dir)
        progress = ProgressReporter(
            "train-router",
            max(int(self.config.max_steps), 1),
            unit="step",
        )
        progress.message("loading the frozen Memory Encoder query runtime")
        query_runtime = LatentMemoryQueryEncoder.from_memory_store(
            self.config.memory_dir,
            memory_meta,
            device=self.config.device,
            model_dtype=self.config.model_dtype,
        )
        query_cache = _build_query_cache(
            query_runtime,
            [*train_records, *validation_records],
            embedding_dim=bank.retrieval_dim,
            progress_label="router-query-cache",
        )
        del query_runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        progress.message("loading the Planner Decoder and initializing Router/DAA")
        decoder = LatentPlannerDecoder.from_artifacts(
            self.config.memory_dir,
            self.config.decoder_artifact_dir,
            device=self.config.device,
            model_dtype=self.config.model_dtype,
            adapter_trainable=True,
        )
        checkpointing = enable_non_reentrant_gradient_checkpointing(
            decoder.bundle.model
        )
        progress.message(
            "enabled non-reentrant gradient checkpointing for Router Decoder"
        )
        pipeline = LatentGuidelineRoutingPipeline(
            memory_bank=bank,
            query_encoder=query_cache.encode_query,
            config=routing_config,
            device=decoder.device,
        )
        model = decoder.bundle.model
        _configure_decoder_training(decoder, False)
        parameters = [parameter for parameter in pipeline.parameters() if parameter.requires_grad]
        decoder_parameters = [
            parameter for name, parameter in model.named_parameters() if "lora_" in name
        ]
        decoder_parameters.extend(decoder.memory_to_decoder_bridge.parameters())
        if not parameters or not decoder_parameters:
            raise RuntimeError("No trainable routing or decoder parameters were found.")
        optimizer = torch.optim.AdamW(
            [
                {"params": parameters, "lr": self.config.learning_rate},
                {
                    "params": decoder_parameters,
                    "lr": self.config.learning_rate
                    * routing_config.training.joint_decoder_lr_multiplier,
                },
            ]
        )
        all_parameters = [*parameters, *decoder_parameters]
        schedule, sampling_audit = PlannerTrajectorySampler(
            train_records,
            seed=self.config.seed,
        ).sample(max(int(self.config.max_steps), 1))
        output_dir = Path(self.config.output_dir)
        refuse_completed_output(output_dir, "routing_checkpoint.pt")
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.jsonl"
        lineage = {
            "source_dataset_hash": routing_manifest.get("source_dataset_hash"),
            "memory_store_fingerprint": memory_meta.get("memory_store_fingerprint"),
            "planner_decoder_adapter_hash": decoder.artifact_meta.get(
                "planner_decoder_adapter_hash"
            ),
            "memory_to_decoder_bridge_hash": decoder.artifact_meta.get(
                "memory_to_decoder_bridge_hash"
            ),
        }
        effective_config_hash = training_config_hash(asdict(self.config))
        resumed = None
        if self.config.auto_resume:
            resumed = load_latest_training_checkpoint(
                torch_module=torch,
                output_dir=output_dir,
                stage="router_daa",
                config_hash=effective_config_hash,
                lineage=lineage,
                map_location=decoder.device,
            )
        start_step = 0
        resumed_last_event: dict[str, Any] | None = None
        if resumed is not None:
            pipeline.load_state_dict(resumed["routing_state"])
            _load_decoder_trainable_state(decoder, resumed["decoder_trainable_state"])
            optimizer.load_state_dict(resumed["optimizer_state"])
            start_step = int(resumed["global_step"])
            resumed_last_event = dict(resumed.get("last_train_event") or {}) or None
        else:
            metrics_path.write_text("", encoding="utf-8")
        accumulation = max(int(self.config.gradient_accumulation_steps), 1)
        optimizer.zero_grad(set_to_none=True)
        log: list[dict[str, Any]] = [resumed_last_event] if resumed_last_event else []
        last_generation_validation: dict[str, Any] | None = None
        pipeline.train()
        current_phase: str | None = None
        progress.start(start_step, status="resumed" if start_step else "training")
        for step in range(start_step, max(int(self.config.max_steps), 1)):
            phase = _training_phase(
                step,
                retrieval_steps,
                warmup_steps,
                distillation_steps,
                joint_start,
            )
            if phase != current_phase:
                _configure_decoder_training(
                    decoder,
                    bool(
                        phase == "joint_calibration"
                        and routing_config.training.train_decoder_lora
                    ),
                )
                current_phase = phase
            record = dict(schedule[step])
            record["_variant_index"] = step
            breakdown = _routing_loss_for_record(
                pipeline,
                decoder,
                record,
                routing_config,
                max_decoder_tokens=self.config.max_decoder_tokens,
                phase=phase,
            )
            (breakdown.total / accumulation).backward()
            if (step + 1) % accumulation == 0 or step + 1 == self.config.max_steps:
                torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            event = {
                "step": step + 1,
                "split": "train",
                "phase": phase,
                **breakdown.detached_metrics(),
            }
            log.append(event)
            _append_jsonl(metrics_path, event)
            progress.update(
                step + 1,
                metrics={"phase": phase, "loss": event.get("total")},
            )
            if _should_eval(step + 1, self.config, validation_records):
                progress.message(f"running validation at step {step + 1}")
                evaluation = _evaluate(
                    pipeline,
                    decoder,
                    validation_records,
                    routing_config,
                    max_decoder_tokens=self.config.max_decoder_tokens,
                    progress_label=f"router-eval@{step + 1}",
                    run_generation=(
                        self.config.generation_eval_steps > 0
                        and (
                            (step + 1) % self.config.generation_eval_steps == 0
                            or step + 1 == self.config.max_steps
                        )
                    ),
                    generation_eval_examples=self.config.generation_eval_examples,
                    generation_max_new_tokens=self.config.generation_max_new_tokens,
                )
                if evaluation.get("generation_schema_pass_rate") is not None:
                    last_generation_validation = {
                        key: value
                        for key, value in evaluation.items()
                        if key.startswith("generation_")
                    }
                evaluation["step"] = step + 1
                _append_jsonl(metrics_path, evaluation)
                progress.update(
                    step + 1,
                    metrics={
                        "status": "validated",
                        "val_loss": evaluation.get("total"),
                        "schema": evaluation.get("generation_schema_pass_rate"),
                    },
                    force=True,
                )
            if self.config.save_steps > 0 and (
                (step + 1) % self.config.save_steps == 0
                or step + 1 == self.config.max_steps
            ):
                progress.message(f"saving checkpoint at step {step + 1}")
                save_training_checkpoint(
                    torch_module=torch,
                    output_dir=output_dir,
                    stage="router_daa",
                    global_step=step + 1,
                    config_hash=effective_config_hash,
                    lineage=lineage,
                    keep_last=self.config.keep_last_checkpoints,
                    state={
                        "routing_state": pipeline.state_dict(),
                        "decoder_trainable_state": _decoder_trainable_state(decoder),
                        "optimizer_state": optimizer.state_dict(),
                        "sampling_audit": sampling_audit,
                        "last_train_event": event,
                    },
                )
        pipeline.eval()
        progress.message("saving final Router/DAA and calibrated Decoder artifacts")
        created_at = datetime.now(timezone.utc).isoformat()
        summary = {
            "steps": self.config.max_steps,
            "training_examples": len(train_records),
            "validation_examples": len(validation_records),
            "final_train_loss": log[-1]["total"] if log else None,
            "base_model": memory_meta.get("base_model"),
            "tokenizer_path": memory_meta.get("tokenizer_path"),
            "retrieval_projection_path": memory_meta.get("retrieval_projection_path"),
            "sampling_audit": sampling_audit,
            "source_dataset_hash": routing_manifest.get("source_dataset_hash"),
            "routing_dataset_manifest": routing_manifest,
            "resumed_from_step": start_step,
            "active_top_k": routing_config.gate.active_top_k,
            "memory_fusion_strategy": routing_config.memory_fusion.strategy,
            "daa_config": routing_config.to_dict()["memory_fusion"].get(
                "dynamic_anchor_attention"
            ),
            "daa_trained": (
                routing_config.memory_fusion.strategy == "dynamic_anchor_attention"
            ),
            "trained_prefix_length": _configured_prefix_length(
                routing_config,
                int(memory_meta.get("memory_tokens") or 0),
            ),
            "train_decoder_lora": routing_config.training.train_decoder_lora,
            "retrieval_pretrain_steps": retrieval_steps,
            "identity_warmup_steps": warmup_steps,
            "distillation_steps": distillation_steps,
            "variable_k_daa_steps": joint_start
            - retrieval_steps
            - warmup_steps
            - distillation_steps,
            "joint_calibration_steps": self.config.max_steps - joint_start,
            "trainable_parameter_count": sum(
                int(parameter.numel()) for parameter in parameters
            ),
            "routing_parameter_count": sum(
                int(parameter.numel()) for parameter in pipeline.parameters()
            ),
            "generation_validation": last_generation_validation,
            "max_decoder_tokens": int(self.config.max_decoder_tokens),
            "gradient_checkpointing": checkpointing,
            "training_memory_strategy": self.config.training_memory_strategy,
        }
        checkpoint = pipeline.checkpoint_payload(
            trained=True,
            created_at=created_at,
            created_from_run=str(output_dir),
            training_summary=summary,
        )
        checkpoint.update(
            {
                "base_model": memory_meta.get("base_model"),
                "base_model_snapshot_path": memory_meta.get("base_model_snapshot_path"),
                "memory_store_lineage_fingerprint": memory_meta.get(
                    "memory_store_fingerprint"
                ),
                "memory_encoder_adapter_hash": memory_meta.get(
                    "memory_encoder_adapter_hash"
                ),
                "tokenizer_path": memory_meta.get("tokenizer_path"),
                "retrieval_projection_path": memory_meta.get("retrieval_projection_path"),
            }
        )
        checkpoint["routing_state_dict"] = {
            key: value.detach().cpu() for key, value in pipeline.state_dict().items()
        }
        checkpoint_path = output_dir / "routing_checkpoint.pt"
        if not hasattr(model, "save_pretrained"):
            raise RuntimeError("Router calibration requires a PEFT Planner Decoder.")
        decoder_adapter = output_dir / "planner_decoder_adapter"
        bridge_path = output_dir / "memory_to_decoder_bridge.pt"
        tokenizer_path = output_dir / "tokenizer"
        model.save_pretrained(decoder_adapter)
        source_tokenizer = Path(str(decoder.artifact_meta["tokenizer_path"]))
        if not source_tokenizer.is_absolute():
            source_tokenizer = Path(self.config.decoder_artifact_dir) / source_tokenizer
        shutil.copytree(source_tokenizer, tokenizer_path, dirs_exist_ok=True)
        torch.save({"state_dict": decoder.memory_to_decoder_bridge.state_dict()}, bridge_path)
        auxiliary_path = None
        auxiliary_value = decoder.artifact_meta.get("auxiliary_heads_path")
        if isinstance(auxiliary_value, str) and auxiliary_value:
            source_auxiliary = Path(auxiliary_value)
            if not source_auxiliary.is_absolute():
                source_auxiliary = (
                    Path(self.config.decoder_artifact_dir) / source_auxiliary
                )
            if not source_auxiliary.is_file():
                raise FileNotFoundError(
                    f"Planner Decoder auxiliary heads are missing: {source_auxiliary}"
                )
            auxiliary_path = output_dir / "planner_aux_heads.pt"
            shutil.copy2(source_auxiliary, auxiliary_path)
        decoder_meta = {
            **dict(decoder.artifact_meta),
            "artifact_role": "planner_decoder",
            "created_at": created_at,
            "memory_store_fingerprint": memory_meta.get("memory_store_fingerprint"),
            "memory_encoder_adapter_hash": memory_meta.get("memory_encoder_adapter_hash"),
            "planner_decoder_adapter_path": decoder_adapter.name,
            "planner_decoder_adapter_hash": sha256_path(decoder_adapter),
            "memory_to_decoder_bridge_path": bridge_path.name,
            "memory_to_decoder_bridge_hash": sha256_path(bridge_path),
            "tokenizer_path": tokenizer_path.name,
            "tokenizer_hash": sha256_path(tokenizer_path),
            "calibrated_by_routing": True,
        }
        if auxiliary_path is not None:
            decoder_meta["auxiliary_heads_path"] = auxiliary_path.name
            decoder_meta["auxiliary_heads_hash"] = sha256_path(auxiliary_path)
        write_json(output_dir / "planner_decoder_meta.json", decoder_meta)
        checkpoint.update(
            {
                "planner_decoder_adapter_hash": decoder_meta[
                    "planner_decoder_adapter_hash"
                ],
                "memory_to_decoder_bridge_hash": decoder_meta[
                    "memory_to_decoder_bridge_hash"
                ],
                "planner_decoder_meta_path": "planner_decoder_meta.json",
            }
        )
        summary["planner_decoder_artifact_dir"] = str(output_dir.resolve())
        checkpoint["training_summary"] = dict(summary)
        torch.save(checkpoint, checkpoint_path)
        write_json(output_dir / "routing_training_config.json", asdict(self.config))
        write_json(output_dir / "routing_training_summary.json", summary)
        progress.finish(metrics={"loss": log[-1]["total"] if log else None})
        return {
            "output_dir": str(output_dir),
            "routing_checkpoint": str(checkpoint_path),
            **summary,
        }


def train_latent_router(config: RoutingTrainerConfig | Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, RoutingTrainerConfig):
        config = RoutingTrainerConfig(**dict(config))
    return LatentRoutingTrainer(config).train()


def _configured_prefix_length(
    config: LatentGuidelineRoutingConfig,
    memory_tokens: int,
) -> int:
    if config.memory_fusion.strategy in {"dynamic_anchor_attention", "top1_only"}:
        return int(memory_tokens)
    active_count = max(int(config.gate.active_top_k), 1)
    return active_count * int(memory_tokens)


def _training_phase(
    step: int,
    retrieval_steps: int,
    warmup_steps: int,
    distillation_steps: int,
    joint_start: int,
) -> str:
    if step < retrieval_steps:
        return "retrieval_pretrain"
    if step < retrieval_steps + warmup_steps:
        return "identity_warmup"
    if step < retrieval_steps + warmup_steps + distillation_steps:
        return "planner_distillation"
    if step < joint_start:
        return "variable_k_daa"
    return "joint_calibration"


def _identity_warmup_loss(
    pipeline: LatentGuidelineRoutingPipeline,
    route: Any,
    config: LatentGuidelineRoutingConfig,
) -> Any:
    if pipeline.fusion_strategy != "dynamic_anchor_attention":
        raise RuntimeError("Identity warm-up is only valid for DAA fusion.")
    anchor_id = str(route.fused_memory.diagnostics.get("anchor_memory_id") or "")
    activation = next(
        (item for item in route.active_memories if item.memory_id == anchor_id),
        route.active_memories[0],
    )
    weight = route.state_repr.new_ones((1,))
    fused, prefix = pipeline.compose_prefix(
        route.state_repr,
        [activation],
        weights=weight,
    )
    target = pipeline.memory_bank.lookup[activation.memory_id].slots.to(prefix)
    loss = identity_loss(
        prefix,
        target,
        cosine_weight=config.training.lambda_identity_cosine,
    )
    zero = prefix.sum() * 0.0
    anchor_loss = (
        anchor_preservation_loss(fused.slots, fused.anchor_slots)
        if fused.anchor_slots is not None
        else zero
    )
    return combine_routing_losses(
        action=zero,
        identity=loss,
        anchor=anchor_loss,
        weights={
            "action": 0.0,
            "identity": config.training.lambda_identity,
            "anchor": config.training.lambda_anchor,
        },
    )


def _routing_loss_for_record(
    pipeline: LatentGuidelineRoutingPipeline,
    decoder: LatentPlannerDecoder,
    record: Mapping[str, Any],
    config: LatentGuidelineRoutingConfig,
    *,
    max_decoder_tokens: int,
    phase: str = "variable_k_daa",
) -> Any:
    patient_state = record.get("patient_state", {})
    if not isinstance(patient_state, Mapping):
        raise ValueError("Routing record patient_state must be an object.")
    history = record.get("trajectory_history", [])
    if not isinstance(history, list):
        history = []
    supervised_candidates = list(
        dict.fromkeys(
            [
                *_string_values(record.get("strong_positive_ids")),
                *_string_values(record.get("weak_positive_ids")),
                *_string_values(record.get("hard_negative_ids")),
                *_string_values(record.get("easy_negative_ids")),
            ]
        )
    )
    route = pipeline(
        patient_state,
        history,
        candidate_memory_ids=supervised_candidates,
    )
    if phase == "identity_warmup":
        return _identity_warmup_loss(pipeline, route, config)
    prompt = _decoder_prompt(
        patient_state,
        history,
        _routing_memories_for_prompt(route),
        route.audit_dict(),
    )
    variants = record.get("target_variants")
    if isinstance(variants, list) and variants:
        target = json.dumps(
            variants[int(record.get("_variant_index") or 0) % len(variants)],
            ensure_ascii=False,
            sort_keys=True,
        )
    else:
        target = str(record.get("target") or "")
    student_output, target_mask = _causal_prefix_forward(
        decoder,
        route.decoder_prefix,
        prompt,
        target,
        max_decoder_tokens=max_decoder_tokens,
    )
    action_loss = student_output.loss
    distillation_loss = action_loss.new_zeros(())
    if phase == "planner_distillation":
        anchor_id = str(route.fused_memory.diagnostics.get("anchor_memory_id") or "")
        if not anchor_id or anchor_id not in pipeline.memory_bank.lookup:
            raise RuntimeError("DAA distillation could not resolve the anchor memory.")
        teacher_prefix = pipeline.memory_bank.lookup[anchor_id].slots.to(
            route.decoder_prefix
        )
        with torch.no_grad():
            teacher_output, _ = _causal_prefix_forward(
                decoder,
                teacher_prefix,
                prompt,
                target,
                max_decoder_tokens=max_decoder_tokens,
            )
        distillation_loss = planner_distillation_loss(
            student_output.logits,
            teacher_output.logits,
            token_mask=target_mask,
            temperature=config.training.distillation_temperature,
        )
    strong_positive_order = _string_values(record.get("strong_positive_ids"))
    weak_positive_order = _string_values(record.get("weak_positive_ids"))
    positive_order = list(
        dict.fromkeys(
            [
                *strong_positive_order,
                *weak_positive_order,
                *_string_values(record.get("positive_memory_ids")),
            ]
        )
    )
    strong_positive_ids = set(strong_positive_order)
    weak_positive_ids = set(weak_positive_order)
    positive_ids = set(positive_order)
    hard_negative_ids = set(_string_values(record.get("hard_negative_ids")))
    easy_negative_ids = set(_string_values(record.get("easy_negative_ids")))
    negative_ids = [
        item
        for field in ("hard_negative_ids", "easy_negative_ids", "negative_memory_ids")
        for item in _string_values(record.get(field))
    ]
    retrieval_ids = list(dict.fromkeys([*positive_order, *negative_ids]))
    for candidate in route.merged_candidates:
        if candidate.memory_id not in retrieval_ids:
            retrieval_ids.append(candidate.memory_id)
        if len(retrieval_ids) >= max(config.retrieval.seed_top_k + len(positive_ids), 2):
            break
    retrieval_memories = [
        pipeline.memory_bank.lookup[memory_id]
        for memory_id in retrieval_ids
        if memory_id in pipeline.memory_bank.lookup
    ]
    projected_state = pipeline.retriever.state_projection(route.state_repr)
    if retrieval_memories and any(
        memory.memory_id in positive_ids for memory in retrieval_memories
    ):
        retrieval_keys = torch.stack(
            [memory.retrieval_key.to(projected_state) for memory in retrieval_memories],
            dim=0,
        )
        positive_mask = torch.tensor(
            [memory.memory_id in positive_ids for memory in retrieval_memories],
            dtype=torch.bool,
            device=projected_state.device,
        )
        positive_weights = torch.tensor(
            [
                1.0
                if memory.memory_id in strong_positive_ids
                else 0.5
                if memory.memory_id in weak_positive_ids
                else 1.0
                if memory.memory_id in positive_ids
                else 0.0
                for memory in retrieval_memories
            ],
            dtype=projected_state.dtype,
            device=projected_state.device,
        )
        candidate_weights = torch.tensor(
            [
                1.5
                if memory.memory_id in hard_negative_ids
                else 1.0
                if memory.memory_id in easy_negative_ids
                else 1.0
                for memory in retrieval_memories
            ],
            dtype=projected_state.dtype,
            device=projected_state.device,
        )
        retrieval_loss = retrieval_contrastive_loss(
            projected_state,
            retrieval_keys,
            positive_mask,
            positive_weights=positive_weights,
            candidate_weights=candidate_weights,
        )
    else:
        retrieval_loss = projected_state.sum() * 0.0
    gate_ids = [item.memory_id for item in route.gate_output.activations]
    gate_positive_mask = torch.tensor(
        [memory_id in positive_ids for memory_id in gate_ids],
        dtype=torch.bool,
        device=route.gate_output.logits.device,
    )
    gate_hard_negative_mask = torch.tensor(
        [memory_id in hard_negative_ids for memory_id in gate_ids],
        dtype=torch.bool,
        device=route.gate_output.logits.device,
    )
    provenance_loss = provenance_multilabel_loss(
        route.gate_output.logits,
        gate_positive_mask,
    )
    margin_loss = gate_margin_loss(
        route.gate_output.logits,
        gate_positive_mask,
        gate_hard_negative_mask,
        margin=config.training.gate_margin,
    )
    positive_gate_mass = float(
        route.gate_output.probabilities[gate_positive_mask].detach().sum().cpu()
    )
    positive_scores = route.gate_output.logits[gate_positive_mask]
    negative_scores = route.gate_output.logits[~gate_positive_mask]
    positive_top1_margin = (
        float((positive_scores.max() - negative_scores.max()).detach().cpu())
        if positive_scores.numel() and negative_scores.numel()
        else 0.0
    )
    utility_loss = action_loss.new_zeros(())
    if config.training.use_utility_distillation and len(route.active_memories) > 1:
        utility_loss = _utility_loss(
            pipeline,
            decoder,
            route,
            prompt,
            target,
            config,
            max_decoder_tokens=max_decoder_tokens,
            full_action_loss=action_loss,
        )
    sparse_loss = sparse_gate_loss(route.gate_output.probabilities)
    attention_entropy = attention_entropy_regularization(
        route.fused_memory.memory_attention_mass
    )
    if attention_entropy is None:
        attention_entropy = action_loss.new_zeros(())
    anchor_loss = (
        anchor_preservation_loss(
            route.fused_memory.slots,
            route.fused_memory.anchor_slots,
        )
        if route.fused_memory.anchor_slots is not None
        else action_loss.new_zeros(())
    )
    return combine_routing_losses(
        action=action_loss,
        retrieval=retrieval_loss,
        utility=utility_loss,
        provenance=provenance_loss,
        gate_margin=margin_loss,
        sparse=sparse_loss,
        distillation=distillation_loss,
        attention_entropy=attention_entropy,
        anchor=anchor_loss,
        weights={
            "action": 0.0
            if phase == "retrieval_pretrain"
            else config.training.lambda_action,
            "retrieval": config.training.lambda_retrieval,
            "utility": config.training.lambda_utility,
            "provenance": config.training.lambda_provenance,
            "gate_margin": config.training.lambda_gate_margin,
            "sparse": config.training.lambda_sparse,
            "distillation": config.training.lambda_distillation,
            "attention_entropy": config.training.lambda_attention_entropy,
            "anchor": config.training.lambda_anchor,
        },
        diagnostics={
            "normalized_gate_entropy": float(
                route.gate_output.diagnostics.get("normalized_entropy") or 0.0
            ),
            "positive_gate_mass": positive_gate_mass,
            "positive_top1_margin": positive_top1_margin,
        },
    )


def _utility_loss(
    pipeline: LatentGuidelineRoutingPipeline,
    decoder: LatentPlannerDecoder,
    route: Any,
    prompt: str,
    target: str,
    config: LatentGuidelineRoutingConfig,
    *,
    max_decoder_tokens: int,
    full_action_loss: torch.Tensor,
) -> torch.Tensor:
    count = min(
        max(config.training.utility_masks_per_sample, 1),
        len(route.active_memories),
    )
    mask_indices = route.gate_output.selected_weights.detach().argsort(descending=True)[:count]
    rewards_without = []
    selected_gate = []
    for index_tensor in mask_indices:
        index = int(index_tensor)
        remaining = [
            activation
            for item_index, activation in enumerate(route.active_memories)
            if item_index != index
        ]
        remaining_weights = torch.cat(
            [
                route.gate_output.selected_weights[:index],
                route.gate_output.selected_weights[index + 1 :],
            ]
        )
        remaining_weights = remaining_weights / remaining_weights.sum().clamp_min(1e-12)
        with torch.no_grad():
            _, prefix = pipeline.compose_prefix(
                route.state_repr,
                remaining,
                weights=remaining_weights,
            )
            without_loss = _causal_prefix_loss(
                decoder,
                prefix,
                prompt,
                target,
                max_decoder_tokens=max_decoder_tokens,
            )
        rewards_without.append(-without_loss.detach())
        selected_gate.append(route.gate_output.selected_weights[index])
    gate_loss, _ = gate_utility_loss(
        torch.stack(selected_gate),
        -full_action_loss.detach(),
        torch.stack(rewards_without),
        utility_temperature=config.training.utility_temperature,
    )
    return gate_loss


def _causal_prefix_loss(
    decoder: LatentPlannerDecoder,
    prefix: torch.Tensor,
    prompt: str,
    target: str,
    *,
    max_decoder_tokens: int,
) -> torch.Tensor:
    output, _ = _causal_prefix_forward(
        decoder,
        prefix,
        prompt,
        target,
        max_decoder_tokens=max_decoder_tokens,
    )
    return output.loss


def _causal_prefix_forward(
    decoder: LatentPlannerDecoder,
    prefix: torch.Tensor,
    prompt: str,
    target: str,
    *,
    max_decoder_tokens: int,
) -> tuple[Any, torch.Tensor]:
    bundle = decoder.bundle
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device
    prompt_ids, target_ids_list, _ = encode_prompt_and_full_target(
        tokenizer,
        prompt,
        target,
        max_decoder_tokens=max_decoder_tokens,
    )
    prompt_ids = prompt_ids.to(device)
    target_ids = torch.tensor([target_ids_list], dtype=torch.long, device=device)
    decoder_ids = torch.cat([prompt_ids["input_ids"], target_ids], dim=1)
    embeddings = _get_input_embeddings(model)
    decoder_embeds = embeddings(decoder_ids)
    model_dtype = decoder_embeds.dtype
    bridged_prefix = decoder.memory_to_decoder_bridge(
        prefix.to(
            device=device,
            dtype=next(decoder.memory_to_decoder_bridge.parameters()).dtype,
        )
    )
    prefix_batch = bridged_prefix.to(dtype=model_dtype).unsqueeze(0)
    inputs_embeds = torch.cat([prefix_batch, decoder_embeds], dim=1)
    labels = torch.full(
        (1, int(prefix_batch.shape[1]) + int(decoder_ids.shape[1])),
        -100,
        dtype=torch.long,
        device=device,
    )
    labels[:, int(prefix_batch.shape[1]) + int(prompt_ids["input_ids"].shape[1]) :] = target_ids
    output = model(inputs_embeds=inputs_embeds, labels=labels, use_cache=False)
    return output, labels.ne(-100)


def _configure_decoder_training(decoder: LatentPlannerDecoder, train_lora: bool) -> None:
    model = decoder.bundle.model
    for name, parameter in model.named_parameters():
        parameter.requires_grad = bool(train_lora and "lora_" in name)
    for parameter in decoder.memory_to_decoder_bridge.parameters():
        parameter.requires_grad = bool(train_lora)
    # Gradient checkpointing is conditioned on module.training by Transformers.
    # Keep the frozen Decoder in training mode so Router/DAA gradients through
    # its inputs are checkpointed; requires_grad still controls LoRA updates.
    model.train()
    decoder.memory_to_decoder_bridge.train(mode=bool(train_lora))


def _evaluate(
    pipeline: LatentGuidelineRoutingPipeline,
    decoder: LatentPlannerDecoder,
    records: Sequence[Mapping[str, Any]],
    config: LatentGuidelineRoutingConfig,
    *,
    max_decoder_tokens: int,
    progress_label: str | None = None,
    run_generation: bool = False,
    generation_eval_examples: int = 8,
    generation_max_new_tokens: int = 1024,
) -> dict[str, Any]:
    if not records:
        return {"split": "validation", "count": 0, "total": None}
    model = decoder.bundle.model
    model_was_training = bool(getattr(model, "training", False))
    pipeline.eval()
    model.eval()
    values = []
    progress = ProgressReporter(
        progress_label or "router-eval",
        len(records),
        unit="transition",
        enabled=progress_label is not None,
    )
    progress.start(status="validating")
    try:
        with torch.no_grad():
            for index, record in enumerate(records):
                value = _routing_loss_for_record(
                    pipeline,
                    decoder,
                    record,
                    config,
                    max_decoder_tokens=max_decoder_tokens,
                ).detached_metrics()
                values.append(value)
                progress.update(index + 1, metrics={"loss": value.get("total")})
    finally:
        pipeline.train()
        model.train(model_was_training)
    result = {
        "split": "validation",
        "count": len(values),
        **{
            key: sum(item[key] for item in values) / len(values)
            for key in values[0]
        },
    }
    if run_generation:
        result.update(
            _evaluate_generation_schema(
                pipeline,
                decoder,
                balanced_validation_records(
                    records,
                    max(int(generation_eval_examples), 1),
                    seed=17,
                ),
                max_new_tokens=generation_max_new_tokens,
                progress_label=(
                    f"{progress_label}-generation" if progress_label else None
                ),
            )
        )
    progress.finish(
        metrics={
            "status": "validated",
            "schema": result.get("generation_schema_pass_rate"),
        }
    )
    return result


def _evaluate_generation_schema(
    pipeline: LatentGuidelineRoutingPipeline,
    decoder: LatentPlannerDecoder,
    records: Sequence[Mapping[str, Any]],
    *,
    max_new_tokens: int,
    progress_label: str | None,
) -> dict[str, Any]:
    progress = ProgressReporter(
        progress_label or "router-generation-eval",
        len(records),
        unit="prediction",
        enabled=progress_label is not None,
    )
    progress.start(status="generating")
    pipeline_was_training = bool(pipeline.training)
    model_was_training = bool(decoder.bundle.model.training)
    bridge_was_training = bool(decoder.memory_to_decoder_bridge.training)
    pipeline.eval()
    decoder.bundle.model.eval()
    decoder.memory_to_decoder_bridge.eval()
    passed = 0
    failures: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for index, record in enumerate(records):
                patient_state = record["patient_state"]
                history = record.get("trajectory_history") or []
                candidate_ids = list(
                    dict.fromkeys(
                        [
                            *_string_values(record.get("strong_positive_ids")),
                            *_string_values(record.get("weak_positive_ids")),
                            *_string_values(record.get("hard_negative_ids")),
                            *_string_values(record.get("easy_negative_ids")),
                        ]
                    )
                )
                raw_text = ""
                route = None
                try:
                    route = pipeline(
                        patient_state,
                        history,
                        candidate_memory_ids=candidate_ids,
                    )
                    memories = _routing_memories_for_prompt(route)
                    prompt_text = _decoder_prompt(
                        patient_state,
                        history,
                        memories,
                        route.audit_dict(),
                    )
                    raw_text = decoder.generate_plan(
                        route.decoder_prefix,
                        prompt_text,
                        max_new_tokens=max_new_tokens,
                    )
                    payload = _parse_planner_json(raw_text)
                    validate_planner_action_v2(
                        payload,
                        patient_state=patient_state,
                        active_memories=memories,
                    )
                    passed += 1
                except Exception as exc:
                    failures.append(
                        {
                            "example_id": record.get("example_id"),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "raw_text": raw_text[:2000],
                        }
                    )
                route = None
                progress.update(
                    index + 1,
                    metrics={"schema_pass_rate": passed / (index + 1)},
                )
    finally:
        pipeline.train(pipeline_was_training)
        decoder.bundle.model.train(model_was_training)
        decoder.memory_to_decoder_bridge.train(bridge_was_training)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rate = passed / len(records) if records else 0.0
    progress.finish(metrics={"schema_pass_rate": rate})
    return {
        "generation_count": len(records),
        "generation_schema_pass_count": passed,
        "generation_schema_pass_rate": rate,
        "generation_failures": failures[:8],
    }


def _should_eval(
    step: int,
    config: RoutingTrainerConfig,
    validation_records: Sequence[Any],
) -> bool:
    if not validation_records or config.eval_steps <= 0:
        return False
    return step % config.eval_steps == 0 or step == config.max_steps


def _string_values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    return [str(value)]


def _validate_routing_records(records: Sequence[Mapping[str, Any]]) -> None:
    required = {
        "schema_version",
        "example_id",
        "split",
        "patient_state",
        "target_variants",
        "action_set",
        "strong_positive_ids",
        "hard_negative_ids",
        "review_status",
    }
    for index, record in enumerate(records):
        missing = sorted(required - set(record))
        if record.get("schema_version") != "routing_example.v2" or missing:
            raise ValueError(
                f"routing record[{index}] is not routing_example.v2; missing={missing}."
            )
        if not _string_values(record.get("strong_positive_ids")):
            raise ValueError(f"routing record[{index}] has no strong positive memory.")
        if not _string_values(record.get("hard_negative_ids")):
            raise ValueError(f"routing record[{index}] has no hard negative memory.")


def _load_routing_splits(
    path_value: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load only train/validation data and validate its admitted lineage."""

    path = Path(path_value)
    if path.is_dir():
        manifest_path = path / "manifest.json"
        train_path = path / "train.jsonl"
        validation_path = path / "validation.jsonl"
    else:
        manifest_path = path.with_suffix(".manifest.json")
        train_path = path
        validation_path = None
    if not manifest_path.is_file():
        raise ValueError("Router training requires routing_dataset_manifest.v2.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "routing_dataset_manifest.v2":
        raise ValueError("Router training received an unsupported dataset manifest.")
    if manifest.get("nonrelease_smoke"):
        raise ValueError("Formal Router training refuses a nonrelease smoke dataset.")
    if not manifest.get("approved_only"):
        raise ValueError("Router training requires approved-only trajectories.")
    if not manifest.get("source_dataset_hash"):
        raise ValueError("Routing dataset manifest is missing source_dataset_hash.")
    if path.is_dir():
        file_hashes = dict(manifest.get("file_hashes") or {})
        for split, split_path in (("train", train_path), ("validation", validation_path)):
            expected = file_hashes.get(split)
            if not split_path or not split_path.is_file() or not expected:
                raise ValueError(f"Routing dataset is missing bound {split} data.")
            if sha256_path(split_path) != expected:
                raise ValueError(f"Routing {split} data hash mismatch.")
        train_records = read_jsonl(train_path)
        validation_records = read_jsonl(validation_path) if validation_path else []
    else:
        # Historical combined files are accepted only when they contain no test
        # rows. Formal workflow uses the split directory form.
        records = read_jsonl(train_path)
        if any(item.get("split") == "test" for item in records):
            raise ValueError("Router trainer refuses test-split leakage in --train-data.")
        train_records = [item for item in records if item.get("split") == "train"]
        validation_records = [
            item for item in records if item.get("split") == "validation"
        ]
    if any(item.get("split") != "train" for item in train_records):
        raise ValueError("Router train.jsonl contains non-train examples.")
    if any(item.get("split") != "validation" for item in validation_records):
        raise ValueError("Router validation.jsonl contains non-validation examples.")
    expected_source = str(manifest["source_dataset_hash"])
    if any(
        str(item.get("source_dataset_hash") or "") != expected_source
        for item in [*train_records, *validation_records]
    ):
        raise ValueError("Routing records do not match their source dataset hash.")
    return train_records, validation_records, manifest


def _decoder_trainable_state(decoder: LatentPlannerDecoder) -> dict[str, Any]:
    values = {
        f"model::{name}": parameter.detach().cpu()
        for name, parameter in decoder.bundle.model.named_parameters()
        if "lora_" in name
    }
    values.update(
        {
            f"bridge::{name}": parameter.detach().cpu()
            for name, parameter in decoder.memory_to_decoder_bridge.named_parameters()
        }
    )
    return values


def _load_decoder_trainable_state(
    decoder: LatentPlannerDecoder,
    state: Mapping[str, Any],
) -> None:
    model_parameters = dict(decoder.bundle.model.named_parameters())
    bridge_parameters = dict(decoder.memory_to_decoder_bridge.named_parameters())
    for key, value in state.items():
        prefix, name = key.split("::", 1)
        parameters = model_parameters if prefix == "model" else bridge_parameters
        if name not in parameters:
            raise RuntimeError(f"Router checkpoint has unknown decoder parameter: {key}")
        parameters[name].data.copy_(value.to(parameters[name]))


def _build_query_cache(
    encoder: LatentMemoryQueryEncoder,
    records: Sequence[Mapping[str, Any]],
    *,
    embedding_dim: int,
    progress_label: str | None = None,
) -> _FrozenQueryCache:
    values: dict[str, Any] = {}
    progress = ProgressReporter(
        progress_label or "router-query-cache",
        len(records),
        unit="state",
        enabled=progress_label is not None,
    )
    progress.start(status="encoding")
    for index, record in enumerate(records):
        state = PatientState.from_mapping(
            record["patient_state"],
            trajectory_history=record.get("trajectory_history") or [],
        )
        text = serialize_patient_state(state, record.get("trajectory_history") or [])
        if text not in values:
            values[text] = encoder.encode_query(text, embedding_dim=embedding_dim)
        progress.update(index + 1, metrics={"unique": len(values)})
    progress.finish(metrics={"unique": len(values)})
    return _FrozenQueryCache(values)


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
