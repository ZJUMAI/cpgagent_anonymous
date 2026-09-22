"""Planner Decoder V2 training with grounded trajectory and auxiliary losses."""

from __future__ import annotations

import gc
import json
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from guideline_planner.artifacts import (
    build_memory_store_fingerprint,
    sha256_json,
    sha256_path,
    validate_artifact_hash,
)
from guideline_planner.io_utils import read_jsonl, write_json
from guideline_planner.latent_memory_retriever import GuidelineMemoryBank
from guideline_planner.memory import load_memory_store_meta, resolve_memory_store_path
from guideline_planner.modeling import (
    _get_input_embeddings,
    enable_non_reentrant_gradient_checkpointing,
    encode_prompt_and_full_target,
    generate_with_memory_slots,
    load_planner_model_bundle,
)
from guideline_planner.phase_machine import PHASE_ORDER
from guideline_planner.planner import (
    _decoder_prompt,
    _parse_planner_json,
)
from guideline_planner.progress import ProgressReporter
from guideline_planner.sampling import (
    PlannerTrajectorySampler,
    balanced_validation_records,
)
from guideline_planner.schemas_v2 import validate_planner_action_v2
from guideline_planner.training_checkpoint import (
    load_latest_training_checkpoint,
    refuse_completed_output,
    save_training_checkpoint,
    training_config_hash,
)
from guideline_planner.trajectory_dataset import require_training_ready_dataset


@dataclass
class PlannerDecoderTrainingConfig:
    trajectory_data: str
    memory_dir: str
    output_dir: str
    device: str = "cuda"
    model_dtype: str = "bf16"
    max_steps: int = 1000
    learning_rate: float = 2e-4
    max_decoder_tokens: int = 2048
    gradient_accumulation_steps: int = 1
    eval_steps: int = 100
    save_steps: int = 500
    keep_last_checkpoints: int = 2
    auto_resume: bool = True
    seed: int = 17
    allow_nonrelease_smoke: bool = False
    overfit_examples: int = 0
    generation_eval_steps: int = 500
    generation_eval_examples: int = 8
    generation_max_new_tokens: int = 1024
    require_generation_schema_pass_rate: float = 0.0
    training_memory_strategy: str = "non_reentrant_segmented_v1"
    lambda_structured_sft: float = 1.0
    lambda_action_preference: float = 0.30
    lambda_state_progress: float = 0.20
    lambda_phase_transition: float = 0.10
    lambda_memory_provenance: float = 0.20


@dataclass
class PlannerLossBreakdown:
    total: torch.Tensor
    structured_sft: torch.Tensor
    action_preference: torch.Tensor
    state_progress: torch.Tensor
    phase_transition: torch.Tensor
    memory_provenance: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            key: float(getattr(self, key).detach().float().cpu())
            for key in (
                "total",
                "structured_sft",
                "action_preference",
                "state_progress",
                "phase_transition",
                "memory_provenance",
            )
        }


class PlannerDecoderTrainer:
    """Fork the encoder LoRA, train Decoder/bridge/heads, and preserve encoder slots."""

    def __init__(self, config: PlannerDecoderTrainingConfig) -> None:
        self.config = config

    def train(self) -> dict[str, Any]:
        torch.manual_seed(self.config.seed)
        if self.config.training_memory_strategy != "non_reentrant_segmented_v1":
            raise ValueError(
                "Planner Decoder only supports training_memory_strategy="
                "'non_reentrant_segmented_v1'."
            )
        if not 0.0 <= float(self.config.require_generation_schema_pass_rate) <= 1.0:
            raise ValueError("require_generation_schema_pass_rate must be in [0, 1].")
        if (
            self.config.require_generation_schema_pass_rate > 0
            and self.config.generation_eval_steps <= 0
        ):
            raise ValueError(
                "A generation schema gate requires generation_eval_steps > 0."
            )
        require_training_ready_dataset(
            self.config.trajectory_data,
            release_gates=not self.config.allow_nonrelease_smoke,
        )
        train_records, validation_records = _load_splits(self.config.trajectory_data)
        if not train_records:
            raise ValueError("Planner Decoder V2 has no train-split trajectories.")
        if self.config.overfit_examples > 0:
            count = min(int(self.config.overfit_examples), len(train_records))
            train_records = train_records[:count]
            validation_records = list(train_records)
        memory_dir = Path(self.config.memory_dir)
        memory_meta = load_memory_store_meta(memory_dir)
        _validate_memory_artifacts(memory_dir, memory_meta)
        bank = GuidelineMemoryBank.from_directory(memory_dir)
        base_model = memory_meta.get("base_model_snapshot_path") or memory_meta.get("base_model")
        progress = ProgressReporter(
            "train-decoder",
            max(int(self.config.max_steps), 1),
            unit="step",
        )
        progress.message(f"loading Planner Decoder base model from {base_model}")
        bundle = load_planner_model_bundle(
            model_name=str(base_model),
            adapter_path=resolve_memory_store_path(
                memory_dir,
                memory_meta.get("memory_encoder_adapter_path"),
                "memory_encoder_adapter_path",
            ),
            tokenizer_path=resolve_memory_store_path(
                memory_dir,
                memory_meta.get("tokenizer_path"),
                "tokenizer_path",
            ),
            retrieval_projection_path=resolve_memory_store_path(
                memory_dir,
                memory_meta.get("retrieval_projection_path"),
                "retrieval_projection_path",
            ),
            device=self.config.device,
            model_dtype=self.config.model_dtype,
            adapter_trainable=True,
        )
        _freeze_non_lora(bundle.model)
        checkpointing = enable_non_reentrant_gradient_checkpointing(bundle.model)
        progress.message(
            "enabled non-reentrant gradient checkpointing for Planner Decoder"
        )
        bridge = torch.nn.Linear(bundle.hidden_size, bundle.hidden_size, bias=False).to(bundle.device)
        torch.nn.init.eye_(bridge.weight)
        progress_head = torch.nn.Linear(bundle.hidden_size, 1).to(bundle.device)
        phase_head = torch.nn.Linear(bundle.hidden_size, len(PHASE_ORDER)).to(bundle.device)
        provenance_head = torch.nn.Linear(bundle.hidden_size, bank.retrieval_dim).to(bundle.device)
        auxiliary = torch.nn.ModuleDict(
            {
                "progress_head": progress_head,
                "phase_head": phase_head,
                "provenance_head": provenance_head,
            }
        )
        parameters = [item for item in bundle.model.parameters() if item.requires_grad]
        parameters.extend(bridge.parameters())
        parameters.extend(auxiliary.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=self.config.learning_rate)
        sampler = PlannerTrajectorySampler(train_records, seed=self.config.seed)
        schedule, sampling_audit = sampler.sample(max(int(self.config.max_steps), 1))
        output_dir = Path(self.config.output_dir)
        refuse_completed_output(output_dir, "planner_decoder_meta.json")
        output_dir.mkdir(parents=True, exist_ok=True)
        token_budget_audit = _audit_decoder_token_budget(
            bundle.tokenizer,
            bank,
            [*train_records, *validation_records],
            max_decoder_tokens=self.config.max_decoder_tokens,
        )
        write_json(output_dir / "token_budget_audit.json", token_budget_audit)
        progress.message(
            "validated full structured targets: "
            f"max={token_budget_audit['target_tokens_with_eos']['max']} tokens, "
            f"prompt_truncated_rate={token_budget_audit['prompt_truncated_rate']:.3f}"
        )
        metrics_path = output_dir / "metrics.jsonl"
        dataset_hash = _trajectory_dataset_hash(self.config.trajectory_data)
        lineage = {
            "trajectory_dataset_hash": dataset_hash,
            "memory_store_fingerprint": memory_meta.get("memory_store_fingerprint"),
        }
        effective_config_hash = training_config_hash(asdict(self.config))
        resumed = None
        if self.config.auto_resume:
            resumed = load_latest_training_checkpoint(
                torch_module=torch,
                output_dir=output_dir,
                stage="planner_decoder",
                config_hash=effective_config_hash,
                lineage=lineage,
                map_location=bundle.device,
            )
        start_step = 0
        resumed_last_metrics: dict[str, float] = {}
        if resumed is not None:
            _load_trainable_parameters(bundle.model, resumed["decoder_trainable_state"])
            bridge.load_state_dict(resumed["bridge_state"])
            auxiliary.load_state_dict(resumed["auxiliary_state"])
            optimizer.load_state_dict(resumed["optimizer_state"])
            start_step = int(resumed["global_step"])
            resumed_last_metrics = dict(resumed.get("last_metrics") or {})
        else:
            metrics_path.write_text("", encoding="utf-8")
        accumulation = max(int(self.config.gradient_accumulation_steps), 1)
        optimizer.zero_grad(set_to_none=True)
        last_metrics: dict[str, float] = resumed_last_metrics
        last_generation_validation: dict[str, Any] | None = None
        progress.start(start_step, status="resumed" if start_step else "training")
        for step in range(start_step, len(schedule)):
            record = schedule[step]
            loss = _backward_loss_for_record(
                bundle,
                bridge,
                auxiliary,
                bank,
                record,
                variant_index=step,
                max_decoder_tokens=self.config.max_decoder_tokens,
                weights=self.config,
                gradient_scale=1.0 / accumulation,
            )
            if (step + 1) % accumulation == 0 or step + 1 == len(schedule):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            last_metrics = loss.detached()
            _append_jsonl(metrics_path, {"step": step + 1, "split": "train", **last_metrics})
            progress.update(
                step + 1,
                metrics={
                    "loss": last_metrics.get("total"),
                    "sft": last_metrics.get("structured_sft"),
                },
            )
            if validation_records and self.config.eval_steps > 0 and (
                (step + 1) % self.config.eval_steps == 0 or step + 1 == len(schedule)
            ):
                progress.message(f"running validation at step {step + 1}")
                event = _evaluate(
                    bundle,
                    bridge,
                    auxiliary,
                    bank,
                    validation_records,
                    self.config,
                    progress_label=f"decoder-eval@{step + 1}",
                    run_generation=(
                        self.config.generation_eval_steps > 0
                        and (
                            (step + 1) % self.config.generation_eval_steps == 0
                            or step + 1 == len(schedule)
                        )
                    ),
                )
                if event.get("generation_schema_pass_rate") is not None:
                    last_generation_validation = {
                        key: value
                        for key, value in event.items()
                        if key.startswith("generation_")
                    }
                event["step"] = step + 1
                _append_jsonl(metrics_path, event)
                progress.update(
                    step + 1,
                    metrics={
                        "status": "validated",
                        "val_loss": event.get("total"),
                        "schema": event.get("generation_schema_pass_rate"),
                    },
                    force=True,
                )
            if self.config.save_steps > 0 and (
                (step + 1) % self.config.save_steps == 0 or step + 1 == len(schedule)
            ):
                progress.message(f"saving checkpoint at step {step + 1}")
                save_training_checkpoint(
                    torch_module=torch,
                    output_dir=output_dir,
                    stage="planner_decoder",
                    global_step=step + 1,
                    config_hash=effective_config_hash,
                    lineage=lineage,
                    keep_last=self.config.keep_last_checkpoints,
                    state={
                        "decoder_trainable_state": _trainable_parameters(bundle.model),
                        "bridge_state": bridge.state_dict(),
                        "auxiliary_state": auxiliary.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "sampling_audit": sampling_audit,
                        "last_metrics": last_metrics,
                    },
                )

        required_schema_rate = float(self.config.require_generation_schema_pass_rate)
        if required_schema_rate > 0.0:
            actual_schema_rate = (
                float(last_generation_validation["generation_schema_pass_rate"])
                if last_generation_validation is not None
                else 0.0
            )
            if actual_schema_rate < required_schema_rate:
                raise RuntimeError(
                    "Planner Decoder generation overfit gate failed: "
                    f"schema_pass_rate={actual_schema_rate:.4f}, "
                    f"required={required_schema_rate:.4f}. Do not start formal "
                    "training until the short-run generation gate passes."
                )
        progress.message("saving final Planner Decoder artifacts")
        adapter_dir = output_dir / "planner_decoder_adapter"
        bundle.model.save_pretrained(adapter_dir)
        source_tokenizer_path = resolve_memory_store_path(
            memory_dir,
            memory_meta.get("tokenizer_path"),
            "tokenizer_path",
        ).resolve()
        tokenizer_path = output_dir / "tokenizer"
        shutil.copytree(source_tokenizer_path, tokenizer_path, dirs_exist_ok=True)
        bridge_path = output_dir / "memory_to_decoder_bridge.pt"
        aux_path = output_dir / "planner_aux_heads.pt"
        torch.save({"state_dict": bridge.state_dict()}, bridge_path)
        torch.save({"state_dict": auxiliary.state_dict(), "phases": list(PHASE_ORDER)}, aux_path)
        meta = {
            "format_version": 2,
            "artifact_role": "planner_decoder",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_model": memory_meta.get("base_model"),
            "base_model_snapshot_path": memory_meta.get("base_model_snapshot_path"),
            "base_model_revision": memory_meta.get("base_model_revision"),
            "memory_store_fingerprint": memory_meta.get("memory_store_fingerprint"),
            "memory_encoder_adapter_hash": memory_meta.get("memory_encoder_adapter_hash"),
            "planner_decoder_adapter_path": adapter_dir.name,
            "planner_decoder_adapter_hash": sha256_path(adapter_dir),
            "memory_to_decoder_bridge_path": bridge_path.name,
            "memory_to_decoder_bridge_hash": sha256_path(bridge_path),
            "auxiliary_heads_path": aux_path.name,
            "auxiliary_heads_hash": sha256_path(aux_path),
            "tokenizer_path": tokenizer_path.name,
            "tokenizer_hash": sha256_path(tokenizer_path),
            "retrieval_projection_hash": memory_meta.get("retrieval_projection_hash"),
            "trajectory_data": str(self.config.trajectory_data),
            "trajectory_dataset_hash": dataset_hash,
            "loss_weights": {
                "structured_sft": self.config.lambda_structured_sft,
                "action_preference": self.config.lambda_action_preference,
                "state_progress": self.config.lambda_state_progress,
                "phase_transition": self.config.lambda_phase_transition,
                "memory_provenance": self.config.lambda_memory_provenance,
            },
            "generation_validation": last_generation_validation,
            "max_decoder_tokens": int(self.config.max_decoder_tokens),
            "token_budget_audit": token_budget_audit,
            "gradient_checkpointing": checkpointing,
            "training_memory_strategy": self.config.training_memory_strategy,
        }
        write_json(output_dir / "planner_decoder_meta.json", meta)
        write_json(output_dir / "training_config.json", asdict(self.config))
        write_json(output_dir / "sampling_audit.json", sampling_audit)
        summary = {
            "output_dir": str(output_dir),
            "steps": len(schedule),
            "resumed_from_step": start_step,
            "training_examples": len(train_records),
            "validation_examples": len(validation_records),
            "final_train_loss": last_metrics,
            "generation_validation": last_generation_validation,
            "overfit_examples": int(self.config.overfit_examples),
            "token_budget_audit": token_budget_audit,
            "gradient_checkpointing": checkpointing,
            "training_memory_strategy": self.config.training_memory_strategy,
            "sampling_audit": sampling_audit,
            "planner_decoder_meta": str(output_dir / "planner_decoder_meta.json"),
        }
        write_json(output_dir / "training_summary.json", summary)
        progress.finish(metrics={"loss": last_metrics.get("total")})
        return summary


def train_planner_decoder_v2(
    config: PlannerDecoderTrainingConfig | Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(config, PlannerDecoderTrainingConfig):
        config = PlannerDecoderTrainingConfig(**dict(config))
    return PlannerDecoderTrainer(config).train()


def _backward_loss_for_record(
    bundle: Any,
    bridge: Any,
    auxiliary: Any,
    bank: GuidelineMemoryBank,
    record: Mapping[str, Any],
    *,
    variant_index: int,
    max_decoder_tokens: int,
    weights: PlannerDecoderTrainingConfig,
    gradient_scale: float,
) -> PlannerLossBreakdown:
    """Backpropagate the exact preference gradient without two resident graphs.

    The positive graph stays resident while a no-grad negative reference pass
    determines ``sigmoid(L_pos - L_neg)``.  We then backpropagate and release
    the positive graph before recomputing/backpropagating the negative graph.
    ``fork_rng`` makes the reference and recomputed negative passes use the
    same dropout masks without advancing the global RNG twice.
    """

    state = record["state_before"]
    labels = record["routing_labels"]
    positive_ids = list(
        dict.fromkeys(
            [
                *labels["strong_positive_memory_ids"],
                *labels["weak_positive_memory_ids"],
            ]
        )
    )
    missing = [item for item in positive_ids if item not in bank.lookup]
    if missing:
        raise ValueError(f"Planner trajectory references missing positive memories: {missing}")
    raw_prefix = torch.stack(
        [bank.lookup[memory_id].slots.to(bundle.device) for memory_id in positive_ids],
        dim=0,
    ).mean(dim=0)
    memories = [_memory_prompt_item(bank.lookup[item]) for item in positive_ids]
    prompt = _decoder_prompt(state, state.get("completed_actions", []), memories, None)
    variants = record["accepted_plan_variants"]
    target_payload = validate_planner_action_v2(
        variants[variant_index % len(variants)],
        patient_state=state,
        active_memories=memories,
    )
    target = json.dumps(target_payload, ensure_ascii=False, sort_keys=True)
    negative_target = json.dumps(
        _negative_plan(record, positive_ids, variant_index=variant_index),
        ensure_ascii=False,
        sort_keys=True,
    )

    candidate_ids = list(
        dict.fromkeys(
            [
                *positive_ids,
                *labels["hard_negative_memory_ids"],
                *labels["easy_negative_memory_ids"],
            ]
        )
    )
    candidate_ids = [item for item in candidate_ids if item in bank.lookup]
    if not candidate_ids or not positive_ids:
        raise ValueError("Provenance loss requires positive and negative memory candidates.")

    positive_prefix = bridge(raw_prefix.to(dtype=bridge.weight.dtype))
    positive_output, positive_hidden = _causal_forward(
        bundle,
        positive_prefix,
        prompt,
        target,
        max_decoder_tokens=max_decoder_tokens,
    )
    positive_sft = positive_output.loss

    # This pass has no autograd graph. Its RNG is restored on exit so the later
    # negative training pass reproduces the same stochastic adapter masks.
    with torch.random.fork_rng(devices=_cuda_rng_devices(bundle.device), enabled=True):
        with torch.no_grad():
            negative_reference_prefix = bridge(
                raw_prefix.to(dtype=bridge.weight.dtype)
            )
            negative_reference_output, _ = _causal_forward(
                bundle,
                negative_reference_prefix,
                prompt,
                negative_target,
                max_decoder_tokens=max_decoder_tokens,
                return_hidden=False,
            )
            negative_reference = negative_reference_output.loss.detach()
            del negative_reference_output, negative_reference_prefix

    preference_weight = torch.sigmoid(
        positive_sft.detach() - negative_reference
    ).detach()
    positive_progress = F.binary_cross_entropy_with_logits(
        _linear_with_module_dtype(
            auxiliary["progress_head"],
            positive_hidden,
        ).reshape(()),
        positive_sft.new_ones(()),
    )
    phase_name = str(record["state_after"]["current_phase"])
    phase_index = PHASE_ORDER.index(phase_name)
    phase_logits = _linear_with_module_dtype(
        auxiliary["phase_head"],
        positive_hidden,
    ).reshape(1, -1)
    phase_loss = F.cross_entropy(
        phase_logits,
        torch.tensor([phase_index], dtype=torch.long, device=phase_logits.device),
    )
    projected = F.normalize(
        _linear_with_module_dtype(auxiliary["provenance_head"], positive_hidden),
        dim=-1,
    )
    candidate_keys = torch.stack(
        [bank.lookup[item].retrieval_key.to(projected) for item in candidate_ids],
        dim=0,
    )
    provenance_logits = torch.matmul(F.normalize(candidate_keys, dim=-1), projected)
    provenance_targets = torch.tensor(
        [1.0 if item in positive_ids else 0.0 for item in candidate_ids],
        dtype=provenance_logits.dtype,
        device=provenance_logits.device,
    )
    provenance = F.binary_cross_entropy_with_logits(
        provenance_logits,
        provenance_targets,
    )
    positive_objective = (
        weights.lambda_structured_sft * positive_sft
        + weights.lambda_action_preference * preference_weight * positive_sft
        + weights.lambda_state_progress * 0.5 * positive_progress
        + weights.lambda_phase_transition * phase_loss
        + weights.lambda_memory_provenance * provenance
    )
    (positive_objective * float(gradient_scale)).backward()
    positive_sft_value = positive_sft.detach()
    positive_progress_value = positive_progress.detach()
    phase_value = phase_loss.detach()
    provenance_value = provenance.detach()
    del (
        positive_objective,
        positive_output,
        positive_hidden,
        positive_prefix,
        positive_progress,
        phase_logits,
        phase_loss,
        projected,
        provenance_logits,
        provenance,
    )

    negative_prefix = bridge(raw_prefix.to(dtype=bridge.weight.dtype))
    negative_output, negative_hidden = _causal_forward(
        bundle,
        negative_prefix,
        prompt,
        negative_target,
        max_decoder_tokens=max_decoder_tokens,
    )
    negative_sft = negative_output.loss
    negative_progress = F.binary_cross_entropy_with_logits(
        _linear_with_module_dtype(
            auxiliary["progress_head"],
            negative_hidden,
        ).reshape(()),
        negative_sft.new_zeros(()),
    )
    negative_objective = (
        -weights.lambda_action_preference * preference_weight * negative_sft
        + weights.lambda_state_progress * 0.5 * negative_progress
    )
    (negative_objective * float(gradient_scale)).backward()
    negative_sft_value = negative_sft.detach()
    negative_progress_value = negative_progress.detach()

    preference_value = F.softplus(positive_sft_value - negative_sft_value)
    progress_value = 0.5 * (
        positive_progress_value + negative_progress_value
    )
    total_value = (
        weights.lambda_structured_sft * positive_sft_value
        + weights.lambda_action_preference * preference_value
        + weights.lambda_state_progress * progress_value
        + weights.lambda_phase_transition * phase_value
        + weights.lambda_memory_provenance * provenance_value
    )
    return PlannerLossBreakdown(
        total_value,
        positive_sft_value,
        preference_value,
        progress_value,
        phase_value,
        provenance_value,
    )


def _loss_for_record(
    bundle: Any,
    bridge: Any,
    auxiliary: Any,
    bank: GuidelineMemoryBank,
    record: Mapping[str, Any],
    *,
    variant_index: int,
    max_decoder_tokens: int,
    weights: PlannerDecoderTrainingConfig,
) -> PlannerLossBreakdown:
    state = record["state_before"]
    labels = record["routing_labels"]
    positive_ids = list(
        dict.fromkeys(
            [
                *labels["strong_positive_memory_ids"],
                *labels["weak_positive_memory_ids"],
            ]
        )
    )
    missing = [item for item in positive_ids if item not in bank.lookup]
    if missing:
        raise ValueError(f"Planner trajectory references missing positive memories: {missing}")
    raw_prefix = torch.stack(
        [bank.lookup[memory_id].slots.to(bundle.device) for memory_id in positive_ids], dim=0
    ).mean(dim=0)
    prefix = bridge(raw_prefix.to(dtype=bridge.weight.dtype))
    memories = [_memory_prompt_item(bank.lookup[item]) for item in positive_ids]
    prompt = _decoder_prompt(state, state.get("completed_actions", []), memories, None)
    variants = record["accepted_plan_variants"]
    target_payload = validate_planner_action_v2(
        variants[variant_index % len(variants)],
        patient_state=state,
        active_memories=memories,
    )
    target = json.dumps(target_payload, ensure_ascii=False, sort_keys=True)
    positive_output, positive_hidden = _causal_forward(
        bundle,
        prefix,
        prompt,
        target,
        max_decoder_tokens=max_decoder_tokens,
    )
    sft = positive_output.loss
    del positive_output
    negative_plan = _negative_plan(record, positive_ids, variant_index=variant_index)
    negative_target = json.dumps(negative_plan, ensure_ascii=False, sort_keys=True)
    negative_output, negative_hidden = _causal_forward(
        bundle,
        prefix,
        prompt,
        negative_target,
        max_decoder_tokens=max_decoder_tokens,
    )
    negative_sft = negative_output.loss
    del negative_output
    preference = F.softplus(sft - negative_sft)
    positive_progress_logits = _linear_with_module_dtype(
        auxiliary["progress_head"],
        positive_hidden,
    ).reshape(())
    negative_progress_logits = _linear_with_module_dtype(
        auxiliary["progress_head"],
        negative_hidden,
    ).reshape(())
    progress = 0.5 * (
        F.binary_cross_entropy_with_logits(
            positive_progress_logits,
            positive_progress_logits.new_ones(()),
        )
        + F.binary_cross_entropy_with_logits(
            negative_progress_logits,
            negative_progress_logits.new_zeros(()),
        )
    )
    phase_name = str(record["state_after"]["current_phase"])
    phase_index = PHASE_ORDER.index(phase_name)
    phase_logits = _linear_with_module_dtype(
        auxiliary["phase_head"],
        positive_hidden,
    ).reshape(1, -1)
    phase_loss = F.cross_entropy(
        phase_logits,
        torch.tensor([phase_index], dtype=torch.long, device=phase_logits.device),
    )
    candidate_ids = list(
        dict.fromkeys(
            [
                *positive_ids,
                *labels["hard_negative_memory_ids"],
                *labels["easy_negative_memory_ids"],
            ]
        )
    )
    candidate_ids = [item for item in candidate_ids if item in bank.lookup]
    if not candidate_ids or not positive_ids:
        raise ValueError("Provenance loss requires positive and negative memory candidates.")
    projected = F.normalize(
        _linear_with_module_dtype(auxiliary["provenance_head"], positive_hidden),
        dim=-1,
    )
    candidate_keys = torch.stack(
        [bank.lookup[item].retrieval_key.to(projected) for item in candidate_ids],
        dim=0,
    )
    provenance_logits = torch.matmul(F.normalize(candidate_keys, dim=-1), projected)
    provenance_targets = torch.tensor(
        [1.0 if item in positive_ids else 0.0 for item in candidate_ids],
        dtype=provenance_logits.dtype,
        device=provenance_logits.device,
    )
    provenance = F.binary_cross_entropy_with_logits(provenance_logits, provenance_targets)
    total = (
        weights.lambda_structured_sft * sft
        + weights.lambda_action_preference * preference
        + weights.lambda_state_progress * progress
        + weights.lambda_phase_transition * phase_loss
        + weights.lambda_memory_provenance * provenance
    )
    return PlannerLossBreakdown(total, sft, preference, progress, phase_loss, provenance)


def _linear_with_module_dtype(module: Any, value: torch.Tensor) -> torch.Tensor:
    """Cross an explicit dtype boundary between BF16 models and FP32 heads.

    Planner auxiliary heads intentionally remain FP32 for numerical stability,
    while Qwen hidden states are BF16 in formal training.  ``nn.Linear`` does
    not implicitly promote mixed input/weight dtypes, so cast only the compact
    pooled hidden state instead of changing the base model or its activations.
    """

    parameter = next(module.parameters())
    return module(value.to(device=parameter.device, dtype=parameter.dtype))


def _causal_forward(
    bundle: Any,
    prefix: torch.Tensor,
    prompt: str,
    target: str,
    *,
    max_decoder_tokens: int,
    return_hidden: bool = True,
) -> tuple[Any, torch.Tensor | None]:
    prompt_ids, target_ids, _ = encode_prompt_and_full_target(
        bundle.tokenizer,
        prompt,
        target,
        max_decoder_tokens=max_decoder_tokens,
    )
    prompt_ids = prompt_ids.to(bundle.device)
    target_tensor = torch.tensor([target_ids], dtype=torch.long, device=bundle.device)
    ids = torch.cat([prompt_ids["input_ids"], target_tensor], dim=1)
    embeddings = _get_input_embeddings(bundle.model)(ids)
    prefix_batch = prefix.to(dtype=embeddings.dtype).unsqueeze(0)
    inputs = torch.cat([prefix_batch, embeddings], dim=1)
    labels = torch.full(inputs.shape[:2], -100, dtype=torch.long, device=bundle.device)
    start = int(prefix_batch.shape[1]) + int(prompt_ids["input_ids"].shape[1])
    labels[:, start:] = target_tensor
    output = bundle.model(
        inputs_embeds=inputs,
        labels=labels,
        output_hidden_states=return_hidden,
        use_cache=False,
    )
    state_hidden = (
        output.hidden_states[-1][0, start - 1]
        if return_hidden
        else None
    )
    return output, state_hidden


def _cuda_rng_devices(device: str | torch.device) -> list[int]:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return []
    return [
        int(resolved.index)
        if resolved.index is not None
        else int(torch.cuda.current_device())
    ]


def _negative_plan(
    record: Mapping[str, Any],
    positive_ids: Sequence[str],
    *,
    variant_index: int,
) -> dict[str, Any]:
    action_set = record["action_set"]
    candidates = [
        *action_set["unsafe"],
        *action_set["premature"],
        *[
            item
            for item in action_set["conditional"]
            if item.get("condition_satisfied") is False
        ],
    ]
    if not candidates:
        raise ValueError("Preference loss requires at least one premature or unsafe action.")
    negative_plans = [
        _bucket_negative_plan(record, item, positive_ids) for item in candidates
    ]

    # Explicitly teach a failure which may not be present in every compiled rule
    # bucket: an action that promises no progress.  Do not treat reuse of a skill
    # as a negative example: report readers and other evidence tools are reusable
    # capabilities and may be invoked multiple times with different objectives.
    no_progress = deepcopy(record["accepted_plan_variants"][0])
    no_progress["actions"][0]["expected_state_delta"] = []
    no_progress["reason"] = "negative preference candidate: no state progress"
    negative_plans.append(no_progress)
    return negative_plans[int(variant_index) % len(negative_plans)]


def _bucket_negative_plan(
    record: Mapping[str, Any],
    item: Mapping[str, Any],
    positive_ids: Sequence[str],
) -> dict[str, Any]:
    provenance = [
        {
            "memory_id": memory_id,
            "rule_ids": list(item["guideline_rule_ids"]),
            "source_spans": list(item["source_spans"]),
        }
        for memory_id in item.get("supporting_memory_ids") or positive_ids
    ]
    return {
        "schema_version": "planner_action.v2",
        "current_phase": record["state_before"]["current_phase"],
        "proposed_phase": None,
        "missing_information": [],
        "actions": [
            {
                "action_id": item["action_id"],
                "objective": item["objective"],
                "action_type": item["action_type"],
                "required_skills": list(item["required_skills"]),
                "preconditions": list(item["preconditions"]),
                "expected_state_delta": list(item["expected_state_delta"]),
                "provenance": provenance,
            }
        ],
        "blocked_actions": [],
        "should_stop": False,
        "reason": "negative preference candidate",
    }


def _memory_prompt_item(memory: Any) -> dict[str, Any]:
    metadata = dict(memory.metadata)
    metadata.update(
        {
            "guideline_memory_id": memory.memory_id,
            "guideline_id": metadata.get("guideline_id") or memory.guideline_name,
            "version": memory.guideline_version,
            "h1_title": memory.section_title,
            "source_rule_ids": list(memory.source_rule_ids),
            "source_span_ids": list(memory.source_chunk_ids),
        }
    )
    return {"guideline_memory_id": memory.memory_id, "score": 1.0, "metadata": metadata}


def _evaluate(
    bundle: Any,
    bridge: Any,
    auxiliary: Any,
    bank: GuidelineMemoryBank,
    records: Sequence[Mapping[str, Any]],
    config: PlannerDecoderTrainingConfig,
    progress_label: str | None = None,
    run_generation: bool = False,
) -> dict[str, Any]:
    values = []
    progress = ProgressReporter(
        progress_label or "decoder-eval",
        len(records),
        unit="transition",
        enabled=progress_label is not None,
    )
    progress.start(status="validating")
    bundle.model.eval()
    bridge.eval()
    auxiliary.eval()
    try:
        with torch.no_grad():
            for index, record in enumerate(records):
                value = _loss_for_record(
                    bundle,
                    bridge,
                    auxiliary,
                    bank,
                    record,
                    variant_index=index,
                    max_decoder_tokens=config.max_decoder_tokens,
                    weights=config,
                ).detached()
                values.append(value)
                progress.update(index + 1, metrics={"loss": value.get("total")})
    finally:
        bundle.model.train()
        bridge.train()
        auxiliary.train()
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
                bundle,
                bridge,
                bank,
                balanced_validation_records(
                    records,
                    max(int(config.generation_eval_examples), 1),
                    seed=config.seed,
                ),
                max_new_tokens=config.generation_max_new_tokens,
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
    bundle: Any,
    bridge: Any,
    bank: GuidelineMemoryBank,
    records: Sequence[Mapping[str, Any]],
    *,
    max_new_tokens: int,
    progress_label: str | None,
) -> dict[str, Any]:
    """Run greedy generation during validation and validate real V2 outputs."""

    generation_progress = ProgressReporter(
        progress_label or "decoder-generation-eval",
        len(records),
        unit="prediction",
        enabled=progress_label is not None,
    )
    generation_progress.start(status="generating")
    passed = 0
    failures: list[dict[str, Any]] = []
    model_was_training = bool(bundle.model.training)
    bridge_was_training = bool(bridge.training)
    bundle.model.eval()
    bridge.eval()
    try:
        with torch.no_grad():
            for index, record in enumerate(records):
                labels = record["routing_labels"]
                positive_ids = list(
                    dict.fromkeys(
                        [
                            *labels["strong_positive_memory_ids"],
                            *labels["weak_positive_memory_ids"],
                        ]
                    )
                )
                raw_prefix = torch.stack(
                    [
                        bank.lookup[memory_id].slots.to(bundle.device)
                        for memory_id in positive_ids
                    ],
                    dim=0,
                ).mean(dim=0)
                prefix = bridge(raw_prefix.to(dtype=bridge.weight.dtype))
                memories = [_memory_prompt_item(bank.lookup[item]) for item in positive_ids]
                state = record["state_before"]
                prompt = _decoder_prompt(
                    state,
                    state.get("completed_actions", []),
                    memories,
                    None,
                )
                raw_text = ""
                try:
                    raw_text = generate_with_memory_slots(
                        bundle,
                        prefix.detach().cpu().float().numpy(),
                        prompt,
                        max_new_tokens=max_new_tokens,
                    )
                    payload = _parse_planner_json(raw_text)
                    validate_planner_action_v2(
                        payload,
                        patient_state=state,
                        active_memories=memories,
                    )
                    passed += 1
                except Exception as exc:
                    failures.append(
                        {
                            "trajectory_id": record.get("trajectory_id"),
                            "turn_index": record.get("turn_index"),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "raw_text": raw_text[:2000],
                        }
                    )
                del prefix, raw_prefix
                generation_progress.update(
                    index + 1,
                    metrics={"schema_pass_rate": passed / (index + 1)},
                )
    finally:
        bundle.model.train(model_was_training)
        bridge.train(bridge_was_training)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rate = passed / len(records) if records else 0.0
    generation_progress.finish(metrics={"schema_pass_rate": rate})
    return {
        "generation_count": len(records),
        "generation_schema_pass_count": passed,
        "generation_schema_pass_rate": rate,
        "generation_failures": failures[:8],
    }


def _audit_decoder_token_budget(
    tokenizer: Any,
    bank: GuidelineMemoryBank,
    records: Sequence[Mapping[str, Any]],
    *,
    max_decoder_tokens: int,
) -> dict[str, Any]:
    """Fail before training if any supervised JSON target would be truncated."""

    target_lengths: list[int] = []
    prompt_lengths: list[int] = []
    prompt_used: list[int] = []
    prompt_truncated = 0
    examples = 0
    for record_index, record in enumerate(records):
        labels = record["routing_labels"]
        positive_ids = list(
            dict.fromkeys(
                [
                    *labels["strong_positive_memory_ids"],
                    *labels["weak_positive_memory_ids"],
                ]
            )
        )
        memories = [_memory_prompt_item(bank.lookup[item]) for item in positive_ids]
        state = record["state_before"]
        prompt = _decoder_prompt(state, state.get("completed_actions", []), memories, None)
        targets = [
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in record["accepted_plan_variants"]
        ]
        targets.append(
            json.dumps(
                _negative_plan(record, positive_ids, variant_index=record_index),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        for target in targets:
            try:
                _, _, audit = encode_prompt_and_full_target(
                    tokenizer,
                    prompt,
                    target,
                    max_decoder_tokens=max_decoder_tokens,
                )
            except ValueError as exc:
                raise ValueError(
                    "Planner target token-budget audit failed for "
                    f"trajectory_id={record.get('trajectory_id')!r}, "
                    f"turn_index={record.get('turn_index')!r}: {exc}"
                ) from exc
            target_lengths.append(int(audit["target_tokens_with_eos"]))
            prompt_lengths.append(int(audit["prompt_tokens_original"]))
            prompt_used.append(int(audit["prompt_tokens_used"]))
            prompt_truncated += int(bool(audit["prompt_truncated"]))
            examples += 1
    if not target_lengths:
        raise ValueError("Planner target token-budget audit received no targets.")
    return {
        "schema_version": "planner_token_budget_audit.v2",
        "max_decoder_tokens": int(max_decoder_tokens),
        "target_truncation_allowed": False,
        "example_count": examples,
        "target_tokens_with_eos": _length_summary(target_lengths),
        "prompt_tokens_original": _length_summary(prompt_lengths),
        "prompt_tokens_used": _length_summary(prompt_used),
        "prompt_truncated_count": prompt_truncated,
        "prompt_truncated_rate": prompt_truncated / examples,
    }


def _length_summary(values: Sequence[int]) -> dict[str, int]:
    ordered = sorted(int(value) for value in values)
    return {
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[min(int(len(ordered) * 0.9), len(ordered) - 1)],
        "max": ordered[-1],
    }


def _load_splits(path_value: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = Path(path_value)
    if path.is_dir():
        return read_jsonl(path / "train.jsonl"), read_jsonl(path / "validation.jsonl")
    rows = read_jsonl(path)
    return (
        [row for row in rows if row.get("split") == "train"],
        [row for row in rows if row.get("split") == "validation"],
    )


def _validate_memory_artifacts(memory_dir: Path, meta: Mapping[str, Any]) -> None:
    if meta.get("artifact_role") != "memory_store" or int(meta.get("format_version") or 0) != 2:
        raise ValueError("Planner Decoder training requires a V2 Memory Store.")
    if meta.get("memory_store_fingerprint") != build_memory_store_fingerprint(meta):
        raise ValueError("Memory Store lineage fingerprint does not match its metadata.")
    for path_field, hash_field in (
        ("memory_encoder_adapter_path", "memory_encoder_adapter_hash"),
        ("tokenizer_path", "tokenizer_hash"),
        ("retrieval_projection_path", "retrieval_projection_hash"),
    ):
        path = resolve_memory_store_path(memory_dir, meta.get(path_field), path_field)
        validate_artifact_hash(label=path_field, path=path, expected_hash=meta.get(hash_field))


def _freeze_non_lora(model: Any) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name


def _trainable_parameters(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _load_trainable_parameters(model: Any, state: Mapping[str, Any]) -> None:
    parameters = dict(model.named_parameters())
    missing = sorted(set(state) - set(parameters))
    if missing:
        raise RuntimeError(f"Planner Decoder checkpoint has unknown parameters: {missing[:5]}")
    for name, value in state.items():
        parameters[name].data.copy_(value.to(parameters[name]))


def _trajectory_dataset_hash(path_value: str | Path) -> str:
    path = Path(path_value)
    if path.is_dir():
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Planner Decoder dataset directory is missing manifest.json.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = str(manifest.get("dataset_hash") or "")
        if not value:
            raise ValueError("Planner Decoder dataset manifest is missing dataset_hash.")
        return value
    return sha256_json(read_jsonl(path))


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")
