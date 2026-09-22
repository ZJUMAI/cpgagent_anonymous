"""ICAE-style Qwen causal-LM wrappers and lightweight training entrypoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from guideline_planner.artifacts import sha256_json, sha256_path
from guideline_planner.constants import (
    DEFAULT_MEMORY_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TASK_RATIOS,
    MEMORY_TOKEN,
    SPECIAL_TOKENS,
)
from guideline_planner.io_utils import read_jsonl, write_json
from guideline_planner.progress import ProgressReporter
from guideline_planner.sampling import (
    DeterministicTaskSampler,
    balanced_validation_records,
)
from guideline_planner.training_checkpoint import (
    load_latest_training_checkpoint,
    refuse_completed_output,
    save_training_checkpoint,
    training_config_hash,
)


@dataclass
class PlannerTrainingConfig:
    train_data_path: str
    output_dir: str
    model_name: str = DEFAULT_MODEL_NAME
    model_revision: str | None = None
    model_snapshot_path: str | None = None
    memory_tokens: int = DEFAULT_MEMORY_TOKENS
    task_ratios: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TASK_RATIOS))
    training_stage: str = "memory_encoder"
    sampling_seed: int = 17
    group_sampling_temperature: float = 0.5
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    max_steps: int = 100
    validation_data_path: str | None = None
    validation_split: float = 0.05
    eval_steps: int = 500
    save_steps: int = 500
    keep_last_checkpoints: int = 2
    auto_resume: bool = True
    eval_max_examples: int = 64
    device: str | None = None
    gradient_accumulation_steps: int = 1
    train_micro_batch_size_per_gpu: int = 1
    mixed_precision: str | None = None
    model_dtype: str = "auto"
    max_encoder_tokens: int = 1024
    max_decoder_tokens: int = 512
    max_query_tokens: int = 256
    max_retrieval_candidates: int = 2
    train_memory_encoder: bool = True
    sharding: str = "none"
    fsdp_cpu_ram_efficient_loading: bool = True
    fsdp_activation_checkpointing: bool = False
    fsdp_transformer_layer_cls_to_wrap: str = "Qwen3DecoderLayer,Qwen2DecoderLayer"
    deepspeed_offload_param_device: str = "none"
    deepspeed_offload_optimizer_device: str = "none"
    deepspeed_nvme_path: str | None = None


@dataclass
class PlannerModelBundle:
    """Loaded causal LM, tokenizer, and projection heads for latent planning."""

    torch: Any
    tokenizer: Any
    model: Any
    device: str
    hidden_size: int
    model_name: str
    model_revision: str | None = None
    model_snapshot_path: str | None = None
    adapter_path: str | None = None
    tokenizer_path: str | None = None
    retrieval_projection_path: str | None = None


def encode_memory_slots_with_model(
    section_text: str,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    model_revision: str | None = None,
    memory_tokens: int = DEFAULT_MEMORY_TOKENS,
    device: str | None = None,
    adapter_path: str | None = None,
    tokenizer_path: str | None = None,
    retrieval_projection_path: str | None = None,
    model_dtype: str = "auto",
    max_encoder_tokens: int = 4096,
) -> np.ndarray:
    """Encode one section as ICAE memory slots using the last N memory-token states."""

    bundle = load_planner_model_bundle(
        model_name=model_name,
        model_revision=model_revision,
        adapter_path=adapter_path,
        tokenizer_path=tokenizer_path,
        retrieval_projection_path=retrieval_projection_path,
        device=device,
        model_dtype=model_dtype,
    )
    return encode_memory_slots_with_bundle(
        bundle,
        section_text,
        memory_tokens=memory_tokens,
        max_encoder_tokens=max_encoder_tokens,
    )


def load_planner_model_bundle(
    *,
    model_name: str,
    model_revision: str | None = None,
    adapter_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    retrieval_projection_path: str | Path | None = None,
    device: str | None = None,
    model_dtype: str = "auto",
    adapter_trainable: bool = False,
) -> PlannerModelBundle:
    """Load the exact model pieces needed to encode and decode latent memories."""

    torch, AutoModelForCausalLM, AutoTokenizer, peft_module = _planner_imports()
    resolved_device = _resolve_torch_device(torch, device)
    tokenizer_source = str(tokenizer_path or model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        **_revision_kwargs(tokenizer_source, model_revision),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        **_revision_kwargs(model_name, model_revision),
        **_from_pretrained_dtype_kwargs(torch, model_dtype),
    )
    tokenizer, model = add_planner_special_tokens(tokenizer, model)
    if adapter_path:
        model = peft_module.PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=bool(adapter_trainable),
        )
    hidden_size = int(getattr(model.config, "hidden_size"))
    _attach_projection_heads(model, torch, hidden_size)
    if retrieval_projection_path:
        projection_state = torch.load(str(retrieval_projection_path), map_location="cpu")
        if "query_projection" not in projection_state or "slot_projection" not in projection_state:
            raise RuntimeError(
                "retrieval_projection.pt must contain query_projection and slot_projection."
            )
        _get_projection_head(model, "guideline_query_projection").load_state_dict(
            projection_state["query_projection"]
        )
        _get_projection_head(model, "guideline_slot_projection").load_state_dict(
            projection_state["slot_projection"]
        )
    model.to(resolved_device)
    _align_projection_heads_to_model(model, torch, resolved_device)
    model.train(mode=bool(adapter_trainable))
    return PlannerModelBundle(
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        device=str(resolved_device),
        hidden_size=hidden_size,
        model_name=model_name,
        model_revision=model_revision,
        model_snapshot_path=str(Path(model_name).resolve())
        if Path(model_name).exists()
        else None,
        adapter_path=str(adapter_path) if adapter_path else None,
        tokenizer_path=str(tokenizer_path) if tokenizer_path else None,
        retrieval_projection_path=str(retrieval_projection_path)
        if retrieval_projection_path
        else None,
    )


def enable_non_reentrant_gradient_checkpointing(model: Any) -> dict[str, Any]:
    """Enable the only checkpointing mode supported by Planner V2 training.

    Non-reentrant checkpointing records the autograd graph during forward and
    therefore remains compatible with frozen base weights plus trainable LoRA,
    bridge, or routing inputs.  Silently falling back to the legacy reentrant
    implementation would reintroduce both correctness and memory surprises, so
    unsupported model runtimes fail before the first optimizer step.
    """

    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise RuntimeError(
            "Planner Decoder model does not support gradient checkpointing."
        )
    try:
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError as exc:
        raise RuntimeError(
            "Planner Decoder requires a Transformers runtime that supports "
            "non-reentrant gradient checkpointing."
        ) from exc
    enable_inputs = getattr(model, "enable_input_require_grads", None)
    if callable(enable_inputs):
        enable_inputs()
    return {
        "enabled": True,
        "use_reentrant": False,
        "input_require_grads_enabled": callable(enable_inputs),
    }


def encode_memory_slots_with_bundle(
    bundle: PlannerModelBundle,
    section_text: str,
    *,
    memory_tokens: int,
    max_encoder_tokens: int = 4096,
) -> np.ndarray:
    """Encode one guideline section with an already-loaded latent planner model."""

    tokenizer = bundle.tokenizer
    torch = bundle.torch
    encoded = _to_device(
        _encode_section_with_memory(
            tokenizer,
            section_text,
            memory_tokens,
            max_length=max_encoder_tokens,
        ),
        bundle.device,
    )
    memory_token_id = tokenizer.convert_tokens_to_ids(MEMORY_TOKEN)
    with torch.no_grad():
        outputs = bundle.model(**encoded, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[-1][0]
    memory_positions = (
        (encoded["input_ids"][0] == memory_token_id)
        .nonzero(as_tuple=False)
        .flatten()
    )
    if len(memory_positions) < memory_tokens:
        raise RuntimeError(
            f"Expected {memory_tokens} memory token states, found {len(memory_positions)}."
        )
    slots = hidden[memory_positions[-memory_tokens:]].detach().cpu().float().numpy()
    return slots.astype("float32")


def projected_slot_embedding_with_bundle(
    bundle: PlannerModelBundle,
    slots: np.ndarray,
) -> np.ndarray:
    """Project latent slots into the trained retrieval space."""

    torch = bundle.torch
    slot_projection = _get_projection_head(bundle.model, "guideline_slot_projection")
    projection_dtype = _module_parameter_dtype(slot_projection) or _input_embedding_dtype(
        bundle.model
    )
    slot_tensor = torch.as_tensor(
        slots,
        dtype=projection_dtype or torch.float32,
        device=bundle.device,
    )
    if slot_tensor.ndim == 2:
        slot_tensor = slot_tensor.unsqueeze(0)
    with torch.no_grad():
        projected = slot_projection(slot_tensor.mean(dim=1))
        projected = torch.nn.functional.normalize(projected, dim=-1)
    return projected[0].detach().cpu().float().numpy().astype("float32")


def projected_query_embedding_with_bundle(
    bundle: PlannerModelBundle,
    query: str,
    *,
    max_query_tokens: int = 256,
) -> np.ndarray:
    """Encode and project a query with the trained query projection head."""

    torch = bundle.torch
    query_projection = _get_projection_head(bundle.model, "guideline_query_projection")
    with torch.no_grad():
        query_emb = _mean_text_embedding(
            bundle.model,
            bundle.tokenizer,
            query,
            max_query_tokens,
            bundle.device,
        )
        projection_dtype = _module_parameter_dtype(query_projection)
        if projection_dtype is not None and getattr(query_emb, "dtype", None) != projection_dtype:
            query_emb = query_emb.to(dtype=projection_dtype)
        projected = query_projection(query_emb)
        projected = torch.nn.functional.normalize(projected, dim=-1)
    return projected[0].detach().cpu().float().numpy().astype("float32")


def generate_with_memory_slots(
    bundle: PlannerModelBundle,
    memory_slots: np.ndarray,
    prompt: str,
    *,
    max_new_tokens: int | str | None = 512,
    generation_diagnostics: dict[str, Any] | None = None,
) -> str:
    """Generate decoder text and optional token-level diagnostics."""

    torch = bundle.torch
    tokenizer = bundle.tokenizer
    slot_dtype = _input_embedding_dtype(bundle.model) or torch.float32
    slot_tensor = torch.as_tensor(memory_slots, dtype=slot_dtype, device=bundle.device)
    if slot_tensor.ndim == 2:
        slot_tensor = slot_tensor.unsqueeze(0)
    if slot_tensor.shape[-1] != bundle.hidden_size:
        raise RuntimeError(
            f"Latent slot hidden size {slot_tensor.shape[-1]} does not match "
            f"model hidden size {bundle.hidden_size}."
        )
    output_ids, prompt_token_count, resolved_max_new_tokens = (
        _generate_from_latent_prefix(
            bundle,
            slot_tensor,
            prompt,
            max_new_tokens=max_new_tokens,
        )
    )
    token_ids = _generated_token_ids(output_ids)
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    diagnostics = {
        "latent_prefix_token_count": int(slot_tensor.shape[1]),
        "prompt_token_count": prompt_token_count,
        "total_input_token_count": int(slot_tensor.shape[1]) + prompt_token_count,
        "requested_max_new_tokens": max_new_tokens,
        "resolved_max_new_tokens": resolved_max_new_tokens,
        "initial_generation": _generation_token_diagnostics(tokenizer, token_ids),
    }

    if generation_diagnostics is not None:
        generation_diagnostics.clear()
        generation_diagnostics.update(diagnostics)
    return text


def _generate_from_latent_prefix(
    bundle: PlannerModelBundle,
    slot_tensor: Any,
    prompt: str,
    *,
    max_new_tokens: int | str | None,
) -> tuple[Any, int, int]:
    torch = bundle.torch
    tokenizer = bundle.tokenizer
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(bundle.device)
    prompt_embeds = _get_input_embeddings(bundle.model)(prompt_ids["input_ids"])
    inputs_embeds = torch.cat([slot_tensor, prompt_embeds], dim=1)
    attention_mask = torch.ones(
        inputs_embeds.shape[:2],
        dtype=torch.long,
        device=bundle.device,
    )
    resolved_max_new_tokens = _resolve_generation_max_new_tokens(
        bundle.model,
        tokenizer,
        int(inputs_embeds.shape[1]),
        max_new_tokens,
    )
    with torch.no_grad():
        output_ids = bundle.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=resolved_max_new_tokens,
            do_sample=False,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    return (
        output_ids,
        int(prompt_ids["input_ids"].shape[1]),
        resolved_max_new_tokens,
    )


def _generated_token_ids(output_ids: Any) -> list[int]:
    sequences = getattr(output_ids, "sequences", output_ids)
    first = sequences[0]
    if hasattr(first, "detach"):
        first = first.detach().cpu()
    values = first.tolist() if hasattr(first, "tolist") else list(first)
    return [int(value) for value in values]


def _generation_token_diagnostics(tokenizer: Any, token_ids: list[int]) -> dict[str, Any]:
    eos_value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_value, (list, tuple, set)):
        eos_ids = [int(value) for value in eos_value]
    elif eos_value is None:
        eos_ids = []
    else:
        eos_ids = [int(eos_value)]
    first_token_id = token_ids[0] if token_ids else None
    return {
        "generated_token_count": len(token_ids),
        "first_token_id": first_token_id,
        "first_token_is_eos": (
            first_token_id in eos_ids if first_token_id is not None else False
        ),
        "eos_token_ids": eos_ids,
        "token_ids_head": token_ids[:16],
        "token_ids_tail": token_ids[-16:] if len(token_ids) > 16 else token_ids[:],
        "decoded_with_special_tokens": tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
        )[:500],
        "decoded_text_empty": not tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        ).strip(),
    }
def _resolve_generation_max_new_tokens(
    model: Any,
    tokenizer: Any,
    prefix_token_count: int,
    requested: int | str | None,
) -> int:
    if not _is_auto_generation_limit(requested):
        try:
            value = int(requested)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"max_new_tokens must be a positive integer or 'auto', got {requested!r}."
            ) from exc
        if value < 1:
            raise ValueError(
                f"max_new_tokens must be a positive integer or 'auto', got {requested!r}."
            )
        return value

    context_window = _model_context_window(model, tokenizer)
    remaining = int(context_window) - int(prefix_token_count)
    if remaining < 1:
        raise RuntimeError(
            "No generation room remains in the model context window: "
            f"context_window={context_window}, prefix_token_count={prefix_token_count}."
        )
    return remaining


def _is_auto_generation_limit(value: int | str | None) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"auto", "none", "unlimited"}:
            return True
        try:
            return int(stripped) <= 0
        except ValueError:
            return False
    return int(value) <= 0


def _model_context_window(model: Any, tokenizer: Any) -> int:
    candidates: list[int] = []
    config = getattr(model, "config", None)
    for name in (
        "max_position_embeddings",
        "max_sequence_length",
        "seq_length",
        "n_positions",
    ):
        value = getattr(config, name, None)
        if isinstance(value, int) and 0 < value < 1_000_000:
            candidates.append(value)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        candidates.append(tokenizer_limit)
    if not candidates:
        return 4096
    return min(candidates)


def add_planner_special_tokens(tokenizer: Any, model: Any) -> tuple[Any, Any]:
    existing = set(tokenizer.get_vocab())
    to_add = [token for token in SPECIAL_TOKENS if token not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    current_vocab_size = _input_embedding_vocab_size(model)
    if current_vocab_size is not None and current_vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


class GuidelinePlannerTrainer:
    """Clean Memory Encoder trainer for AE/RETRIEVE/CONTINUE only."""

    def __init__(self, config: PlannerTrainingConfig) -> None:
        self.config = config

    def train(self) -> dict[str, Any]:
        """Run a small LoRA-capable training loop and save adapter/checkpoint metadata."""

        torch, AutoModelForCausalLM, AutoTokenizer, peft_module = _planner_imports()
        accelerator = _build_accelerator(self.config)
        model_source = _resolve_training_model_source(self.config)
        progress = ProgressReporter(
            "train-memory",
            self.config.max_steps,
            unit="step",
            enabled=accelerator.is_main_process,
        )
        progress.message(
            f"loading model and tokenizer from {model_source['model_load_path']}"
        )
        loaded_records = read_jsonl(Path(self.config.train_data_path))
        if not loaded_records:
            raise ValueError(f"No training records found: {self.config.train_data_path}")
        _validate_memory_training_records(loaded_records, self.config)
        records, validation_records = _training_and_validation_records(
            loaded_records,
            self.config,
        )
        if not records:
            raise ValueError("Validation split consumed all training records.")
        output_dir = Path(self.config.output_dir)
        refuse_completed_output(output_dir, "memory_encoder_meta.json")
        if accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()

        model_load_path = model_source["model_load_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_load_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_load_path,
            trust_remote_code=True,
            **_from_pretrained_dtype_kwargs(torch, self.config.model_dtype),
        )
        tokenizer, model = add_planner_special_tokens(tokenizer, model)
        if self.config.fsdp_activation_checkpointing and hasattr(
            model, "gradient_checkpointing_enable"
        ):
            model.gradient_checkpointing_enable()
        if self.config.use_lora:
            lora_config = peft_module.LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            model = peft_module.get_peft_model(model, lora_config)
        hidden_size = int(getattr(model.config, "hidden_size"))
        _attach_projection_heads(model, torch, hidden_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        model, optimizer = accelerator.prepare(model, optimizer)
        device = accelerator.device
        model.train()
        lineage = {
            "train_data_hash": sha256_path(self.config.train_data_path),
            "validation_data_hash": sha256_path(self.config.validation_data_path)
            if self.config.validation_data_path
            else None,
            "base_model_revision": model_source.get("model_revision"),
            "base_model_snapshot_path": model_source.get("model_snapshot_path"),
        }
        effective_config_hash = training_config_hash(asdict(self.config))
        resumed = None
        if self.config.auto_resume:
            resumed = load_latest_training_checkpoint(
                torch_module=torch,
                output_dir=output_dir,
                stage="memory_encoder",
                config_hash=effective_config_hash,
                lineage=lineage,
                map_location=device,
            )
        start_step = 0
        resumed_last_event: dict[str, Any] | None = None
        if resumed is not None:
            _load_named_trainable_state(
                accelerator.unwrap_model(model), resumed["model_trainable_state"]
            )
            optimizer.load_state_dict(resumed["optimizer_state"])
            start_step = int(resumed["global_step"])
            resumed_last_event = dict(resumed.get("last_train_event") or {}) or None
        elif accelerator.is_main_process:
            (output_dir / "metrics.jsonl").write_text("", encoding="utf-8")
        accelerator.wait_for_everyone()
        section_text_by_id = _section_text_by_memory_id(records + validation_records)
        sampler = DeterministicTaskSampler(
            records,
            task_ratios=self.config.task_ratios,
            seed=self.config.sampling_seed,
            group_temperature=self.config.group_sampling_temperature,
        )
        sampled_records, sampling_audit = sampler.sample(
            self.config.max_steps,
            world_size=accelerator.num_processes,
            rank=accelerator.process_index,
        )
        progress.start(start_step, status="resumed" if start_step else "training")

        loss_log: list[dict[str, Any]] = [resumed_last_event] if resumed_last_event else []
        eval_log: list[dict[str, Any]] = []
        for step in range(start_step, self.config.max_steps):
            example = sampled_records[step]
            with accelerator.accumulate(model):
                if example.get("task") == "RETRIEVE":
                    loss = _retrieval_contrastive_loss(
                        model,
                        tokenizer,
                        example,
                        section_text_by_id,
                        self.config.memory_tokens,
                        self.config.max_query_tokens,
                        self.config.max_encoder_tokens,
                        self.config.max_retrieval_candidates,
                        self.config.train_memory_encoder,
                        device,
                    )
                    if loss is None:
                        loss = _zero_loss(model)
                else:
                    loss = _causal_memory_loss(
                        model,
                        tokenizer,
                        example,
                        self.config.memory_tokens,
                        self.config.max_encoder_tokens,
                        self.config.max_decoder_tokens,
                        self.config.train_memory_encoder,
                        device,
                    )
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            mean_loss = accelerator.reduce(loss.detach(), reduction="mean")
            if accelerator.is_main_process:
                train_event = {
                    "step": step + 1,
                    "split": "train",
                    "task": str(example.get("task") or "unknown"),
                    "cancer_type": example.get("cancer_type"),
                    "guideline_id": example.get("guideline_id"),
                    "loss": float(mean_loss.cpu()),
                }
                loss_log.append(train_event)
                _append_jsonl(output_dir / "metrics.jsonl", train_event)
                progress.update(
                    step + 1,
                    metrics={
                        "task": train_event["task"],
                        "loss": train_event["loss"],
                    },
                )
            if _should_evaluate(step + 1, self.config, validation_records):
                progress.message(f"running validation at step {step + 1}")
                eval_event = _evaluate_records(
                    accelerator,
                    model,
                    tokenizer,
                    validation_records,
                    section_text_by_id,
                    self.config,
                    device,
                    step + 1,
                    progress_label=(
                        f"memory-eval@{step + 1}"
                        if accelerator.is_main_process
                        else None
                    ),
                )
                if accelerator.is_main_process and eval_event:
                    eval_log.append(eval_event)
                    _append_jsonl(output_dir / "metrics.jsonl", eval_event)
                    progress.update(
                        step + 1,
                        metrics={"status": "validated", "val_loss": eval_event.get("loss")},
                        force=True,
                    )
            if self.config.save_steps > 0 and (
                (step + 1) % self.config.save_steps == 0
                or step + 1 == self.config.max_steps
            ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    progress.message(f"saving checkpoint at step {step + 1}")
                    save_training_checkpoint(
                        torch_module=torch,
                        output_dir=output_dir,
                        stage="memory_encoder",
                        global_step=step + 1,
                        config_hash=effective_config_hash,
                        lineage=lineage,
                        keep_last=self.config.keep_last_checkpoints,
                        state={
                            "model_trainable_state": _named_trainable_state(
                                accelerator.unwrap_model(model)
                            ),
                            "optimizer_state": optimizer.state_dict(),
                            "last_train_event": loss_log[-1] if loss_log else None,
                            "sampling_audit": sampling_audit.to_dict(),
                        },
                    )
                accelerator.wait_for_everyone()

        accelerator.wait_for_everyone()
        progress.message("saving final Memory Encoder artifacts")
        state_dict = None
        if self.config.use_lora:
            state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            adapter_dir = output_dir / (
                "memory_encoder_adapter" if self.config.use_lora else "memory_encoder_model"
            )
            if hasattr(unwrapped_model, "save_pretrained"):
                save_kwargs = {}
                if state_dict is not None:
                    save_kwargs["state_dict"] = state_dict
                try:
                    unwrapped_model.save_pretrained(
                        adapter_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                        **save_kwargs,
                    )
                except TypeError:
                    unwrapped_model.save_pretrained(
                        adapter_dir,
                        **save_kwargs,
                    )
            torch.save(
                {
                    "query_projection": _get_projection_head(
                        unwrapped_model, "guideline_query_projection"
                    ).state_dict(),
                    "slot_projection": _get_projection_head(
                        unwrapped_model, "guideline_slot_projection"
                    ).state_dict(),
                },
                output_dir / "retrieval_projection.pt",
            )
            tokenizer.save_pretrained(output_dir / "tokenizer")
            training_config = dict(self.config.__dict__)
            training_config.update(model_source)
            training_config["artifact_role"] = "memory_encoder"
            training_config["memory_encoder_adapter_path"] = str(adapter_dir.resolve())
            write_json(output_dir / "training_config.json", training_config)
            write_json(output_dir / "sampling_audit.json", sampling_audit.to_dict())
            write_json(
                output_dir / "memory_encoder_meta.json",
                {
                    "format_version": 2,
                    "artifact_role": "memory_encoder",
                    "base_model_name": self.config.model_name,
                    "base_model_revision": self.config.model_revision,
                    "base_model_snapshot_path": model_source.get("model_snapshot_path"),
                    "memory_encoder_adapter_path": str(adapter_dir.resolve()),
                    "memory_encoder_adapter_hash": sha256_path(adapter_dir),
                    "tokenizer_path": str((output_dir / "tokenizer").resolve()),
                    "tokenizer_hash": sha256_path(output_dir / "tokenizer"),
                    "retrieval_projection_path": str(
                        (output_dir / "retrieval_projection.pt").resolve()
                    ),
                    "retrieval_projection_hash": sha256_path(
                        output_dir / "retrieval_projection.pt"
                    ),
                    "task_ratios": dict(self.config.task_ratios),
                    "task_schema_hash": sha256_json(
                        {
                            "tasks": sorted(DEFAULT_TASK_RATIOS),
                            "ratios": self.config.task_ratios,
                            "memory_tokens": self.config.memory_tokens,
                        }
                    ),
                },
            )
            write_json(output_dir / "training_log.json", loss_log)
            write_json(output_dir / "eval_log.json", eval_log)
            write_json(
                output_dir / "eval_metrics.json",
                eval_log[-1]
                if eval_log
                else {
                    "status": "skipped",
                    "reason": "no validation records or eval disabled",
                    "validation_examples": len(validation_records),
                },
            )
            progress.finish(metrics={"loss": loss_log[-1]["loss"] if loss_log else None})
        accelerator.wait_for_everyone()
        return {
            "output_dir": str(output_dir),
            "steps": self.config.max_steps,
            "resumed_from_step": start_step,
            "validation_examples": len(validation_records),
            "final_eval": eval_log[-1] if eval_log else None,
            "model_name": self.config.model_name,
            "model_load_path": model_source["model_load_path"],
            "model_snapshot_path": model_source.get("model_snapshot_path"),
            "model_revision": self.config.model_revision,
            "use_lora": self.config.use_lora,
            "training_stage": self.config.training_stage,
            "memory_encoder_adapter": str(
                output_dir
                / ("memory_encoder_adapter" if self.config.use_lora else "memory_encoder_model")
            ),
            "sampling": sampling_audit.to_dict(),
            "num_processes": accelerator.num_processes,
            "mixed_precision": accelerator.mixed_precision,
            "sharding": self.config.sharding,
            "is_main_process": accelerator.is_main_process,
        }


def train_planner(config: PlannerTrainingConfig | Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, PlannerTrainingConfig):
        config = PlannerTrainingConfig(**dict(config))
    return GuidelinePlannerTrainer(config).train()


def _validate_memory_training_records(
    records: list[dict[str, Any]],
    config: PlannerTrainingConfig,
) -> None:
    if config.training_stage != "memory_encoder":
        raise ValueError(
            "GuidelinePlannerTrainer is the V2 Memory Encoder trainer; use the "
            "Planner Decoder V2 trainer for trajectory supervision."
        )
    allowed = set(DEFAULT_TASK_RATIOS)
    tasks = {str(record.get("task") or "") for record in records}
    unsupported = tasks - allowed
    if unsupported:
        raise ValueError(
            "Memory Encoder data may contain only AE/RETRIEVE/CONTINUE; found: "
            + ", ".join(sorted(unsupported))
        )
    missing = allowed - tasks
    if missing:
        raise ValueError(
            "Memory Encoder data is missing configured task families: "
            + ", ".join(sorted(missing))
        )
    if any("patient_state" in record or record.get("task") == "PLAN" for record in records):
        raise ValueError("Chunk-derived PLAN supervision is forbidden in Memory Encoder V2 data.")


def _named_trainable_state(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _load_named_trainable_state(model: Any, state: Mapping[str, Any]) -> None:
    parameters = dict(model.named_parameters())
    missing = sorted(set(state) - set(parameters))
    if missing:
        raise RuntimeError(f"Memory Encoder checkpoint has unknown parameters: {missing[:5]}")
    for name, value in state.items():
        parameters[name].data.copy_(value.to(parameters[name]))


def _training_and_validation_records(
    records: list[dict[str, Any]],
    config: PlannerTrainingConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if config.validation_data_path:
        return records, read_jsonl(Path(config.validation_data_path))
    return _split_train_validation_records(records, config.validation_split)


def _split_train_validation_records(
    records: list[dict[str, Any]],
    validation_split: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split = max(min(float(validation_split or 0.0), 0.9), 0.0)
    if split <= 0.0 or len(records) < 2:
        return list(records), []
    desired_count = max(1, round(len(records) * split))
    desired_count = min(desired_count, len(records) - 1)
    indices_by_task: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        task = str(record.get("task") or "unknown")
        indices_by_task.setdefault(task, []).append(index)

    remaining_by_task = {
        task: list(indices)
        for task, indices in indices_by_task.items()
        if len(indices) > 1
    }
    validation_indices: set[int] = set()
    task_order = sorted(remaining_by_task)
    while len(validation_indices) < desired_count:
        progressed = False
        for task in task_order:
            remaining = remaining_by_task.get(task)
            if not remaining or len(remaining) <= 1:
                continue
            validation_indices.add(remaining.pop())
            progressed = True
            if len(validation_indices) >= desired_count:
                break
        if not progressed:
            break

    if not validation_indices:
        return list(records), []
    train_records = [
        record for index, record in enumerate(records) if index not in validation_indices
    ]
    validation_records = [
        record for index, record in enumerate(records) if index in validation_indices
    ]
    return train_records, validation_records


def _should_evaluate(
    step: int,
    config: PlannerTrainingConfig,
    validation_records: list[dict[str, Any]],
) -> bool:
    if not validation_records or int(config.eval_steps) <= 0:
        return False
    return step == int(config.max_steps) or step % int(config.eval_steps) == 0


def _evaluate_records(
    accelerator: Any,
    model: Any,
    tokenizer: Any,
    validation_records: list[dict[str, Any]],
    section_text_by_id: Mapping[str, str],
    config: PlannerTrainingConfig,
    device: str,
    step: int,
    progress_label: str | None = None,
) -> dict[str, Any]:
    import math
    import torch

    max_examples = int(config.eval_max_examples)
    records = balanced_validation_records(
        validation_records,
        max_examples,
        seed=config.sampling_seed,
    )
    if not records:
        return {}

    was_training = bool(getattr(model, "training", False))
    model.eval()
    loss_sum = 0.0
    count = 0
    skipped = 0
    causal_loss_sum = 0.0
    causal_count = 0
    retrieval_loss_sum = 0.0
    retrieval_count = 0
    retrieval_top1 = 0.0
    retrieval_mrr = 0.0
    per_task: dict[str, dict[str, float]] = {}
    progress = ProgressReporter(
        progress_label or "memory-eval",
        len(records),
        unit="example",
        enabled=progress_label is not None,
    )
    progress.start(status="validating")

    try:
        with torch.no_grad():
            for index, example in enumerate(records):
                task = str(example.get("task") or "unknown")
                if task == "RETRIEVE":
                    result = _retrieval_contrastive_result(
                        model,
                        tokenizer,
                        example,
                        section_text_by_id,
                        config.memory_tokens,
                        config.max_query_tokens,
                        config.max_encoder_tokens,
                        config.max_retrieval_candidates,
                        config.train_memory_encoder,
                        device,
                    )
                    if result is None:
                        skipped += 1
                        progress.update(
                            index + 1,
                            metrics={"task": task, "status": "skipped"},
                        )
                        continue
                    loss_value = float(result["loss"].detach().float().cpu())
                    retrieval_loss_sum += loss_value
                    retrieval_count += 1
                    retrieval_top1 += float(result["top1_correct"])
                    retrieval_mrr += float(result["mrr"])
                else:
                    loss = _causal_memory_loss(
                        model,
                        tokenizer,
                        example,
                        config.memory_tokens,
                        config.max_encoder_tokens,
                        config.max_decoder_tokens,
                        config.train_memory_encoder,
                        device,
                    )
                    loss_value = float(loss.detach().float().cpu())
                    causal_loss_sum += loss_value
                    causal_count += 1

                loss_sum += loss_value
                count += 1
                task_entry = per_task.setdefault(task, {"loss_sum": 0.0, "count": 0.0})
                task_entry["loss_sum"] += loss_value
                task_entry["count"] += 1.0
                progress.update(
                    index + 1,
                    metrics={"task": task, "loss": loss_value},
                )
    finally:
        if was_training:
            model.train()
    progress.finish(metrics={"status": "validated"})

    metrics: dict[str, Any] = {
        "step": step,
        "split": "validation",
        "num_examples": count,
        "skipped_examples": skipped,
        "loss": (loss_sum / count) if count else None,
        "per_task": {
            task: {
                "loss": values["loss_sum"] / values["count"],
                "count": int(values["count"]),
            }
            for task, values in sorted(per_task.items())
            if values["count"] > 0
        },
    }
    if causal_count:
        causal_loss = causal_loss_sum / causal_count
        metrics["causal_lm"] = {
            "loss": causal_loss,
            "perplexity": math.exp(min(causal_loss, 20.0)),
            "count": causal_count,
        }
    if retrieval_count:
        metrics["retrieval"] = {
            "loss": retrieval_loss_sum / retrieval_count,
            "top1_accuracy": retrieval_top1 / retrieval_count,
            "mrr": retrieval_mrr / retrieval_count,
            "count": retrieval_count,
        }
    return metrics


def _causal_memory_loss(
    model: Any,
    tokenizer: Any,
    example: Mapping[str, Any],
    memory_tokens: int,
    max_encoder_tokens: int,
    max_decoder_tokens: int,
    train_memory_encoder: bool,
    device: str,
) -> Any:
    import torch

    section_text = str(example.get("encoder_text") or "")
    prompt = f"{example.get('task_token', '')}\n{example.get('prompt', '')}\n"
    target = str(example.get("target") or "")
    memory_token_id = tokenizer.convert_tokens_to_ids(MEMORY_TOKEN)
    encoder_ids = _to_device(
        _encode_section_with_memory(
            tokenizer,
            section_text,
            memory_tokens,
            max_length=max_encoder_tokens,
        ),
        device,
    )
    if train_memory_encoder:
        encoder_outputs = model(**encoder_ids, output_hidden_states=True, use_cache=False)
    else:
        with torch.no_grad():
            encoder_outputs = model(**encoder_ids, output_hidden_states=True, use_cache=False)
    hidden = encoder_outputs.hidden_states[-1]
    memory_positions = (encoder_ids["input_ids"][0] == memory_token_id).nonzero(as_tuple=False).flatten()
    memory_slots = hidden[:, memory_positions[-memory_tokens:], :]
    if not train_memory_encoder:
        memory_slots = memory_slots.detach()

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max(max_decoder_tokens // 2, 1),
    ).to(device)
    target_token_ids = _encode_target_with_eos(
        tokenizer,
        target,
        max_length=max(max_decoder_tokens - prompt_ids["input_ids"].shape[1], 1),
    )
    target_ids = torch.tensor(
        [target_token_ids],
        dtype=torch.long,
        device=device,
    )
    decoder_ids = torch.cat([prompt_ids["input_ids"], target_ids], dim=1)
    decoder_embeds = _get_input_embeddings(model)(decoder_ids)
    inputs_embeds = torch.cat([memory_slots, decoder_embeds], dim=1)
    labels = torch.full(
        (1, memory_tokens + decoder_ids.shape[1]),
        -100,
        dtype=torch.long,
        device=device,
    )
    labels[:, memory_tokens + prompt_ids["input_ids"].shape[1] :] = target_ids
    outputs = model(inputs_embeds=inputs_embeds, labels=labels, use_cache=False)
    return outputs.loss


def _encode_target_with_eos(
    tokenizer: Any,
    target: str,
    *,
    max_length: int,
) -> list[int]:
    """Encode a causal target while reserving its final supervised token for EOS."""

    if max_length < 1:
        raise ValueError("Target max_length must be at least 1 to include EOS.")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise RuntimeError("Planner tokenizer does not define eos_token_id.")
    content_budget = max_length - 1
    content_ids: list[int] = []
    if content_budget > 0:
        encoded = tokenizer.encode(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=content_budget,
        )
        content_ids = [int(token_id) for token_id in encoded[:content_budget]]
    return [*content_ids, int(eos_token_id)]


def encode_prompt_and_full_target(
    tokenizer: Any,
    prompt: str,
    target: str,
    *,
    max_decoder_tokens: int,
) -> tuple[Any, list[int], dict[str, Any]]:
    """Tokenize a causal example without ever truncating its supervised target.

    Planner V2 targets are structured JSON. Truncating one and then appending EOS
    teaches the model to terminate inside an object, so the target receives first
    priority and only the prompt may be truncated to the remaining budget.
    """

    limit = int(max_decoder_tokens)
    if limit < 2:
        raise ValueError("max_decoder_tokens must leave room for prompt and target.")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise RuntimeError("Planner tokenizer does not define eos_token_id.")
    content_ids = tokenizer.encode(
        target,
        add_special_tokens=False,
        truncation=False,
    )
    target_ids = [int(token_id) for token_id in content_ids]
    target_ids.append(int(eos_token_id))
    prompt_budget = limit - len(target_ids)
    if prompt_budget < 1:
        raise ValueError(
            "Planner structured target cannot fit without truncation: "
            f"target_tokens_with_eos={len(target_ids)}, "
            f"max_decoder_tokens={limit}. Increase --max-decoder-tokens; "
            "the target will not be silently truncated."
        )
    untruncated_prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
        truncation=False,
    )
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=prompt_budget,
    )
    actual_prompt_tokens = int(prompt_ids["input_ids"].shape[1])
    return prompt_ids, target_ids, {
        "max_decoder_tokens": limit,
        "target_tokens_with_eos": len(target_ids),
        "prompt_token_budget": prompt_budget,
        "prompt_tokens_original": len(untruncated_prompt_ids),
        "prompt_tokens_used": actual_prompt_tokens,
        "prompt_truncated": len(untruncated_prompt_ids) > actual_prompt_tokens,
    }


def _retrieval_contrastive_loss(
    model: Any,
    tokenizer: Any,
    example: Mapping[str, Any],
    section_text_by_id: Mapping[str, str],
    memory_tokens: int,
    max_query_tokens: int,
    max_encoder_tokens: int,
    max_retrieval_candidates: int,
    train_memory_encoder: bool,
    device: str,
) -> Any | None:
    result = _retrieval_contrastive_result(
        model,
        tokenizer,
        example,
        section_text_by_id,
        memory_tokens,
        max_query_tokens,
        max_encoder_tokens,
        max_retrieval_candidates,
        train_memory_encoder,
        device,
    )
    if result is None:
        return None
    return result["loss"]


def _retrieval_contrastive_result(
    model: Any,
    tokenizer: Any,
    example: Mapping[str, Any],
    section_text_by_id: Mapping[str, str],
    memory_tokens: int,
    max_query_tokens: int,
    max_encoder_tokens: int,
    max_retrieval_candidates: int,
    train_memory_encoder: bool,
    device: str,
) -> dict[str, Any] | None:
    import torch
    import torch.nn.functional as F

    positive_ids = list(example.get("strong_positive") or []) + list(
        example.get("weak_positive") or []
    )
    negative_ids = list(example.get("hard_negative") or []) + list(
        example.get("easy_negative") or []
    )
    selected_positive_ids = [
        memory_id for memory_id in positive_ids if memory_id in section_text_by_id
    ][:1]
    selected_negative_ids = [
        memory_id for memory_id in negative_ids if memory_id in section_text_by_id
    ][: max(max_retrieval_candidates - len(selected_positive_ids), 1)]
    candidate_ids = selected_positive_ids + selected_negative_ids
    if not positive_ids or not negative_ids or len(candidate_ids) < 2:
        return None

    query_emb = _mean_text_embedding(
        model,
        tokenizer,
        str(example.get("query") or example.get("prompt") or ""),
        max_query_tokens,
        device,
    )
    query_projection = _get_projection_head(model, "guideline_query_projection")
    slot_projection = _get_projection_head(model, "guideline_slot_projection")
    query_emb = F.normalize(query_projection(query_emb), dim=-1)
    slot_embeddings = []
    labels = []
    for memory_id in candidate_ids:
        slots = _memory_slots_tensor(
            model,
            tokenizer,
            section_text_by_id[memory_id],
            memory_tokens,
            max_encoder_tokens,
            train_memory_encoder,
            device,
        )
        slot_emb = F.normalize(slot_projection(slots.mean(dim=1)), dim=-1)
        slot_embeddings.append(slot_emb)
        labels.append(1.0 if memory_id in positive_ids else 0.0)
    slot_matrix = torch.cat(slot_embeddings, dim=0)
    logits = torch.matmul(slot_matrix, query_emb.transpose(0, 1)).squeeze(1) / 0.07
    label_tensor = torch.tensor(labels, device=device, dtype=torch.float32)
    positive_mask = label_tensor > 0
    if not positive_mask.any():
        return None
    log_probs = logits - torch.logsumexp(logits, dim=0)
    sorted_indices = logits.detach().argsort(descending=True).tolist()
    positive_indices = {
        index for index, label in enumerate(labels) if label > 0.0
    }
    best_positive_rank = min(
        (
            rank
            for rank, candidate_index in enumerate(sorted_indices, start=1)
            if candidate_index in positive_indices
        ),
        default=0,
    )
    return {
        "loss": -(log_probs[positive_mask]).mean(),
        "candidate_count": len(candidate_ids),
        "top1_correct": 1.0 if sorted_indices and sorted_indices[0] in positive_indices else 0.0,
        "mrr": (1.0 / best_positive_rank) if best_positive_rank else 0.0,
    }


def _zero_loss(model: Any) -> Any:
    loss = next(_iter_parameters(model)).sum() * 0.0
    loss = loss + next(
        _get_projection_head(model, "guideline_query_projection").parameters()
    ).sum() * 0.0
    loss = loss + next(
        _get_projection_head(model, "guideline_slot_projection").parameters()
    ).sum() * 0.0
    return loss


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _mean_text_embedding(
    model: Any,
    tokenizer: Any,
    text: str,
    max_query_tokens: int,
    device: str,
) -> Any:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max(max_query_tokens, 1),
    ).to(device)
    outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[-1]
    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _memory_slots_tensor(
    model: Any,
    tokenizer: Any,
    section_text: str,
    memory_tokens: int,
    max_encoder_tokens: int,
    train_memory_encoder: bool,
    device: str,
) -> Any:
    import torch

    memory_token_id = tokenizer.convert_tokens_to_ids(MEMORY_TOKEN)
    encoded = _to_device(
        _encode_section_with_memory(
            tokenizer,
            section_text,
            memory_tokens,
            max_length=max_encoder_tokens,
        ),
        device,
    )
    if train_memory_encoder:
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    else:
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[-1]
    memory_positions = (encoded["input_ids"][0] == memory_token_id).nonzero(as_tuple=False).flatten()
    slots = hidden[:, memory_positions[-memory_tokens:], :]
    if not train_memory_encoder:
        slots = slots.detach()
    return slots


def _attach_projection_heads(model: Any, torch: Any, hidden_size: int) -> None:
    model.add_module("guideline_query_projection", torch.nn.Linear(hidden_size, hidden_size))
    model.add_module("guideline_slot_projection", torch.nn.Linear(hidden_size, hidden_size))


def _get_wrapped_module(model: Any) -> Any:
    if hasattr(model, "module"):
        return model.module
    return model


def _get_input_embeddings(model: Any) -> Any:
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    module = _get_wrapped_module(model)
    if hasattr(module, "get_input_embeddings"):
        return module.get_input_embeddings()
    raise AttributeError("Wrapped model does not expose get_input_embeddings().")


def _input_embedding_vocab_size(model: Any) -> int | None:
    try:
        embeddings = _get_input_embeddings(model)
    except Exception:
        return None
    weight = getattr(embeddings, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) < 1:
        return None
    return int(shape[0])


def _input_embedding_dtype(model: Any) -> Any | None:
    try:
        embeddings = _get_input_embeddings(model)
    except Exception:
        return None
    weight = getattr(embeddings, "weight", None)
    return getattr(weight, "dtype", None)


def _module_parameter_dtype(module: Any) -> Any | None:
    if not hasattr(module, "parameters"):
        return None
    try:
        parameter = next(module.parameters())
    except StopIteration:
        return None
    return getattr(parameter, "dtype", None)


def _align_projection_heads_to_model(model: Any, torch: Any, device: Any) -> None:
    dtype = _input_embedding_dtype(model)
    for name in ("guideline_query_projection", "guideline_slot_projection"):
        head = _get_projection_head(model, name)
        if not hasattr(head, "to"):
            continue
        kwargs: dict[str, Any] = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        try:
            head.to(**kwargs)
        except TypeError:
            if dtype is not None:
                head.to(dtype=dtype)
            head.to(device)


def _get_projection_head(model: Any, name: str) -> Any:
    if hasattr(model, name):
        return getattr(model, name)
    module = _get_wrapped_module(model)
    if hasattr(module, name):
        return getattr(module, name)
    raise AttributeError(f"Wrapped model does not expose projection head {name!r}.")


def _iter_parameters(model: Any) -> Any:
    if hasattr(model, "parameters"):
        return iter(model.parameters())
    return iter(_get_wrapped_module(model).parameters())


def _section_text_by_memory_id(records: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for record in records:
        memory_id = record.get("guideline_memory_id")
        encoder_text = record.get("encoder_text")
        if isinstance(memory_id, str) and isinstance(encoder_text, str) and memory_id not in mapping:
            mapping[memory_id] = encoder_text
    return mapping


def _encode_section_with_memory(
    tokenizer: Any,
    section_text: str,
    memory_tokens: int,
    *,
    max_length: int = 4096,
) -> Any:
    import torch

    memory_token_id = tokenizer.convert_tokens_to_ids(MEMORY_TOKEN)
    section_ids = tokenizer.encode(section_text, add_special_tokens=False)
    max_section_tokens = max(max_length - memory_tokens, 1)
    section_ids = section_ids[:max_section_tokens]
    input_ids = section_ids + [memory_token_id] * memory_tokens
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
    }


def _to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _planner_imports() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "guideline_planner training requires torch and transformers. "
            "Install with: pip install -e '.[planner]'"
        ) from exc
    try:
        import peft
    except Exception as exc:
        raise RuntimeError(
            "LoRA training requires peft. Install with: pip install -e '.[planner]'"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, peft


def _resolve_training_model_source(config: PlannerTrainingConfig) -> dict[str, Any]:
    """Resolve a floating model id into an exact local snapshot before training."""

    if config.model_snapshot_path:
        snapshot = Path(config.model_snapshot_path).expanduser().resolve()
        if not snapshot.exists():
            raise FileNotFoundError(f"model_snapshot_path does not exist: {snapshot}")
        return {
            "model_name": config.model_name,
            "model_revision": config.model_revision,
            "model_snapshot_path": str(snapshot),
            "model_load_path": str(snapshot),
        }
    model_path = Path(config.model_name).expanduser()
    if model_path.exists():
        snapshot = model_path.resolve()
        return {
            "model_name": config.model_name,
            "model_revision": config.model_revision,
            "model_snapshot_path": str(snapshot),
            "model_load_path": str(snapshot),
        }
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "Resolving a fixed model snapshot requires huggingface_hub. "
            "Install planner dependencies or pass --model-snapshot-path."
        ) from exc
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=config.model_name,
                revision=config.model_revision,
            )
        ).resolve()
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve fixed snapshot for {config.model_name!r}. "
            "Pass --model-snapshot-path with an exact local snapshot directory."
        ) from exc
    return {
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "model_snapshot_path": str(snapshot),
        "model_load_path": str(snapshot),
    }


def _build_accelerator(config: PlannerTrainingConfig) -> Any:
    try:
        from accelerate import Accelerator
        from accelerate.utils import DistributedDataParallelKwargs
    except Exception as exc:
        raise RuntimeError(
            "distributed/single-process planner training requires accelerate. "
            "Install with: pip install -e '.[planner]'"
        ) from exc

    kwargs = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    mixed_precision = config.mixed_precision
    if mixed_precision == "no":
        mixed_precision = None
    sharding = (config.sharding or "none").lower()
    accelerator_kwargs: dict[str, Any] = {
        "gradient_accumulation_steps": max(int(config.gradient_accumulation_steps), 1),
        "mixed_precision": mixed_precision,
        "kwargs_handlers": kwargs,
    }
    if sharding == "fsdp":
        accelerator_kwargs["fsdp_plugin"] = _build_fsdp_plugin(config)
    elif sharding in {"deepspeed-zero3", "deepspeed_zero3", "zero3"}:
        accelerator_kwargs["deepspeed_plugin"] = _build_deepspeed_zero3_plugin(config)
    elif sharding not in {"none", "ddp"}:
        raise ValueError(
            "Unsupported sharding mode. Use one of: none, fsdp, deepspeed-zero3."
        )
    return Accelerator(**accelerator_kwargs)


def _build_fsdp_plugin(config: PlannerTrainingConfig) -> Any:
    try:
        from accelerate import FullyShardedDataParallelPlugin
    except Exception:
        from accelerate.utils import FullyShardedDataParallelPlugin

    transformer_layers = [
        item.strip()
        for item in config.fsdp_transformer_layer_cls_to_wrap.split(",")
        if item.strip()
    ]
    kwargs = {
        "fsdp_version": 1,
        "sharding_strategy": "FULL_SHARD",
        "auto_wrap_policy": "transformer_based_wrap",
        "transformer_cls_names_to_wrap": transformer_layers or None,
        "cpu_ram_efficient_loading": bool(config.fsdp_cpu_ram_efficient_loading),
        "sync_module_states": bool(config.fsdp_cpu_ram_efficient_loading),
        "use_orig_params": True,
        "activation_checkpointing": bool(config.fsdp_activation_checkpointing),
        "state_dict_type": "FULL_STATE_DICT",
    }
    return _construct_with_supported_kwargs(FullyShardedDataParallelPlugin, kwargs)


def _build_deepspeed_zero3_plugin(config: PlannerTrainingConfig) -> Any:
    try:
        from accelerate.utils import DeepSpeedPlugin
    except Exception:
        from accelerate import DeepSpeedPlugin

    kwargs = {
        "zero_stage": 3,
        "gradient_accumulation_steps": max(int(config.gradient_accumulation_steps), 1),
        "train_micro_batch_size_per_gpu": max(
            int(config.train_micro_batch_size_per_gpu), 1
        ),
        "offload_param_device": config.deepspeed_offload_param_device,
        "offload_optimizer_device": config.deepspeed_offload_optimizer_device,
        "zero3_init_flag": True,
        "zero3_save_16bit_model": False,
    }
    if config.deepspeed_nvme_path:
        kwargs["offload_param_nvme_path"] = config.deepspeed_nvme_path
        kwargs["offload_optimizer_nvme_path"] = config.deepspeed_nvme_path
    plugin = _construct_with_supported_kwargs(DeepSpeedPlugin, kwargs)
    _patch_deepspeed_plugin_config(plugin, config)
    return plugin


def _patch_deepspeed_plugin_config(plugin: Any, config: PlannerTrainingConfig) -> None:
    """Force config keys that older Accelerate versions may drop from kwargs."""

    micro_batch_size = max(int(config.train_micro_batch_size_per_gpu), 1)
    gradient_accumulation_steps = max(int(config.gradient_accumulation_steps), 1)
    updates = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    deepspeed_config = getattr(plugin, "deepspeed_config", None)
    if isinstance(deepspeed_config, dict):
        deepspeed_config.update(updates)

    hf_ds_config = getattr(plugin, "hf_ds_config", None)
    hf_config = getattr(hf_ds_config, "config", None)
    if isinstance(hf_config, dict):
        hf_config.update(updates)


def _construct_with_supported_kwargs(cls: Any, kwargs: Mapping[str, Any]) -> Any:
    import inspect

    signature = inspect.signature(cls)
    if any(parameter.kind == parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return cls(**dict(kwargs))
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }
    return cls(**supported)


def _revision_kwargs(model_source: str, model_revision: str | None) -> dict[str, Any]:
    if not model_revision:
        return {}
    if Path(model_source).expanduser().exists():
        return {}
    return {"revision": model_revision}


def _resolve_torch_device(torch: Any, device: str | None) -> str:
    requested = (device or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for guideline planner, but torch.cuda is not available.")
    if requested.startswith("cuda:") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for guideline planner, but torch.cuda is not available.")
    if requested not in {"cpu", "cuda"} and not requested.startswith("cuda:"):
        raise ValueError("device must be one of: auto, cpu, cuda, cuda:<index>.")
    return requested


def _from_pretrained_dtype_kwargs(torch: Any, model_dtype: str) -> dict[str, Any]:
    dtype = (model_dtype or "auto").lower()
    if dtype == "auto":
        return {"torch_dtype": "auto"}
    if dtype in {"no", "none"}:
        return {}
    mapping = {
        "bf16": getattr(torch, "bfloat16"),
        "bfloat16": getattr(torch, "bfloat16"),
        "fp16": getattr(torch, "float16"),
        "float16": getattr(torch, "float16"),
        "fp32": getattr(torch, "float32"),
        "float32": getattr(torch, "float32"),
    }
    if dtype not in mapping:
        raise ValueError("model_dtype must be one of: auto, bf16, fp16, fp32, none.")
    return {"torch_dtype": mapping[dtype]}
