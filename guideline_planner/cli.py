"""Command line interface for the latent guideline-memory planner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from guideline_planner.chunking import chunk_guidelines
from guideline_planner.constants import (
    DEFAULT_GUIDELINE_DIR,
    DEFAULT_MEMORY_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TASK_RATIOS,
)
from guideline_planner.dataset import build_training_data
from guideline_planner.medclaw_adapter import to_medclaw_plan
from guideline_planner.memory import extract_memory_slots
from guideline_planner.modeling import PlannerTrainingConfig, train_planner
from guideline_planner.planner import LatentGuidelinePlanner
from guideline_planner.retrieval import retrieve_latent_guideline_memory

CANONICAL_DATASET_ROOT = "datasets/planner_v2_pilot"
CANONICAL_TRAJECTORY_DATA = f"{CANONICAL_DATASET_ROOT}/dataset"
CANONICAL_CHUNKS = f"{CANONICAL_DATASET_ROOT}/guideline_chunks.jsonl"
CANONICAL_OUTPUT_ROOT = "guideline_planner/outputs/planner_v2_lung_endometrial"
CANONICAL_MEMORY_TRAIN_DATA = f"{CANONICAL_OUTPUT_ROOT}/memory_training_data"
CANONICAL_MEMORY_ENCODER = f"{CANONICAL_OUTPUT_ROOT}/memory_encoder"
CANONICAL_MEMORY_STORE = f"{CANONICAL_OUTPUT_ROOT}/memory_store"
CANONICAL_DECODER = f"{CANONICAL_OUTPUT_ROOT}/planner_decoder"
CANONICAL_ROUTING_DATA = f"{CANONICAL_OUTPUT_ROOT}/routing_data"
CANONICAL_ROUTER = f"{CANONICAL_OUTPUT_ROOT}/router_daa"
CANONICAL_EVALUATION = f"{CANONICAL_OUTPUT_ROOT}/evaluation"
CANONICAL_CONFIG = "guideline_planner/configs/planner_v2_lung_endometrial.yaml"
CANONICAL_ROUTING_CONFIG = "guideline_planner/configs/latent_guideline_routing.yaml"
GPT_TRAJECTORY_CONFIG = (
    "guideline_planner/configs/planner_v2_lung_endometrial_npc.yaml"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    chunk = subparsers.add_parser("chunk-guidelines")
    chunk.add_argument("--guideline-dir", default=DEFAULT_GUIDELINE_DIR)
    chunk.add_argument("--output", required=True)

    data = subparsers.add_parser("build-train-data")
    data.add_argument("--chunks", default=CANONICAL_CHUNKS)
    data.add_argument("--output-dir", default=CANONICAL_MEMORY_TRAIN_DATA)
    data.add_argument("--task-ratios-json", default=None)

    train = subparsers.add_parser("train")
    train.add_argument("--train-data", default=f"{CANONICAL_MEMORY_TRAIN_DATA}/mixed_train.jsonl")
    train.add_argument("--output-dir", default=CANONICAL_MEMORY_ENCODER)
    train.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    train.add_argument(
        "--model-revision",
        default=None,
        help="Optional Hugging Face revision/commit to resolve before training.",
    )
    train.add_argument(
        "--model-snapshot-path",
        default=None,
        help="Exact local model snapshot directory. Overrides --model-name for loading.",
    )
    train.add_argument("--memory-tokens", type=int, default=DEFAULT_MEMORY_TOKENS)
    train.add_argument("--max-steps", type=int, default=5000)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument(
        "--validation-data",
        default=None,
        help="Optional JSONL validation data. If omitted, a validation split is held out from --train-data.",
    )
    train.add_argument(
        "--validation-split",
        type=float,
        default=0.05,
        help="Fraction of --train-data to hold out for validation when --validation-data is not set.",
    )
    train.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Run validation every N optimizer steps and at the final step. Set 0 to disable.",
    )
    train.add_argument(
        "--eval-max-examples",
        type=int,
        default=64,
        help="Maximum validation examples evaluated each time. Set <=0 to evaluate all validation records.",
    )
    train.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train.add_argument("--save-steps", type=int, default=500)
    train.add_argument("--keep-last-checkpoints", type=int, default=2)
    train.add_argument("--no-auto-resume", action="store_true")
    train.add_argument(
        "--train-micro-batch-size-per-gpu",
        type=int,
        default=1,
        help="DeepSpeed train_micro_batch_size_per_gpu. This trainer currently consumes one example per rank by default.",
    )
    train.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16"),
        default=None,
        help="Accelerate mixed precision mode. Defaults to the accelerate config.",
    )
    train.add_argument(
        "--model-dtype",
        choices=("auto", "bf16", "fp16", "fp32", "none"),
        default="auto",
        help="dtype passed to from_pretrained; use bf16/fp16 to reduce load memory.",
    )
    train.add_argument(
        "--max-encoder-tokens",
        type=int,
        default=1024,
        help="Maximum section tokens before appending memory tokens. Lower this to reduce attention memory.",
    )
    train.add_argument(
        "--max-decoder-tokens",
        type=int,
        default=512,
        help="Maximum prompt+target tokens in AE/CONTINUE decoder loss.",
    )
    train.add_argument(
        "--max-query-tokens",
        type=int,
        default=256,
        help="Maximum query tokens for retrieval contrastive loss.",
    )
    train.add_argument(
        "--max-retrieval-candidates",
        type=int,
        default=2,
        help="Maximum positive/negative slot candidates encoded per retrieval step.",
    )
    train.add_argument(
        "--freeze-memory-encoder",
        action="store_false",
        dest="train_memory_encoder",
        default=True,
        help=(
            "Freeze the memory encoder path. V2 trains it by default on "
            "AE/RETRIEVE/CONTINUE only."
        ),
    )
    train.add_argument(
        "--sharding",
        choices=("none", "fsdp", "deepspeed-zero3"),
        default="none",
        help="Parameter sharding backend. Use deepspeed-zero3 or fsdp when one GPU cannot hold the model.",
    )
    train.add_argument(
        "--fsdp-transformer-layer-cls-to-wrap",
        default="Qwen3DecoderLayer,Qwen2DecoderLayer",
        help="Comma-separated transformer layer class names for FSDP auto wrapping.",
    )
    train.add_argument(
        "--no-fsdp-cpu-ram-efficient-loading",
        action="store_true",
        help="Disable FSDP CPU RAM efficient loading.",
    )
    train.add_argument(
        "--fsdp-activation-checkpointing",
        action="store_true",
        help="Enable model gradient checkpointing / FSDP activation checkpointing.",
    )
    train.add_argument(
        "--deepspeed-offload-param-device",
        choices=("none", "cpu", "nvme"),
        default="none",
        help="DeepSpeed ZeRO-3 parameter offload device.",
    )
    train.add_argument(
        "--deepspeed-offload-optimizer-device",
        choices=("none", "cpu", "nvme"),
        default="none",
        help="DeepSpeed ZeRO-3 optimizer offload device.",
    )
    train.add_argument(
        "--deepspeed-nvme-path",
        default=None,
        help="NVMe folder for DeepSpeed offload when an offload device is nvme.",
    )
    train.add_argument("--no-lora", action="store_true")

    extract = subparsers.add_parser("extract-memory")
    extract.add_argument("--chunks", default=CANONICAL_CHUNKS)
    extract.add_argument("--output-dir", default=CANONICAL_MEMORY_STORE)
    extract.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    extract.add_argument("--model-revision", default=None)
    extract.add_argument("--model-snapshot-path", default=None)
    extract.add_argument("--memory-tokens", type=int, default=None)
    extract.add_argument("--training-run-dir", default=CANONICAL_MEMORY_ENCODER)
    extract.add_argument("--memory-encoder-adapter-path", default=None)
    extract.add_argument("--tokenizer-path", default=None)
    extract.add_argument("--retrieval-projection-path", default=None)
    extract.add_argument("--device", default=None)
    extract.add_argument(
        "--model-dtype",
        choices=("auto", "bf16", "fp16", "fp32", "none"),
        default="auto",
    )
    extract.add_argument("--max-encoder-tokens", type=int, default=4096)
    extract.add_argument("--mock", action="store_true")

    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("--memory-dir", required=True)
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--top-k", type=int, default=3)
    retrieve.add_argument("--filter", action="append", default=[])
    retrieve.add_argument("--device", default=None)
    retrieve.add_argument(
        "--model-dtype",
        choices=("auto", "bf16", "fp16", "fp32", "none"),
        default="auto",
    )
    retrieve.add_argument("--max-query-tokens", type=int, default=256)

    demo = subparsers.add_parser("demo-next-step")
    demo.add_argument("--memory-dir", required=True)
    demo.add_argument("--decoder-artifact-dir", required=True)
    demo.add_argument("--patient-state-json", default=None)
    demo.add_argument("--top-k", type=int, default=3)
    demo.add_argument("--device", default=None)
    demo.add_argument("--query-encoder-device", default="cpu")
    demo.add_argument(
        "--max-new-tokens",
        default="512",
        help="Planner generation budget. Use a positive integer or 'auto' to use remaining model context.",
    )
    demo.add_argument(
        "--output-mode",
        choices=("json",),
        default="json",
        help="Planner V2 only accepts strict planner_action.v2 JSON.",
    )
    demo.add_argument(
        "--model-dtype",
        choices=("auto", "bf16", "fp16", "fp32", "none"),
        default="auto",
    )
    demo.add_argument(
        "--routing-config",
        default=None,
        help="YAML/JSON patient-state-conditioned routing config.",
    )
    demo.add_argument(
        "--routing-checkpoint",
        default=None,
        help="Trained routing checkpoint bound to --memory-dir.",
    )
    routing_data = subparsers.add_parser(
        "build-routing-data",
        help="Build Router/DAA examples from validated planner_trajectory.v2 data.",
    )
    routing_data.add_argument("--trajectory-data", default=CANONICAL_TRAJECTORY_DATA)
    routing_data.add_argument("--output", default=CANONICAL_ROUTING_DATA)
    routing_data.add_argument("--allow-nonrelease-smoke", action="store_true")

    router_train = subparsers.add_parser(
        "train-router",
        help="Train current-state retrieval, gate, and DAA fusion.",
    )
    router_train.add_argument("--train-data", default=CANONICAL_ROUTING_DATA)
    router_train.add_argument("--memory-dir", default=CANONICAL_MEMORY_STORE)
    router_train.add_argument("--decoder-artifact-dir", default=CANONICAL_DECODER)
    router_train.add_argument("--output-dir", default=CANONICAL_ROUTER)
    router_train.add_argument(
        "--routing-config",
        default=CANONICAL_ROUTING_CONFIG,
    )
    router_train.add_argument("--device", default="cuda")
    router_train.add_argument(
        "--model-dtype",
        choices=("auto", "bf16", "fp16", "fp32", "none"),
        default="bf16",
    )
    router_train.add_argument("--max-steps", type=int, default=5000)
    router_train.add_argument("--learning-rate", type=float, default=1e-4)
    router_train.add_argument("--max-decoder-tokens", type=int, default=2048)
    router_train.add_argument("--gradient-accumulation-steps", type=int, default=4)
    router_train.add_argument("--eval-steps", type=int, default=100)
    router_train.add_argument("--save-steps", type=int, default=500)
    router_train.add_argument("--keep-last-checkpoints", type=int, default=2)
    router_train.add_argument("--no-auto-resume", action="store_true")
    router_train.add_argument("--seed", type=int, default=17)
    router_train.add_argument("--generation-eval-steps", type=int, default=500)
    router_train.add_argument("--generation-eval-examples", type=int, default=8)
    router_train.add_argument("--generation-max-new-tokens", type=int, default=1024)

    trajectory = subparsers.add_parser(
        "build-planner-data-v2",
        help="Validate, group-split, and audit planner_trajectory.v2 candidates.",
    )
    trajectory.add_argument("--input", required=True)
    trajectory.add_argument("--output-dir", required=True)
    trajectory.add_argument("--seed", type=int, default=17)
    trajectory.add_argument("--rule-registry", default=None)
    trajectory.add_argument("--memory-dir", default=None)
    trajectory.add_argument(
        "--release-scope",
        default=f"{CANONICAL_DATASET_ROOT}/release_scope.json",
    )
    trajectory.add_argument("--allow-incomplete", action="store_true")
    trajectory.add_argument("--fail-if-not-ready", action="store_true")

    compile_rules = subparsers.add_parser(
        "compile-action-set-v2",
        help="Compile rule-permitted action buckets for one patient_state.v2.",
    )
    compile_rules.add_argument("--patient-state-json", required=True)
    compile_rules.add_argument("--rule-registry", required=True)
    compile_rules.add_argument("--output", default=None)

    validate_data = subparsers.add_parser("validate-planner-data-v2")
    validate_data.add_argument("--input", required=True)
    validate_data.add_argument("--allow-incomplete", action="store_true")
    validate_data.add_argument("--rule-registry", default=None)
    validate_data.add_argument("--memory-dir", default=None)
    validate_data.add_argument(
        "--release-scope",
        default=f"{CANONICAL_DATASET_ROOT}/release_scope.json",
    )

    observations = subparsers.add_parser("extract-observation-transitions")
    observations.add_argument("--run-dir", action="append", default=[])
    observations.add_argument("--runs-root", default=None)
    observations.add_argument("--output", required=True)

    approve_reviews = subparsers.add_parser(
        "approve-planner-reviews",
        help="Synchronize a completed human review across all Planner V2 artifacts.",
    )
    approve_reviews.add_argument("--dataset-root", default=CANONICAL_DATASET_ROOT)
    approve_reviews.add_argument("--reviewer-id", default="manual-review-team")
    approve_reviews.add_argument("--reviewed-at", default=None)
    approve_reviews.add_argument(
        "--release-scope",
        default=f"{CANONICAL_DATASET_ROOT}/release_scope.json",
    )

    gpt_data = subparsers.add_parser(
        "gpt-planner-data-v2",
        help=(
            "Prepare, inspect, validate, and merge the resumable GPT-reviewed "
            "three-cancer planner_trajectory.v2 dataset."
        ),
    )
    gpt_data.add_argument(
        "stage",
        choices=(
            "prepare",
            "generate-npc",
            "review-existing",
            "repair",
            "verify",
            "approve-manual",
            "validate",
            "merge",
            "status",
        ),
    )
    gpt_data.add_argument("--config", default=GPT_TRAJECTORY_CONFIG)
    gpt_data.add_argument("--model", default="gpt-5.6-sol")
    gpt_data.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="high",
    )
    gpt_data.add_argument("--max-parallel", type=int, default=3)
    gpt_data.add_argument("--max-repair-cycles", type=int, default=2)
    gpt_data.add_argument(
        "--reviewer-id", default="professional-clinician-review-team"
    )
    gpt_data.add_argument("--reviewed-at", default=None)
    gpt_data.add_argument(
        "--force",
        action="store_true",
        help="Rebuild stale stage artifacts; valid hash-matched outputs remain protected.",
    )

    validate_gpt_output = subparsers.add_parser(
        "validate-gpt-planner-output",
        help="Validate one case-isolated GPT teacher/review/repair/verifier output.",
    )
    validate_gpt_output.add_argument("--packet", required=True)
    validate_gpt_output.add_argument("--output", required=True)
    validate_gpt_output.add_argument("--config", default=GPT_TRAJECTORY_CONFIG)

    decoder_train = subparsers.add_parser("train-planner-decoder")
    decoder_train.add_argument("--trajectory-data", default=CANONICAL_TRAJECTORY_DATA)
    decoder_train.add_argument("--memory-dir", default=CANONICAL_MEMORY_STORE)
    decoder_train.add_argument("--output-dir", default=CANONICAL_DECODER)
    decoder_train.add_argument("--device", default="cuda")
    decoder_train.add_argument("--model-dtype", default="bf16")
    decoder_train.add_argument("--max-steps", type=int, default=5000)
    decoder_train.add_argument("--learning-rate", type=float, default=2e-4)
    decoder_train.add_argument("--max-decoder-tokens", type=int, default=2048)
    decoder_train.add_argument("--gradient-accumulation-steps", type=int, default=4)
    decoder_train.add_argument("--eval-steps", type=int, default=100)
    decoder_train.add_argument("--save-steps", type=int, default=500)
    decoder_train.add_argument("--keep-last-checkpoints", type=int, default=2)
    decoder_train.add_argument("--no-auto-resume", action="store_true")
    decoder_train.add_argument("--seed", type=int, default=17)
    decoder_train.add_argument("--allow-nonrelease-smoke", action="store_true")
    decoder_train.add_argument("--overfit-examples", type=int, default=0)
    decoder_train.add_argument("--generation-eval-steps", type=int, default=500)
    decoder_train.add_argument("--generation-eval-examples", type=int, default=8)
    decoder_train.add_argument("--generation-max-new-tokens", type=int, default=1024)
    decoder_train.add_argument(
        "--require-generation-schema-pass-rate",
        type=float,
        default=0.0,
    )

    routing_eval = subparsers.add_parser(
        "evaluate-routing",
        help="Aggregate current-state routing metrics from benchmark runs.",
    )
    routing_eval.add_argument("--run-dir", action="append", default=[])
    routing_eval.add_argument("--runs-root", default=None)
    routing_eval.add_argument("--output", default=None)

    planner_v2_eval = subparsers.add_parser(
        "evaluate-planner-v2",
        help="Apply the fixed planner_action.v2 acceptance gates to test predictions.",
    )
    planner_v2_eval.add_argument("--examples", required=True)
    planner_v2_eval.add_argument("--output", default=None)

    router_v2_eval = subparsers.add_parser(
        "evaluate-router-v2",
        help="Apply Router/DAA V2 acceptance gates to fixed test predictions.",
    )
    router_v2_eval.add_argument("--examples", required=True)
    router_v2_eval.add_argument("--output", default=None)

    daa_eval = subparsers.add_parser(
        "evaluate-daa-comparison",
        help="Evaluate paired DAA versus latent_topk action scores.",
    )
    daa_eval.add_argument(
        "--comparison-json",
        required=True,
        help="JSON containing daa_scores and latent_topk_scores arrays.",
    )
    daa_eval.add_argument("--bootstrap-samples", type=int, default=2000)
    daa_eval.add_argument("--seed", type=int, default=17)
    daa_eval.add_argument("--output", default=None)

    predict = subparsers.add_parser(
        "predict-planner-v2",
        help="Generate latent_topk/DAA predictions over the immutable test split.",
    )
    predict.add_argument("--trajectory-data", default=CANONICAL_TRAJECTORY_DATA)
    predict.add_argument("--release-dir", default=None)
    predict.add_argument("--memory-dir", default=CANONICAL_MEMORY_STORE)
    predict.add_argument("--decoder-artifact-dir", default=CANONICAL_DECODER)
    predict.add_argument("--runtime-decoder-artifact-dir", default=CANONICAL_ROUTER)
    predict.add_argument("--output-dir", default=CANONICAL_EVALUATION)
    predict.add_argument(
        "--mode",
        action="append",
        choices=("latent_topk", "daa_full"),
        default=None,
    )
    predict.add_argument("--routing-config", default=CANONICAL_ROUTING_CONFIG)
    predict.add_argument(
        "--routing-checkpoint",
        default=f"{CANONICAL_ROUTER}/routing_checkpoint.pt",
    )
    predict.add_argument("--top-k", type=int, default=4)
    predict.add_argument("--device", default="cuda")
    predict.add_argument("--query-encoder-device", default="cpu")
    predict.add_argument("--model-dtype", default="bf16")
    predict.add_argument("--max-new-tokens", default="1024")
    predict.add_argument("--bootstrap-samples", type=int, default=2000)
    predict.add_argument("--seed", type=int, default=17)
    predict.add_argument(
        "--warn-only-quality-gates",
        action="store_true",
        help="Record failed model-quality gates without returning a failing exit code.",
    )

    resolve_snapshot = subparsers.add_parser(
        "resolve-model-snapshot",
        help="Download/resolve an exact Hugging Face snapshot for formal training.",
    )
    resolve_snapshot.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    resolve_snapshot.add_argument("--revision", default=None)
    resolve_snapshot.add_argument(
        "--output",
        default=f"{CANONICAL_OUTPUT_ROOT}/model_snapshot.json",
    )

    package_release = subparsers.add_parser(
        "package-planner-release",
        help="Build a hash-bound planner_release.v2 after fixed-test gates pass.",
    )
    package_release.add_argument("--output-dir", default=f"{CANONICAL_OUTPUT_ROOT}/release")
    package_release.add_argument("--release-id", default="planner-v2-lung-endometrial")
    package_release.add_argument("--dataset-dir", default=CANONICAL_TRAJECTORY_DATA)
    package_release.add_argument(
        "--release-scope", default=f"{CANONICAL_DATASET_ROOT}/release_scope.json"
    )
    package_release.add_argument("--memory-encoder-dir", default=CANONICAL_MEMORY_ENCODER)
    package_release.add_argument("--memory-dir", default=CANONICAL_MEMORY_STORE)
    package_release.add_argument("--baseline-decoder-dir", default=CANONICAL_DECODER)
    package_release.add_argument("--runtime-decoder-dir", default=CANONICAL_ROUTER)
    package_release.add_argument("--routing-config", default=CANONICAL_ROUTING_CONFIG)
    package_release.add_argument(
        "--routing-checkpoint", default=f"{CANONICAL_ROUTER}/routing_checkpoint.pt"
    )
    package_release.add_argument(
        "--default-mode",
        choices=("auto", "latent_topk", "daa_full"),
        default="auto",
    )
    package_release.add_argument(
        "--evaluation-summary", default=f"{CANONICAL_EVALUATION}/summary.json"
    )
    package_release.add_argument("--top-k", type=int, default=4)
    package_release.add_argument(
        "--warn-only-quality-gates",
        action="store_true",
        help="Package an experimental release even when fixed-test quality gates fail.",
    )

    args = parser.parse_args(argv)
    if args.command == "chunk-guidelines":
        chunks = chunk_guidelines(args.guideline_dir, output_path=args.output)
        _print_json({"chunks": len(chunks), "output": args.output})
        return 0
    if args.command == "build-train-data":
        ratios = _load_ratios(args.task_ratios_json)
        manifest = build_training_data(args.chunks, args.output_dir, task_ratios=ratios)
        _print_json(manifest)
        return 0
    if args.command == "train":
        result = train_planner(
            PlannerTrainingConfig(
                train_data_path=args.train_data,
                output_dir=args.output_dir,
                model_name=args.model_name,
                model_revision=args.model_revision,
                model_snapshot_path=args.model_snapshot_path,
                memory_tokens=args.memory_tokens,
                use_lora=not args.no_lora,
                max_steps=args.max_steps,
                learning_rate=args.learning_rate,
                validation_data_path=args.validation_data,
                validation_split=args.validation_split,
                eval_steps=args.eval_steps,
                save_steps=args.save_steps,
                keep_last_checkpoints=args.keep_last_checkpoints,
                auto_resume=not args.no_auto_resume,
                eval_max_examples=args.eval_max_examples,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                train_micro_batch_size_per_gpu=args.train_micro_batch_size_per_gpu,
                mixed_precision=args.mixed_precision,
                model_dtype=args.model_dtype,
                max_encoder_tokens=args.max_encoder_tokens,
                max_decoder_tokens=args.max_decoder_tokens,
                max_query_tokens=args.max_query_tokens,
                max_retrieval_candidates=args.max_retrieval_candidates,
                train_memory_encoder=args.train_memory_encoder,
                sharding=args.sharding,
                fsdp_cpu_ram_efficient_loading=not args.no_fsdp_cpu_ram_efficient_loading,
                fsdp_activation_checkpointing=args.fsdp_activation_checkpointing,
                fsdp_transformer_layer_cls_to_wrap=args.fsdp_transformer_layer_cls_to_wrap,
                deepspeed_offload_param_device=args.deepspeed_offload_param_device,
                deepspeed_offload_optimizer_device=args.deepspeed_offload_optimizer_device,
                deepspeed_nvme_path=args.deepspeed_nvme_path,
            )
        )
        if result.get("is_main_process", True):
            _print_json(result)
        return 0
    if args.command == "extract-memory":
        result = extract_memory_slots(
            args.chunks,
            args.output_dir,
            model_name=args.model_name,
            model_revision=args.model_revision,
            model_snapshot_path=args.model_snapshot_path,
            memory_tokens=args.memory_tokens,
            mock=args.mock,
            training_run_dir=args.training_run_dir,
            adapter_path=args.memory_encoder_adapter_path,
            tokenizer_path=args.tokenizer_path,
            retrieval_projection_path=args.retrieval_projection_path,
            device=args.device,
            model_dtype=args.model_dtype,
            max_encoder_tokens=args.max_encoder_tokens,
        )
        _print_json(result)
        return 0
    if args.command == "retrieve":
        from guideline_planner.latent_decoder import LatentMemoryQueryEncoder
        from guideline_planner.memory import load_memory_store_meta

        meta = load_memory_store_meta(args.memory_dir)
        selector = LatentMemoryQueryEncoder.from_memory_store(
            args.memory_dir,
            meta,
            device=args.device,
            model_dtype=args.model_dtype,
            max_query_tokens=args.max_query_tokens,
        )
        result = retrieve_latent_guideline_memory(
            args.query,
            args.top_k,
            memory_dir=args.memory_dir,
            filters=_parse_filters(args.filter),
            selector=selector,
            load_slots=False,
        )
        _print_json(_serializable_retrieval(result))
        return 0
    if args.command == "demo-next-step":
        patient_state = _patient_state(args.patient_state_json)
        planner = LatentGuidelinePlanner(
            memory_dir=args.memory_dir,
            decoder_artifact_dir=args.decoder_artifact_dir,
            top_k=args.top_k,
            device=args.device,
            query_encoder_device=args.query_encoder_device,
            max_new_tokens=args.max_new_tokens,
            model_dtype=args.model_dtype,
            output_mode=args.output_mode,
            routing_config=args.routing_config,
            routing_checkpoint=args.routing_checkpoint,
        )
        result = planner.plan(
            patient_state,
            trajectory_history=[],
        )
        _print_json(
            {
                "planner_output": result.action,
                "planner_step": result.to_dict(),
                "medclaw_plan": to_medclaw_plan(result.action),
            }
        )
        return 0
    if args.command == "build-routing-data":
        from guideline_planner.routing_dataset import build_routing_training_data

        result = build_routing_training_data(
            args.trajectory_data,
            args.output,
            allow_nonrelease_smoke=args.allow_nonrelease_smoke,
        )
        _print_json(result)
        return 0
    if args.command == "train-router":
        from guideline_planner.routing_training import (
            RoutingTrainerConfig,
            train_latent_router,
        )

        result = train_latent_router(
            RoutingTrainerConfig(
                train_data_path=args.train_data,
                memory_dir=args.memory_dir,
                decoder_artifact_dir=args.decoder_artifact_dir,
                output_dir=args.output_dir,
                routing_config_path=args.routing_config,
                device=args.device,
                model_dtype=args.model_dtype,
                max_steps=args.max_steps,
                learning_rate=args.learning_rate,
                max_decoder_tokens=args.max_decoder_tokens,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                eval_steps=args.eval_steps,
                save_steps=args.save_steps,
                keep_last_checkpoints=args.keep_last_checkpoints,
                auto_resume=not args.no_auto_resume,
                seed=args.seed,
                generation_eval_steps=args.generation_eval_steps,
                generation_eval_examples=args.generation_eval_examples,
                generation_max_new_tokens=args.generation_max_new_tokens,
            )
        )
        _print_json(result)
        return 0
    if args.command == "build-planner-data-v2":
        from guideline_planner.trajectory_dataset import build_trajectory_dataset_v2

        result = build_trajectory_dataset_v2(
            args.input,
            args.output_dir,
            seed=args.seed,
            release_gates=not args.allow_incomplete,
            fail_if_not_ready=args.fail_if_not_ready,
            rule_registry_path=args.rule_registry,
            memory_dir=args.memory_dir,
            release_scope=args.release_scope,
        )
        _print_json(result)
        return 0
    if args.command == "compile-action-set-v2":
        from guideline_planner.io_utils import read_jsonl, write_json
        from guideline_planner.rule_engine import compile_action_set

        state = json.loads(Path(args.patient_state_json).read_text(encoding="utf-8"))
        result = compile_action_set(state, read_jsonl(Path(args.rule_registry)))
        if args.output:
            write_json(Path(args.output), result)
        _print_json(result)
        return 0
    if args.command == "validate-planner-data-v2":
        from guideline_planner.grounding import validate_grounding_assets
        from guideline_planner.schemas_v2 import (
            audit_trajectory_dataset_v2,
            load_v2_jsonl,
        )

        input_path = Path(args.input)
        if input_path.is_dir() and (input_path / "admission_report.json").is_file():
            if args.allow_incomplete:
                result = json.loads(
                    (input_path / "admission_report.json").read_text(encoding="utf-8")
                )
            else:
                from guideline_planner.trajectory_dataset import (
                    require_training_ready_dataset,
                )

                result = require_training_ready_dataset(
                    input_path,
                    release_gates=True,
                ).to_dict()
        else:
            rows = load_v2_jsonl(input_path)
            report = audit_trajectory_dataset_v2(
                rows,
                release_gates=not args.allow_incomplete,
                release_scope=args.release_scope,
            )
            result = report.to_dict()
            if not args.allow_incomplete:
                grounding_errors = validate_grounding_assets(
                    rows,
                    rule_registry_path=args.rule_registry,
                    memory_dir=args.memory_dir,
                )
                if grounding_errors:
                    result["ready"] = False
                    result["errors"].extend(grounding_errors)
        _print_json(result)
        return 0 if result["ready"] else 2
    if args.command == "extract-observation-transitions":
        from guideline_planner.trajectory_dataset import (
            extract_observation_transitions_from_runs,
        )

        result = extract_observation_transitions_from_runs(
            _resolve_run_dirs(args.run_dir, args.runs_root),
            args.output,
        )
        _print_json(result)
        return 0
    if args.command == "approve-planner-reviews":
        from guideline_planner.review_sync import approve_all_trajectory_reviews_v2

        result = approve_all_trajectory_reviews_v2(
            args.dataset_root,
            reviewer_id=args.reviewer_id,
            reviewed_at=args.reviewed_at,
            release_scope=args.release_scope,
        )
        _print_json(result)
        return 0 if result.get("ready") else 2
    if args.command == "gpt-planner-data-v2":
        from guideline_planner.gpt_trajectory_pipeline import (
            run_gpt_trajectory_stage,
        )

        result = run_gpt_trajectory_stage(
            args.stage,
            args.config,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_parallel=args.max_parallel,
            max_repair_cycles=args.max_repair_cycles,
            reviewer_id=args.reviewer_id,
            reviewed_at=args.reviewed_at,
            force=args.force,
        )
        _print_json(result)
        if str(result.get("status", "")).lower() in {"error", "failed"}:
            return 2
        if args.stage in {"validate", "merge"} and result.get("ready") is False:
            return 2
        return 0
    if args.command == "validate-gpt-planner-output":
        from guideline_planner.gpt_trajectory_pipeline import (
            validate_gpt_case_output,
        )

        result = validate_gpt_case_output(
            args.packet,
            args.output,
            args.config,
        )
        _print_json(result)
        return 0 if result.get("ready") else 2
    if args.command == "train-planner-decoder":
        from guideline_planner.planner_decoder_training import (
            PlannerDecoderTrainingConfig,
            train_planner_decoder_v2,
        )

        result = train_planner_decoder_v2(
            PlannerDecoderTrainingConfig(
                trajectory_data=args.trajectory_data,
                memory_dir=args.memory_dir,
                output_dir=args.output_dir,
                device=args.device,
                model_dtype=args.model_dtype,
                max_steps=args.max_steps,
                learning_rate=args.learning_rate,
                max_decoder_tokens=args.max_decoder_tokens,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                eval_steps=args.eval_steps,
                save_steps=args.save_steps,
                keep_last_checkpoints=args.keep_last_checkpoints,
                auto_resume=not args.no_auto_resume,
                seed=args.seed,
                allow_nonrelease_smoke=args.allow_nonrelease_smoke,
                overfit_examples=args.overfit_examples,
                generation_eval_steps=args.generation_eval_steps,
                generation_eval_examples=args.generation_eval_examples,
                generation_max_new_tokens=args.generation_max_new_tokens,
                require_generation_schema_pass_rate=(
                    args.require_generation_schema_pass_rate
                ),
            )
        )
        _print_json(result)
        return 0
    if args.command == "evaluate-routing":
        from guideline_planner.io_utils import write_json
        from guideline_planner.routing_metrics import evaluate_routing_runs

        run_dirs = _resolve_run_dirs(args.run_dir, args.runs_root)
        result = evaluate_routing_runs(run_dirs)
        if args.output:
            write_json(Path(args.output), result)
        _print_json(result)
        return 0
    if args.command == "evaluate-planner-v2":
        from guideline_planner.evaluation_v2 import evaluate_planner_predictions
        from guideline_planner.io_utils import read_jsonl, write_json

        result = evaluate_planner_predictions(read_jsonl(Path(args.examples)))
        if args.output:
            write_json(Path(args.output), result)
        _print_json(result)
        return 0 if result["accepted"] else 2
    if args.command == "evaluate-router-v2":
        from guideline_planner.evaluation_v2 import evaluate_router_predictions
        from guideline_planner.io_utils import read_jsonl, write_json

        result = evaluate_router_predictions(read_jsonl(Path(args.examples)))
        if args.output:
            write_json(Path(args.output), result)
        _print_json(result)
        return 0 if result["accepted"] else 2
    if args.command == "evaluate-daa-comparison":
        from guideline_planner.evaluation_v2 import evaluate_daa_comparison
        from guideline_planner.io_utils import write_json

        payload = json.loads(Path(args.comparison_json).read_text(encoding="utf-8"))
        result = evaluate_daa_comparison(
            payload["daa_scores"],
            payload["latent_topk_scores"],
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
        if args.output:
            write_json(Path(args.output), result)
        _print_json(result)
        return 0 if result["accepted_as_default"] else 2
    if args.command == "predict-planner-v2":
        from guideline_planner.fixed_prediction import predict_planner_v2_fixed_test

        result = predict_planner_v2_fixed_test(
            trajectory_data=args.trajectory_data,
            memory_dir=args.memory_dir,
            decoder_artifact_dir=args.decoder_artifact_dir,
            runtime_decoder_artifact_dir=args.runtime_decoder_artifact_dir,
            output_dir=args.output_dir,
            release_dir=args.release_dir,
            modes=args.mode or ("latent_topk", "daa_full"),
            routing_config=args.routing_config,
            routing_checkpoint=args.routing_checkpoint,
            top_k=args.top_k,
            device=args.device,
            query_encoder_device=args.query_encoder_device,
            model_dtype=args.model_dtype,
            max_new_tokens=args.max_new_tokens,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        _print_json(result)
        # A failed DAA ablation must not prevent packaging a passing
        # latent_topk release. package-planner-release applies the stricter DAA
        # and paired-comparison gates when selecting the default mode.
        reports = result["reports"]
        if "latent_topk_planner" in reports:
            accepted = bool(reports["latent_topk_planner"].get("accepted", False))
        else:
            accepted = all(
                report.get("accepted", False)
                for key, report in reports.items()
                if key.endswith(("_planner", "_router"))
            )
        if not accepted and args.warn_only_quality_gates:
            print(
                "[planner-v2] WARNING: fixed-test quality gates failed; "
                "continuing because warn-only mode is enabled.",
                file=sys.stderr,
            )
            return 0
        return 0 if accepted else 2
    if args.command == "resolve-model-snapshot":
        output_path = Path(args.output)
        if output_path.is_file():
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "model_snapshot_lock.v2":
                raise RuntimeError(
                    f"Existing model snapshot lock has an unsupported schema: {output_path}"
                )
            if payload.get("model_name") != args.model_name:
                raise RuntimeError(
                    "Existing model snapshot lock belongs to a different model; "
                    "use a new output root instead of overwriting it."
                )
            if payload.get("requested_revision") != args.revision:
                raise RuntimeError(
                    "Existing model snapshot lock was created for a different revision; "
                    "use a new output root instead of overwriting it."
                )
            snapshot = Path(str(payload.get("snapshot_path") or ""))
            if not snapshot.is_dir():
                raise FileNotFoundError(
                    f"Existing model snapshot lock points to a missing directory: {snapshot}"
                )
            if not str(payload.get("resolved_revision") or "").strip():
                raise RuntimeError("Existing model snapshot lock has no resolved revision.")
            _print_json(payload)
            return 0
        try:
            from huggingface_hub import HfApi, snapshot_download
        except Exception as exc:
            raise RuntimeError(
                "resolve-model-snapshot requires huggingface_hub; install .[planner]."
            ) from exc
        snapshot = Path(
            snapshot_download(repo_id=args.model_name, revision=args.revision)
        ).resolve()
        info = HfApi().model_info(args.model_name, revision=args.revision)
        payload = {
            "schema_version": "model_snapshot_lock.v2",
            "model_name": args.model_name,
            "requested_revision": args.revision,
            "resolved_revision": str(info.sha),
            "snapshot_path": str(snapshot),
        }
        from guideline_planner.io_utils import write_json

        write_json(output_path, payload)
        _print_json(payload)
        return 0
    if args.command == "package-planner-release":
        from guideline_planner.artifacts import sha256_path
        from guideline_planner.release import build_planner_release_manifest

        evaluation_path = Path(args.evaluation_summary)
        if not evaluation_path.is_file():
            raise FileNotFoundError(
                f"Release packaging requires fixed-test evaluation: {evaluation_path}"
            )
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        reports = dict(evaluation.get("reports") or {})
        baseline_ok = bool(
            (reports.get("latent_topk_planner") or {}).get("accepted", False)
        )
        daa_ok = bool(
            (reports.get("daa_full_planner") or {}).get("accepted", False)
            and (reports.get("daa_full_router") or {}).get("accepted", False)
            and (evaluation.get("comparison") or {}).get(
                "accepted_as_default", False
            )
        )
        default_mode = args.default_mode
        if args.warn_only_quality_gates:
            failed = []
            if not baseline_ok:
                failed.append("latent_topk")
            if not daa_ok:
                failed.append("daa_full")
            if failed:
                print(
                    "[planner-v2] WARNING: packaging an experimental release despite "
                    f"failed quality gates: {', '.join(failed)}.",
                    file=sys.stderr,
                )
            if default_mode == "auto":
                default_mode = "daa_full"
            include_daa = True
        else:
            if not baseline_ok:
                raise RuntimeError(
                    "Planner fixed-test gates failed; a runnable release cannot be packaged."
                )
            if default_mode == "auto":
                default_mode = "daa_full" if daa_ok else "latent_topk"
            if default_mode == "daa_full" and not daa_ok:
                raise RuntimeError(
                    "daa_full cannot be packaged as default because its gates failed."
                )
            include_daa = daa_ok
        result = build_planner_release_manifest(
            args.output_dir,
            release_id=args.release_id,
            dataset_dir=args.dataset_dir,
            release_scope=args.release_scope,
            memory_encoder_dir=args.memory_encoder_dir,
            memory_dir=args.memory_dir,
            baseline_decoder_dir=args.baseline_decoder_dir,
            runtime_decoder_dir=args.runtime_decoder_dir if include_daa else args.baseline_decoder_dir,
            routing_config=args.routing_config if include_daa else None,
            routing_checkpoint=args.routing_checkpoint if include_daa else None,
            default_mode=default_mode,
            top_k=args.top_k,
            quality_gate_report={
                "enforcement": (
                    "warn_only" if args.warn_only_quality_gates else "strict"
                ),
                "latent_topk_passed": baseline_ok,
                "daa_full_passed": daa_ok,
                "evaluation_summary_hash": sha256_path(evaluation_path),
            },
        )
        _print_json(result)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


def _load_ratios(value: str | None) -> dict[str, float]:
    if not value:
        return dict(DEFAULT_TASK_RATIOS)
    path = Path(value)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(value)
    return {key: float(payload[key]) for key in DEFAULT_TASK_RATIOS if key in payload}


def _parse_filters(values: list[str]) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Filter must use key=value syntax: {value}")
        key, item = value.split("=", 1)
        filters[key] = item
    return filters


def _patient_state(value: str | None) -> dict[str, Any]:
    if not value:
        return {
            "schema_version": "patient_state.v2",
            "case_id": "demo-nsclc",
            "cancer_family": "lung",
            "disease_subtype": "nsclc",
            "known_diagnosis": "lung adenocarcinoma",
            "known_stage": None,
            "known_biomarkers": {},
            "risk_stratification": {},
            "current_phase": "diagnostic_workup",
            "decision_date": "2026-01-01",
            "guideline_context": {
                "decision_date": "2026-01-01",
                "guidelines": [{"guideline_id": "nccn-nsclc", "version": "2025"}],
            },
            "available_modalities": ["clinical"],
            "completed_skills": [],
            "completed_actions": [],
            "treatment_history": [],
            "current_treatment_line": 0,
            "evidence_ledger": [],
            "last_transition": None,
            "pending_actions": [],
            "blocked_actions": [],
            "unresolved_information": ["stage", "biomarkers"],
        }
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def _serializable_retrieval(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "score": item["score"],
            "guideline_memory_id": item["guideline_memory_id"],
            "metadata": item["metadata"],
        }
        for item in results
    ]


def _resolve_run_dirs(
    explicit: list[str],
    runs_root: str | None,
    *,
    allow_empty: bool = False,
) -> list[Path]:
    paths = [Path(value) for value in explicit]
    if runs_root:
        root = Path(runs_root)
        if not root.is_dir():
            raise FileNotFoundError(f"Runs root does not exist: {root}")
        paths.extend(path.parent for path in root.rglob("memory_routing.json"))
    unique = sorted({path.resolve() for path in paths})
    if not unique and not allow_empty:
        raise ValueError("Provide at least one --run-dir or --runs-root.")
    return unique


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
