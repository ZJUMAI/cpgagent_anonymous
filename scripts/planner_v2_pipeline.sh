#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1
STAGE="${1:-}"
OUTPUT_ROOT="${PLANNER_OUTPUT_ROOT:-guideline_planner/outputs/planner_v2_lung_endometrial}"
DATASET_ROOT="${PLANNER_DATASET_ROOT:-datasets/planner_v2_pilot}"
TRAJECTORY_DATA="${DATASET_ROOT}/dataset"
CHUNKS="${DATASET_ROOT}/guideline_chunks.jsonl"
MEMORY_DATA="${OUTPUT_ROOT}/memory_training_data"
MEMORY_ENCODER="${PLANNER_MEMORY_ENCODER_DIR:-${OUTPUT_ROOT}/memory_encoder}"
MEMORY_STORE="${PLANNER_MEMORY_STORE_DIR:-${OUTPUT_ROOT}/memory_store}"
DECODER="${OUTPUT_ROOT}/planner_decoder"
ROUTING_DATA="${OUTPUT_ROOT}/routing_data"
ROUTER="${OUTPUT_ROOT}/router_daa"
EVALUATION="${OUTPUT_ROOT}/evaluation"
RELEASE="${OUTPUT_ROOT}/release"
SNAPSHOT_LOCK="${OUTPUT_ROOT}/model_snapshot.json"
ROUTING_CONFIG="${PLANNER_ROUTING_CONFIG:-guideline_planner/configs/latent_guideline_routing.yaml}"
MODEL_NAME="${PLANNER_MODEL_NAME:-Qwen/Qwen3.5-9B}"
WARN_ONLY_QUALITY_GATES="${PLANNER_WARN_ONLY_QUALITY_GATES:-0}"
ROUTER_EVAL_STEPS="${PLANNER_ROUTER_EVAL_STEPS:-100}"
ROUTER_GENERATION_EVAL_STEPS="${PLANNER_ROUTER_GENERATION_EVAL_STEPS:-500}"

PREDICT_QUALITY_ARGS=()
PACKAGE_QUALITY_ARGS=()
OVERFIT_QUALITY_ARGS=(--require-generation-schema-pass-rate 0.95)
if [[ "${WARN_ONLY_QUALITY_GATES}" == "1" ]]; then
  PREDICT_QUALITY_ARGS+=(--warn-only-quality-gates)
  PACKAGE_QUALITY_ARGS+=(--warn-only-quality-gates)
  OVERFIT_QUALITY_ARGS=()
  echo "[planner-v2] quality_gates=warn-only"
fi

if [[ -z "${STAGE}" ]]; then
  echo "Usage: bash scripts/planner_v2_pipeline.sh {data-preflight|preflight|smoke|train-memory|export-memory|overfit-decoder|train-decoder|build-routing|train-router|evaluate|package-release}" >&2
  exit 2
fi

PIPELINE_START_SECONDS=${SECONDS}
echo "[planner-v2] stage=${STAGE} started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
pipeline_finish() {
  local status=$?
  local elapsed=$((SECONDS - PIPELINE_START_SECONDS))
  if (( status == 0 )); then
    echo "[planner-v2] stage=${STAGE} status=done elapsed=${elapsed}s"
  else
    echo "[planner-v2] stage=${STAGE} status=failed exit_code=${status} elapsed=${elapsed}s" >&2
  fi
}
trap pipeline_finish EXIT

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

require_file() {
  [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 2; }
}

snapshot_value() {
  "${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' "${SNAPSHOT_LOCK}" "$1"
}

build_memory_data() {
  if [[ ! -f "${MEMORY_DATA}/manifest.json" ]]; then
    run "${PYTHON_BIN}" -m guideline_planner.cli build-train-data \
      --chunks "${CHUNKS}" --output-dir "${MEMORY_DATA}"
  fi
}

data_preflight() {
  require_file "${TRAJECTORY_DATA}/manifest.json"
  require_file "${DATASET_ROOT}/release_scope.json"
  require_file "${CHUNKS}"
  run "${PYTHON_BIN}" -m guideline_planner.cli validate-planner-data-v2 \
    --input "${TRAJECTORY_DATA}" --release-scope "${DATASET_ROOT}/release_scope.json"
  run "${PYTHON_BIN}" -c 'import sys; from guideline_planner.trajectory_dataset import require_training_ready_dataset; report=require_training_ready_dataset(sys.argv[1], release_gates=True); assert report.ready, report.errors; print(report.to_dict())' "${TRAJECTORY_DATA}"
  run "${PYTHON_BIN}" -c 'import sys; rows=sum(1 for line in open(sys.argv[1],encoding="utf-8") if line.strip()); assert rows == 79, f"expected 79 chunks, got {rows}"; print({"guideline_chunks": rows})' "${CHUNKS}"
}

case "${STAGE}" in
  data-preflight)
    data_preflight
    ;;
  preflight)
    data_preflight
    run "${PYTHON_BIN}" -c 'import accelerate, datasets, peft, safetensors, torch, transformers; assert torch.cuda.is_available(), "CUDA is unavailable"; p=torch.cuda.get_device_properties(0); assert p.total_memory >= 75*1024**3, f"formal training requires an A100/H100 80GB-class GPU, got {p.total_memory/1024**3:.1f} GiB"; print(p.name, p.total_memory)'
    command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required." >&2; exit 2; }
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    SNAPSHOT_ARGS=(--model-name "${MODEL_NAME}" --output "${SNAPSHOT_LOCK}")
    if [[ -n "${PLANNER_MODEL_REVISION:-}" ]]; then
      SNAPSHOT_ARGS+=(--revision "${PLANNER_MODEL_REVISION}")
    fi
    run "${PYTHON_BIN}" -m guideline_planner.cli resolve-model-snapshot "${SNAPSHOT_ARGS[@]}"
    ;;
  smoke)
    require_file "${SNAPSHOT_LOCK}"
    SMOKE_ROOT="${PLANNER_SMOKE_ROOT:-${OUTPUT_ROOT}/smoke}"
    REVISION="$(snapshot_value resolved_revision)"
    SNAPSHOT="$(snapshot_value snapshot_path)"
    run "${PYTHON_BIN}" -m guideline_planner.cli build-train-data --chunks "${CHUNKS}" --output-dir "${SMOKE_ROOT}/memory_data"
    run "${PYTHON_BIN}" -m guideline_planner.cli train \
      --train-data "${SMOKE_ROOT}/memory_data/mixed_train.jsonl" --output-dir "${SMOKE_ROOT}/memory_encoder" \
      --model-name "${MODEL_NAME}" --model-revision "${REVISION}" --model-snapshot-path "${SNAPSHOT}" \
      --max-steps 10 --eval-steps 5 --save-steps 5 --keep-last-checkpoints 2 \
      --gradient-accumulation-steps 1 --mixed-precision bf16 --model-dtype bf16
    run "${PYTHON_BIN}" -m guideline_planner.cli extract-memory \
      --chunks "${CHUNKS}" --output-dir "${SMOKE_ROOT}/memory_store" \
      --training-run-dir "${SMOKE_ROOT}/memory_encoder" --memory-tokens 64 --device cuda --model-dtype bf16
    run "${PYTHON_BIN}" -m guideline_planner.cli train-planner-decoder \
      --trajectory-data "${TRAJECTORY_DATA}" --memory-dir "${SMOKE_ROOT}/memory_store" \
      --output-dir "${SMOKE_ROOT}/planner_decoder" --max-steps 10 --eval-steps 5 \
      --save-steps 5 --gradient-accumulation-steps 1
    run "${PYTHON_BIN}" -m guideline_planner.cli build-routing-data \
      --trajectory-data "${TRAJECTORY_DATA}" --output "${SMOKE_ROOT}/routing_data"
    run "${PYTHON_BIN}" -m guideline_planner.cli train-router \
      --train-data "${SMOKE_ROOT}/routing_data" --memory-dir "${SMOKE_ROOT}/memory_store" \
      --decoder-artifact-dir "${SMOKE_ROOT}/planner_decoder" --output-dir "${SMOKE_ROOT}/router_daa" \
      --routing-config guideline_planner/configs/planner_v2_smoke_routing.yaml \
      --max-steps 10 --eval-steps 5 --save-steps 5 --gradient-accumulation-steps 1
    ;;
  train-memory)
    require_file "${SNAPSHOT_LOCK}"
    build_memory_data
    REVISION="$(snapshot_value resolved_revision)"
    SNAPSHOT="$(snapshot_value snapshot_path)"
    run "${PYTHON_BIN}" -m guideline_planner.cli train \
      --train-data "${MEMORY_DATA}/mixed_train.jsonl" --output-dir "${MEMORY_ENCODER}" \
      --model-name "${MODEL_NAME}" --model-revision "${REVISION}" --model-snapshot-path "${SNAPSHOT}" \
      --max-steps 5000 --learning-rate 2e-4 --gradient-accumulation-steps 4 \
      --eval-steps 100 --save-steps 500 --keep-last-checkpoints 2 \
      --mixed-precision bf16 --model-dtype bf16
    ;;
  export-memory)
    require_file "${MEMORY_ENCODER}/memory_encoder_meta.json"
    run "${PYTHON_BIN}" -m guideline_planner.cli extract-memory \
      --chunks "${CHUNKS}" --output-dir "${MEMORY_STORE}" \
      --training-run-dir "${MEMORY_ENCODER}" --memory-tokens 64 --device cuda --model-dtype bf16
    ;;
  overfit-decoder)
    require_file "${MEMORY_STORE}/memory_store_meta.json"
    OVERFIT_STEPS="${PLANNER_OVERFIT_STEPS:-200}"
    OVERFIT_EXAMPLES="${PLANNER_OVERFIT_EXAMPLES:-8}"
    run "${PYTHON_BIN}" -m guideline_planner.cli train-planner-decoder \
      --trajectory-data "${TRAJECTORY_DATA}" --memory-dir "${MEMORY_STORE}" \
      --output-dir "${OUTPUT_ROOT}/decoder_overfit" \
      --max-steps "${OVERFIT_STEPS}" --overfit-examples "${OVERFIT_EXAMPLES}" \
      --learning-rate 2e-4 --max-decoder-tokens 2048 --gradient-accumulation-steps 1 \
      --eval-steps 25 --generation-eval-steps 100 \
      --generation-eval-examples "${OVERFIT_EXAMPLES}" \
      --generation-max-new-tokens 1024 \
      "${OVERFIT_QUALITY_ARGS[@]}" \
      --save-steps 0 --no-auto-resume
    ;;
  train-decoder)
    require_file "${MEMORY_STORE}/memory_store_meta.json"
    if [[ "${WARN_ONLY_QUALITY_GATES}" == "1" ]]; then
      if [[ -f "${OUTPUT_ROOT}/decoder_overfit/planner_decoder_meta.json" ]]; then
        run "${PYTHON_BIN}" -c 'import json,sys; p=sys.argv[1]; m=json.load(open(p,encoding="utf-8")); g=m.get("generation_validation") or {}; actual=float(g.get("generation_schema_pass_rate",0.0)); print({"warning":"decoder overfit schema gate is advisory","decoder_overfit_schema_pass_rate":actual})' \
          "${OUTPUT_ROOT}/decoder_overfit/planner_decoder_meta.json"
      else
        echo "[planner-v2] WARNING: decoder overfit artifact is absent; continuing formal training in warn-only mode." >&2
      fi
    else
      require_file "${OUTPUT_ROOT}/decoder_overfit/planner_decoder_meta.json"
      run "${PYTHON_BIN}" -c 'import json,sys; p=sys.argv[1]; m=json.load(open(p,encoding="utf-8")); g=m.get("generation_validation") or {}; actual=float(g.get("generation_schema_pass_rate",0.0)); required=0.95; assert actual >= required, f"decoder overfit schema gate failed: {actual:.4f} < {required:.4f}"; print({"decoder_overfit_schema_pass_rate":actual,"gate":required})' \
        "${OUTPUT_ROOT}/decoder_overfit/planner_decoder_meta.json"
    fi
    run "${PYTHON_BIN}" -m guideline_planner.cli train-planner-decoder \
      --trajectory-data "${TRAJECTORY_DATA}" --memory-dir "${MEMORY_STORE}" --output-dir "${DECODER}" \
      --max-steps 5000 --learning-rate 2e-4 --max-decoder-tokens 2048 --gradient-accumulation-steps 4 \
      --generation-eval-steps 500 --generation-eval-examples 8 --generation-max-new-tokens 1024 \
      --eval-steps 100 --save-steps 500 --keep-last-checkpoints 2
    ;;
  build-routing)
    require_file "${DECODER}/planner_decoder_meta.json"
    run "${PYTHON_BIN}" -m guideline_planner.cli build-routing-data \
      --trajectory-data "${TRAJECTORY_DATA}" --output "${ROUTING_DATA}"
    ;;
  train-router)
    require_file "${ROUTING_DATA}/manifest.json"
    run "${PYTHON_BIN}" -m guideline_planner.cli train-router \
      --train-data "${ROUTING_DATA}" --memory-dir "${MEMORY_STORE}" \
      --decoder-artifact-dir "${DECODER}" --output-dir "${ROUTER}" \
      --routing-config "${ROUTING_CONFIG}" --max-steps 5000 --learning-rate 1e-4 --max-decoder-tokens 2048 \
      --generation-eval-steps "${ROUTER_GENERATION_EVAL_STEPS}" \
      --generation-eval-examples 8 --generation-max-new-tokens 1024 \
      --gradient-accumulation-steps 4 --eval-steps "${ROUTER_EVAL_STEPS}" \
      --save-steps 500 --keep-last-checkpoints 2
    ;;
  evaluate)
    require_file "${ROUTER}/routing_checkpoint.pt"
    run "${PYTHON_BIN}" -m guideline_planner.cli predict-planner-v2 \
      --trajectory-data "${TRAJECTORY_DATA}" --memory-dir "${MEMORY_STORE}" \
      --decoder-artifact-dir "${DECODER}" --runtime-decoder-artifact-dir "${ROUTER}" \
      --routing-config "${ROUTING_CONFIG}" \
      --routing-checkpoint "${ROUTER}/routing_checkpoint.pt" --output-dir "${EVALUATION}" \
      --mode latent_topk --mode daa_full --top-k 4 --device cuda --model-dtype bf16 \
      --max-new-tokens 1024 "${PREDICT_QUALITY_ARGS[@]}"
    ;;
  package-release)
    require_file "${EVALUATION}/summary.json"
    run "${PYTHON_BIN}" -m guideline_planner.cli package-planner-release \
      --output-dir "${RELEASE}" --dataset-dir "${TRAJECTORY_DATA}" \
      --release-id "${PLANNER_RELEASE_ID:-planner-v2-lung-endometrial}" \
      --release-scope "${DATASET_ROOT}/release_scope.json" \
      --memory-encoder-dir "${MEMORY_ENCODER}" --memory-dir "${MEMORY_STORE}" \
      --baseline-decoder-dir "${DECODER}" --runtime-decoder-dir "${ROUTER}" \
      --routing-config "${ROUTING_CONFIG}" --routing-checkpoint "${ROUTER}/routing_checkpoint.pt" \
      --evaluation-summary "${EVALUATION}/summary.json" --default-mode auto --top-k 4 \
      "${PACKAGE_QUALITY_ARGS[@]}"
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac
