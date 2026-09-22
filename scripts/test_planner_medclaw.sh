#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET="${1:-}"
OUTPUT_ROOT="${PLANNER_OUTPUT_ROOT:-guideline_planner/outputs/planner_v2_lung_endometrial}"
RELEASE_DIR="${PLANNER_RELEASE_DIR:-${OUTPUT_ROOT}/release}"
DATASET="${PLANNER_TRAJECTORY_DATA:-datasets/planner_v2_pilot/dataset}"
RUNS_ROOT="${PLANNER_MEDCLAW_RUNS_ROOT:-${OUTPUT_ROOT}/planner_medclaw_runs}"
CASE_ID="${CASE_ID:-TCGA-38-4626}"
CASES_ROOT="${CASES_ROOT:-examples/cases}"
AGENT="${AGENT:-local_openai}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-local_openai}"
ENABLE_EVALUATION="${ENABLE_EVALUATION:-0}"
LIMIT="${LIMIT:-2}"

case "${ENABLE_EVALUATION,,}" in
  1|true|yes|on) EVALUATION_ENABLED=1 ;;
  0|false|no|off|"") EVALUATION_ENABLED=0 ;;
  *) echo "ENABLE_EVALUATION must be 0/1 or false/true; got ${ENABLE_EVALUATION}." >&2; exit 2 ;;
esac

if [[ -z "${TARGET}" ]]; then
  echo "Usage: bash scripts/test_planner_medclaw.sh {planner-fixed|single|batch-smoke}" >&2
  exit 2
fi
[[ -f "${RELEASE_DIR}/planner_release.json" ]] || {
  echo "Planner release is missing: ${RELEASE_DIR}/planner_release.json" >&2
  exit 2
}

release_mode() {
  "${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["default_mode"])' "${RELEASE_DIR}/planner_release.json"
}

MODE="${PLANNER_MODE:-$(release_mode)}"

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

validate_run() {
  local run_dir="$1"
  run "${PYTHON_BIN}" -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
required = ["planner_outputs.jsonl", "patient_state_history.jsonl", "routing_metrics.json", "final_answers.json"]
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit(f"missing run artifacts in {root}: {missing}")
planner = [json.loads(line) for line in (root / "planner_outputs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
states = [json.loads(line) for line in (root / "patient_state_history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
if not planner or not states:
    raise SystemExit(f"empty planner/state trace in {root}")
if any(item.get("validation_error") for item in planner):
    raise SystemExit(f"planner validator errors detected in {root}")
print({"run_dir": str(root), "planner_rounds": len(planner), "state_events": len(states)})
' "${run_dir}"
}

case "${TARGET}" in
  planner-fixed)
    run "${PYTHON_BIN}" -m guideline_planner.cli predict-planner-v2 \
      --release-dir "${RELEASE_DIR}" --trajectory-data "${DATASET}" \
      --output-dir "${OUTPUT_ROOT}/release_evaluation_${MODE}" --mode "${MODE}" \
      --device "${PLANNER_DEVICE:-cuda}" --model-dtype "${PLANNER_MODEL_DTYPE:-bf16}"
    ;;
  single)
    CASE_DIR="${CASES_ROOT}/${CASE_ID}"
    [[ -d "${CASE_DIR}" ]] || { echo "Case directory is missing: ${CASE_DIR}" >&2; exit 2; }
    RUN_ID="${RUN_ID:-planner_v2_single_${MODE}}"
    RUN_DIR="${RUNS_ROOT}/${CASE_ID}/${RUN_ID}"
    if [[ "${EVALUATION_ENABLED}" == "1" ]]; then
      if [[ -f "${RUN_DIR}/judge_scores.json" ]]; then
        echo "Evaluation already exists; skipping: ${RUN_DIR}/judge_scores.json" >&2
      elif [[ -f "${RUN_DIR}/final_answers.json" && \
              -f "${RUN_DIR}/planner_outputs.jsonl" && \
              -f "${RUN_DIR}/patient_state_history.jsonl" ]]; then
        RUBRIC_PATH="$(
          "${PYTHON_BIN}" -c \
            'import sys; from medclaw_benchmark.case_paths import find_rubric_path; print(find_rubric_path(sys.argv[1]))' \
            "${CASE_DIR}"
        )"
        run "${PYTHON_BIN}" -m medclaw_benchmark.cli judge \
          --run-dir "${RUN_DIR}" --rubric-path "${RUBRIC_PATH}" \
          --judge-provider "${JUDGE_PROVIDER}"
      else
        run "${PYTHON_BIN}" -m medclaw_benchmark.cli dual-agent-run-and-judge \
          --case-dir "${CASE_DIR}" --run-id "${RUN_ID}" --runs-root "${RUNS_ROOT}" \
          --agent "${AGENT}" --judge-provider "${JUDGE_PROVIDER}" \
          --planner-release-dir "${RELEASE_DIR}" --planner-mode "${MODE}" \
          --planner-device "${PLANNER_DEVICE:-auto}" --planner-query-encoder-device cpu
      fi
    else
      run "${PYTHON_BIN}" -m medclaw_benchmark.cli dual-agent-run \
        --case-dir "${CASE_DIR}" --run-id "${RUN_ID}" --runs-root "${RUNS_ROOT}" \
        --agent "${AGENT}" \
        --planner-release-dir "${RELEASE_DIR}" --planner-mode "${MODE}" \
        --planner-device "${PLANNER_DEVICE:-auto}" --planner-query-encoder-device cpu
    fi
    validate_run "${RUN_DIR}"
    ;;
  batch-smoke)
    RUN_ID="${RUN_ID:-planner_v2_batch_smoke_${MODE}}"
    evaluation_args=()
    if [[ "${EVALUATION_ENABLED}" == "1" ]]; then
      evaluation_args+=(--evaluate --judge-provider "${JUDGE_PROVIDER}")
    fi
    run "${PYTHON_BIN}" -m medclaw_benchmark.cli dual-agent-batch-run \
      --cases-root "${CASES_ROOT}" --run-id "${RUN_ID}" --runs-root "${RUNS_ROOT}" \
      --agent "${AGENT}" "${evaluation_args[@]}" \
      --planner-release-dir "${RELEASE_DIR}" --planner-mode "${MODE}" \
      --planner-device "${PLANNER_DEVICE:-auto}" --planner-query-encoder-device cpu \
      --limit "${LIMIT}"
    mapfile -t completed < <(find "${RUNS_ROOT}" -mindepth 2 -maxdepth 2 -type d -name "${RUN_ID}" | sort)
    if (( ${#completed[@]} < LIMIT )); then
      echo "Expected ${LIMIT} completed batch runs, found ${#completed[@]}." >&2
      exit 2
    fi
    for run_dir in "${completed[@]:0:${LIMIT}}"; do
      validate_run "${run_dir}"
    done
    ;;
  *)
    echo "Unknown test target: ${TARGET}" >&2
    exit 2
    ;;
esac
