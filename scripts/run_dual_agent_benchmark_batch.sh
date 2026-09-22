#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-cpg}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" && "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV_NAME}" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi

if [[ -z "${PYTHON_BIN}" ]] && command -v conda >/dev/null 2>&1; then
  CPG_ENV_PREFIX="$(
    conda env list 2>/dev/null \
      | awk -v name="${CONDA_ENV_NAME}" '$1 == name {print $NF; exit}'
  )"
  if [[ -n "${CPG_ENV_PREFIX}" ]]; then
    PYTHON_BIN="${CPG_ENV_PREFIX}/bin/python"
  fi
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in \
    "${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python" \
    "${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python" \
    "${HOME}/anaconda3/envs/${CONDA_ENV_NAME}/bin/python"; do
    if [[ -x "${candidate}" ]]; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Could not find Python from Conda environment '${CONDA_ENV_NAME}'." >&2
  echo "Activate it with 'conda activate ${CONDA_ENV_NAME}' or set PYTHON_BIN=/path/to/python." >&2
  exit 2
fi

CASES_ROOT="${CASES_ROOT:-/data4/share/cpgtrajbench/LUNG}"
RUN_ID="${RUN_ID:-dual_agent_001}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
# MedClaw skills execute with this same active Python interpreter.
AGENT="${AGENT:-local_openai}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-local_openai}"
ENABLE_EVALUATION="${ENABLE_EVALUATION:-0}"
PLANNER_RELEASE_DIR="${PLANNER_RELEASE_DIR:-guideline_planner/outputs/planner_v2_lung_endometrial/release}"
PLANNER_MODE="${PLANNER_MODE:-}"
PLANNER_ABLATION="${PLANNER_ABLATION:-none}"
PLANNER_ABLATION_SEED="${PLANNER_ABLATION_SEED:-17}"
# Raw artifact variables are intentionally empty by default. Setting one is an
# explicit debug override of the hash-bound release, not a production default.
PLANNER_MEMORY_DIR="${PLANNER_MEMORY_DIR:-}"
PLANNER_DECODER_ARTIFACT_DIR="${PLANNER_DECODER_ARTIFACT_DIR:-}"
PLANNER_TOP_K="${PLANNER_TOP_K:-}"
PLANNER_DEVICE="${PLANNER_DEVICE:-auto}"
PLANNER_QUERY_ENCODER_DEVICE="${PLANNER_QUERY_ENCODER_DEVICE:-cuda}"
PLANNER_MAX_NEW_TOKENS="${PLANNER_MAX_NEW_TOKENS:-auto}"
PLANNER_OUTPUT_MODE="json"
PLANNER_ROUTING_CONFIG="${PLANNER_ROUTING_CONFIG:-}"
PLANNER_ROUTING_CHECKPOINT="${PLANNER_ROUTING_CHECKPOINT:-}"
MAX_PLANNER_ROUNDS="${MAX_PLANNER_ROUNDS:-4}"
MAX_TOOL_ROUNDS_PER_STEP="${MAX_TOOL_ROUNDS_PER_STEP:-12}"
LIMIT="${LIMIT:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-5}}"

export MEDCLAW_CASES_ROOT="${CASES_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
case "${ENABLE_EVALUATION,,}" in
  1|true|yes|on) EVALUATION_ENABLED=1 ;;
  0|false|no|off|"") EVALUATION_ENABLED=0 ;;
  *) echo "ENABLE_EVALUATION must be 0/1 or false/true; got ${ENABLE_EVALUATION}." >&2; exit 2 ;;
esac
if [[ -z "${MEDCLAW_GUIDELINE_RERANK_PROVIDER:-}" ]] &&
   [[ "${AGENT}" == "local_openai" || "${AGENT}" == "qwen" ]]; then
  export MEDCLAW_GUIDELINE_RERANK_PROVIDER="${AGENT}"
fi

if [[ "${AGENT}" == "local_openai" || ( "${EVALUATION_ENABLED}" == "1" && "${JUDGE_PROVIDER}" == "local_openai" ) ]]; then
  if [[ -z "${MEDCLAW_LOCAL_API_KEY:-${VLLM_API_KEY:-}}" ]]; then
    echo "Warning: set MEDCLAW_LOCAL_API_KEY to the vLLM --api-key value." >&2
  fi
fi
if [[ "${AGENT}" == "qwen" || ( "${EVALUATION_ENABLED}" == "1" && "${JUDGE_PROVIDER}" == "qwen" ) ]]; then
  if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
    echo "Warning: DASHSCOPE_API_KEY is required for the selected qwen provider." >&2
  fi
fi
echo "Using MEDCLAW_CASES_ROOT=${MEDCLAW_CASES_ROOT}" >&2
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
echo "Using Python=${PYTHON_BIN} (default Conda env: ${CONDA_ENV_NAME})" >&2
echo "Using AGENT=${AGENT}" >&2
echo "Using ENABLE_EVALUATION=${EVALUATION_ENABLED}" >&2
if [[ "${EVALUATION_ENABLED}" == "1" ]]; then
  echo "Using JUDGE_PROVIDER=${JUDGE_PROVIDER}" >&2
fi
if [[ ! -f "${PLANNER_RELEASE_DIR}/planner_release.json" ]]; then
  echo "Planner release is missing: ${PLANNER_RELEASE_DIR}/planner_release.json" >&2
  exit 2
fi
echo "Using PLANNER_RELEASE_DIR=${PLANNER_RELEASE_DIR}" >&2
if [[ -n "${PLANNER_MODE}" ]]; then
  echo "Using PLANNER_MODE=${PLANNER_MODE}" >&2
fi
echo "Using PLANNER_ABLATION=${PLANNER_ABLATION}" >&2
echo "Using PLANNER_ABLATION_SEED=${PLANNER_ABLATION_SEED}" >&2
echo "Using PLANNER_DEVICE=${PLANNER_DEVICE}" >&2
echo "Using PLANNER_QUERY_ENCODER_DEVICE=${PLANNER_QUERY_ENCODER_DEVICE}" >&2
echo "Using PLANNER_OUTPUT_MODE=${PLANNER_OUTPUT_MODE}" >&2
if [[ -n "${PLANNER_ROUTING_CONFIG}" ]]; then
  echo "Using PLANNER_ROUTING_CONFIG=${PLANNER_ROUTING_CONFIG}" >&2
fi

PLANNER_DEVICE_CHECK="${PLANNER_DEVICE}" \
PLANNER_QUERY_DEVICE_CHECK="${PLANNER_QUERY_ENCODER_DEVICE}" \
"${PYTHON_BIN}" - <<'PY'
import os
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"Failed to import torch from {sys.executable}: {exc}") from exc

print(f"Python executable: {sys.executable}", file=sys.stderr)
print(
    f"Torch: {torch.__version__}, build CUDA: {torch.version.cuda}, "
    f"CUDA available: {torch.cuda.is_available()}",
    file=sys.stderr,
)
requested = {
    "PLANNER_DEVICE": os.environ.get("PLANNER_DEVICE_CHECK", "").lower(),
    "PLANNER_QUERY_ENCODER_DEVICE": os.environ.get(
        "PLANNER_QUERY_DEVICE_CHECK", ""
    ).lower(),
}
for name, device in requested.items():
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"{name}={device}, but CUDA is unavailable in the selected Python environment."
        )
PY

ARGS=(
  -m medclaw_benchmark.cli dual-agent-batch-run
  --cases-root "${CASES_ROOT}"
  --run-id "${RUN_ID}"
  --runs-root "${RUNS_ROOT}"
  --agent "${AGENT}"
  --planner-release-dir "${PLANNER_RELEASE_DIR}"
  --planner-device "${PLANNER_DEVICE}"
  --planner-query-encoder-device "${PLANNER_QUERY_ENCODER_DEVICE}"
  --planner-max-new-tokens "${PLANNER_MAX_NEW_TOKENS}"
  --planner-output-mode "${PLANNER_OUTPUT_MODE}"
  --planner-ablation "${PLANNER_ABLATION}"
  --planner-ablation-seed "${PLANNER_ABLATION_SEED}"
  --max-planner-rounds "${MAX_PLANNER_ROUNDS}"
  --max-tool-rounds-per-step "${MAX_TOOL_ROUNDS_PER_STEP}"
  --limit "${LIMIT}"
)

if [[ "${EVALUATION_ENABLED}" == "1" ]]; then
  ARGS+=(--evaluate --judge-provider "${JUDGE_PROVIDER}")
fi

if [[ -n "${PLANNER_MODE}" ]]; then
  ARGS+=(--planner-mode "${PLANNER_MODE}")
fi
if [[ -n "${PLANNER_MEMORY_DIR}" ]]; then
  ARGS+=(--planner-memory-dir "${PLANNER_MEMORY_DIR}")
fi
if [[ -n "${PLANNER_DECODER_ARTIFACT_DIR}" ]]; then
  ARGS+=(--planner-decoder-artifact-dir "${PLANNER_DECODER_ARTIFACT_DIR}")
fi
if [[ -n "${PLANNER_TOP_K}" ]]; then
  ARGS+=(--planner-top-k "${PLANNER_TOP_K}")
fi

if [[ -n "${PLANNER_ROUTING_CONFIG}" ]]; then
  ARGS+=(--planner-routing-config "${PLANNER_ROUTING_CONFIG}")
fi
if [[ -n "${PLANNER_ROUTING_CHECKPOINT}" ]]; then
  ARGS+=(--planner-routing-checkpoint "${PLANNER_ROUTING_CHECKPOINT}")
fi

if [[ "${FORCE}" == "1" || "${FORCE}" == "true" ]]; then
  ARGS+=(--force)
fi
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  ARGS+=(--dry-run)
fi

"${PYTHON_BIN}" "${ARGS[@]}"
