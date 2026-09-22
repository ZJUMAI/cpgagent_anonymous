#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi

CASES_ROOT="${CASES_ROOT:-/data4/share/cpgtrajbench/LUNG}"
RUN_ID="${RUN_ID:-run_001}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
AGENT="${AGENT:-local_openai}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-local_openai}"
ENABLE_EVALUATION="${ENABLE_EVALUATION:-0}"
LIMIT="${LIMIT:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-5}}"
GUIDELINE_CHUNK_MODE="${GUIDELINE_CHUNK_MODE:-${MEDCLAW_GUIDELINE_CHUNK_MODE:-}}"
GUIDELINE_MAX_SNIPPETS="${GUIDELINE_MAX_SNIPPETS:-${MEDCLAW_GUIDELINE_MAX_SNIPPETS:-}}"
GUIDELINE_RERANK_CANDIDATE_COUNT="${GUIDELINE_RERANK_CANDIDATE_COUNT:-${MEDCLAW_GUIDELINE_RERANK_CANDIDATE_COUNT:-}}"
GUIDELINE_RETRIEVAL_MODE="${GUIDELINE_RETRIEVAL_MODE:-${MEDCLAW_GUIDELINE_RETRIEVAL_MODE:-}}"
GUIDELINE_RERANK_MODE="${GUIDELINE_RERANK_MODE:-${MEDCLAW_GUIDELINE_RERANK_MODE:-}}"

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
if [[ -n "${GUIDELINE_CHUNK_MODE}" ]]; then
  export MEDCLAW_GUIDELINE_CHUNK_MODE="${GUIDELINE_CHUNK_MODE}"
fi
if [[ -n "${GUIDELINE_MAX_SNIPPETS}" ]]; then
  export MEDCLAW_GUIDELINE_MAX_SNIPPETS="${GUIDELINE_MAX_SNIPPETS}"
fi
if [[ -n "${GUIDELINE_RERANK_CANDIDATE_COUNT}" ]]; then
  export MEDCLAW_GUIDELINE_RERANK_CANDIDATE_COUNT="${GUIDELINE_RERANK_CANDIDATE_COUNT}"
fi
if [[ -n "${GUIDELINE_RETRIEVAL_MODE}" ]]; then
  export MEDCLAW_GUIDELINE_RETRIEVAL_MODE="${GUIDELINE_RETRIEVAL_MODE}"
fi
if [[ -n "${GUIDELINE_RERANK_MODE}" ]]; then
  export MEDCLAW_GUIDELINE_RERANK_MODE="${GUIDELINE_RERANK_MODE}"
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
echo "Using AGENT=${AGENT}" >&2
echo "Using ENABLE_EVALUATION=${EVALUATION_ENABLED}" >&2
if [[ "${EVALUATION_ENABLED}" == "1" ]]; then
  echo "Using JUDGE_PROVIDER=${JUDGE_PROVIDER}" >&2
fi
if [[ -n "${MEDCLAW_GUIDELINE_CHUNK_MODE:-}" ]]; then
  echo "Using MEDCLAW_GUIDELINE_CHUNK_MODE=${MEDCLAW_GUIDELINE_CHUNK_MODE}" >&2
fi
if [[ -n "${MEDCLAW_GUIDELINE_MAX_SNIPPETS:-}" ]]; then
  echo "Using MEDCLAW_GUIDELINE_MAX_SNIPPETS=${MEDCLAW_GUIDELINE_MAX_SNIPPETS}" >&2
fi
if [[ -n "${MEDCLAW_GUIDELINE_RERANK_CANDIDATE_COUNT:-}" ]]; then
  echo "Using MEDCLAW_GUIDELINE_RERANK_CANDIDATE_COUNT=${MEDCLAW_GUIDELINE_RERANK_CANDIDATE_COUNT}" >&2
fi

ARGS=(
  -m medclaw_benchmark.cli batch-run
  --cases-root "${CASES_ROOT}"
  --run-id "${RUN_ID}"
  --runs-root "${RUNS_ROOT}"
  --agent "${AGENT}"
  --limit "${LIMIT}"
)

if [[ "${EVALUATION_ENABLED}" == "1" ]]; then
  ARGS+=(--evaluate --judge-provider "${JUDGE_PROVIDER}")
fi

if [[ "${FORCE}" == "1" || "${FORCE}" == "true" ]]; then
  ARGS+=(--force)
fi
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  ARGS+=(--dry-run)
fi

"${PYTHON}" "${ARGS[@]}"
