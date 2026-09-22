#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi

CASE_DIR="${CASE_DIR:-examples/cases/TCGA-38-4625}"
RUN_ID="${RUN_ID:-run_001}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
AGENT="${AGENT:-local_openai}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-local_openai}"
CASE_ID="$(basename "${CASE_DIR}")"
if [[ -z "${RUBRIC_PATH:-}" ]]; then
  if [[ -f "${CASE_DIR}/evaluation/${CASE_ID}_rubric.json" ]]; then
    RUBRIC_PATH="${CASE_DIR}/evaluation/${CASE_ID}_rubric.json"
  else
    RUBRIC_PATH="${CASE_DIR}/evaluation/${CASE_ID}_T1_rubric.json"
  fi
fi
GUIDELINE_CHUNK_MODE="${GUIDELINE_CHUNK_MODE:-${MEDCLAW_GUIDELINE_CHUNK_MODE:-}}"
GUIDELINE_MAX_SNIPPETS="${GUIDELINE_MAX_SNIPPETS:-${MEDCLAW_GUIDELINE_MAX_SNIPPETS:-}}"
GUIDELINE_RERANK_CANDIDATE_COUNT="${GUIDELINE_RERANK_CANDIDATE_COUNT:-${MEDCLAW_GUIDELINE_RERANK_CANDIDATE_COUNT:-}}"
GUIDELINE_RETRIEVAL_MODE="${GUIDELINE_RETRIEVAL_MODE:-${MEDCLAW_GUIDELINE_RETRIEVAL_MODE:-}}"
GUIDELINE_RERANK_MODE="${GUIDELINE_RERANK_MODE:-${MEDCLAW_GUIDELINE_RERANK_MODE:-}}"

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

if [[ "${AGENT}" == "local_openai" || "${JUDGE_PROVIDER}" == "local_openai" ]]; then
  if [[ -z "${MEDCLAW_LOCAL_API_KEY:-${VLLM_API_KEY:-}}" ]]; then
    echo "Warning: set MEDCLAW_LOCAL_API_KEY to the vLLM --api-key value." >&2
  fi
fi
if [[ "${AGENT}" == "qwen" || "${JUDGE_PROVIDER}" == "qwen" ]]; then
  if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
    echo "Warning: DASHSCOPE_API_KEY is required for the selected qwen provider." >&2
  fi
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

"${PYTHON}" -m medclaw_benchmark.cli run-and-judge \
  --case-dir "${CASE_DIR}" \
  --rubric-path "${RUBRIC_PATH}" \
  --agent "${AGENT}" \
  --judge-provider "${JUDGE_PROVIDER}" \
  --judge llm \
  --run-id "${RUN_ID}" \
  --runs-root "${RUNS_ROOT}"
