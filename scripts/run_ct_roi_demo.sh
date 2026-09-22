#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi

if [[ -z "${MEDCLAW_LOCAL_API_KEY:-${VLLM_API_KEY:-}}" ]]; then
  echo "Warning: set MEDCLAW_LOCAL_API_KEY to the vLLM --api-key value." >&2
fi

"${PYTHON}" examples/run_lung_tumor_roi_demo.py "$@"
