#!/usr/bin/env bash
# Fallback: run batch with nohup if tmux is unavailable.
set -euo pipefail

TOP_K="${TOP_K:-10}"
DEVICE="${CUDA_VISIBLE_DEVICES:-2}"
LIMIT="${LIMIT:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${MEDCLAW_PATHOLOGY_LOG_DIR:-${SKILL_DIR}/logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/nohup_conch_roi_${TS}.log"
PID_FILE="${LOG_DIR}/conch_roi_batch.pid"

PY_ARGS="--device cuda:0 --top-k ${TOP_K}"
if [[ -n "${LIMIT}" && "${LIMIT}" != "0" ]]; then
  PY_ARGS+=" --limit ${LIMIT}"
fi

nohup bash -lc "
  export CUDA_VISIBLE_DEVICES=${DEVICE}
  cd '${SKILL_DIR}'
  '${PYTHON_BIN}' scripts/run_conch_prompt_roi.py ${PY_ARGS}
" >> "${RUN_LOG}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Started nohup PID=$(cat ${PID_FILE})"
echo "  log: ${RUN_LOG}"
echo "  tail -f ${RUN_LOG}"
