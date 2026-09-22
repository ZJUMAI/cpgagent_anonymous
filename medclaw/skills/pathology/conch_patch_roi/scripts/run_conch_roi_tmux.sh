#!/usr/bin/env bash
# Start CONCH prompt ROI batch in tmux.
set -euo pipefail

SESSION="${1:-conch_roi}"
TOP_K="${TOP_K:-10}"
LIMIT="${LIMIT:-0}"
DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${MEDCLAW_PATHOLOGY_LOG_DIR:-${SKILL_DIR}/logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/tmux_conch_roi_${TS}.log"

PY_ARGS="--device cuda:0 --top-k ${TOP_K}"
if [[ -n "${LIMIT}" && "${LIMIT}" != "0" ]]; then
  PY_ARGS+=" --limit ${LIMIT}"
fi

CMD=$(cat <<EOF
export CUDA_VISIBLE_DEVICES=${DEVICE}
cd '${SKILL_DIR}'
exec > >(tee -a '${RUN_LOG}') 2>&1
echo "=== CONCH prompt ROI batch @ \$(date -Iseconds) ==="
echo "session=${SESSION} GPU=\${CUDA_VISIBLE_DEVICES} top_k=${TOP_K}"
'${PYTHON_BIN}' scripts/run_conch_prompt_roi.py ${PY_ARGS}
echo "=== finished @ \$(date -Iseconds) exit=\$? ==="
EOF
)

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session '${SESSION}' already exists."
  echo "  attach: tmux attach -t ${SESSION}"
  echo "  kill:   tmux kill-session -t ${SESSION}"
  exit 1
fi

tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "$CMD")"
echo "Started tmux session: ${SESSION}"
echo "  attach: tmux attach -t ${SESSION}"
echo "  log:    ${RUN_LOG}"
