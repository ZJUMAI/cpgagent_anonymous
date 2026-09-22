#!/usr/bin/env bash
# Start LUAD multiscale pipeline in tmux.
set -euo pipefail

SESSION="${1:-luad_multiscale}"
SCALE="${SCALE:-all}"
LIMIT="${LIMIT:-0}"
DEVICE="${CUDA_VISIBLE_DEVICES:-4}"
SKIP_SEG="${SKIP_SEG:-1}"
SKIP_CONCH="${SKIP_CONCH:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONCH_SKIP_EXISTING="${CONCH_SKIP_EXISTING:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${MEDCLAW_PATHOLOGY_LOG_DIR:-${SKILL_DIR}/logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/tmux_luad_multiscale_${TS}.log"

CMD=$(cat <<EOF
export CUDA_VISIBLE_DEVICES=${DEVICE}
export PYTHON_BIN='${PYTHON_BIN}'
export SCALE=${SCALE}
export LIMIT=${LIMIT}
export SKIP_SEG=${SKIP_SEG}
export SKIP_CONCH=${SKIP_CONCH}
export SKIP_EXISTING=${SKIP_EXISTING}
export CONCH_SKIP_EXISTING=${CONCH_SKIP_EXISTING}
cd '${SKILL_DIR}'
exec > >(tee -a '${RUN_LOG}') 2>&1
echo "=== LUAD multiscale @ \$(date -Iseconds) ==="
echo "session=${SESSION} SCALE=\${SCALE} GPU=\${CUDA_VISIBLE_DEVICES} LIMIT=\${LIMIT}"
bash scripts/run_luad_multiscale.sh
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
