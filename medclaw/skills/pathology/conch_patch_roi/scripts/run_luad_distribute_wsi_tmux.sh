#!/usr/bin/env bash
# Copy LUAD WSI slides into processed/LUNG/{case_id}/pathology/wsi/ (tmux).
set -euo pipefail

SESSION="${1:-luad_wsi_copy}"
WSI_DIR="${WSI_DIR:-/data4/share/TCGA/TCGA_LUAD/wsi}"
OUT_ROOT="${OUT_ROOT:-/data4/tujiayong/processed/LUNG}"
METHOD="${METHOD:-copy}"
LIMIT="${LIMIT:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${MEDCLAW_PATHOLOGY_LOG_DIR:-${SKILL_DIR}/logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/tmux_luad_wsi_copy_${TS}.log"

PY_ARGS="--wsi-dir ${WSI_DIR} --out-root ${OUT_ROOT} --method ${METHOD}"
if [[ -n "${LIMIT}" && "${LIMIT}" != "0" ]]; then
  PY_ARGS+=" --limit ${LIMIT}"
fi

CMD=$(cat <<EOF
cd '${SCRIPT_DIR}'
exec > >(tee -a '${RUN_LOG}') 2>&1
echo "=== LUAD WSI copy @ \$(date -Iseconds) ==="
echo "session=${SESSION} method=${METHOD} wsi_dir=${WSI_DIR}"
echo "out_root=${OUT_ROOT}"
'${PYTHON_BIN}' luad_distribute_wsi.py ${PY_ARGS}
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
