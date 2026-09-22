#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1

STAGE="${1:-}"
CONFIG="${PLANNER_GPT_DATA_CONFIG:-guideline_planner/configs/planner_v2_lung_endometrial_npc.yaml}"
TEACHER_MODEL="${PLANNER_GPT_MODEL:-gpt-5.6-sol}"
REASONING_EFFORT="${PLANNER_GPT_REASONING_EFFORT:-high}"
MAX_PARALLEL="${PLANNER_GPT_MAX_PARALLEL:-3}"
MAX_REPAIR_CYCLES="${PLANNER_GPT_MAX_REPAIR_CYCLES:-2}"
REVIEWER_ID="${PLANNER_MANUAL_REVIEWER_ID:-professional-clinician-review-team}"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/planner_v2_gpt_data_pipeline.sh <stage>

Stages:
  prepare          Build sanitized NPC teacher and Lung/UCEC review packets.
  generate-npc     Report/advance the resumable NPC teacher-output queue.
  review-existing  Report/advance the Lung/UCEC case-review queue.
  repair           Report/advance rejected-case repair packets.
  verify           Report/advance independent verifier packets.
  approve-manual   Bind the professional review to the current 109 cases by hash.
  validate         Run schema, grounding, split, provenance, and QC gates.
  merge            Build the immutable three-cancer dataset when all gates pass.
  status           Print case/record-level progress without changing outputs.

Environment overrides:
  PLANNER_GPT_DATA_CONFIG
  PLANNER_GPT_MODEL                 (default: gpt-5.6-sol)
  PLANNER_GPT_REASONING_EFFORT      (default: high)
  PLANNER_GPT_MAX_PARALLEL          (default: 3)
  PLANNER_GPT_MAX_REPAIR_CYCLES     (default: 2)
  PLANNER_MANUAL_REVIEWER_ID         (default: professional-clinician-review-team)
  PLANNER_MANUAL_REVIEWED_AT         optional ISO-8601 UTC timestamp
  FORCE=1                           rebuild stale stage artifacts

Generation, review, repair, and verification packets are one-case-per-task.
The command is resumable: valid outputs whose request hashes match are skipped.
EOF
}

case "${STAGE}" in
  prepare|generate-npc|review-existing|repair|verify|approve-manual|validate|merge|status)
    ;;
  -h|--help|help|"")
    usage
    [[ -n "${STAGE}" ]] && exit 0 || exit 2
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    usage
    exit 2
    ;;
esac

[[ -f "${CONFIG}" ]] || {
  echo "Planner GPT data config is missing: ${CONFIG}" >&2
  exit 2
}

ARGS=(
  "${PYTHON_BIN}" -m guideline_planner.cli gpt-planner-data-v2 "${STAGE}"
  --config "${CONFIG}"
  --model "${TEACHER_MODEL}"
  --reasoning-effort "${REASONING_EFFORT}"
  --max-parallel "${MAX_PARALLEL}"
  --max-repair-cycles "${MAX_REPAIR_CYCLES}"
  --reviewer-id "${REVIEWER_ID}"
)
if [[ -n "${PLANNER_MANUAL_REVIEWED_AT:-}" ]]; then
  ARGS+=(--reviewed-at "${PLANNER_MANUAL_REVIEWED_AT}")
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  ARGS+=(--force)
fi

START_SECONDS=${SECONDS}
echo "[planner-v2-gpt-data] stage=${STAGE} started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '+ '
printf '%q ' "${ARGS[@]}"
printf '\n'

set +e
"${ARGS[@]}"
STATUS=$?
set -e

ELAPSED=$((SECONDS - START_SECONDS))
if (( STATUS == 0 )); then
  echo "[planner-v2-gpt-data] stage=${STAGE} status=done elapsed=${ELAPSED}s"
else
  echo "[planner-v2-gpt-data] stage=${STAGE} status=failed exit_code=${STATUS} elapsed=${ELAPSED}s" >&2
fi
exit "${STATUS}"
