#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

COMMAND="${1:-}"
TARGET="${2:-}"
RUNS_ROOT="${RUNS_ROOT:-/data2/zhenglujie/cpg_trajbench/runs}"
PLANNER_RELEASE_DIR="${PLANNER_RELEASE_DIR:-${PROJECT_ROOT}/guideline_planner/outputs/planner_v2_lung_endometrial_npc/release}"
LUNG_CASES_ROOT="${LUNG_CASES_ROOT:-/data4/share/cpgtrajbench/LUNG}"
UCEC_CASES_ROOT="${UCEC_CASES_ROOT:-/data4/share/cpgtrajbench/UCEC}"
NPC_CASES_ROOT="${NPC_CASES_ROOT:-/data4/share/cpgtrajbench/NPC}"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/run_planner_v2_three_cancer_experiments.sh <planner|medclaw-only> <lung|ucec|npc|all>

The Planner runs use the new three-cancer release.  Existing Lung/UCEC
MedClaw-only run IDs are reused and completed cases are skipped. Evaluation is
disabled unless ENABLE_EVALUATION=1 is explicitly set.
EOF
}

case "${COMMAND}" in
  planner|medclaw-only) ;;
  *) usage; exit 2 ;;
esac
case "${TARGET}" in
  lung|ucec|npc|all) ;;
  *) usage; exit 2 ;;
esac

case_root() {
  case "$1" in
    lung) printf '%s' "${LUNG_CASES_ROOT}" ;;
    ucec) printf '%s' "${UCEC_CASES_ROOT}" ;;
    npc) printf '%s' "${NPC_CASES_ROOT}" ;;
  esac
}

run_id() {
  local method="$1"
  local cancer="$2"
  if [[ "${method}" == "planner" ]]; then
    printf 'planner_v2_three_cancer_%s_daa' "${cancer}"
    return
  fi
  case "${cancer}" in
    lung) printf '%s' 'medclaw_only_lung' ;;
    ucec) printf '%s' 'medclaw_only_ucec_full' ;;
    npc) printf '%s' 'medclaw_only_npc' ;;
  esac
}

run_one() {
  local method="$1"
  local cancer="$2"
  local root
  local id
  root="$(case_root "${cancer}")"
  id="$(run_id "${method}" "${cancer}")"
  echo "[three-cancer-experiment] method=${method} cancer=${cancer} run_id=${id} cases=${root}" >&2
  if [[ "${method}" == "planner" ]]; then
    CASES_ROOT="${root}" RUN_ID="${id}" RUNS_ROOT="${RUNS_ROOT}" \
      PLANNER_RELEASE_DIR="${PLANNER_RELEASE_DIR}" PLANNER_MODE="daa_full" \
      bash "${SCRIPT_DIR}/run_dual_agent_benchmark_batch.sh"
  else
    CASES_ROOT="${root}" RUN_ID="${id}" RUNS_ROOT="${RUNS_ROOT}" \
      bash "${SCRIPT_DIR}/run_benchmark_batch.sh"
  fi
}

if [[ "${TARGET}" == "all" ]]; then
  for cancer in lung ucec npc; do
    run_one "${COMMAND}" "${cancer}"
  done
else
  run_one "${COMMAND}" "${TARGET}"
fi
