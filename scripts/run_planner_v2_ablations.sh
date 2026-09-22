#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

COMMAND="${1:-}"
EXPERIMENT="${2:-}"
COHORT="${3:-all}"
if [[ "${COMMAND}" == "run-all" || "${COMMAND}" == "status" ]]; then
  COHORT="${2:-all}"
fi

RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/runs}"
LUNG_CASES_ROOT="${LUNG_CASES_ROOT:-/data4/share/cpgtrajbench/LUNG}"
UCEC_CASES_ROOT="${UCEC_CASES_ROOT:-/data4/share/cpgtrajbench/UCEC}"
NPC_CASES_ROOT="${NPC_CASES_ROOT:-/data4/share/cpgtrajbench/NPC}"
PLANNER_RELEASE_DIR="${PLANNER_RELEASE_DIR:-${PROJECT_ROOT}/guideline_planner/outputs/planner_v2_lung_endometrial_npc/release}"
PLANNER_ABLATION_SEED="${PLANNER_ABLATION_SEED:-17}"
ENABLE_EVALUATION="${ENABLE_EVALUATION:-0}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"

case "${ENABLE_EVALUATION,,}" in
  1|true|yes|on) EVALUATION_ENABLED=1 ;;
  0|false|no|off|"") EVALUATION_ENABLED=0 ;;
  *) echo "ENABLE_EVALUATION must be 0/1 or false/true; got ${ENABLE_EVALUATION}." >&2; exit 2 ;;
esac

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_planner_v2_ablations.sh run <A0-A6> <lung|ucec|npc|all>
  bash scripts/run_planner_v2_ablations.sh run-all <lung|ucec|npc|all>
  bash scripts/run_planner_v2_ablations.sh status <lung|ucec|npc|all>

A0 and A5 are reuse-only and are never executed, including with FORCE=1.
run-all executes A1, A2, A3, A4, and A6 only.
All A1-A6 run IDs are isolated from the legacy two-cancer ablation runs.
Evaluation is disabled by default; set ENABLE_EVALUATION=1 to run/backfill it.
EOF
}

normalize_experiment() {
  local value="${1^^}"
  case "${value}" in
    A0|A1|A2|A3|A4|A5|A6) printf '%s' "${value}" ;;
    *) echo "Unknown experiment: ${1}" >&2; usage >&2; exit 2 ;;
  esac
}

validate_cohort() {
  case "${1}" in
    lung|ucec|npc|all) ;;
    *) echo "Unknown cohort: ${1}" >&2; usage >&2; exit 2 ;;
  esac
}

run_id_for() {
  local experiment="${1}"
  local cohort="${2}"
  case "${experiment}:${cohort}" in
    A0:lung) printf '%s' "medclaw_only_lung" ;;
    A0:ucec) printf '%s' "medclaw_only_ucec_full" ;;
    A0:npc) printf '%s' "medclaw_only_npc" ;;
    A1:*) printf 'planner_v2_three_cancer_ablation_a1_no_memory_%s' "${cohort}" ;;
    A2:*) printf 'planner_v2_three_cancer_ablation_a2_latent_topk_runtime_decoder_%s' "${cohort}" ;;
    A3:*) printf 'planner_v2_three_cancer_ablation_a3_router_topk_no_daa_%s' "${cohort}" ;;
    A4:*) printf 'planner_v2_three_cancer_ablation_a4_daa_uniform_gate_%s' "${cohort}" ;;
    A5:*) printf 'planner_v2_three_cancer_%s_daa' "${cohort}" ;;
    A6:*) printf 'planner_v2_three_cancer_ablation_a6_random_memory_global_%s' "${cohort}" ;;
    *) echo "No run ID mapping for ${experiment}/${cohort}" >&2; exit 2 ;;
  esac
}

ablation_for() {
  case "${1}" in
    A1) printf '%s' "no_memory" ;;
    A2) printf '%s' "latent_topk" ;;
    A3) printf '%s' "router_topk_no_daa" ;;
    A4) printf '%s' "daa_uniform_gate" ;;
    A6) printf '%s' "random_memory_global" ;;
    *) printf '%s' "none" ;;
  esac
}

cases_root_for() {
  case "${1}" in
    lung) printf '%s' "${LUNG_CASES_ROOT}" ;;
    ucec) printf '%s' "${UCEC_CASES_ROOT}" ;;
    npc) printf '%s' "${NPC_CASES_ROOT}" ;;
  esac
}

completion_kind_for() {
  if [[ "${1}" == "A0" ]]; then
    printf '%s' "medclaw"
  else
    printf '%s' "planner"
  fi
}

status_one() {
  local experiment="${1}"
  local cohort="${2}"
  local run_id
  local cases_root
  local completion_kind
  run_id="$(run_id_for "${experiment}" "${cohort}")"
  cases_root="$(cases_root_for "${cohort}")"
  completion_kind="$(completion_kind_for "${experiment}")"
  if [[ -z "${PYTHON_BIN}" ]]; then
    echo "Python is unavailable; cannot inspect status." >&2
    exit 2
  fi
  STATUS_CASES_ROOT="${cases_root}" STATUS_RUNS_ROOT="${RUNS_ROOT}" \
    STATUS_RUN_ID="${run_id}" STATUS_EXPERIMENT="${experiment}" \
    STATUS_COHORT="${cohort}" STATUS_KIND="${completion_kind}" \
    STATUS_REQUIRE_EVALUATION="${EVALUATION_ENABLED}" \
    "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

cases_root = Path(os.environ["STATUS_CASES_ROOT"])
runs_root = Path(os.environ["STATUS_RUNS_ROOT"])
run_id = os.environ["STATUS_RUN_ID"]
kind = os.environ["STATUS_KIND"]
cases = sorted(
    path.name
    for path in cases_root.iterdir()
    if path.is_dir() and (path / "evaluation").is_dir()
) if cases_root.is_dir() else []
required = (
    ("final_answers.json",)
    if kind == "medclaw"
    else (
        "final_answers.json",
        "planner_outputs.jsonl",
        "patient_state_history.jsonl",
    )
)
if os.environ["STATUS_REQUIRE_EVALUATION"] == "1":
    required = (*required, "judge_scores.json")
complete = []
for case_id in cases:
    run_dir = runs_root / case_id / run_id
    if all((run_dir / name).is_file() for name in required):
        complete.append(case_id)
print(json.dumps({
    "experiment": os.environ["STATUS_EXPERIMENT"],
    "cohort": os.environ["STATUS_COHORT"],
    "run_id": run_id,
    "reuse_only": os.environ["STATUS_EXPERIMENT"] in {"A0", "A5"},
    "evaluation_required": os.environ["STATUS_REQUIRE_EVALUATION"] == "1",
    "cases_root": str(cases_root),
    "runs_root": str(runs_root),
    "total": len(cases),
    "complete": len(complete),
    "missing_or_incomplete": len(cases) - len(complete),
}, ensure_ascii=False))
PY
}

for_selected_cohorts() {
  local function_name="${1}"
  local experiment="${2}"
  case "${COHORT}" in
    lung) "${function_name}" "${experiment}" lung ;;
    ucec) "${function_name}" "${experiment}" ucec ;;
    npc) "${function_name}" "${experiment}" npc ;;
    all)
      "${function_name}" "${experiment}" lung
      "${function_name}" "${experiment}" ucec
      "${function_name}" "${experiment}" npc
      ;;
  esac
}

run_one() {
  local experiment="${1}"
  local cohort="${2}"
  if [[ "${experiment}" == "A0" || "${experiment}" == "A5" ]]; then
    echo "[ablation] ${experiment}/${cohort} is reuse-only; no run will be started." >&2
    status_one "${experiment}" "${cohort}"
    return
  fi
  if [[ ! -f "${PLANNER_RELEASE_DIR}/planner_release.json" ]]; then
    echo "Planner release is missing: ${PLANNER_RELEASE_DIR}/planner_release.json" >&2
    exit 2
  fi
  local run_id
  local cases_root
  local ablation
  run_id="$(run_id_for "${experiment}" "${cohort}")"
  cases_root="$(cases_root_for "${cohort}")"
  ablation="$(ablation_for "${experiment}")"
  echo "[ablation] run ${experiment}/${cohort}: ${ablation} -> ${run_id}" >&2
  CASES_ROOT="${cases_root}" RUNS_ROOT="${RUNS_ROOT}" RUN_ID="${run_id}" \
    PLANNER_RELEASE_DIR="${PLANNER_RELEASE_DIR}" PLANNER_MODE="daa_full" \
    PLANNER_ABLATION="${ablation}" \
    PLANNER_ABLATION_SEED="${PLANNER_ABLATION_SEED}" \
    ENABLE_EVALUATION="${EVALUATION_ENABLED}" \
    bash "${SCRIPT_DIR}/run_dual_agent_benchmark_batch.sh"
}

if [[ -z "${COMMAND}" ]]; then
  usage >&2
  exit 2
fi
validate_cohort "${COHORT}"

case "${COMMAND}" in
  run)
    if [[ -z "${EXPERIMENT}" ]]; then
      usage >&2
      exit 2
    fi
    EXPERIMENT="$(normalize_experiment "${EXPERIMENT}")"
    for_selected_cohorts run_one "${EXPERIMENT}"
    ;;
  run-all)
    echo "[ablation] A0 and A5 are reuse-only; reporting their current status." >&2
    for_selected_cohorts status_one A0
    for_selected_cohorts status_one A5
    for experiment in A1 A2 A3 A4 A6; do
      for_selected_cohorts run_one "${experiment}"
    done
    ;;
  status)
    for experiment in A0 A1 A2 A3 A4 A5 A6; do
      for_selected_cohorts status_one "${experiment}"
    done
    ;;
  *)
    echo "Unknown command: ${COMMAND}" >&2
    usage >&2
    exit 2
    ;;
esac
