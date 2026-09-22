#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

METHOD="both"
FORCE_RUN="${FORCE:-0}"
DRY_RUN_VALUE="${DRY_RUN:-0}"
PLAN_ONLY="0"
GPU_ID_VALUE="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-5}}"

usage() {
  printf '%s\n' \
    "Usage: bash scripts/run_case_study_cases.sh [options]" \
    "" \
    "Options:" \
    "  --method planner|medclaw-only|both  Methods to run (default: both)." \
    "  --force                              Replace completed runs." \
    "  --dry-run                            Run benchmark preflight only." \
    "  --plan-only                          Print the resolved plan; run nothing." \
    "  --gpu ID                             CUDA device, e.g. 2 or 2,3." \
    "  -h, --help                           Show this help." \
    "" \
    "Environment overrides:" \
    "  CASE_STUDY_RUNS_ROOT, LUNG_CASES_ROOT, UCEC_CASES_ROOT, NPC_CASES_ROOT" \
    "  PLANNER_RUN_ID, MEDCLAW_RUN_ID, PLANNER_RELEASE_DIR, PYTHON_BIN, GPU_ID" \
    "  AGENT, ENABLE_EVALUATION, JUDGE_PROVIDER"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      [[ $# -ge 2 ]] || { echo "--method requires a value" >&2; exit 2; }
      METHOD="$2"
      shift 2
      ;;
    --force)
      FORCE_RUN="1"
      shift
      ;;
    --dry-run)
      DRY_RUN_VALUE="1"
      shift
      ;;
    --plan-only)
      PLAN_ONLY="1"
      shift
      ;;
    --gpu)
      [[ $# -ge 2 ]] || { echo "--gpu requires a value" >&2; exit 2; }
      GPU_ID_VALUE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${METHOD}" in
  planner|medclaw-only|both) ;;
  *) echo "--method must be planner, medclaw-only, or both; got ${METHOD}" >&2; exit 2 ;;
esac

choose_cases_root() {
  local server_root="$1"
  local local_root="$2"
  if [[ -d "${server_root}" ]]; then
    printf '%s' "${server_root}"
  elif [[ -d "${local_root}" ]]; then
    printf '%s' "${local_root}"
  else
    printf '%s' "${server_root}"
  fi
}

LUNG_CASES_ROOT="${LUNG_CASES_ROOT:-$(choose_cases_root "/data4/share/cpgtrajbench/LUNG" "${PROJECT_ROOT}/data/cpgtrajbench/LUNG")}"
UCEC_CASES_ROOT="${UCEC_CASES_ROOT:-$(choose_cases_root "/data4/share/cpgtrajbench/UCEC" "${PROJECT_ROOT}/data/cpgtrajbench/UCEC")}"
NPC_CASES_ROOT="${NPC_CASES_ROOT:-$(choose_cases_root "/data4/share/cpgtrajbench/NPC" "${PROJECT_ROOT}/data/patient_data/NPC")}"
RUNS_ROOT="${CASE_STUDY_RUNS_ROOT:-${PROJECT_ROOT}/runs_casestudy}"
PLANNER_RUN_ID="${PLANNER_RUN_ID:-planner_v2_three_cancer_casestudy_daa}"
MEDCLAW_RUN_ID="${MEDCLAW_RUN_ID:-medclaw_only_casestudy}"
PLANNER_RELEASE_DIR="${PLANNER_RELEASE_DIR:-${PROJECT_ROOT}/guideline_planner/outputs/planner_v2_lung_endometrial_npc/release}"

CASE_IDS=("TCGA-78-8640" "03797212" "M519805")
CANCER_TYPES=("LUNG" "UCEC" "NPC")
CASE_ROOTS=("${LUNG_CASES_ROOT}" "${UCEC_CASES_ROOT}" "${NPC_CASES_ROOT}")
CASE_REASONS=(
  "state propagation error"
  "conflicting staging and conditional decisions"
  "three-cancer NPC multimodal trajectory"
)

mkdir -p "${RUNS_ROOT}" "${RUNS_ROOT}/logs"
TMP_BASE="${TMPDIR:-/tmp}"
TMP_BASE="${TMP_BASE%/}"
STAGING_ROOT=""
cleanup() {
  if [[ -n "${STAGING_ROOT:-}" && -d "${STAGING_ROOT}" ]] &&
     [[ "${STAGING_ROOT}" == "${TMP_BASE}/cpgtraj-casestudy."* ]]; then
    rm -rf -- "${STAGING_ROOT}"
  fi
}
trap cleanup EXIT

MANIFEST="${RUNS_ROOT}/case_study_cases.tsv"
printf 'case_id\tcancer_type\tsource_dir\treason\n' > "${MANIFEST}"
for index in "${!CASE_IDS[@]}"; do
  case_id="${CASE_IDS[$index]}"
  cancer_type="${CANCER_TYPES[$index]}"
  source_root="${CASE_ROOTS[$index]}"
  source_dir="${source_root}/${case_id}"
  reason="${CASE_REASONS[$index]}"
  if [[ ! -d "${source_dir}" ]]; then
    echo "Case directory does not exist: ${source_dir}" >&2
    exit 2
  fi
  if [[ ! -d "${source_dir}/evaluation" ]]; then
    echo "Case is not benchmark-ready (missing evaluation/): ${source_dir}" >&2
    exit 2
  fi
  source_dir="$(cd "${source_dir}" && pwd -P)"
  printf '%s\t%s\t%s\t%s\n' \
    "${case_id}" "${cancer_type}" "${source_dir}" "${reason}" >> "${MANIFEST}"
done

print_plan() {
  echo "Case-study input root: ${STAGING_ROOT:-<temporary selected-case root>}"
  echo "Results root: ${RUNS_ROOT}"
  echo "Method selection: ${METHOD}"
  echo "Planner run ID: ${PLANNER_RUN_ID}"
  echo "MedClaw-only run ID: ${MEDCLAW_RUN_ID}"
  echo "Evaluation enabled: ${ENABLE_EVALUATION:-0}"
  echo "CUDA_VISIBLE_DEVICES: ${GPU_ID_VALUE}"
  echo "Force: ${FORCE_RUN}; dry-run: ${DRY_RUN_VALUE}"
  printf 'Cases:\n'
  for index in "${!CASE_IDS[@]}"; do
    printf '  - %s (%s): %s\n' \
      "${CASE_IDS[$index]}" "${CANCER_TYPES[$index]}" "${CASE_REASONS[$index]}"
  done
}

run_planner() {
  local log_path="${RUNS_ROOT}/logs/${PLANNER_RUN_ID}.log"
  echo "[case-study] Starting Planner V2 + DAA; log=${log_path}" >&2
  CASES_ROOT="${STAGING_ROOT}" \
  RUNS_ROOT="${RUNS_ROOT}" \
  RUN_ID="${PLANNER_RUN_ID}" \
  PLANNER_RELEASE_DIR="${PLANNER_RELEASE_DIR}" \
  PLANNER_MODE="daa_full" \
  ENABLE_EVALUATION="${ENABLE_EVALUATION:-0}" \
  GPU_ID="${GPU_ID_VALUE}" \
  FORCE="${FORCE_RUN}" \
  DRY_RUN="${DRY_RUN_VALUE}" \
    bash "${SCRIPT_DIR}/run_dual_agent_benchmark_batch.sh" 2>&1 | tee "${log_path}"
}

run_medclaw_only() {
  local log_path="${RUNS_ROOT}/logs/${MEDCLAW_RUN_ID}.log"
  echo "[case-study] Starting MedClaw only; log=${log_path}" >&2
  CASES_ROOT="${STAGING_ROOT}" \
  RUNS_ROOT="${RUNS_ROOT}" \
  RUN_ID="${MEDCLAW_RUN_ID}" \
  ENABLE_EVALUATION="${ENABLE_EVALUATION:-0}" \
  GPU_ID="${GPU_ID_VALUE}" \
  FORCE="${FORCE_RUN}" \
  DRY_RUN="${DRY_RUN_VALUE}" \
    bash "${SCRIPT_DIR}/run_benchmark_batch.sh" 2>&1 | tee "${log_path}"
}

verify_outputs() {
  local run_id="$1"
  local missing=0
  for case_id in "${CASE_IDS[@]}"; do
    result_path="${RUNS_ROOT}/${case_id}/${run_id}/final_answers.json"
    if [[ -f "${result_path}" ]]; then
      echo "[case-study] complete: ${result_path}" >&2
    else
      echo "[case-study] missing: ${result_path}" >&2
      missing=1
    fi
  done
  return "${missing}"
}

if [[ "${PLAN_ONLY}" == "1" ]]; then
  print_plan
  exit 0
fi

STAGING_ROOT="$(mktemp -d "${TMP_BASE}/cpgtraj-casestudy.XXXXXX")"
for index in "${!CASE_IDS[@]}"; do
  source_dir="${CASE_ROOTS[$index]}/${CASE_IDS[$index]}"
  source_dir="$(cd "${source_dir}" && pwd -P)"
  ln -s "${source_dir}" "${STAGING_ROOT}/${CASE_IDS[$index]}"
done
print_plan

case "${METHOD}" in
  planner)
    run_planner
    ;;
  medclaw-only)
    run_medclaw_only
    ;;
  both)
    run_planner
    run_medclaw_only
    ;;
esac

if [[ "${DRY_RUN_VALUE}" != "1" && "${DRY_RUN_VALUE}" != "true" ]]; then
  if [[ "${METHOD}" == "planner" || "${METHOD}" == "both" ]]; then
    verify_outputs "${PLANNER_RUN_ID}"
  fi
  if [[ "${METHOD}" == "medclaw-only" || "${METHOD}" == "both" ]]; then
    verify_outputs "${MEDCLAW_RUN_ID}"
  fi
fi

echo "[case-study] Finished. Results: ${RUNS_ROOT}" >&2
