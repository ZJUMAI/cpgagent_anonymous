#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# The 79-chunk Memory Encoder and Memory Store already include NSCLC, SCLC,
# UCEC, and NPC.  Retraining starts from a clean three-cancer Decoder/Router
# namespace while keeping those frozen, hash-checked artifacts.
export PLANNER_DATASET_ROOT="${PLANNER_DATASET_ROOT:-datasets/planner_v2_lung_endometrial_npc_gpt}"
export PLANNER_OUTPUT_ROOT="${PLANNER_OUTPUT_ROOT:-guideline_planner/outputs/planner_v2_lung_endometrial_npc}"
export PLANNER_MEMORY_ENCODER_DIR="${PLANNER_MEMORY_ENCODER_DIR:-guideline_planner/outputs/planner_v2_lung_endometrial/memory_encoder}"
export PLANNER_MEMORY_STORE_DIR="${PLANNER_MEMORY_STORE_DIR:-guideline_planner/outputs/planner_v2_lung_endometrial/memory_store}"
export PLANNER_RELEASE_ID="${PLANNER_RELEASE_ID:-planner-v2-lung-endometrial-npc}"
export PLANNER_WARN_ONLY_QUALITY_GATES="${PLANNER_WARN_ONLY_QUALITY_GATES:-1}"
export PLANNER_ROUTER_EVAL_STEPS="${PLANNER_ROUTER_EVAL_STEPS:-500}"
export PLANNER_ROUTER_GENERATION_EVAL_STEPS="${PLANNER_ROUTER_GENERATION_EVAL_STEPS:-500}"

exec bash "${SCRIPT_DIR}/planner_v2_pipeline.sh" "$@"
