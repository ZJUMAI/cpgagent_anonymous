#!/usr/bin/env bash
# LUAD multiscale patch coords + CONCH features (256 / 512 / 1024).
set -euo pipefail
set -o pipefail

# Usage:
#   SCALE=512 CUDA_VISIBLE_DEVICES=0 bash run_luad_multiscale.sh
#   SCALE=all LIMIT=5 bash run_luad_multiscale.sh

SCALE="${SCALE:-512}"          # 256 | 512 | 1024 | all
LIMIT="${LIMIT:-0}"
DEVICE="${CUDA_VISIBLE_DEVICES:-4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_SEG="${SKIP_SEG:-1}"      # 1=reuse masks/masks/*.jpg
SKIP_CONCH="${SKIP_CONCH:-0}"  # 1=coords only
SKIP_EXISTING="${SKIP_EXISTING:-1}"      # skip existing coords h5
CONCH_SKIP_EXISTING="${CONCH_SKIP_EXISTING:-0}"  # 0=force re-extract .pt

TCGA_ROOT="/data4/share/TCGA/TCGA_LUAD"
TCGA_WRITE="${TCGA_LUAD_WRITE_ROOT:-/data4/tujiayong/share/TCGA/TCGA_LUAD}"
CONCH_WRITE="${CONCH_LUAD_WRITE_ROOT:-/data4/tujiayong/share/CONCH_TCGA/LUAD/features}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${TCGA_WRITE}/patches" "${CONCH_WRITE}"

export CUDA_VISIBLE_DEVICES="${DEVICE}"

LIMIT_ARG=()
if [[ -n "${LIMIT}" && "${LIMIT}" != "0" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

EXIST_ARG=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  EXIST_ARG=(--skip-existing)
fi

CONCH_EXIST_ARG=()
if [[ "${CONCH_SKIP_EXISTING}" == "1" ]]; then
  CONCH_EXIST_ARG=(--skip-existing)
fi

run_one() {
  local ps="$1"
  local do_conch=1
  [[ "${ps}" == "1024" ]] && do_conch=0
  [[ "${ps}" == "256" && "${SKIP_CONCH}" == "1" ]] && do_conch=0

  echo "========== patch_size=${ps} @ $(date -Iseconds) =========="

  if [[ "${ps}" == "256" ]]; then
    PY=("${PYTHON_BIN}" "${SCRIPT_DIR}/luad_create_patches.py" --scale 256 --mode verify_only
      --out-dir "${TCGA_ROOT}/patches/patches"
    )
  else
    # Align 512/1024 tissue footprint with CLAM 256 coords (2×2 / 4×4 aggregation).
    PY=("${PYTHON_BIN}" "${SCRIPT_DIR}/luad_create_patches.py"
      --scale "${ps}"
      --mode derive_from_256
      --coords256-dir "${TCGA_ROOT}/patches/patches"
      --out-dir "${TCGA_WRITE}/patches/patches_${ps}"
      "${EXIST_ARG[@]}"
    )
  fi
  PY+=("${LIMIT_ARG[@]}")
  "${PY[@]}"

  if [[ "${SKIP_CONCH}" == "0" && "${do_conch}" == "1" ]]; then
    local bs=256
    [[ "${ps}" == "512" ]] && bs=32
    "${PYTHON_BIN}" "${SCRIPT_DIR}/luad_extract_conch.py" \
      --scale "${ps}" \
      --wsi-dir "${TCGA_ROOT}/wsi" \
      --coords-dir "${TCGA_WRITE}/patches/patches_${ps}" \
      --out-pt-dir "${CONCH_WRITE}/pt_files_${ps}" \
      --out-h5-dir "${CONCH_WRITE}/h5_files_${ps}" \
      --batch-size "${bs}" \
      --device cuda:0 \
      "${CONCH_EXIST_ARG[@]}" \
      "${LIMIT_ARG[@]}"
  fi

  local verify_args=(--scale "${ps}" --expect-size "${ps}" --wsi-dir "${TCGA_ROOT}/wsi")
  if [[ "${ps}" == "256" ]]; then
    verify_args+=(--coords-dir "${TCGA_ROOT}/patches/patches"
      --pt-dir "/data4/share/CONCH_TCGA/LUAD/features/pt_files")
  else
    verify_args+=(--coords-dir "${TCGA_WRITE}/patches/patches_${ps}")
    if [[ "${do_conch}" == "0" || "${SKIP_CONCH}" == "1" ]]; then
      verify_args+=(--skip-pt)
    else
      verify_args+=(--pt-dir "${CONCH_WRITE}/pt_files_${ps}")
    fi
  fi
  if [[ "${ps}" == "256" && "${SKIP_CONCH}" == "1" ]]; then
    verify_args+=(--skip-pt)
  fi
  local sample_reads=3
  [[ "${ps}" == "256" ]] && sample_reads=0
  "${PYTHON_BIN}" "${SCRIPT_DIR}/luad_verify_scale.py" \
    "${verify_args[@]}" \
    --sample-reads "${sample_reads}" \
    "${LIMIT_ARG[@]}"
}

if [[ "${SCALE}" == "all" ]]; then
  SKIP_SEG=1 SKIP_CONCH=1 run_one 256
  SKIP_SEG=1 SKIP_CONCH=0 run_one 512
  SKIP_SEG=1 SKIP_CONCH=0 run_one 1024
else
  run_one "${SCALE}"
fi

echo "========== done @ $(date -Iseconds) =========="
