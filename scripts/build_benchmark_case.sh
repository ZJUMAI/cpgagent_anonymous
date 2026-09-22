#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi

CASE_DIR="${1:-examples/cases/TCGA-38-4626}"
shift || true

"${PYTHON}" -m medclaw_benchmark.cli build-case --case-dir "${CASE_DIR}" "$@"
