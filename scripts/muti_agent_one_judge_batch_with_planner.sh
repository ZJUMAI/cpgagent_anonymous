#!/usr/bin/env bash
set -euo pipefail

# =========================
# 项目路径（有 Planner 消融；结果写到 new_run）
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CASES_ROOT="${PROJECT_ROOT}/examples/cases"
RUNS_ROOT="${PROJECT_ROOT}/new_run"
CASE_IDS=(
  TCGA-38-4625
  TCGA-38-A44F
  TCGA-50-6590
  TCGA-50-8459
  TCGA-J2-A4AE
  TCGA-J2-A4AG
)

# agent:run_id:model_id（与无 Planner 的 run_* 区分开）
MODEL_SPECS=(
  "openai:run_openai_5.5_planner:gpt-5.5"
  "gemini:run_gemini_3.1_pro_preview_planner:gemini-3.1-pro-preview"
)

# Judge 固定 GPT-5.5（与 agent 的 openai 模型共用 MEDCLAW_OPENAI_MODEL）
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openai}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.5}"

# Planner（与 run_dual_agent_benchmark_batch.sh 对齐；可 export 覆盖）
PLANNER_RELEASE_DIR="${PLANNER_RELEASE_DIR:-${PROJECT_ROOT}/guideline_planner/outputs/planner_v2_lung_endometrial/release}"
PLANNER_MODE="${PLANNER_MODE:-}"
PLANNER_MEMORY_DIR="${PLANNER_MEMORY_DIR:-}"
PLANNER_DECODER_ARTIFACT_DIR="${PLANNER_DECODER_ARTIFACT_DIR:-}"
PLANNER_TOP_K="${PLANNER_TOP_K:-}"
PLANNER_DEVICE="${PLANNER_DEVICE:-auto}"
PLANNER_QUERY_ENCODER_DEVICE="${PLANNER_QUERY_ENCODER_DEVICE:-cpu}"
PLANNER_MAX_NEW_TOKENS="${PLANNER_MAX_NEW_TOKENS:-auto}"
PLANNER_OUTPUT_MODE="json"
PLANNER_ROUTING_CONFIG="${PLANNER_ROUTING_CONFIG:-}"
PLANNER_ROUTING_CHECKPOINT="${PLANNER_ROUTING_CHECKPOINT:-}"
MAX_PLANNER_ROUNDS="${MAX_PLANNER_ROUNDS:-4}"
MAX_TOOL_ROUNDS_PER_STEP="${MAX_TOOL_ROUNDS_PER_STEP:-12}"

# =========================
# 0) 自动选卡（最先做；没有空闲卡就立刻退出，不装依赖、不跑 case）
# =========================
# 规则：
#   - 全机扫描 nvidia-smi
#   - total >= MIN_GPU_MEM_MIB（默认 20GB，排除 2080 Ti）
#   - free  >= MIN_GPU_FREE_MIB（默认 50GB，给 planner + agent 留足余量）
#   - 每张入选卡同时只跑 1 个 case
#   - 0 张可用 -> exit 2
# 可选：GPU_IDS=2,3 只在候选里挑；不设则扫全部。
MIN_GPU_MEM_MIB="${MIN_GPU_MEM_MIB:-20000}"
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-50000}"
MAX_GPUS="${MAX_GPUS:-0}" # 0 = 不限制，用上所有合格卡

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "STOP: nvidia-smi 不可用，无法自动选卡。" >&2
  exit 2
fi

CANDIDATE_GPU_CSV="${GPU_IDS:-${GPU_ID:-}}"
echo "========== 自动选卡 ==========" >&2
echo "门槛: total>=${MIN_GPU_MEM_MIB} MiB, free>=${MIN_GPU_FREE_MIB} MiB" >&2
if [[ -n "${CANDIDATE_GPU_CSV}" ]]; then
  echo "候选范围: ${CANDIDATE_GPU_CSV}" >&2
else
  echo "候选范围: 全机扫描" >&2
fi

GPU_PICK_FILE="$(mktemp)"
set +e
MIN_GPU_MEM_MIB="${MIN_GPU_MEM_MIB}" \
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB}" \
CANDIDATE_GPU_CSV="${CANDIDATE_GPU_CSV}" \
MAX_GPUS="${MAX_GPUS}" \
python3 - <<'PY' >"${GPU_PICK_FILE}"
import csv
import os
import subprocess
import sys

min_total = int(os.environ["MIN_GPU_MEM_MIB"])
min_free = int(os.environ["MIN_GPU_FREE_MIB"])
max_gpus = int(os.environ.get("MAX_GPUS", "0") or "0")
candidates_csv = os.environ.get("CANDIDATE_GPU_CSV", "").strip()
wanted = {x.strip() for x in candidates_csv.split(",") if x.strip()} if candidates_csv else None

out = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
eligible = []
for row in csv.reader(out.splitlines()):
    if len(row) < 4:
        continue
    idx, name, total, free = [c.strip() for c in row[:4]]
    if wanted is not None and idx not in wanted:
        continue
    total_i, free_i = int(float(total)), int(float(free))
    ok = total_i >= min_total and free_i >= min_free
    tag = "OK" if ok else "SKIP"
    print(
        f"[{tag}] GPU {idx}: {name}, total={total_i} MiB, free={free_i} MiB",
        file=sys.stderr,
    )
    if ok:
        eligible.append((free_i, idx))

eligible.sort(reverse=True)
if max_gpus > 0:
    eligible = eligible[:max_gpus]

if not eligible:
    print(
        "STOP: 没有空闲可用 GPU，退出不跑。"
        f"（需要 total>={min_total} MiB 且 free>={min_free} MiB）",
        file=sys.stderr,
    )
    sys.exit(2)

ids = [idx for _, idx in eligible]
print(f"选中 {len(ids)} 张卡: {', '.join(ids)}", file=sys.stderr)
print("\n".join(ids))
PY
GPU_PICK_RC=$?
set -e

if (( GPU_PICK_RC != 0 )); then
  rm -f "${GPU_PICK_FILE}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv >&2 || true
  exit 2
fi

mapfile -t GPU_ID_LIST < "${GPU_PICK_FILE}"
rm -f "${GPU_PICK_FILE}"
# 去掉空行
_cleaned=()
for _g in "${GPU_ID_LIST[@]}"; do
  [[ -n "${_g}" ]] && _cleaned+=("${_g}")
done
GPU_ID_LIST=("${_cleaned[@]}")
unset _cleaned _g

if (( ${#GPU_ID_LIST[@]} == 0 )); then
  echo "STOP: 没有空闲可用 GPU，退出不跑。" >&2
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv >&2 || true
  exit 2
fi

# 并行度 = 选中卡数（每卡 1 个 case）
PARALLEL_CASES="${PARALLEL_CASES:-${#GPU_ID_LIST[@]}}"
if (( PARALLEL_CASES > ${#GPU_ID_LIST[@]} )); then
  PARALLEL_CASES=${#GPU_ID_LIST[@]}
fi
if (( PARALLEL_CASES < 1 )); then
  echo "STOP: PARALLEL_CASES < 1，退出不跑。" >&2
  exit 2
fi

echo "将用 GPU_ID_LIST=${GPU_ID_LIST[*]} 并行跑 PARALLEL_CASES=${PARALLEL_CASES}" >&2
echo "========== 选卡完成，开始准备环境 ==========" >&2

cd "${PROJECT_ROOT}"

# =========================
# 1) 创建并激活虚拟环境
# =========================
USE_CURRENT_ENV="${USE_CURRENT_ENV:-1}"
if [[ "${USE_CURRENT_ENV}" == "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
  echo "Using current Python environment: ${PYTHON_BIN}" >&2
  INSTALL_PROJECT_DEPS="${INSTALL_PROJECT_DEPS:-0}"
else
  if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PYTHON_BIN="${PYTHON_BIN:-python}"
  INSTALL_PROJECT_DEPS="${INSTALL_PROJECT_DEPS:-1}"
  "${PYTHON_BIN}" -m pip install -U pip
fi
# planner extra: torch / transformers / peft（跑 LatentGuidelinePlanner 必需）
if [[ "${INSTALL_PROJECT_DEPS}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install -e ".[dev,planner]"
else
  "${PYTHON_BIN}" -m pip install -e . --no-deps
fi

"${PYTHON_BIN}" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"Python: {sys.executable}", file=sys.stderr)
    print(f"STOP: failed to import torch in the selected environment: {exc}", file=sys.stderr)
    raise SystemExit(2)

print(f"Python: {sys.executable}", file=sys.stderr)
print(f"Torch: {torch.__version__}, torch CUDA: {torch.version.cuda}", file=sys.stderr)
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}", file=sys.stderr)
PY

# =========================
# 2) API Keys（改成你的；也可 export 覆盖）
# =========================
export OPENAI_API_KEY="${OPENAI_API_KEY:-$MEDCLAW_OPENAI_API_KEY}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-$MEDCLAW_GEMINI_API_KEY}"

export MEDCLAW_JUDGE_PROVIDER="${JUDGE_PROVIDER}"
export MEDCLAW_OPENAI_MODEL="${JUDGE_MODEL}"
export MEDCLAW_CASES_ROOT="${CASES_ROOT}"
# 注意：不要在这里全局 export CUDA_VISIBLE_DEVICES；每个 case worker 独占一张卡。

# LLM 连接失败自动重试（指数退避，默认最多 4 次、初始间隔 2s）
export MEDCLAW_LLM_MAX_RETRIES="${MEDCLAW_LLM_MAX_RETRIES:-4}"
export MEDCLAW_LLM_RETRY_BASE_SEC="${MEDCLAW_LLM_RETRY_BASE_SEC:-2}"

# 代理（国外 API 需要；跑前也可手动 export 覆盖）
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7897}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"
if [[ -n "${ALL_PROXY:-}${all_proxy:-}" ]]; then
  echo "Warning: 检测到 ALL_PROXY/all_proxy（SOCKS）。若报 socksio 错误请 unset，或 pip install 'httpx[socks]'" >&2
fi

echo "Using RUNS_ROOT=${RUNS_ROOT}" >&2
if [[ ! -f "${PLANNER_RELEASE_DIR}/planner_release.json" ]]; then
  echo "Planner release 不存在: ${PLANNER_RELEASE_DIR}/planner_release.json" >&2
  exit 2
fi
echo "Using PLANNER_RELEASE_DIR=${PLANNER_RELEASE_DIR}" >&2
echo "Using PLANNER_DEVICE=${PLANNER_DEVICE}" >&2
echo "Using PLANNER_OUTPUT_MODE=${PLANNER_OUTPUT_MODE}" >&2
echo "Using GPU_ID_LIST=${GPU_ID_LIST[*]} PARALLEL_CASES=${PARALLEL_CASES}" >&2

# =========================
# 3) 预检 Agent + Judge API 是否可调用
# =========================
echo ""
echo "========== API 预检（MODEL_SPECS + ${JUDGE_PROVIDER}-judge/${JUDGE_MODEL}）=========="
export PRECHECK_MODEL_SPECS="${MODEL_SPECS[*]}"
export PRECHECK_JUDGE_PROVIDER="${JUDGE_PROVIDER}"
export PRECHECK_JUDGE_MODEL="${JUDGE_MODEL}"
"${PYTHON_BIN}" - <<'PY'
import os
import sys

from medclaw.llm.factory import create_llm_client
from medclaw.llm.protocols import LLMAPIError, LLMConfigurationError

ENV_BY_AGENT = {
    "claude": "MEDCLAW_CLAUDE_MODEL",
    "openai": "MEDCLAW_OPENAI_MODEL",
    "gemini": "MEDCLAW_GEMINI_MODEL",
}

failed: list[str] = []
seen: set[tuple[str, str]] = set()

for spec in os.environ.get("PRECHECK_MODEL_SPECS", "").split():
    agent, _run_id, model_id = spec.split(":", 2)
    key = (agent, model_id)
    if key in seen:
        continue
    seen.add(key)
    os.environ[ENV_BY_AGENT[agent]] = model_id
    label = f"{agent} ({model_id})"
    try:
        client = create_llm_client(agent)
        print(f"[check] {label} ...", flush=True)
        completion = client.complete(
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            tools=[],
            tool_choice="none",
        )
        content = completion.message.get("content", "")
        print(f"[ok]   {label}: {str(content)[:120]!r}")
    except LLMConfigurationError as exc:
        print(f"[fail] {label}: 配置错误 - {exc}", file=sys.stderr)
        failed.append(label)
    except LLMAPIError as exc:
        print(f"[fail] {label}: API 调用失败 - {exc}", file=sys.stderr)
        failed.append(label)
    except Exception as exc:
        print(f"[fail] {label}: {type(exc).__name__} - {exc}", file=sys.stderr)
        failed.append(label)

judge_provider = os.environ.get("PRECHECK_JUDGE_PROVIDER", "openai")
judge_model = os.environ.get("PRECHECK_JUDGE_MODEL", "gpt-5.5")
if judge_provider == "openai":
    os.environ["MEDCLAW_OPENAI_MODEL"] = judge_model
label = f"{judge_provider} (judge, {judge_model})"
try:
    client = create_llm_client(judge_provider)
    model = getattr(getattr(client, "config", None), "model", "?")
    print(f"[check] {label} model={model} ...", flush=True)
    completion = client.complete(
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        tools=[],
        tool_choice="none",
    )
    content = completion.message.get("content", "")
    print(f"[ok]   {label}: {str(content)[:120]!r}")
except LLMConfigurationError as exc:
    print(f"[fail] {label}: 配置错误 - {exc}", file=sys.stderr)
    failed.append(label)
except LLMAPIError as exc:
    print(f"[fail] {label}: API 调用失败 - {exc}", file=sys.stderr)
    failed.append(label)
except Exception as exc:
    print(f"[fail] {label}: {type(exc).__name__} - {exc}", file=sys.stderr)
    failed.append(label)

if failed:
    print(f"\nAPI 预检失败: {', '.join(failed)}", file=sys.stderr)
    print("请检查 API Key、HTTP_PROXY/HTTPS_PROXY、模型名、账号余额/权限", file=sys.stderr)
    sys.exit(1)

print("\nAPI 预检全部通过。")
PY

# =========================
# 4) Agent + Planner + GPT-5.5 Judge：case 间并行，case 内串行
# =========================
run_one() {
  local case_dir="$1"
  local rubric_path="$2"
  local case_id="$3"
  local agent="$4"
  local run_id="$5"
  local model_id="$6"

  case "${agent}" in
    claude) export MEDCLAW_CLAUDE_MODEL="${model_id}" ;;
    openai) export MEDCLAW_OPENAI_MODEL="${model_id}" ;;
    gemini) export MEDCLAW_GEMINI_MODEL="${model_id}" ;;
    *) echo "未知 agent: ${agent}" >&2; return 1 ;;
  esac

  # Judge 固定用 JUDGE_MODEL；openai agent 与 judge 同为 gpt-5.5 时共用该变量
  if [[ "${JUDGE_PROVIDER}" == "openai" && "${agent}" != "openai" ]]; then
    export MEDCLAW_OPENAI_MODEL="${JUDGE_MODEL}"
  fi

  echo ""
  echo "[${case_id}] agent=${agent}, model=${model_id}, run_id=${run_id}, judge=${JUDGE_PROVIDER}/${JUDGE_MODEL}, planner=on"
  local routing_args=(--planner-release-dir "${PLANNER_RELEASE_DIR}")
  if [[ -n "${PLANNER_MODE}" ]]; then
    routing_args+=(--planner-mode "${PLANNER_MODE}")
  fi
  if [[ -n "${PLANNER_MEMORY_DIR}" ]]; then
    routing_args+=(--planner-memory-dir "${PLANNER_MEMORY_DIR}")
  fi
  if [[ -n "${PLANNER_DECODER_ARTIFACT_DIR}" ]]; then
    routing_args+=(--planner-decoder-artifact-dir "${PLANNER_DECODER_ARTIFACT_DIR}")
  fi
  if [[ -n "${PLANNER_TOP_K}" ]]; then
    routing_args+=(--planner-top-k "${PLANNER_TOP_K}")
  fi
  if [[ -n "${PLANNER_ROUTING_CONFIG}" ]]; then
    routing_args+=(--planner-routing-config "${PLANNER_ROUTING_CONFIG}")
  fi
  if [[ -n "${PLANNER_ROUTING_CHECKPOINT}" ]]; then
    routing_args+=(--planner-routing-checkpoint "${PLANNER_ROUTING_CHECKPOINT}")
  fi
  "${PYTHON_BIN}" -m medclaw_benchmark.cli dual-agent-run-and-judge \
    --case-dir "${case_dir}" \
    --rubric-path "${rubric_path}" \
    --agent "${agent}" \
    --judge-provider "${JUDGE_PROVIDER}" \
    --judge llm \
    --run-id "${run_id}" \
    --runs-root "${RUNS_ROOT}" \
    --planner-device "${PLANNER_DEVICE}" \
    --planner-query-encoder-device "${PLANNER_QUERY_ENCODER_DEVICE}" \
    --planner-max-new-tokens "${PLANNER_MAX_NEW_TOKENS}" \
    --planner-output-mode "${PLANNER_OUTPUT_MODE}" \
    --max-planner-rounds "${MAX_PLANNER_ROUNDS}" \
    --max-tool-rounds-per-step "${MAX_TOOL_ROUNDS_PER_STEP}" \
    "${routing_args[@]}"
}

run_case_worker() {
  local case_id="$1"
  local gpu_id="$2"
  local case_dir="${CASES_ROOT}/${case_id}"
  local rubric_path="${case_dir}/evaluation/${case_id}_rubric.json"
  local log_file="${BATCH_LOG_DIR}/${case_id}.log"
  local case_failed=0

  # 每个 worker 独占一张物理 GPU；进程内看到的是 cuda:0
  export CUDA_VISIBLE_DEVICES="${gpu_id}"

  {
    echo "################################################################"
    echo "# Case: ${case_id} (worker PID $$, GPU ${gpu_id} -> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
    echo "################################################################"

    for spec in "${MODEL_SPECS[@]}"; do
      IFS=':' read -r agent run_id model_id <<< "${spec}"
      run_one "${case_dir}" "${rubric_path}" "${case_id}" "${agent}" "${run_id}" "${model_id}" || case_failed=1
    done

    echo ""
    echo "# Case ${case_id} done, failed=${case_failed}, gpu=${gpu_id}"
  } >> "${log_file}" 2>&1

  return "${case_failed}"
}

BATCH_LOG_DIR="${RUNS_ROOT}/_batch_logs"
mkdir -p "${BATCH_LOG_DIR}"

failed=0
ready_cases=()
for case_id in "${CASE_IDS[@]}"; do
  case_dir="${CASES_ROOT}/${case_id}"
  rubric_path="${case_dir}/evaluation/${case_id}_rubric.json"

  if [[ ! -d "${case_dir}" ]]; then
    echo "跳过 ${case_id}: 目录不存在 ${case_dir}" >&2
    failed=1
    continue
  fi
  if [[ ! -f "${rubric_path}" ]]; then
    echo "跳过 ${case_id}: rubric 不存在 ${rubric_path}" >&2
    failed=1
    continue
  fi

  ready_cases+=("${case_id}")
done

if (( ${#ready_cases[@]} == 0 )); then
  echo "没有可运行的 case" >&2
  exit 1
fi

echo ""
echo "========== build-case（串行预构建 ${#ready_cases[@]} 个 case）=========="
for case_id in "${ready_cases[@]}"; do
  echo "build-case: ${case_id}"
  if ! "${PYTHON_BIN}" -m medclaw_benchmark.cli build-case --case-dir "${CASES_ROOT}/${case_id}"; then
    echo "Warning: build-case 失败，若 case 已有 clinical/molecular 等文件将继续尝试运行: ${case_id}" >&2
  fi
done

echo ""
echo "========== 启动多卡 case worker（GPUs=${GPU_ID_LIST[*]}, PARALLEL_CASES=${PARALLEL_CASES}）=========="
echo "调度：空闲 GPU 池；每卡同时最多 1 个 case"
echo "各 case 日志: ${BATCH_LOG_DIR}/<case_id>.log"

# 空闲 GPU 池 + pid->gpu 映射，避免轮询把新 case 派到仍占用的卡上
FREE_GPUS=("${GPU_ID_LIST[@]}")
declare -A PID_TO_GPU=()

wait_for_free_gpu() {
  while (( ${#FREE_GPUS[@]} == 0 )); do
    if ! wait -n -p finished_pid; then
      failed=1
    fi
    if [[ -n "${finished_pid:-}" && -n "${PID_TO_GPU[$finished_pid]:-}" ]]; then
      FREE_GPUS+=("${PID_TO_GPU[$finished_pid]}")
      unset "PID_TO_GPU[$finished_pid]"
    fi
    finished_pid=""
  done
}

for case_id in "${ready_cases[@]}"; do
  # 并行上限：不超过 PARALLEL_CASES，且必须有空闲 GPU
  while (( $(jobs -rp | wc -l) >= PARALLEL_CASES )); do
    if ! wait -n -p finished_pid; then
      failed=1
    fi
    if [[ -n "${finished_pid:-}" && -n "${PID_TO_GPU[$finished_pid]:-}" ]]; then
      FREE_GPUS+=("${PID_TO_GPU[$finished_pid]}")
      unset "PID_TO_GPU[$finished_pid]"
    fi
    finished_pid=""
  done
  wait_for_free_gpu

  gpu_id="${FREE_GPUS[0]}"
  FREE_GPUS=("${FREE_GPUS[@]:1}")
  echo "启动 worker: ${case_id} -> GPU ${gpu_id} (free_gpus_left=${#FREE_GPUS[@]})"
  run_case_worker "${case_id}" "${gpu_id}" &
  PID_TO_GPU[$!]="${gpu_id}"
done

while (( $(jobs -rp | wc -l) > 0 )); do
  if ! wait -n; then failed=1; fi
done

echo ""
echo "========== 全部 worker 结束 =========="
for case_id in "${ready_cases[@]}"; do
  echo "--- tail ${BATCH_LOG_DIR}/${case_id}.log ---"
  tail -n 3 "${BATCH_LOG_DIR}/${case_id}.log" 2>/dev/null || true
done

if (( failed != 0 )); then
  echo "一个或多个 case / agent 运行失败" >&2
  exit 1
fi

# =========================
# 5) 汇总分数
# =========================
echo ""
echo "========== 分数汇总 =========="
for case_id in "${CASE_IDS[@]}"; do
  echo ""
  echo "--- ${case_id} ---"
  for spec in "${MODEL_SPECS[@]}"; do
    IFS=':' read -r _agent run_id _model_id <<< "${spec}"
    score_file="${RUNS_ROOT}/${case_id}/${run_id}/judge_scores.json"
    if [[ -f "${score_file}" ]]; then
      SCORE_FILE="${score_file}" RUN_ID="${run_id}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

data = json.loads(Path(os.environ["SCORE_FILE"]).read_text(encoding="utf-8"))
print(
    f"  {os.environ['RUN_ID']}: final_total={data.get('final_total')}, "
    f"status={data.get('status')}"
)
PY
    else
      echo "  ${run_id}: judge_scores.json 不存在"
    fi
  done
done

echo ""
echo "结果目录: ${RUNS_ROOT}/<case_id>/run_*_planner"
echo "case 日志: ${BATCH_LOG_DIR}/<case_id>.log"
echo "每个 run 可看: final_answers.json, planner_outputs.jsonl, dual_agent_rounds.jsonl, error_analysis.md"
