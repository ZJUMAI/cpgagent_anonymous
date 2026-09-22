#!/usr/bin/env bash
set -euo pipefail

# =========================
# 项目路径
# =========================
PROJECT_ROOT="/data4/tujiayong/cpg_trajbench"
CASES_ROOT="${PROJECT_ROOT}/examples/cases"
RUNS_ROOT="${PROJECT_ROOT}/runs"
PARALLEL_CASES="${PARALLEL_CASES:-3}"

CASE_IDS=(
  TCGA-38-4625
  TCGA-38-4626
  TCGA-50-6673
  TCGA-50-8459
  TCGA-J2-8192
  TCGA-J2-8194
)

# agent:run_id:model_id
MODEL_SPECS=(
  "openai:run_openai_5.5:gpt-5.5"
  "gemini:run_gemini_3.1_pro_preview:gemini-3.1-pro-preview"
)

# Judge 固定 GPT-5.5（与 agent 的 openai 模型共用 MEDCLAW_OPENAI_MODEL）
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openai}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.5}"

cd "${PROJECT_ROOT}"

# =========================
# 1) 创建并激活虚拟环境
# =========================
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip
pip install -e ".[dev]"

# =========================
# 2) API Keys（改成你的；也可 export 覆盖）
# =========================
export OPENAI_API_KEY="${OPENAI_API_KEY:-$MEDCLAW_OPENAI_API_KEY}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-$MEDCLAW_GEMINI_API_KEY}"

export MEDCLAW_JUDGE_PROVIDER="${JUDGE_PROVIDER}"
export MEDCLAW_OPENAI_MODEL="${JUDGE_MODEL}"

# LLM 连接失败自动重试（指数退避，默认最多 4 次、初始间隔 2s）
export MEDCLAW_LLM_MAX_RETRIES="${MEDCLAW_LLM_MAX_RETRIES:-4}"
export MEDCLAW_LLM_RETRY_BASE_SEC="${MEDCLAW_LLM_RETRY_BASE_SEC:-2}"

# 代理（国外 API 需要；跑前也可手动 export 覆盖）
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7897}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"
if [[ -n "${ALL_PROXY:-}${all_proxy:-}" ]]; then
  echo "Warning: 检测到 ALL_PROXY/all_proxy（SOCKS）。若报 socksio 错误请 unset，或 pip install 'httpx[socks]'" >&2
fi

# =========================
# 3) 预检 Agent + Judge API 是否可调用
# =========================
echo ""
echo "========== API 预检（MODEL_SPECS + ${JUDGE_PROVIDER}-judge/${JUDGE_MODEL}）=========="
export PRECHECK_MODEL_SPECS="${MODEL_SPECS[*]}"
export PRECHECK_JUDGE_PROVIDER="${JUDGE_PROVIDER}"
export PRECHECK_JUDGE_MODEL="${JUDGE_MODEL}"
python - <<'PY'
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
# 4) Agent + GPT-5.5 Judge：case 间并行，case 内串行
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
  echo "[${case_id}] agent=${agent}, model=${model_id}, run_id=${run_id}, judge=${JUDGE_PROVIDER}/${JUDGE_MODEL}"
  python -m medclaw_benchmark.cli run-and-judge \
    --case-dir "${case_dir}" \
    --rubric-path "${rubric_path}" \
    --agent "${agent}" \
    --judge-provider "${JUDGE_PROVIDER}" \
    --judge llm \
    --run-id "${run_id}" \
    --runs-root "${RUNS_ROOT}"
}

run_case_worker() {
  local case_id="$1"
  local case_dir="${CASES_ROOT}/${case_id}"
  local rubric_path="${case_dir}/evaluation/${case_id}_rubric.json"
  local log_file="${BATCH_LOG_DIR}/${case_id}.log"
  local case_failed=0

  {
    echo "################################################################"
    echo "# Case: ${case_id} (worker PID $$)"
    echo "################################################################"

    for spec in "${MODEL_SPECS[@]}"; do
      IFS=':' read -r agent run_id model_id <<< "${spec}"
      run_one "${case_dir}" "${rubric_path}" "${case_id}" "${agent}" "${run_id}" "${model_id}" || case_failed=1
    done

    echo ""
    echo "# Case ${case_id} done, failed=${case_failed}"
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
  if ! python -m medclaw_benchmark.cli build-case --case-dir "${CASES_ROOT}/${case_id}"; then
    echo "Warning: build-case 失败，若 case 已有 clinical/molecular 等文件将继续尝试运行: ${case_id}" >&2
  fi
done

echo ""
echo "========== 启动 case 并行 worker（PARALLEL_CASES=${PARALLEL_CASES}）=========="
echo "各 case 日志: ${BATCH_LOG_DIR}/<case_id>.log"

for case_id in "${ready_cases[@]}"; do
  while (( $(jobs -rp | wc -l) >= PARALLEL_CASES )); do
    if ! wait -n; then failed=1; fi
  done
  echo "启动 worker: ${case_id}"
  run_case_worker "${case_id}" &
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
      python - <<PY
import json
from pathlib import Path
p = Path("${score_file}")
data = json.loads(p.read_text(encoding="utf-8"))
print(f"  ${run_id}: final_total={data.get('final_total')}, status={data.get('status')}")
PY
    else
      echo "  ${run_id}: judge_scores.json 不存在"
    fi
  done
done

echo ""
echo "结果目录: ${RUNS_ROOT}/{case_id}/run_*"
echo "case 日志: ${BATCH_LOG_DIR}/<case_id>.log"
echo "每个 run 可看: final_answers.json, agent_context.json, conversation_log.jsonl, error_analysis.md"
