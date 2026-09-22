# 一键运行脚本

这些脚本是仓库级 wrapper，只负责定位项目根目录、选择 `.venv` 中的 Python
（如果存在）并调用已有 CLI 或 demo。真实逻辑仍在 `medclaw_benchmark`、
`examples/` 和各个 skill 中。

## Benchmark

生成/刷新 TCGA-38-4626 轻量病例包：

```powershell
.\scripts\build_benchmark_case.ps1
```

```bash
bash scripts/build_benchmark_case.sh
```

运行 model-driven benchmark；核心多模态模型通过 `tool_choice=auto` 自己规划工具调用。
图像工具切出 ROI 后会立刻回传给模型，最终用完整可审计 trajectory 和
LLM-as-Judge 评分：

```powershell
.\scripts\run_benchmark_llm_judge.ps1
```

```bash
bash scripts/run_benchmark_llm_judge.sh
```

批量运行服务器上的已处理病例；默认扫描 `/data4/share/cpgtrajbench/LUNG`，只跑含
`evaluation/` 的 case，已有完整结果会自动跳过：

```powershell
.\scripts\run_benchmark_batch.ps1 -CasesRoot /data4/share/cpgtrajbench/LUNG
```

```bash
bash scripts/run_benchmark_batch.sh
```

常用参数可通过环境变量覆盖：

```bash
CASES_ROOT=/data4/share/cpgtrajbench/LUNG RUN_ID=run_001 bash scripts/run_benchmark_batch.sh
DRY_RUN=1 bash scripts/run_benchmark_batch.sh
LIMIT=10 bash scripts/run_benchmark_batch.sh
FORCE=1 bash scripts/run_benchmark_batch.sh
GPU_ID=5 bash scripts/run_benchmark_batch.sh
```

批量脚本会自动导出 `MEDCLAW_CASES_ROOT=$CASES_ROOT`，因此真实 skill 会从同一
数据集根目录解析 CT 和 WSI。

LLM Judge 和核心多模态 Agent 默认连接
`http://127.0.0.1:8087/v1` 的 `qwen3.6-27b`。先设置：

```bash
export MEDCLAW_LOCAL_API_KEY=local-qwen-key
python scripts/check_local_llm.py
```

API 对照实验仍可使用：

```bash
export DASHSCOPE_API_KEY=你的Key
AGENT=qwen JUDGE_PROVIDER=qwen bash scripts/run_benchmark_batch.sh
```

脚本不会读取仓库文件或保存 key。

## Agent / Demo

运行交互式 CancerClaw agent（默认本地 Qwen3.6-27B）：

```powershell
.\scripts\run_qwen_agent.ps1
```

运行 CT ROI demo：

```powershell
.\scripts\run_ct_roi_demo.ps1
```

运行 CONCH patch ROI demo：

```powershell
.\scripts\run_conch_patch_roi_demo.ps1
```

Linux/macOS 可使用对应 `.sh` 脚本。

## Tests

运行默认 pytest：

```powershell
.\scripts\run_tests.ps1
```

只跑 benchmark 测试：

```powershell
.\scripts\run_tests.ps1 tests\test_benchmark.py -q
```

```bash
bash scripts/run_tests.sh tests/test_benchmark.py -q
```

## Dual-Agent Planner

正式 Planner V2 训练、固定集评估和 release 打包请按
`guideline_planner/README_zh.md` 执行。推荐的一体化入口为：

```bash
bash scripts/planner_v2_pipeline.sh preflight
bash scripts/planner_v2_pipeline.sh smoke
# 依次执行 train-memory、export-memory、overfit-decoder、train-decoder、build-routing、
# train-router、evaluate、package-release

bash scripts/test_planner_medclaw.sh planner-fixed
bash scripts/test_planner_medclaw.sh single
bash scripts/test_planner_medclaw.sh batch-smoke
```

`run_dual_agent_benchmark_batch.sh` 默认定位 Conda `cpg` 环境，不再优先使用
项目 `.venv`。可通过 `CONDA_ENV_NAME` 更换环境名，或用 `PYTHON_BIN` 指定解释器：

```bash
PYTHON_BIN="$CONDA_PREFIX/bin/python" \
bash scripts/run_dual_agent_benchmark_batch.sh
```

脚本启动时会打印实际 Python 路径、Torch 版本、Torch CUDA build 和 CUDA
可用状态；找不到目标环境时会直接退出。

正式批量入口中的所有 skills、Planner 和 query encoder 都使用脚本选定的同一个
Python（默认是 Conda `cpg` 环境，也可由 `PYTHON_BIN` 覆盖）。Planner 和 query
encoder 默认使用 GPU（`PLANNER_DEVICE=auto`、
`PLANNER_QUERY_ENCODER_DEVICE=cuda`）；不存在单独的 per-skill Conda 环境。
显存不足时可将 query encoder 单独改为 `PLANNER_QUERY_ENCODER_DEVICE=cpu`。

该脚本现在默认读取
`guideline_planner/outputs/planner_v2_lung_endometrial/release/planner_release.json`。
可用 `PLANNER_RELEASE_DIR` 和 `PLANNER_MODE=latent_topk|daa_full` 覆盖。
`PLANNER_MEMORY_DIR`、`PLANNER_DECODER_ARTIFACT_DIR`、`PLANNER_TOP_K` 仅作为显式
调试覆盖，正式运行不应使用它们拼装 artifact。

在加载 9B Planner 前，脚本会先对全部待运行病例执行 model-free preflight：
只允许当前 release 的 `lung/nsclc` 与 `endometrial/ucec` action scope，并核对
rubric 中显式声明的指南年份。比如 2025 rubric 不能与 NSCLC 2010 release
静默混跑，必须更新 rubric 或选择时间一致的 release。可用 `DRY_RUN=1` 只执行
release/hash/case/rubric 预检。预检通过后 Planner 在整个 batch 中只加载一次，
不会按病例重复加载模型。

Planner 输出中的 `pathology.read_diagnostic_report`、
`molecular.complete_biomarker_profile`、`radiology.review_staging_extent` 等是训练
监督的语义 skill。运行时会将它们显式映射到可调用的 MedClaw 工具，并在成功
后把对应语义 alias 写回 `completed_skills`；`treatment.*` 决策标签不会被误当
成工具调用。`guideline.retrieve` 的癌种、指南体系和版本则由 release 中的
Patient State context 强制绑定，Agent 传入的冲突过滤条件不会覆盖它。

Planner routing 仅使用当前 `patient_state.v2`、版本过滤后的 memory 检索结果、
Gate 和 DAA。脚本不再读取或更新 transition graph，运行顺序不会改变其他
case 的候选 memory。

### 使用旧版轨迹评分批量重评

`rescore_trajectory_batch.py` 使用当前仓库中的 `dyn_traj_alignment_v1`，不调用
LLM。它重算 `trajectory_scores.json`，复用原 `judge_scores.json.final_total` 作为
Static Rubric Micro 分，并更新 `0.5 × Micro + 0.5 × Macro` 综合分。默认在第一次
覆盖前生成 `*.before_legacy_rescore.json` 备份；不会读取或修改任何 `*.v2.json`。

```bash
python scripts/rescore_trajectory_batch.py \
  --runs-root data/runs \
  --run-id planner_v2_lung_daa \
  --run-id medclaw_only_lung \
  --run-id planner_v2_ucec_daa \
  --run-id medclaw_only_ucec_full \
  --pair planner_v2_lung_daa medclaw_only_lung \
  --pair planner_v2_ucec_daa medclaw_only_ucec_full
```

运行前可追加 `--dry-run` 仅检查输入。汇总写入
`data/runs/trajectory_rescore_legacy.json`；若任一 run 缺少
`guideline_trajectory.json` 或 `final_answers.json`，脚本会列出该 run 并以非零状态退出。

### 重新调用 LLM Judge 完整重评

`rejudge_llm_batch.py` 会重新读取每个病例的原始 rubric、最终答案、运行轨迹、
工具调用和 evidence board，真实调用 `MEDCLAW_JUDGE_PROVIDER` 指定的 Judge。
每例会重新生成 Micro，同时调用当前 trajectory scorer 更新 Macro 和 50/50 综合分。
旧评测文件默认备份到 `<runs-root>/.llm_rejudge_backups/<UTC时间>/`；脚本每完成
一例就更新汇总，因此中断后可用 `--resume` 继续。

```bash
python scripts/rejudge_llm_batch.py \
  --runs-root runs \
  --cases-root /data4/share/cpgtrajbench/LUNG \
  --cases-root /data4/share/cpgtrajbench/UCEC \
  --run-id planner_v2_lung_daa \
  --run-id medclaw_only_lung \
  --run-id planner_v2_ucec_daa \
  --run-id medclaw_only_ucec_full \
  --pair planner_v2_lung_daa medclaw_only_lung \
  --pair planner_v2_ucec_daa medclaw_only_ucec_full \
  --judge-provider local_openai
```

先追加 `--dry-run` 可只检查 case、rubric 和运行目录，不调用模型或覆盖文件。
只重评个别病例时重复传入 `--case-id`。
