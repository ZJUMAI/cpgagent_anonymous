# Guideline Planner V2：正式训练与运行

本版本把 Memory Encoder、Memory Store、Planner Decoder、Router/DAA 和运行时 release 完全拆开。正式数据位于 `datasets/planner_v2_pilot`，420/420 条轨迹均已标记为 `approved`。训练与运行入口不再接受模板 PLAN 数据、V1 八字段输出、旧 Planner activation 标签或隐式共用的单一 adapter。

## 当前发布范围

- Planner action：`lung/nsclc`（`NSCLC_2010@2010`）和 `endometrial/ucec`（`CSCO子宫内膜癌2023@2023`）。
- Memory Encoder：NSCLC、SCLC、子宫内膜癌、鼻咽癌四份指南，共 79 个 chunk。
- SCLC 和鼻咽癌只进入记忆训练；当前 release 调用 Planner action 会明确失败。
- 默认基础模型为 `Qwen/Qwen3.5-9B`。`preflight` 会解析并锁定精确 Hugging Face snapshot commit。
- 正式配置见 `guideline_planner/configs/planner_v2_lung_endometrial.yaml`，默认 seed 17、top-K 4、memory tokens 64、BF16、单卡 A100/H100 80GB。

`release_scope.json` 区分 schema、Memory 和 Planner action 三种范围。数据集、模型 snapshot、Encoder、Store、Bridge、Decoder、Router 和 tokenizer 均由独立 hash 绑定；加载时任一不匹配都会失败。

## 环境

在训练机创建 Python 3.10+ 环境并安装：

```bash
python -m pip install -e ".[planner,dev]"
```

如果使用 DeepSpeed，再安装：

```bash
python -m pip install -e ".[planner-deepspeed]"
```

训练脚本默认调用 `python`，可用 `PYTHON_BIN` 指定解释器。模型下载需要 Hugging Face 网络和相应缓存权限；若模型要求认证，先设置 `HF_TOKEN`。API key 不写入仓库。

## 数据准入与人工审核

当前正式状态应满足：420 条 schema 有效、420 条 approved、grounding error 为 0、`ready=true`，且 train/validation/test 为 288/66/66。只读检查：

```bash
python -m guideline_planner.cli validate-planner-data-v2 \
  --input datasets/planner_v2_pilot/dataset
```

以后人工审核完成后，可重复执行严格同步。该命令会拒绝未知 ID、重复冲突决定和不完整同步：

```bash
python -m guideline_planner.cli approve-planner-reviews \
  --dataset-root datasets/planner_v2_pilot \
  --reviewer-id manual-review-team
```

正式训练只读取通过 admission 的 split。Router 构建会生成独立的 train/validation/test 文件，但训练器只加载 train 与 validation，绝不读取 test。

## 完整训练流程

统一入口是 `scripts/planner_v2_pipeline.sh`。每个阶段可单独执行；训练阶段遇到未完成输出时自动从最新 checkpoint 恢复，完整 artifact 不会被意外覆盖。

```bash
bash scripts/planner_v2_pipeline.sh preflight
bash scripts/planner_v2_pipeline.sh smoke
bash scripts/planner_v2_pipeline.sh train-memory
bash scripts/planner_v2_pipeline.sh export-memory
bash scripts/planner_v2_pipeline.sh overfit-decoder
bash scripts/planner_v2_pipeline.sh train-decoder
bash scripts/planner_v2_pipeline.sh build-routing
bash scripts/planner_v2_pipeline.sh train-router
bash scripts/planner_v2_pipeline.sh evaluate
bash scripts/planner_v2_pipeline.sh package-release
```

阶段含义：

1. `preflight`：检查 80GB CUDA、依赖、79 个 chunk、正式数据准入，并锁定模型 commit。
2. `smoke`：各阶段 10 step GPU 小规模串联测试。
3. `train-memory`：仅以 AE:RETRIEVE:CONTINUE=`5:5:3` 训练 Memory Encoder，5000 steps，LR `2e-4`。
4. `export-memory`：冻结 Encoder，重新导出 79 个 chunk 的 Memory Store。
5. `overfit-decoder`：默认选 8 条 train trajectory 重复训练 200 steps，并对真实 greedy generation 执行 schema gate；8/8 全部通过后才成功退出。
6. `train-decoder`：冻结 Encoder/Store，训练独立 Decoder LoRA、bridge 和辅助 heads，5000 steps，LR `2e-4`。
7. `build-routing`：由人工审核 action provenance 构造 Router label；不构造 transition graph，也不使用模型自己的历史 activation。
8. `train-router`：100-step retrieval pretrain → 100-step K=1 identity warm-up → variable-K DAA → 最后 10% 以 0.1 倍 LR 联合校准 Decoder，合计 5000 steps。
9. `evaluate`：在不可变 test split 上配对评估 `latent_topk` 与 `daa_full`。
10. `package-release`：门槛通过后生成 hash-bound `planner_release.v2`。DAA 未过门槛但基础 Planner 通过时，默认模式自动回退到 `latent_topk`。

Decoder 与 Router 的结构化训练上限默认为 2048 tokens。训练器先完整编码
`planner_action.v2` target，只允许压缩 prompt；target 无法完整装入时立即失败，
不再截断 JSON 后强制追加 EOS。每 500 steps 还会对 8 条 validation 状态执行
真实 greedy generation，`metrics.jsonl` 会记录 `generation_schema_pass_rate` 和
失败时的原始文本。`latent_topk` 会对四份 slot-aligned memory 做归一化加权求和，
Decoder prefix 始终保持 64 slots，但四份 memory 都保留为 active provenance。

Decoder 和 Router Decoder 训练默认启用 non-reentrant gradient checkpointing。
Planner preference loss 采用低显存分段反向：正样本图保留期间仅运行一次无梯度
负样本参考前向，正样本 backward 后释放其图，再以相同 RNG 状态重算负样本并
backward；梯度与原 `softplus(L_pos-L_neg)` 保持一致，但不再同时驻留两条完整
2048-token autograd graph。真实 generation validation 结束后会执行 Python GC 和
`torch.cuda.empty_cache()`，避免 KV/cache 影响后续训练。

默认输出根目录：

```text
guideline_planner/outputs/planner_v2_lung_endometrial/
  model_snapshot.json
  memory_training_data/
  memory_encoder/
  memory_store/
  planner_decoder/
  routing_data/
  router_daa/
  evaluation/
  release/planner_release.json
```

可用环境变量覆盖根路径，例如：

```bash
PYTHON_BIN=/opt/conda/envs/planner/bin/python \
PLANNER_OUTPUT_ROOT=/data/planner_v2_run \
bash scripts/planner_v2_pipeline.sh preflight
```

checkpoint 包含 trainable weights、optimizer、global step、RNG、sampler 状态、配置 hash 与上游 lineage hash。默认每 500 step 保存、保留最近 2 个；重新执行同一阶段即可恢复。若要重训已完整阶段，请先把旧 artifact 移到新的备份目录，或改用新的 `PLANNER_OUTPUT_ROOT`，不要混写。

### 仅重训 Decoder 与 Router/DAA

如果 Encoder 与 Store 已经训练完成，可把新下游实验写入独立目录，同时显式复用
旧的上游 artifact：

```bash
OLD_ROOT=guideline_planner/outputs/planner_v2_lung_endometrial
export PLANNER_OUTPUT_ROOT=guideline_planner/outputs/planner_v2_lung_endometrial_2048
export PLANNER_MEMORY_ENCODER_DIR="${OLD_ROOT}/memory_encoder"
export PLANNER_MEMORY_STORE_DIR="${OLD_ROOT}/memory_store"

# 必须先通过短程生成 overfit；默认 200 steps、8 个样本、要求 8/8 schema 合法。
bash scripts/planner_v2_pipeline.sh overfit-decoder

bash scripts/planner_v2_pipeline.sh train-decoder
bash scripts/planner_v2_pipeline.sh build-routing
bash scripts/planner_v2_pipeline.sh train-router
bash scripts/planner_v2_pipeline.sh evaluate
bash scripts/planner_v2_pipeline.sh package-release
```

可用 `PLANNER_OVERFIT_STEPS=100..300` 和 `PLANNER_OVERFIT_EXAMPLES` 调整短程
实验。不要让新的 Decoder/Router 恢复旧的 768-token checkpoint；新的
`PLANNER_OUTPUT_ROOT` 会从干净 Decoder LoRA 开始，同时保持 Encoder/Store 不变。

## Loss 与采样

Memory task batch schedule 按最大余数法精确实现 `5:5:3`，按 seed 打乱并按 rank 分片。Planner 先平衡癌种，再平衡 phase 和 action bucket；验证和测试不重采样。

Planner Decoder 默认：

```text
1.00 structured SFT
+ 0.30 action preference
+ 0.20 state progress
+ 0.10 phase transition
+ 0.20 memory provenance
```

Router/DAA 默认：

```text
1.00 action
+ 0.30 retrieval contrastive
+ 0.20 provenance
+ 0.10 gate margin
+ 0.01 sparse
```

normalized entropy 只作诊断，不施加统一降熵 loss。运行时 phase transition 仍由确定性状态机验证，Decoder 只能提出 `proposed_phase`。

## 固定集评估与发布门槛

`evaluate` 记录结构化 action、全部 active memories、gate diagnostics、validator 错误和配对分数。主要门槛包括 Action Hit@1 ≥ 0.85、unsafe/premature ≤ 1%、state progress ≥ 80%、phase F1 ≥ 0.85、schema/provenance 合法率 100%，以及 Router Recall@4 ≥ 0.85、positive mass ≥ 0.60、margin ≥ 0.05。DAA 相对 latent_topk 的配对 bootstrap 95% CI 下界必须 ≥ -0.02，才会成为默认模式。

已打包 release 可再次运行固定集：

```bash
PLANNER_RELEASE_DIR=guideline_planner/outputs/planner_v2_lung_endometrial/release \
bash scripts/test_planner_medclaw.sh planner-fixed
```

## Planner + MedClaw

运行时优先且默认只传 release 目录；`--planner-memory-dir`、`--planner-decoder-artifact-dir` 等旧参数仅用于显式调试覆盖。单病例默认使用未进入训练轨迹的 `TCGA-38-4626`：

```bash
bash scripts/test_planner_medclaw.sh single
bash scripts/test_planner_medclaw.sh batch-smoke
```

所有 skill 都使用脚本选定的当前 Python；batch smoke 只跑 2 例。GPU 环境验证：

```bash
PLANNER_DEVICE=cuda \
bash scripts/test_planner_medclaw.sh single
```

本地 OpenAI-compatible provider 默认读取项目已有的 `local_openai` 配置，可用相应的 MedClaw 环境变量覆盖 API 地址、key 和模型。Planner+MedClaw 运行时只解析 JSON 和八个必需顶层字段，不执行 `planner_action.v2` validator；phase、precondition、version、provenance 等 validator 指标仅在离线数据审计和评估中记录，不会中止病例运行。脚本检查 `planner_outputs.jsonl`、`patient_state_history.jsonl`、`routing_metrics.json`、离线 validator 结果及最终答案；batch smoke 成功后再扩大批量。

## 测试与打包

不依赖 GPU 的核心测试：

```bash
python -m pytest \
  tests/test_guideline_planner.py \
  tests/test_guideline_planner_v2.py \
  tests/test_guideline_routing_core.py \
  tests/test_planner_review_scope.py \
  tests/test_planner_release.py
```

安装 Torch 后再执行 checkpoint reload、CPU micro-overfit、variable-K DAA 和真实 reload 集成测试：

```bash
python -m pytest tests/test_planner_workflow_v2.py tests/test_guideline_routing_torch.py
```

构建并检查 wheel：

```bash
python -m build
python -m zipfile -l dist/*.whl
```

wheel 必须包含 `medclaw/knowledge/guidelines` 下四份根目录指南。正式大规模训练应只在 `preflight` 与 `smoke` 全部成功后启动。
