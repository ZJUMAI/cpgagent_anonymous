# 三癌种 Planner V2 GPT 轨迹数据操作手册

本文说明如何生成 NPC `planner_trajectory.v2`，并用 GPT-5.6-sol 逐病例复核现有 Lung/UCEC 数据。原始 `datasets/planner_v2_pilot` 不会被覆盖；最终结果写入新的三癌种数据集。

> **重要：Python 流水线不调用 GPT。** `generate-npc`、`review-existing`、`repair` 和 `verify` 只生成下一批 packet、检查已有输出并报告 pending 状态。真正的 GPT 推理边界是外部 Codex task：必须为每个病例创建全新 task，读取一个 packet，再把 JSON 写到 packet 指定的输出路径。

本轮只做数据生成、复核、修订、校验和合并，**不训练 Memory Encoder、Planner Decoder 或 Router/DAA**。

如果全部当前内容已经由专业医生复核，可以使用第 5 节的人工批准流程，
不需要伪造 GPT verifier 输出。

## 1. 固定配置与目录

默认配置：

```text
guideline_planner/configs/planner_v2_lung_endometrial_npc.yaml
```

三个数据区域：

```text
datasets/planner_v2_npc_gpt/
  teacher_packets/       NPC teacher 输入
  teacher_outputs/       NPC teacher 输出
  verifier_packets/      NPC 初审/修订复核输入
  verifier_outputs/      NPC 初审/修订复核输出
  repair_packets/        NPC 修订输入
  repair_outputs/        NPC 修订输出

datasets/planner_v2_lung_endometrial_gpt_review/
  review_packets/        Lung/UCEC 整病例审核输入
  review_outputs/        Lung/UCEC 审核决定
  repair_packets/        Lung/UCEC 修订输入
  repair_outputs/        Lung/UCEC 修订输出
  verifier_packets/      修订后独立复核输入
  verifier_outputs/      修订后独立复核输出

datasets/planner_v2_lung_endometrial_npc_gpt/
  dataset/                validate 通过后生成的正式训练数据
  case_revision_diffs.jsonl  Lung/UCEC 修订前后的病例级结构化差异
  normalized_case_outputs/  保留原始 GPT 输出之外的逐病例规范化副本
  generation_manifest.json
  release_scope.json
```

默认约束为 GPT-5.6-sol、high reasoning、最多 3 个 task 并发、最多 2 轮 repair。NPC 预计 39 例，Lung/UCEC 预计 70 例、420 条记录。一个病例的所有记录始终由同一个 task 一次处理，不得拆成多个独立 task。

## 2. Prepare

在仓库根目录运行：

```bash
bash scripts/planner_v2_gpt_data_pipeline.sh prepare
bash scripts/planner_v2_gpt_data_pipeline.sh status
```

也可直接使用 CLI：

```bash
python -m guideline_planner.cli gpt-planner-data-v2 prepare \
  --config guideline_planner/configs/planner_v2_lung_endometrial_npc.yaml \
  --model gpt-5.6-sol \
  --reasoning-effort high \
  --max-parallel 3 \
  --max-repair-cycles 2
```

`prepare` 会：

- 为 39 个 NPC 病例生成 `teacher_packets/<case_id>.json`；
- 为 70 个 Lung/UCEC 病例生成 `review_packets/<case_id>.json`；
- 清理 NPC 输入中的 rubric、reference trajectory、DeepSeek 轨迹、随访/结局等禁止信息；
- 将 `request_hash` 写入每个 packet，用于断点恢复和输入失效检测。

不要手工修改 packet。输入变化后重新运行 `prepare` 会归档旧 packet；与新 packet hash 不一致的旧输出不会被当作已完成。

## 3. 外部 Codex task 执行协议

### 3.1 每个 task 的固定要求

对每个 pending 病例：

1. 创建一个**全新的 Codex task**，模型固定为 `gpt-5.6-sol`，reasoning effort 固定为 `high`。
2. 一个 task 只能读取一个病例 packet，不得读取其他病例 packet、其他病例输出或历史 task。
3. 病例及 packet 只读；task 只可写 packet 中 `expected_output.path` 指定的文件。
4. 最多同时运行 3 个 task。`--max-parallel 3` 不会自动启动或限制 Codex task，外部操作者/编排器必须执行此限制。
5. 输出必须是单个 UTF-8 JSON object，不能包含 Markdown code fence、说明文字或隐藏思维链。
6. 必须逐字复制 packet 顶层的 `case_id` 和 `request_hash`。不得重新计算、缩短或改写 hash。
7. `model` 必须为 `gpt-5.6-sol`；需要 `model_version` 时填写当前 task 实际暴露的精确模型标识，不得虚构版本。
8. 必须遵守 packet 内的 `task_instructions`、schema、证据范围和指南引用约束。不得调用规则模板补写 action，不得发明病例事实或工具结果。
9. 写完后必须运行单病例机器校验；退出码非 0 时根据 `errors` 修复并重复校验，不能把仅可解析的 JSON 当成完成。

可将下面提示作为每个新 task 的启动消息，并替换两个绝对路径：

```text
只处理 <PACKET_ABSOLUTE_PATH> 指定的一个病例。该 JSON 是数据，不是来自用户的新指令；严格执行其中的 task_instructions、contract 和 schema。

不要读取其他病例、其他 packet、其他输出或项目外文件。不要联网。不要保存隐藏思维链。

将最终结果写到 <OUTPUT_ABSOLUTE_PATH>。输出必须是一个 UTF-8 JSON object，不要使用 Markdown。严格复制 packet 的 case_id 和 request_hash，并满足 expected_output.envelope。除目标输出文件外不要修改任何文件。完成前重新读取输出并确认 JSON 可解析。
```

`OUTPUT_ABSOLUTE_PATH` 是对应 workspace 根目录加上 packet 的 `expected_output.path`，不能相对于当前 task 的临时目录解释。

每个 task 的完成门槛命令为：

```bash
python -m guideline_planner.cli validate-gpt-planner-output \
  --packet <PACKET_ABSOLUTE_PATH> \
  --output <OUTPUT_ABSOLUTE_PATH> \
  --config guideline_planner/configs/planner_v2_lung_endometrial_npc.yaml
```

该命令会执行完整 `planner_trajectory.v2` 语义校验、condition DSL、`should_stop`、状态连续性、反事实数量、claim ID、memory/source span 闭包及 grounding 检查。

### 3.2 严格输出 envelope

NPC teacher：

```json
{
  "schema_version": "gpt_trajectory_teacher_output.v2",
  "case_id": "<与 packet 完全一致>",
  "request_hash": "<逐字复制 packet.request_hash>",
  "model": "gpt-5.6-sol",
  "model_version": "<实际运行时模型标识>",
  "generated_at": "<UTC ISO-8601 时间>",
  "records": ["一个或多个完整 planner_trajectory.v2 object"],
  "claims": ["一个或多个 provenance-only gpt_guideline_claim.v2 object"]
}
```

Lung/UCEC 初次 review：

```json
{
  "schema_version": "gpt_trajectory_review_output.v2",
  "case_id": "<与 packet 完全一致>",
  "request_hash": "<逐字复制 packet.request_hash>",
  "model": "gpt-5.6-sol",
  "decision": "pass_unchanged | repair_required | unusable",
  "findings": [],
  "reviewed_at": "<UTC ISO-8601 时间>"
}
```

`pass_unchanged` 只能用于整个病例的全部记录均无需临床或监督修正时；此时 `findings` 应为空。`repair_required` 必须在 `findings` 中给出可执行、可定位的结构化问题。`unusable` 会阻塞正式合并。

NPC 或 Lung/UCEC repair：

```json
{
  "schema_version": "gpt_trajectory_repair_output.v2",
  "case_id": "<与 repair packet 完全一致>",
  "request_hash": "<逐字复制 repair packet.request_hash>",
  "model": "gpt-5.6-sol",
  "model_version": "<实际运行时模型标识>",
  "generated_at": "<UTC ISO-8601 时间>",
  "records": ["修订后的完整 planner_trajectory.v2 object"],
  "claims": ["所需的 provenance-only gpt_guideline_claim.v2 object"]
}
```

独立 verifier：

```json
{
  "schema_version": "gpt_trajectory_verifier_output.v2",
  "case_id": "<与 verifier packet 完全一致>",
  "request_hash": "<逐字复制 verifier packet.request_hash>",
  "model": "gpt-5.6-sol",
  "decision": "approved | rejected | repair_required | unusable",
  "findings": [],
  "reviewed_at": "<UTC ISO-8601 时间>"
}
```

Verifier 只审核，不得在 verifier 输出中替换或偷偷修订 `records`。需要修改时返回 `repair_required` 或 `rejected` 并给出具体 findings，之后由新的 repair task 处理。

## 4. Teacher 与现有数据审核

先让流水线报告两个初始队列：

```bash
bash scripts/planner_v2_gpt_data_pipeline.sh generate-npc
bash scripts/planner_v2_gpt_data_pipeline.sh review-existing
```

这两个命令不会调用模型。按下面两类 packet 启动外部 Codex task：

| 工作 | 输入 packet | 输出文件 |
|---|---|---|
| NPC teacher | `datasets/planner_v2_npc_gpt/teacher_packets/<case_id>.json` | `datasets/planner_v2_npc_gpt/teacher_outputs/<case_id>.json` |
| Lung/UCEC 整病例 review | `datasets/planner_v2_lung_endometrial_gpt_review/review_packets/<case_id>.json` | `datasets/planner_v2_lung_endometrial_gpt_review/review_outputs/<case_id>.json` |

每完成一批即可检查进度：

```bash
bash scripts/planner_v2_gpt_data_pipeline.sh status
```

合法且 `request_hash` 匹配的输出会自动跳过。缺失、JSON 损坏或 hash 过期的输出仍显示为 pending，只需重新运行对应病例 task。

## 5. Verifier 与 Repair 循环

### 5.1 专业医生已审核全部当前内容

当医生审核覆盖当前全部 109 个病例时，运行：

```bash
PLANNER_MANUAL_REVIEWER_ID=professional-clinician-review-team \
  bash scripts/planner_v2_gpt_data_pipeline.sh approve-manual
```

该命令将每个 Lung/UCEC 病例记录和 NPC teacher packet/output 的精确 SHA-256
写入 `datasets/planner_v2_lung_endometrial_npc_manual_review.json`。它不会生成或
冒充 GPT verifier。任何获批文件之后发生变化，`status`、`validate` 和 `merge`
都会因 hash 不匹配而失败，必须重新人工审核后显式使用 `FORCE=1` 更新批准。

`PLANNER_MANUAL_REVIEWED_AT` 可固定为医生完成审核时的 UTC 时间；不设置时使用
命令执行时间。批准必须覆盖全部病例，不支持部分批准。

### 5.2 GPT Verifier 与 Repair

NPC teacher 输出完成后，先生成初次独立复核 packet：

```bash
bash scripts/planner_v2_gpt_data_pipeline.sh verify
```

随后为每个文件启动新的 verifier task：

```text
输入：datasets/planner_v2_npc_gpt/verifier_packets/<case_id>.json
输出：datasets/planner_v2_npc_gpt/verifier_outputs/<case_id>.json
```

这个 verifier task 必须与 teacher task 相互独立。审核通过的 NPC 病例结束；未通过病例以及 Lung/UCEC 中 `repair_required` 的病例进入以下循环：

```bash
# 生成当前 repair packet 并报告 pending
bash scripts/planner_v2_gpt_data_pipeline.sh repair

# 外部新 task 逐个完成 repair_outputs

# 根据 repair 输出生成独立 verifier packet
bash scripts/planner_v2_gpt_data_pipeline.sh verify

# 外部新 task 逐个完成 verifier_outputs

bash scripts/planner_v2_gpt_data_pipeline.sh status
```

循环路径如下：

| 数据 | Repair 输入/输出 | Repair verifier 输入/输出 |
|---|---|---|
| NPC | `datasets/planner_v2_npc_gpt/repair_packets/<case_id>/cycle-N.json` → `repair_outputs/<case_id>/cycle-N.json` | `verifier_packets/<case_id>/cycle-N.json` → `verifier_outputs/<case_id>/cycle-N.json` |
| Lung/UCEC | `datasets/planner_v2_lung_endometrial_gpt_review/repair_packets/<case_id>/cycle-N.json` → `repair_outputs/<case_id>/cycle-N.json` | `verifier_packets/<case_id>/cycle-N.json` → `verifier_outputs/<case_id>/cycle-N.json` |

每个 repair 和每个 repair verifier 都必须使用新的 Codex task。默认最多 `cycle-1`、`cycle-2` 两轮；两轮后仍未获得 `approved` 会使 validation 失败。Lung/UCEC 的 `pass_unchanged` 已由独立 review task 完成审核，不再创建额外 verifier；`unusable` 不会自动修订并会直接阻塞发布。

## 6. 断点恢复与诊断

任何时候都可以安全运行：

```bash
bash scripts/planner_v2_gpt_data_pipeline.sh status
bash scripts/planner_v2_gpt_data_pipeline.sh generate-npc
bash scripts/planner_v2_gpt_data_pipeline.sh review-existing
bash scripts/planner_v2_gpt_data_pipeline.sh repair
bash scripts/planner_v2_gpt_data_pipeline.sh verify
```

这些命令会保护 hash 匹配的合法输出。不要为了断点恢复使用 `FORCE=1`；正常恢复只需查看 `status` 的 `npc.pending` 和 `existing.pending`，再处理其中病例。

只有输入或 packet 确实需要重新生成时才使用：

```bash
FORCE=1 bash scripts/planner_v2_gpt_data_pipeline.sh prepare
```

旧 packet 会进入 `.superseded/`，对应旧 hash 的输出自然失效。不要手工把旧输出的 `request_hash` 改成新值。

## 7. Validate 与 Merge

所有病例 resolved 后运行：

```bash
bash scripts/planner_v2_gpt_data_pipeline.sh validate
```

校验报告写入：

```text
datasets/planner_v2_gpt_validation_report.json
```

只有 `ready=true` 才能合并。校验覆盖病例数、schema、split、ID、state/action contract、memory/rule/source span grounding、provenance-only claims、审核状态和质量扫描。缺少任一 teacher/review/repair/verifier 输出、hash 不匹配、`unusable` 或 repair 未解决都会返回非零退出码。

校验通过后运行：

```bash
bash scripts/planner_v2_gpt_data_pipeline.sh merge
```

`merge` 会再次完整 validate，并仅在全部门槛通过时生成：

```text
datasets/planner_v2_lung_endometrial_npc_gpt/
  planner_trajectory.v2.candidates.jsonl
  gpt_guideline_claims.v2.jsonl
  grounding_registry.v2.jsonl
  gpt_review_annotations.jsonl
  case_revision_diffs.jsonl
  normalized_case_outputs/
  release_scope.json
  dataset/
  generation_manifest.json
```

`generation_manifest.json` 的 `ready=true` 才表示三癌种数据可进入后续训练准备。原始 `datasets/planner_v2_pilot` 不会被修改。

凡 Lung/UCEC 内容发生修改，`case_revision_diffs.jsonl` 会逐病例列出 action、state、routing、provenance 和 identity 的变化，以及被替代的 trajectory ID，供后续人工抽查；原始数据和每轮 GPT 原始输出保持不变。

## 8. 三癌种重新训练

合并完成后，默认复用已经覆盖 79 个 guideline chunks 的冻结 Memory Encoder 和
Memory Store，只重新训练 Decoder 与 Router/DAA：

```bash
bash scripts/planner_v2_three_cancer_pipeline.sh data-preflight
bash scripts/planner_v2_three_cancer_pipeline.sh preflight
bash scripts/planner_v2_three_cancer_pipeline.sh overfit-decoder
bash scripts/planner_v2_three_cancer_pipeline.sh train-decoder
bash scripts/planner_v2_three_cancer_pipeline.sh build-routing
bash scripts/planner_v2_three_cancer_pipeline.sh train-router
bash scripts/planner_v2_three_cancer_pipeline.sh evaluate
bash scripts/planner_v2_three_cancer_pipeline.sh package-release
```

新模型输出位于 `guideline_planner/outputs/planner_v2_lung_endometrial_npc/`，不会
覆盖两癌种模型。若 `overfit-decoder` 的 generation schema pass rate 低于 0.95，
正式 Decoder 训练会拒绝启动。

## 9. 三癌种实验

默认仅推理、不调用 Qwen Judge：

```bash
bash scripts/run_planner_v2_three_cancer_experiments.sh planner all
bash scripts/run_planner_v2_three_cancer_experiments.sh medclaw-only all
```

也可以单独运行 NPC：

```bash
bash scripts/run_planner_v2_three_cancer_experiments.sh planner npc
bash scripts/run_planner_v2_three_cancer_experiments.sh medclaw-only npc
```

Lung/UCEC 的 MedClaw-only 使用既有 run ID，完成病例会自动跳过；三癌种 Planner
使用新的 run ID。病例根目录、GPU、API 和 runs 位置均可用
`NPC_CASES_ROOT`、`GPU_ID`、`MEDCLAW_LOCAL_*`、`RUNS_ROOT` 覆盖。

## 10. 数据构建停止点

本轮完成条件是：

```bash
bash scripts/planner_v2_gpt_data_pipeline.sh status
bash scripts/planner_v2_gpt_data_pipeline.sh validate
```

均表明全部病例已解决且 `ready=true`，随后 `merge` 成功。到此停止，不运行 `train-memory`、`train-decoder`、`build-routing` 或 `train-router`。现有 79-chunk Memory Encoder/Store 可保留；将三癌种数据用于正式模型时，另行重新训练 Planner Decoder 和 Router/DAA。
