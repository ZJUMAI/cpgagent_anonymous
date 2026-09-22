# Planner V2 A0-A6 inference ablations

These experiments reuse the hash-bound three-cancer `daa_full` release. A1-A6
use the A5 runtime decoder; no training command is run. A0 and A5 are read-only
references. Lung, UCEC, and NPC are supported by the same entrypoint.

| ID | Runtime behavior | Run policy |
|---|---|---|
| A0 | MedClaw without Planner | reuse only |
| A1 | zero 64-slot prefix, no active memory | run |
| A2 | Encoder latent Top-4, fixed-length slot fusion | run |
| A3 | learned Router Top-4, fixed-length slot fusion, no DAA | run |
| A4 | learned Router Top-4, uniform gate weights into DAA | run |
| A5 | full Router + DAA | reuse only |
| A6 | deterministic global random Top-4, uniform weights into DAA | run |

A6 derives its random seed from `SHA256(seed, case_id, current_time)` and
excludes the normal A5 active memories before sampling from the entire store.

## Commands

```bash
cd /data2/zhenglujie/cpg_trajbench
conda activate cpg

export MEDCLAW_LOCAL_BASE_URL=http://127.0.0.1:8087/v1
export MEDCLAW_LOCAL_API_KEY=local-qwen-key
export MEDCLAW_LOCAL_MODEL=qwen3.6-27b
export MEDCLAW_JUDGE_PROVIDER=local_openai
export PLANNER_RELEASE_DIR=/data2/zhenglujie/cpg_trajbench/guideline_planner/outputs/planner_v2_lung_endometrial_npc/release
export RUNS_ROOT=/data2/zhenglujie/cpg_trajbench/runs
export GPU_ID=3

# Evaluation is disabled by default. Set this to 1 only for the legacy
# Qwen/LLM Judge; Codex evaluation remains a separate post-processing step.
export ENABLE_EVALUATION=0

bash scripts/run_planner_v2_ablations.sh status all
DRY_RUN=1 bash scripts/run_planner_v2_ablations.sh run-all all
bash scripts/run_planner_v2_ablations.sh run-all all
```

To run or backfill the legacy Judge without rerunning completed inference:

```bash
ENABLE_EVALUATION=1 bash scripts/run_planner_v2_ablations.sh run-all all
```

Resume one experiment with:

```bash
bash scripts/run_planner_v2_ablations.sh run A3 all
```

`run A0` and `run A5` only print existing-result coverage. `FORCE=1` does not
change that protection. Other runs retain the normal per-case resume behavior.

The three-cancer run IDs are:

```text
A1 planner_v2_three_cancer_ablation_a1_no_memory_{lung|ucec|npc}
A2 planner_v2_three_cancer_ablation_a2_latent_topk_runtime_decoder_{lung|ucec|npc}
A3 planner_v2_three_cancer_ablation_a3_router_topk_no_daa_{lung|ucec|npc}
A4 planner_v2_three_cancer_ablation_a4_daa_uniform_gate_{lung|ucec|npc}
A5 planner_v2_three_cancer_{lung|ucec|npc}_daa
A6 planner_v2_three_cancer_ablation_a6_random_memory_global_{lung|ucec|npc}
```

A0 reuses `medclaw_only_lung`, `medclaw_only_ucec_full`, and
`medclaw_only_npc`; it is intentionally not copied into new directories.
