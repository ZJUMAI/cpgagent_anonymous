from __future__ import annotations

from pathlib import Path

import pytest

from guideline_planner.planner import (
    PLANNER_ABLATION_IDS,
    PLANNER_ABLATION_MODES,
    normalize_planner_ablation,
)
from medclaw_benchmark.dual_agent_runner import DualAgentBatchConfig


def test_ablation_names_and_ids_are_stable() -> None:
    assert PLANNER_ABLATION_MODES == (
        "none",
        "no_memory",
        "latent_topk",
        "router_topk_no_daa",
        "daa_uniform_gate",
        "random_memory_global",
    )
    assert PLANNER_ABLATION_IDS == {
        "none": "A5",
        "no_memory": "A1",
        "latent_topk": "A2",
        "router_topk_no_daa": "A3",
        "daa_uniform_gate": "A4",
        "random_memory_global": "A6",
    }
    assert normalize_planner_ablation("DAA-UNIFORM-GATE") == "daa_uniform_gate"
    with pytest.raises(ValueError, match="Unknown Planner ablation"):
        normalize_planner_ablation("transition_graph")


def test_batch_config_carries_ablation_seed() -> None:
    config = DualAgentBatchConfig(
        cases_root=Path("cases"),
        runs_root=Path("runs"),
        planner_mode="daa_full",
        planner_ablation="random_memory_global",
        planner_ablation_seed=29,
    )
    assert config.planner_ablation == "random_memory_global"
    assert config.planner_ablation_seed == 29
    assert config.evaluate is False


def test_shell_entrypoint_hard_codes_a0_a5_as_reuse_only() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_planner_v2_ablations.sh").read_text(
        encoding="utf-8"
    )
    assert '"${experiment}" == "A0" || "${experiment}" == "A5"' in script
    assert "run-all executes A1, A2, A3, A4, and A6 only" in script
    assert "medclaw_only_lung" in script
    assert "medclaw_only_ucec_full" in script
    assert "medclaw_only_npc" in script
    assert "planner_v2_three_cancer_%s_daa" in script
    assert "planner_v2_three_cancer_ablation_a1_no_memory_%s" in script
    assert "planner_v2_three_cancer_ablation_a6_random_memory_global_%s" in script
    assert "planner_v2_lung_endometrial_npc/release" in script
    assert 'NPC_CASES_ROOT="${NPC_CASES_ROOT:-/data4/share/cpgtrajbench/NPC}"' in script
    assert "lung|ucec|npc|all" in script
    assert 'npc) "${function_name}" "${experiment}" npc ;;' in script
    assert '"${function_name}" "${experiment}" npc' in script
    assert 'ENABLE_EVALUATION="${ENABLE_EVALUATION:-0}"' in script
    assert 'STATUS_REQUIRE_EVALUATION="${EVALUATION_ENABLED}"' in script


def test_regular_shell_entrypoints_disable_evaluation_by_default() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "scripts/run_benchmark_batch.sh",
        "scripts/run_dual_agent_benchmark_batch.sh",
        "scripts/run_planner_v2_ablations.sh",
        "scripts/test_planner_medclaw.sh",
    ):
        script = (root / relative).read_text(encoding="utf-8")
        assert 'ENABLE_EVALUATION="${ENABLE_EVALUATION:-0}"' in script
    for relative in (
        "scripts/run_benchmark_batch.sh",
        "scripts/run_dual_agent_benchmark_batch.sh",
    ):
        script = (root / relative).read_text(encoding="utf-8")
        assert 'ARGS+=(--evaluate --judge-provider "${JUDGE_PROVIDER}")' in script
