from __future__ import annotations

import json
from pathlib import Path

import pytest

from guideline_planner.cli import main as planner_cli_main
from guideline_planner.io_utils import read_jsonl, write_jsonl
from guideline_planner.routing_config import LatentGuidelineRoutingConfig
from guideline_planner.routing_dataset import build_routing_training_data
from guideline_planner.schemas_v2 import V2SchemaError
from guideline_planner.routing_metrics import (
    evaluate_routing_runs,
    trajectory_quality_details,
)


def test_routing_config_is_opt_in_and_profiles_control_ablation() -> None:
    assert LatentGuidelineRoutingConfig.from_value(None).enabled is False

    config = LatentGuidelineRoutingConfig.from_value(
        {
            "enabled": True,
            "profile": "dynamic_anchor",
            "retrieval": {"seed_top_k": 4},
            "gate": {"active_top_k": 2},
        }
    )
    assert config.enabled is True
    assert config.retrieval.seed_top_k == 4
    assert config.gate.active_top_k == 2

    with pytest.raises(ValueError, match="pathway routing was removed"):
        LatentGuidelineRoutingConfig.from_value(
            {"enabled": True, "pathway": {"enabled": True}}
        )

    top1 = config.apply_profile("latent_top1")
    assert top1.gate.enabled is False
    assert top1.gate.active_top_k == 1
    assert top1.memory_fusion.strategy == "top1_only"

    dynamic_anchor = config.apply_profile("dynamic_anchor")
    assert dynamic_anchor.gate.enabled is True
    assert dynamic_anchor.memory_fusion.strategy == "dynamic_anchor_attention"


def test_official_routing_config_trains_fixed_length_daa_and_decoder_lora() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "guideline_planner"
        / "configs"
        / "latent_guideline_routing.yaml"
    )
    config = LatentGuidelineRoutingConfig.from_value(config_path)

    assert config.gate.active_top_k == 4
    assert config.memory_fusion.strategy == "dynamic_anchor_attention"
    assert config.memory_fusion.dynamic_anchor_attention.max_num_memories == 4
    assert config.memory_fusion.dynamic_anchor_attention.num_heads == 8
    assert config.training.train_decoder_lora is True
    assert config.training.retrieval_pretrain_steps == 100
    assert config.training.identity_warmup_steps == 100
    assert config.training.lambda_retrieval == 0.3
    assert config.training.lambda_provenance == 0.2
    assert config.training.lambda_gate_margin == 0.1


def test_routing_config_rejects_stage_or_biomarker_filters() -> None:
    with pytest.raises(ValueError, match="Unknown keys"):
        LatentGuidelineRoutingConfig.from_value(
            {
                "enabled": True,
                "retrieval": {"filter_stage": True},
            }
        )


@pytest.mark.parametrize(
    ("profile", "enabled", "gate", "strategy"),
    [
        ("text_rag", False, True, "dynamic_anchor_attention"),
        ("latent_top1", True, False, "top1_only"),
        ("latent_topk_concatenation", True, False, "weighted_concatenation"),
        ("soft_memory_mixture", True, True, "dynamic_anchor_attention"),
        ("dynamic_anchor", True, True, "dynamic_anchor_attention"),
    ],
)
def test_all_routing_ablation_profiles(
    profile: str,
    enabled: bool,
    gate: bool,
    strategy: str,
) -> None:
    config = LatentGuidelineRoutingConfig(enabled=True).apply_profile(profile)
    assert config.enabled is enabled
    assert config.gate.enabled is gate
    assert config.memory_fusion.strategy == strategy


def test_legacy_pathway_profile_maps_to_current_state_daa_with_warning() -> None:
    with pytest.warns(DeprecationWarning, match="pathway routing was removed"):
        config = LatentGuidelineRoutingConfig.from_value(
            {
                "enabled": True,
                "profile": "soft_expert_pathway",
            }
        )
    assert config.profile == "dynamic_anchor"
    assert config.memory_fusion.strategy == "dynamic_anchor_attention"


def test_build_routing_data_rejects_legacy_plan_retrieval_bootstrap(
    tmp_path: Path,
) -> None:
    planner_data = tmp_path / "mixed_train.jsonl"
    write_jsonl(
        planner_data,
        [
            {
                "task": "RETRIEVE",
                "guideline_memory_id": "m1",
                "query": "NSCLC diagnostic workup",
                "positive_slot_ids": ["m1", "m2"],
                "negative_slot_ids": ["m3"],
                "strong_positive": ["m1"],
                "weak_positive": ["m2"],
                "hard_negative": ["m3"],
                "easy_negative": ["m4"],
            },
            {
                "task": "PLAN",
                "guideline_memory_id": "m1",
                "patient_state": {
                    "case_id": "case-1",
                    "cancer_type": "NSCLC",
                    "current_phase": "diagnosis",
                },
                "prompt": "patient state",
                "target": json.dumps(
                    {
                        "current_phase": "diagnosis",
                        "missing_information": ["pathology"],
                        "next_step": "collect pathology",
                        "required_skills": ["pathology.read_report"],
                        "guideline_memory_id": "m1",
                        "supporting_rule_ids": ["r1"],
                        "blocked_pathways": [],
                        "reason": "missing evidence",
                    }
                ),
            },
        ],
    )

    with pytest.raises(V2SchemaError, match="planner_trajectory.v2"):
        build_routing_training_data(planner_data, tmp_path / "routing_train.jsonl")


def test_build_routing_data_cli_rejects_non_v2_bootstrap(
    tmp_path: Path,
) -> None:
    planner_data = tmp_path / "mixed_train.jsonl"
    output = tmp_path / "routing_train.jsonl"
    write_jsonl(
        planner_data,
        [
            {
                "task": "PLAN",
                "guideline_memory_id": "m1",
                "patient_state": {"current_phase": "diagnosis"},
                "prompt": "patient state",
                "target": "{}",
            }
        ],
    )

    with pytest.raises(V2SchemaError, match="planner_trajectory.v2"):
        planner_cli_main(
            [
                "build-routing-data",
                "--trajectory-data",
                str(planner_data),
                "--output",
                str(output),
            ]
        )


def test_quality_renormalizes_available_components_without_deductions() -> None:
    details = trajectory_quality_details(
        {"next_step": "collect evidence"},
        {"required_score": 1.0, "unsafe_score": 0.0},
        None,
        {"known_stage": None},
        {"known_stage": "IIA"},
    )

    assert details["quality"] == pytest.approx(1.0)
    assert details["components"]["acceptable"] is None
    assert details["unsafe_known"] is True


def test_routing_evaluation_reports_metric_coverage(tmp_path: Path) -> None:
    run_dir = tmp_path / "case" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "memory_routing.json").write_text(
        json.dumps(
            {
                "case_id": "case",
                "run_id": "run",
                "steps": [
                    {
                        "active_memories": [
                            {"memory_id": "m1", "weight": 0.6},
                            {"memory_id": "m2", "weight": 0.4},
                        ],
                        "gate_entropy": 0.67,
                        "unsafe_score": 0.0,
                    },
                    {
                        "active_memories": [
                            {"memory_id": "m3", "weight": 0.7},
                            {"memory_id": "m4", "weight": 0.3},
                        ],
                        "gate_entropy": 0.67,
                        "unsafe_score": 0.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_routing_runs([run_dir])

    assert result["run_count"] == 1
    assert result["aggregate"]["average_active_memory_count"] == 2
    assert result["aggregate"]["gate_entropy"] == pytest.approx(0.67)
    assert result["aggregate"]["action_score_coverage"] == 0.0
