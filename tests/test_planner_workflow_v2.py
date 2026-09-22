from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
torch = pytest.importorskip("torch", reason="install the planner extra to run Torch tests")

import guideline_planner.cli as planner_cli
from guideline_planner.fixed_prediction import predict_planner_v2_fixed_test
from guideline_planner.io_utils import read_jsonl
from guideline_planner.planner_decoder_training import _linear_with_module_dtype
from guideline_planner.progress import ProgressReporter
from guideline_planner.routing_dataset import build_routing_training_data
from guideline_planner.routing_training import _load_routing_splits
from guideline_planner.routing_types import MemoryActivation, PlannerStepResult
from guideline_planner.training_checkpoint import (
    load_latest_training_checkpoint,
    save_training_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_DATASET = PROJECT_ROOT / "datasets" / "planner_v2_pilot" / "dataset"


def test_progress_reporter_shows_metrics_elapsed_and_eta() -> None:
    stream = io.StringIO()
    now = [0.0]
    progress = ProgressReporter(
        "unit-stage",
        10,
        unit="step",
        stream=stream,
        clock=lambda: now[0],
        min_interval=0.0,
    )

    progress.start(status="training")
    now[0] = 5.0
    progress.update(5, metrics={"loss": 1.25})
    now[0] = 10.0
    progress.finish(metrics={"loss": 0.5})

    output = stream.getvalue()
    assert "[unit-stage]" in output
    assert "5/10 step" in output
    assert "loss=1.25" in output
    assert "elapsed=00:05" in output
    assert "eta=00:05" in output
    assert "10/10 step" in output


def test_planner_auxiliary_head_accepts_bfloat16_hidden_state() -> None:
    head = torch.nn.Linear(8, 3)
    hidden = torch.randn(8, dtype=torch.bfloat16, requires_grad=True)

    output = _linear_with_module_dtype(head, hidden)

    assert output.dtype == head.weight.dtype == torch.float32
    output.sum().backward()
    assert hidden.grad is not None
    assert head.weight.grad is not None


def test_memory_train_cli_forwards_learning_rate(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_train(config):
        captured["config"] = config
        return {"is_main_process": True}

    monkeypatch.setattr(planner_cli, "train_planner", fake_train)
    exit_code = planner_cli.main(
        [
            "train",
            "--train-data",
            str(tmp_path / "train.jsonl"),
            "--output-dir",
            str(tmp_path / "output"),
            "--learning-rate",
            "2e-4",
        ]
    )

    assert exit_code == 0
    assert captured["config"].learning_rate == pytest.approx(2e-4)


def test_atomic_checkpoint_resume_is_lineage_bound_and_pruned(tmp_path: Path) -> None:
    for step in (1, 2, 3):
        save_training_checkpoint(
            torch_module=torch,
            output_dir=tmp_path,
            stage="unit",
            global_step=step,
            config_hash="config",
            lineage={"dataset": "hash"},
            state={"optimizer_state": {"step": step}},
            keep_last=2,
        )
    checkpoints = sorted((tmp_path / "checkpoints").glob("step-*"))
    assert [item.name for item in checkpoints] == ["step-00000002", "step-00000003"]
    resumed = load_latest_training_checkpoint(
        torch_module=torch,
        output_dir=tmp_path,
        stage="unit",
        config_hash="config",
        lineage={"dataset": "hash"},
    )
    assert resumed is not None
    assert resumed["global_step"] == 3
    assert resumed["optimizer_state"] == {"step": 3}
    with pytest.raises(RuntimeError, match="lineage mismatch"):
        load_latest_training_checkpoint(
            torch_module=torch,
            output_dir=tmp_path,
            stage="unit",
            config_hash="config",
            lineage={"dataset": "different"},
        )


def test_formal_routing_build_is_split_and_trainer_never_loads_test(tmp_path: Path) -> None:
    output = tmp_path / "routing"
    manifest = build_routing_training_data(PILOT_DATASET, output)
    assert manifest["source_dataset_hash"]
    assert manifest["approved_only"] is True
    assert manifest["nonrelease_smoke"] is False
    assert set(manifest["files"]) == {"train", "validation", "test"}
    train, validation, loaded = _load_routing_splits(output)
    assert train and validation
    assert all(item["split"] == "train" for item in train)
    assert all(item["split"] == "validation" for item in validation)
    assert len(train) + len(validation) < manifest["example_count"]
    assert loaded["source_dataset_hash"] == manifest["source_dataset_hash"]


def test_fixed_prediction_uses_test_only_without_loading_a_model(tmp_path: Path) -> None:
    trajectories = read_jsonl(PILOT_DATASET / "test.jsonl")

    class FakePlanner:
        def __init__(self) -> None:
            self.index = 0
            self.last_attempt = {}

        def plan(self, patient_state, trajectory_history):
            record = trajectories[self.index]
            self.index += 1
            self.last_attempt = {
                "raw_text": json.dumps(
                    record["accepted_plan_variants"][0],
                    ensure_ascii=False,
                ),
                "status": "parsed",
            }
            labels = record["routing_labels"]
            positive = [
                *labels["strong_positive_memory_ids"],
                *labels["weak_positive_memory_ids"],
            ]
            return PlannerStepResult(
                action=record["accepted_plan_variants"][0],
                current_objective=None,
                expected_state_change=[],
                active_memories=[
                    MemoryActivation(memory_id=item, weight=1.0 / len(positive), selected=True)
                    for item in positive
                ],
                expert_outputs=[],
                diagnostics={"active_memories": []},
            )

    summary = predict_planner_v2_fixed_test(
        trajectory_data=PILOT_DATASET,
        memory_dir=tmp_path / "unused-memory",
        decoder_artifact_dir=tmp_path / "unused-decoder",
        output_dir=tmp_path / "predictions",
        modes=("latent_topk",),
        planner_factory=lambda **_: FakePlanner(),
    )
    assert summary["test_count"] == len(trajectories)
    rows = read_jsonl(tmp_path / "predictions" / "latent_topk.predictions.jsonl")
    assert rows and all(item["trajectory"]["split"] == "test" for item in rows)
    assert not any(item["error"] for item in rows)
    assert all(item["planner_attempt"]["status"] == "parsed" for item in rows)
    assert all(item["planner_attempt"]["raw_text"] for item in rows)
    report = json.loads(
        (tmp_path / "predictions" / "latent_topk.planner_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["schema_version"] == "planner_evaluation.v2"
