"""Run deterministic Planner V2 predictions over the immutable test split."""

from __future__ import annotations

import gc
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from guideline_planner.evaluation_v2 import (
    evaluate_daa_comparison,
    evaluate_planner_predictions,
    evaluate_router_predictions,
    planner_action_macro_score,
)
from guideline_planner.io_utils import read_jsonl, write_json, write_jsonl
from guideline_planner.progress import ProgressReporter
from guideline_planner.trajectory_dataset import require_training_ready_dataset


PlannerFactory = Callable[..., Any]


def predict_planner_v2_fixed_test(
    *,
    trajectory_data: str | Path,
    memory_dir: str | Path,
    decoder_artifact_dir: str | Path,
    runtime_decoder_artifact_dir: str | Path | None = None,
    output_dir: str | Path,
    release_dir: str | Path | None = None,
    modes: Sequence[str] = ("latent_topk", "daa_full"),
    routing_config: str | Path | Mapping[str, Any] | None = None,
    routing_checkpoint: str | Path | None = None,
    top_k: int = 4,
    device: str | None = None,
    query_encoder_device: str = "cpu",
    model_dtype: str = "bf16",
    max_new_tokens: int | str = 1024,
    seed: int = 17,
    bootstrap_samples: int = 2000,
    planner_factory: PlannerFactory | None = None,
) -> dict[str, Any]:
    """Predict both ablations without ever training on or rewriting test data."""

    dataset = Path(trajectory_data)
    require_training_ready_dataset(dataset, release_gates=True)
    if not dataset.is_dir():
        raise ValueError("Fixed Planner evaluation requires a built dataset directory.")
    test_path = dataset / "test.jsonl"
    trajectories = read_jsonl(test_path)
    if not trajectories or any(item.get("split") != "test" for item in trajectories):
        raise ValueError("Fixed Planner evaluation requires a non-empty, test-only split.")
    requested = list(dict.fromkeys(str(item) for item in modes))
    unsupported = set(requested) - {"latent_topk", "daa_full"}
    if unsupported:
        raise ValueError(f"Unsupported prediction modes: {sorted(unsupported)}")
    if "daa_full" in requested and release_dir is None and not routing_checkpoint:
        raise ValueError("daa_full fixed prediction requires --routing-checkpoint.")
    if planner_factory is None:
        from guideline_planner.planner import LatentGuidelinePlanner

        planner_factory = LatentGuidelinePlanner

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mode_rows: dict[str, list[dict[str, Any]]] = {}
    reports: dict[str, Any] = {}
    scores: dict[str, list[float]] = {}
    progress = ProgressReporter(
        "evaluate",
        len(requested) * len(trajectories),
        unit="prediction",
    )
    completed = 0
    progress.start(status="loading")
    for mode in requested:
        routing_enabled = mode == "daa_full"
        progress.message(f"loading {mode} Planner runtime")
        if release_dir is not None and hasattr(planner_factory, "from_release"):
            planner = planner_factory.from_release(
                release_dir,
                mode=mode,
                device=device,
                query_encoder_device=query_encoder_device,
                max_new_tokens=max_new_tokens,
                model_dtype=model_dtype,
            )
        else:
            planner = planner_factory(
                memory_dir=memory_dir,
                decoder_artifact_dir=(
                    runtime_decoder_artifact_dir
                    if routing_enabled and runtime_decoder_artifact_dir is not None
                    else decoder_artifact_dir
                ),
                top_k=top_k,
                device=device,
                query_encoder_device=query_encoder_device,
                max_new_tokens=max_new_tokens,
                model_dtype=model_dtype,
                output_mode="json",
                routing_config=routing_config if routing_enabled else {"enabled": False},
                routing_checkpoint=routing_checkpoint if routing_enabled else None,
            )
        rows = []
        for trajectory in trajectories:
            key = (str(trajectory["trajectory_id"]), int(trajectory["turn_index"]))
            error: dict[str, Any] | None = None
            # The planner instance is reused for the fixed split. Clear the
            # previous row so an early routing failure cannot inherit a stale
            # decoder attempt from another trajectory.
            planner.last_attempt = {}
            try:
                step = planner.plan(
                    trajectory["state_before"],
                    trajectory_history=list(
                        trajectory["state_before"].get("completed_actions") or []
                    ),
                )
                prediction = step.action
                active_memories = [item.to_dict() for item in step.active_memories]
                routing = dict(step.diagnostics)
            except Exception as exc:  # one bad generation must remain auditable
                prediction = {}
                active_memories = []
                routing = {}
                error = {"type": type(exc).__name__, "message": str(exc)}
            planner_attempt = deepcopy(
                dict(getattr(planner, "last_attempt", {}) or {})
            )
            row = {
                "schema_version": "planner_fixed_prediction.v2",
                "mode": mode,
                "example_id": f"{key[0]}::{key[1]:04d}",
                "trajectory": trajectory,
                "prediction": prediction,
                "active_memories": active_memories,
                "routing": routing,
                "planner_attempt": planner_attempt,
                "error": error,
            }
            rows.append(row)
            completed += 1
            progress.update(
                completed,
                metrics={
                    "mode": mode,
                    "example": row["example_id"],
                    "errors": sum(item["error"] is not None for item in rows),
                },
            )
        progress.message(f"computing {mode} fixed-set metrics")
        write_jsonl(output / f"{mode}.predictions.jsonl", rows)
        planner_report = evaluate_planner_predictions(rows)
        write_json(output / f"{mode}.planner_evaluation.json", planner_report)
        reports[f"{mode}_planner"] = planner_report
        mode_rows[mode] = rows
        scores[mode] = [planner_action_macro_score(row) for row in rows]
        if routing_enabled:
            router_rows = [
                {
                    "schema_version": "router_fixed_prediction.v2",
                    "example_id": row["example_id"],
                    "routing_labels": row["trajectory"]["routing_labels"],
                    "routing": row["routing"],
                    "error": row["error"],
                }
                for row in rows
            ]
            write_jsonl(output / "daa_full.routing_predictions.jsonl", router_rows)
            router_report = evaluate_router_predictions(router_rows)
            write_json(output / "daa_full.router_evaluation.json", router_report)
            reports["daa_full_router"] = router_report
        del planner
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    comparison = None
    if {"latent_topk", "daa_full"}.issubset(mode_rows):
        baseline_ids = [item["example_id"] for item in mode_rows["latent_topk"]]
        daa_ids = [item["example_id"] for item in mode_rows["daa_full"]]
        if baseline_ids != daa_ids:
            raise RuntimeError("DAA and latent_topk predictions are not pair-aligned.")
        comparison = evaluate_daa_comparison(
            scores["daa_full"],
            scores["latent_topk"],
            samples=bootstrap_samples,
            seed=seed,
        )
        write_json(output / "daa_vs_latent_topk.json", comparison)
    summary = {
        "schema_version": "planner_fixed_test_run.v2",
        "trajectory_data": str(dataset),
        "release_dir": str(release_dir) if release_dir is not None else None,
        "baseline_decoder_artifact_dir": str(decoder_artifact_dir),
        "runtime_decoder_artifact_dir": str(runtime_decoder_artifact_dir)
        if runtime_decoder_artifact_dir is not None
        else None,
        "test_file": str(test_path),
        "test_count": len(trajectories),
        "modes": requested,
        "top_k": int(top_k),
        "reports": reports,
        "comparison": comparison,
    }
    write_json(output / "summary.json", summary)
    progress.finish(metrics={"modes": ",".join(requested)})
    return summary
