"""Atomic, lineage-bound checkpoints shared by Planner V2 training stages."""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Mapping

from guideline_planner.artifacts import sha256_json, sha256_path
from guideline_planner.io_utils import write_json


def training_config_hash(config: Mapping[str, Any]) -> str:
    """Hash effective training settings while ignoring only the output location."""

    return sha256_json({key: value for key, value in config.items() if key != "output_dir"})


def capture_rng_state(torch_module: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch_module.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch_module.cuda.is_available():
        state["torch_cuda"] = torch_module.cuda.get_rng_state_all()
    return state


def restore_rng_state(torch_module: Any, state: Mapping[str, Any]) -> None:
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("torch_cpu") is not None:
        torch_module.set_rng_state(state["torch_cpu"])
    if state.get("numpy") is not None:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    if state.get("torch_cuda") is not None and torch_module.cuda.is_available():
        torch_module.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(
    *,
    torch_module: Any,
    output_dir: str | Path,
    stage: str,
    global_step: int,
    config_hash: str,
    lineage: Mapping[str, Any],
    state: Mapping[str, Any],
    keep_last: int = 2,
) -> Path:
    """Atomically save one checkpoint and advance the ``latest`` pointer."""

    root = Path(output_dir) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    final_dir = root / f"step-{int(global_step):08d}"
    temporary_dir = root / f".step-{int(global_step):08d}.{os.getpid()}.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    payload = {
        "schema_version": "planner_training_checkpoint.v2",
        "stage": stage,
        "global_step": int(global_step),
        "config_hash": config_hash,
        "lineage": dict(lineage),
        "rng_state": capture_rng_state(torch_module),
        **dict(state),
    }
    payload_path = temporary_dir / "checkpoint.pt"
    temporary_payload = temporary_dir / "checkpoint.pt.tmp"
    torch_module.save(payload, temporary_payload)
    temporary_payload.replace(payload_path)
    checkpoint_hash = sha256_path(payload_path)
    write_json(
        temporary_dir / "metadata.json",
        {
            key: payload[key]
            for key in ("schema_version", "stage", "global_step", "config_hash", "lineage")
        },
    )
    metadata = json.loads((temporary_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["checkpoint_sha256"] = checkpoint_hash
    write_json(temporary_dir / "metadata.json", metadata)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    temporary_dir.replace(final_dir)
    latest_tmp = root / "latest.json.tmp"
    write_json(
        latest_tmp,
        {
            "checkpoint_dir": final_dir.name,
            "global_step": int(global_step),
            "checkpoint_sha256": checkpoint_hash,
        },
    )
    latest_tmp.replace(root / "latest.json")
    _prune_checkpoints(root, keep_last=max(int(keep_last), 1))
    return final_dir


def load_latest_training_checkpoint(
    *,
    torch_module: Any,
    output_dir: str | Path,
    stage: str,
    config_hash: str,
    lineage: Mapping[str, Any],
    map_location: Any = "cpu",
) -> dict[str, Any] | None:
    """Load the latest checkpoint, failing closed on config or lineage drift."""

    root = Path(output_dir) / "checkpoints"
    latest = root / "latest.json"
    if not latest.is_file():
        return None
    pointer = json.loads(latest.read_text(encoding="utf-8"))
    path = root / str(pointer.get("checkpoint_dir") or "") / "checkpoint.pt"
    if not path.is_file():
        raise RuntimeError(f"Latest checkpoint pointer is incomplete: {path}")
    if sha256_path(path) != pointer.get("checkpoint_sha256"):
        raise RuntimeError("Latest Planner training checkpoint content hash mismatch.")
    try:
        payload = torch_module.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch<2.0 compatibility
        payload = torch_module.load(path, map_location=map_location)
    if payload.get("schema_version") != "planner_training_checkpoint.v2":
        raise RuntimeError("Unsupported Planner training checkpoint schema.")
    if payload.get("stage") != stage:
        raise RuntimeError(
            f"Checkpoint stage mismatch: expected={stage}, actual={payload.get('stage')}."
        )
    if payload.get("config_hash") != config_hash:
        raise RuntimeError("Checkpoint training configuration hash mismatch.")
    expected_lineage = dict(lineage)
    actual_lineage = dict(payload.get("lineage") or {})
    if actual_lineage != expected_lineage:
        raise RuntimeError(
            f"Checkpoint lineage mismatch: expected={expected_lineage}, actual={actual_lineage}."
        )
    restore_rng_state(torch_module, payload.get("rng_state") or {})
    return dict(payload)


def refuse_completed_output(output_dir: str | Path, completion_marker: str) -> None:
    marker = Path(output_dir) / completion_marker
    if marker.is_file():
        raise FileExistsError(
            f"Completed artifact already exists and will not be overwritten: {marker}"
        )


def _prune_checkpoints(root: Path, *, keep_last: int) -> None:
    checkpoints = sorted(
        (path for path in root.glob("step-*") if path.is_dir()),
        key=lambda path: path.name,
    )
    for stale in checkpoints[:-keep_last]:
        shutil.rmtree(stale)
