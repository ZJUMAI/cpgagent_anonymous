"""Build Router/DAA supervision exclusively from admitted V2 trajectories."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from guideline_planner.artifacts import sha256_path
from guideline_planner.io_utils import read_jsonl, write_json, write_jsonl
from guideline_planner.progress import ProgressReporter
from guideline_planner.schemas_v2 import V2SchemaError, validate_trajectory_record_v2
from guideline_planner.trajectory_dataset import require_training_ready_dataset


def build_routing_training_data(
    planner_train_data: str | Path,
    output_path: str | Path,
    *,
    run_dirs: Iterable[str | Path] | None = None,
    allow_nonrelease_smoke: bool = False,
) -> dict[str, Any]:
    """Convert admitted trajectories into split-safe routing supervision.

    Formal builds accept only a dataset directory with a ready admission report
    and manifest. ``allow_nonrelease_smoke`` is intentionally explicit and is
    recorded in the output manifest. A directory output receives separate split
    files; a historical ``.jsonl`` output also receives sibling split files.
    """

    if list(run_dirs or []):
        raise V2SchemaError(
            "Router V2 does not accept legacy run activations. Extract observation "
            "state deltas separately and label memory provenance from rules."
        )
    source_path = Path(planner_train_data)
    source_manifest: dict[str, Any] = {}
    source_dataset_hash: str | None = None
    source_manifest_hash: str | None = None
    admission_hash: str | None = None
    if source_path.is_dir():
        require_training_ready_dataset(source_path, release_gates=not allow_nonrelease_smoke)
        manifest_path = source_path / "manifest.json"
        admission_path = source_path / "admission_report.json"
        if not allow_nonrelease_smoke and not manifest_path.is_file():
            raise V2SchemaError("Router V2 requires the admitted dataset manifest.json.")
        if manifest_path.is_file():
            source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_dataset_hash = str(source_manifest.get("dataset_hash") or "") or None
            source_manifest_hash = sha256_path(manifest_path)
        if admission_path.is_file():
            admission_hash = sha256_path(admission_path)
        if not allow_nonrelease_smoke and not source_dataset_hash:
            raise V2SchemaError("Router V2 source manifest is missing dataset_hash.")
        raw_records = [
            record
            for split in ("train", "validation", "test")
            for record in read_jsonl(source_path / f"{split}.jsonl")
        ]
    else:
        if not allow_nonrelease_smoke:
            raise V2SchemaError(
                "Router V2 formal planner_trajectory.v2 data must be built from an "
                "admitted dataset directory."
            )
        raw_records = read_jsonl(source_path)

    progress = ProgressReporter("build-routing", len(raw_records), unit="transition")
    progress.message(f"validating {len(raw_records)} admitted Planner transitions")
    source_records = [validate_trajectory_record_v2(record) for record in raw_records]
    if not source_records:
        raise V2SchemaError("Router V2 requires non-empty planner_trajectory.v2 data.")
    if not allow_nonrelease_smoke:
        pending = [
            f"{item['trajectory_id']}::{int(item['turn_index']):04d}"
            for item in source_records
            if item["provenance"]["review_status"] != "approved"
        ]
        if pending:
            raise V2SchemaError(
                "Router V2 formal data contains non-approved transitions: "
                + ", ".join(pending[:10])
            )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in source_records:
        grouped[str(source["trajectory_id"])].append(source)
    for trajectory in grouped.values():
        trajectory.sort(key=lambda item: int(item["turn_index"]))

    records: list[dict[str, Any]] = []
    processed = 0
    progress.start(status="compiling")
    for trajectory_id, trajectory in sorted(grouped.items()):
        for source in trajectory:
            labels = source["routing_labels"]
            strong = _unique_strings(labels["strong_positive_memory_ids"])
            weak = _unique_strings(labels["weak_positive_memory_ids"])
            hard = _unique_strings(labels["hard_negative_memory_ids"])
            easy = _unique_strings(labels["easy_negative_memory_ids"])
            if not strong:
                raise V2SchemaError(
                    f"Routing example {trajectory_id}/{source['turn_index']} needs a strong positive."
                )
            if not hard:
                raise V2SchemaError(
                    f"Routing example {trajectory_id}/{source['turn_index']} needs a hard negative."
                )
            variants = source["accepted_plan_variants"]
            records.append(
                {
                    "schema_version": "routing_example.v2",
                    "example_id": f"{trajectory_id}::{int(source['turn_index']):04d}",
                    "trajectory_id": trajectory_id,
                    "base_case_id": source["base_case_id"],
                    "case_id": source["case_id"],
                    "turn_index": source["turn_index"],
                    "split": source["split"],
                    "source": "planner_trajectory.v2_rule_provenance",
                    "source_dataset_hash": source_dataset_hash,
                    "guideline_context": source["guideline_context"],
                    "patient_state": source["state_before"],
                    "trajectory_history": _compact_history(trajectory, source["turn_index"]),
                    "positive_memory_ids": _unique_strings([*strong, *weak]),
                    "negative_memory_ids": _unique_strings([*hard, *easy]),
                    "strong_positive_ids": strong,
                    "weak_positive_ids": weak,
                    "hard_negative_ids": hard,
                    "easy_negative_ids": easy,
                    "retrieval_query": _retrieval_query(source),
                    "prompt": json.dumps(source["state_before"], ensure_ascii=False, sort_keys=True),
                    "target": json.dumps(variants[0], ensure_ascii=False, sort_keys=True),
                    "target_variants": variants,
                    "action_set": source["action_set"],
                    "state_delta": source["state_delta"],
                    "progress_label": 1.0,
                    "phase_target": source["state_after"]["current_phase"],
                    "trajectory_quality": 1.0
                    if source["provenance"]["review_status"] == "approved"
                    else 0.0,
                    "unsafe_score": 0.0,
                    "review_status": source["provenance"]["review_status"],
                }
            )
            processed += 1
            progress.update(
                processed,
                metrics={"split": source["split"], "trajectory": trajectory_id},
            )

    split_records = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "validation", "test")
    }
    if not split_records["train"]:
        raise V2SchemaError("Router V2 training data has no train-split examples.")
    output = Path(output_path)
    files: dict[str, str] = {}
    if output.suffix.lower() == ".jsonl":
        write_jsonl(output, records)
        for split, values in split_records.items():
            split_path = output.with_name(f"{output.stem}.{split}.jsonl")
            write_jsonl(split_path, values)
            files[split] = str(split_path)
        manifest_path = output.with_suffix(".manifest.json")
    else:
        output.mkdir(parents=True, exist_ok=True)
        for split, values in split_records.items():
            split_path = output / f"{split}.jsonl"
            write_jsonl(split_path, values)
            files[split] = str(split_path)
        manifest_path = output / "manifest.json"
    manifest = {
        "schema_version": "routing_dataset_manifest.v2",
        "output_path": str(output),
        "example_count": len(records),
        "split_counts": dict(sorted(Counter(item["split"] for item in records).items())),
        "train_example_count": len(split_records["train"]),
        "validation_example_count": len(split_records["validation"]),
        "test_example_count": len(split_records["test"]),
        "trajectory_count": len(grouped),
        "source_train_data": str(planner_train_data),
        "source_dataset_hash": source_dataset_hash,
        "source_dataset_manifest_hash": source_manifest_hash,
        "source_admission_report_hash": admission_hash,
        "release_scope": source_manifest.get("release_scope"),
        "approved_only": all(item["review_status"] == "approved" for item in records),
        "nonrelease_smoke": bool(allow_nonrelease_smoke),
        "legacy_run_labels_used": False,
        "files": files,
        "file_hashes": {split: sha256_path(path) for split, path in files.items()},
    }
    write_json(manifest_path, manifest)
    progress.finish(metrics={"examples": len(records)})
    return manifest


def _compact_history(
    trajectory: list[dict[str, Any]],
    turn_index: int,
) -> list[dict[str, Any]]:
    result = []
    for source in trajectory:
        if int(source["turn_index"]) >= int(turn_index):
            break
        result.append(
            {
                "turn_index": source["turn_index"],
                "planner_output": source["accepted_plan_variants"][0],
                "state_delta": source["state_delta"],
            }
        )
    return result[-4:]


def _retrieval_query(source: Mapping[str, Any]) -> str:
    state = source["state_before"]
    return " ".join(
        str(value)
        for value in (
            state.get("cancer_family"),
            state.get("disease_subtype"),
            state.get("known_diagnosis"),
            state.get("known_stage"),
            state.get("current_phase"),
            source["guideline_context"].get("decision_date"),
        )
        if value not in (None, "")
    )


def _unique_strings(value: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in value if str(item).strip()))
