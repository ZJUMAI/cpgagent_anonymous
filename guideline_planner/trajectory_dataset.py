"""V2 trajectory dataset assembly, grouped splitting, and safe run-log extraction."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from guideline_planner.io_utils import read_jsonl, write_json, write_jsonl
from guideline_planner.grounding import validate_grounding_assets
from guideline_planner.release_scope import (
    copy_release_scope,
    load_release_scope,
    release_scope_hash,
)
from guideline_planner.schemas_v2 import (
    DatasetAdmissionReport,
    V2SchemaError,
    audit_trajectory_dataset_v2,
    stable_case_splits,
    validate_trajectory_record_v2,
)


def build_trajectory_dataset_v2(
    records: str | Path | Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    seed: int = 17,
    release_gates: bool = True,
    fail_if_not_ready: bool = False,
    rule_registry_path: str | Path | None = None,
    memory_dir: str | Path | None = None,
    release_scope: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate V2 transitions, enforce grouped splits, and write split JSONL files."""

    try:
        resolved_scope = load_release_scope(release_scope)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise V2SchemaError(f"Invalid Planner release scope: {exc}") from exc
    scope_hash = release_scope_hash(resolved_scope)
    rows = _load_records(records)
    case_families: dict[str, str] = {}
    for index, row in enumerate(rows):
        base_case_id = str(row.get("base_case_id") or row.get("case_id") or "")
        state = row.get("state_before")
        family = str(state.get("cancer_family") or "") if isinstance(state, Mapping) else ""
        if not base_case_id or not family:
            raise V2SchemaError(
                f"record[{index}] needs base_case_id and state_before.cancer_family before splitting."
            )
        previous = case_families.get(base_case_id)
        if previous is not None and previous != family:
            raise V2SchemaError(
                f"base_case_id {base_case_id!r} appears under multiple cancer families."
            )
        case_families[base_case_id] = family
    assignments = stable_case_splits(case_families, seed=seed)
    normalized = []
    for row in rows:
        item = deepcopy(row)
        item["split"] = assignments[str(item.get("base_case_id") or item.get("case_id"))]
        normalized.append(validate_trajectory_record_v2(item))

    report = audit_trajectory_dataset_v2(
        normalized,
        release_gates=release_gates,
        release_scope=resolved_scope,
    )
    grounding_errors = validate_grounding_assets(
        normalized,
        rule_registry_path=rule_registry_path,
        memory_dir=memory_dir,
    ) if release_gates else []
    if grounding_errors:
        report = DatasetAdmissionReport(
            ready=False,
            errors=tuple([*report.errors, *grounding_errors]),
            warnings=report.warnings,
            statistics=report.statistics,
            release_scope=copy_release_scope(resolved_scope),
            release_scope_hash=scope_hash,
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        write_jsonl(
            output / f"{split}.jsonl",
            [row for row in normalized if row["split"] == split],
        )
    write_json(output / "admission_report.json", report.to_dict())
    manifest = {
        "schema_version": "planner_dataset_manifest.v2",
        "record_count": len(normalized),
        "case_count": len(case_families),
        "seed": seed,
        "release_gates": release_gates,
        "release_scope": copy_release_scope(resolved_scope),
        "release_scope_hash": scope_hash,
        "rule_registry_path": str(rule_registry_path) if rule_registry_path else None,
        "memory_dir": str(memory_dir) if memory_dir else None,
        "ready": report.ready,
        "split_counts": dict(sorted(Counter(row["split"] for row in normalized).items())),
        "dataset_hash": _records_hash(normalized),
        "files": {
            split: str(output / f"{split}.jsonl")
            for split in ("train", "validation", "test")
        },
        "admission_report": str(output / "admission_report.json"),
    }
    write_json(output / "manifest.json", manifest)
    if fail_if_not_ready and not report.ready:
        raise V2SchemaError(
            "Planner V2 dataset failed release admission gates: "
            + "; ".join(report.errors[:10])
        )
    return manifest


def require_training_ready_dataset(
    data_path: str | Path,
    *,
    release_gates: bool = True,
) -> DatasetAdmissionReport:
    path = Path(data_path)
    if path.is_dir():
        report_path = path / "admission_report.json"
        manifest_path = path / "manifest.json"
        manifest: dict[str, Any] | None = None
        if release_gates:
            if not manifest_path.is_file():
                raise V2SchemaError(
                    "Planner training requires a built dataset directory with manifest.json."
                )
            if not report_path.is_file():
                raise V2SchemaError(
                    "Planner training requires a built dataset directory with admission_report.json."
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            admission = json.loads(report_path.read_text(encoding="utf-8"))
            if not manifest.get("ready"):
                raise V2SchemaError(
                    "Planner V2 dataset manifest.json is not ready."
                )
            if not admission.get("ready"):
                raise V2SchemaError(
                    "Planner V2 dataset admission_report.json is not ready."
                )
        rows = [
            row
            for split in ("train", "validation", "test")
            for row in read_jsonl(path / f"{split}.jsonl")
        ]
        if release_gates:
            assert manifest is not None
            scope_payload = manifest.get("release_scope")
            stored_scope_hash = str(manifest.get("release_scope_hash") or "")
            if not isinstance(scope_payload, Mapping) or not stored_scope_hash:
                raise V2SchemaError(
                    "Planner dataset manifest is missing its bound release scope."
                )
            try:
                resolved_scope = load_release_scope(scope_payload)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise V2SchemaError(f"Invalid manifest-bound release scope: {exc}") from exc
            actual_scope_hash = release_scope_hash(resolved_scope)
            if stored_scope_hash != actual_scope_hash:
                raise V2SchemaError("Planner dataset release scope hash mismatch.")
            if admission.get("release_scope") != copy_release_scope(resolved_scope):
                raise V2SchemaError(
                    "Planner dataset admission report release scope differs from manifest."
                )
            if admission.get("release_scope_hash") != actual_scope_hash:
                raise V2SchemaError(
                    "Planner dataset admission report release scope hash mismatch."
                )
            actual_dataset_hash = _records_hash(rows)
            if manifest.get("dataset_hash") != actual_dataset_hash:
                raise V2SchemaError("Planner dataset content hash mismatch.")
            if int(manifest.get("record_count", -1)) != len(rows):
                raise V2SchemaError("Planner dataset record count differs from manifest.")
            actual_split_counts = dict(
                sorted(Counter(str(row.get("split") or "") for row in rows).items())
            )
            if manifest.get("split_counts") != actual_split_counts:
                raise V2SchemaError("Planner dataset split counts differ from manifest.")
        else:
            resolved_scope = load_release_scope(None)
    else:
        if release_gates:
            raise V2SchemaError(
                "Release training requires a build-planner-data-v2 directory so rule/memory grounding is auditable."
            )
        rows = read_jsonl(path)
        resolved_scope = load_release_scope(None)
    report = audit_trajectory_dataset_v2(
        rows,
        release_gates=release_gates,
        release_scope=resolved_scope,
    )
    if not report.ready:
        raise V2SchemaError(
            "Planner V2 training data is not admissible: "
            + "; ".join(report.errors[:10])
        )
    return report


def extract_observation_transitions_from_runs(
    run_dirs: Iterable[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Extract only observed state deltas from legacy runs.

    Old planner outputs and memory activations are intentionally never read, so
    this artifact cannot accidentally turn collapsed model decisions into gold
    V2 action or routing labels.
    """

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    run_count = 0
    for value in run_dirs:
        run_dir = Path(value)
        history_path = run_dir / "patient_state_history.jsonl"
        if not history_path.is_file():
            continue
        run_count += 1
        for event in read_jsonl(history_path):
            if event.get("event_type") != "patient_state_updated":
                continue
            before = event.get("patient_state_before")
            after = event.get("patient_state_after")
            delta = event.get("delta")
            if not all(isinstance(item, Mapping) for item in (before, after, delta)):
                continue
            payload = {
                "schema_version": "observed_state_transition.v2",
                "case_id": str(event.get("case_id") or before.get("case_id") or ""),
                "run_id": str(event.get("run_id") or run_dir.name),
                "round_index": int(event.get("round_index") or 0),
                "state_before": dict(before),
                "state_after": dict(after),
                "state_delta": dict(delta),
                "source": "legacy_observation_only",
            }
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "case_id": payload["case_id"],
                        "state_before": payload["state_before"],
                        "state_after": payload["state_after"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            payload["observation_hash"] = fingerprint
            records.append(payload)
    write_jsonl(Path(output_path), records)
    manifest = {
        "schema_version": "observed_state_transition_manifest.v2",
        "run_count": run_count,
        "transition_count": len(records),
        "output_path": str(output_path),
        "excluded_supervision": ["planner_output", "active_memories"],
    }
    write_json(Path(output_path).with_suffix(".manifest.json"), manifest)
    return manifest


def _load_records(
    records: str | Path | Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(records, (str, Path)):
        return read_jsonl(Path(records))
    return [dict(record) for record in records]


def _records_hash(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(
        records,
        key=lambda item: (str(item.get("trajectory_id")), int(item.get("turn_index", 0))),
    ):
        digest.update(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()
