"""Fail-closed synchronization of Planner V2 human-review decisions."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from guideline_planner.io_utils import read_jsonl, write_json, write_jsonl
from guideline_planner.release_scope import (
    PLANNER_V2_LUNG_ENDOMETRIAL_RELEASE_SCOPE,
    copy_release_scope,
    load_release_scope,
    release_scope_hash,
)
from guideline_planner.schemas_v2 import V2SchemaError, validate_trajectory_record_v2
from guideline_planner.trajectory_dataset import (
    _records_hash,
    build_trajectory_dataset_v2,
)


ReviewKey = tuple[str, int]


def approve_all_trajectory_reviews_v2(
    dataset_root: str | Path,
    *,
    reviewer_id: str = "manual-review-team",
    reviewed_at: str | None = None,
    release_scope: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Approve every canonical candidate and synchronize all derived copies.

    If the release was already approved, omitting ``reviewed_at`` reuses the
    one stored timestamp and makes this operation idempotent.  A conflicting
    reviewer or decision is never overwritten.
    """

    root = Path(dataset_root)
    candidates = read_jsonl(root / "planner_trajectory.v2.candidates.jsonl")
    if not candidates:
        raise V2SchemaError(f"No Planner V2 candidates found under {root}.")
    if reviewed_at is None:
        decided = {
            (
                str(row.get("provenance", {}).get("review_status") or ""),
                str(row.get("provenance", {}).get("reviewer_id") or ""),
                str(row.get("provenance", {}).get("reviewed_at") or ""),
            )
            for row in candidates
            if row.get("provenance", {}).get("review_status") != "pending"
        }
        if decided:
            if len(decided) != 1:
                raise V2SchemaError(
                    "Existing candidate review metadata is inconsistent; refusing to resynchronize."
                )
            status, existing_reviewer, existing_time = next(iter(decided))
            if status != "approved" or existing_reviewer != reviewer_id or not existing_time:
                raise V2SchemaError(
                    "Existing candidate review metadata conflicts with approve-all request."
                )
            reviewed_at = existing_time
        else:
            reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
                "+00:00", "Z"
            )
    decisions = [
        {
            "trajectory_id": row["trajectory_id"],
            "turn_index": row["turn_index"],
            "review_status": "approved",
        }
        for row in candidates
    ]
    return synchronize_trajectory_reviews_v2(
        root,
        decisions,
        reviewer_id=reviewer_id,
        reviewed_at=reviewed_at,
        release_scope=(
            release_scope
            if release_scope is not None
            else PLANNER_V2_LUNG_ENDOMETRIAL_RELEASE_SCOPE
        ),
    )


def synchronize_trajectory_reviews_v2(
    dataset_root: str | Path,
    decisions: Iterable[Mapping[str, Any]],
    *,
    reviewer_id: str,
    reviewed_at: str,
    release_scope: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically synchronize a complete decision set across V2 artifacts.

    The decision set must cover every canonical candidate exactly once.  Split
    rows must be exact derived copies of candidates before review metadata is
    changed, and review-queue keys must be a duplicate-free subset.  These
    invariants prevent a partially reviewed or stale copy from being admitted.
    """

    root = Path(dataset_root)
    reviewer = str(reviewer_id or "").strip()
    if not reviewer:
        raise V2SchemaError("reviewer_id must be non-empty.")
    reviewed = _normalize_utc_timestamp(reviewed_at)
    try:
        scope = load_release_scope(release_scope)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise V2SchemaError(f"Invalid Planner release scope: {exc}") from exc
    scope_digest = release_scope_hash(scope)

    candidates_path = root / "planner_trajectory.v2.candidates.jsonl"
    queue_path = root / "review_queue.jsonl"
    dataset_dir = root / "dataset"
    rules_path = root / "guideline_rules.v2.jsonl"
    memory_dir = root / "memory_catalog_mock"
    required_paths = [
        candidates_path,
        queue_path,
        rules_path,
        memory_dir / "memory_store_meta.json",
        *(dataset_dir / f"{split}.jsonl" for split in ("train", "validation", "test")),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise V2SchemaError("Review synchronization is missing artifacts: " + ", ".join(missing))

    candidates = read_jsonl(candidates_path)
    candidate_index, candidate_order = _unique_record_index(candidates, "candidates")
    decision_index = validate_review_decisions_v2(candidate_order, decisions)

    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_index: dict[ReviewKey, dict[str, Any]] = {}
    split_orders: dict[str, list[ReviewKey]] = {}
    for split in ("train", "validation", "test"):
        rows = read_jsonl(dataset_dir / f"{split}.jsonl")
        local_index, local_order = _unique_record_index(rows, f"dataset/{split}")
        for key, row in local_index.items():
            if key in split_index:
                raise V2SchemaError(f"Split artifacts duplicate review key {_render_key(key)}.")
            if row.get("split") != split:
                raise V2SchemaError(
                    f"Split artifact {split!r} contains row declared as {row.get('split')!r}."
                )
            split_index[key] = row
        split_rows[split] = rows
        split_orders[split] = local_order
    if set(split_index) != set(candidate_index):
        _raise_key_difference(
            expected=set(candidate_index),
            actual=set(split_index),
            label="split artifacts",
        )
    for key, candidate in candidate_index.items():
        # Raw candidates retain their builder-time placeholder split; the
        # dataset copies carry the deterministic grouped split.  Split is the
        # only derived field allowed to differ between these artifacts.
        if _without_review_or_split(candidate) != _without_review_or_split(split_index[key]):
            raise V2SchemaError(
                f"Split copy {_render_key(key)} differs from its canonical candidate."
            )
        _assert_existing_review_is_compatible(
            split_index[key].get("provenance"),
            decision_index[key],
            reviewer_id=reviewer,
            reviewed_at=reviewed,
            label=f"Split copy {_render_key(key)}",
        )

    queue = read_jsonl(queue_path)
    queue_index, _ = _unique_record_index(queue, "review queue")
    unknown_queue = set(queue_index) - set(candidate_index)
    if unknown_queue:
        raise V2SchemaError(
            "Review queue contains unknown candidate keys: "
            + ", ".join(_render_key(key) for key in sorted(unknown_queue))
        )
    for key, row in queue_index.items():
        _assert_existing_review_is_compatible(
            row,
            decision_index[key],
            reviewer_id=reviewer,
            reviewed_at=reviewed,
            label=f"Review queue row {_render_key(key)}",
        )

    updated_candidates = [
        _apply_decision(
            row,
            decision_index[_review_key(row)],
            reviewer_id=reviewer,
            reviewed_at=reviewed,
        )
        for row in candidates
    ]
    for row in updated_candidates:
        validate_trajectory_record_v2(row)
    updated_index = {_review_key(row): row for row in updated_candidates}
    updated_queue = []
    for row in queue:
        item = deepcopy(row)
        decision = decision_index[_review_key(row)]
        item["review_status"] = decision
        item["reviewer_id"] = reviewer
        item["reviewed_at"] = reviewed
        updated_queue.append(item)

    with tempfile.TemporaryDirectory(prefix=".review-sync-", dir=root) as temp_value:
        stage_root = Path(temp_value)
        stage_dataset = stage_root / "dataset"
        staged_manifest = build_trajectory_dataset_v2(
            updated_candidates,
            stage_dataset,
            seed=17,
            release_gates=True,
            fail_if_not_ready=True,
            rule_registry_path=rules_path,
            memory_dir=memory_dir,
            release_scope=scope,
        )
        if not staged_manifest["ready"]:
            raise V2SchemaError("Reviewed dataset did not pass formal admission.")
        for split, original_order in split_orders.items():
            staged_rows = read_jsonl(stage_dataset / f"{split}.jsonl")
            staged_order = [_review_key(row) for row in staged_rows]
            if staged_order != original_order:
                raise V2SchemaError(
                    f"Review synchronization would alter {split} row order or split membership."
                )
            if any(
                {**updated_index[key], "split": split} != row
                for key, row in zip(staged_order, staged_rows)
            ):
                raise V2SchemaError(
                    f"Review synchronization would alter non-review fields in {split}."
                )

        final_manifest = deepcopy(staged_manifest)
        final_manifest["files"] = {
            split: str(dataset_dir / f"{split}.jsonl")
            for split in ("train", "validation", "test")
        }
        final_manifest["admission_report"] = str(dataset_dir / "admission_report.json")
        final_manifest["review_status"] = "approved"
        final_manifest["reviewer_id"] = reviewer
        final_manifest["reviewed_at"] = reviewed

        generation = _load_json_object(root / "generation_manifest.json", required=False)
        generation.update(
            {
                "schema_version": "real_case_trajectory_generation_manifest.v2",
                "dataset_manifest": final_manifest,
                "release_scope": copy_release_scope(scope),
                "release_scope_hash": scope_digest,
                "scope": [
                    item["cancer_family"] for item in scope["action_targets"]
                ],
                "deferred": [
                    f"{item['cancer_family']}/" + ",".join(item["disease_subtypes"])
                    for item in scope["deferred_action_targets"]
                ],
                "review_status": "approved",
                "reviewer_id": reviewer,
                "reviewed_at": reviewed,
                "reviewed_record_count": len(updated_candidates),
                "reviewed_queue_count": len(updated_queue),
                "candidate_hash": _records_hash(updated_candidates),
                "training_authorization": "approved_for_planner_action_supervision",
            }
        )
        artifacts = generation.setdefault("artifacts", {})
        if isinstance(artifacts, dict):
            artifacts["release_scope"] = str(root / "release_scope.json")

        automated_qc = _load_json_object(root / "automated_qc_report.json", required=False)
        automated_qc.update(
            {
                "automated_qc_passed": True,
                "blocking_reason": None,
                "grouped_split_manifest_ready": True,
                "human_review_required_count": 0,
                "human_reviewed_count": len(updated_candidates),
                "memory_catalog_trainable": False,
                "planner_decoder_training_requires_formal_memory_store": True,
                "review_status": "approved",
                "reviewer_id": reviewer,
                "reviewed_at": reviewed,
                "training_ready": True,
                "release_scope_hash": scope_digest,
            }
        )

        write_jsonl(stage_root / "candidates.jsonl", updated_candidates)
        write_jsonl(stage_root / "review_queue.jsonl", updated_queue)
        write_json(stage_root / "release_scope.json", copy_release_scope(scope))
        write_json(stage_root / "dataset_manifest.json", final_manifest)
        write_json(stage_root / "generation_manifest.json", generation)
        write_json(stage_root / "automated_qc_report.json", automated_qc)

        commit_pairs = [
            (stage_root / "candidates.jsonl", candidates_path),
            (stage_root / "review_queue.jsonl", queue_path),
            (stage_root / "release_scope.json", root / "release_scope.json"),
            *[
                (stage_dataset / f"{split}.jsonl", dataset_dir / f"{split}.jsonl")
                for split in ("train", "validation", "test")
            ],
            (stage_dataset / "admission_report.json", dataset_dir / "admission_report.json"),
            (stage_root / "dataset_manifest.json", dataset_dir / "manifest.json"),
            (stage_root / "generation_manifest.json", root / "generation_manifest.json"),
            (stage_root / "automated_qc_report.json", root / "automated_qc_report.json"),
        ]
        for source, target in commit_pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)

    return {
        "schema_version": "planner_review_sync.v2",
        "dataset_root": str(root),
        "candidate_count": len(updated_candidates),
        "queue_count": len(updated_queue),
        "review_status": "approved"
        if all(value == "approved" for value in decision_index.values())
        else "mixed",
        "reviewer_id": reviewer,
        "reviewed_at": reviewed,
        "release_scope_hash": scope_digest,
        "candidate_hash": _records_hash(updated_candidates),
        "dataset_hash": final_manifest["dataset_hash"],
        "ready": True,
    }


def validate_review_decisions_v2(
    expected_keys: Iterable[ReviewKey],
    decisions: Iterable[Mapping[str, Any]],
) -> dict[ReviewKey, str]:
    """Validate exact decision coverage and reject duplicates/conflicts."""

    expected = set(expected_keys)
    result: dict[ReviewKey, str] = {}
    for index, raw in enumerate(decisions):
        if not isinstance(raw, Mapping):
            raise V2SchemaError(f"review decision[{index}] must be an object.")
        key = _review_key(raw)
        if key in result:
            raise V2SchemaError(f"Duplicate review decision for {_render_key(key)}.")
        status = str(raw.get("review_status") or "").strip().lower()
        if status not in {"approved", "rejected"}:
            raise V2SchemaError(
                f"Review decision {_render_key(key)} must be approved or rejected."
            )
        result[key] = status
    unknown = set(result) - expected
    if unknown:
        raise V2SchemaError(
            "Review decisions contain unknown candidate keys: "
            + ", ".join(_render_key(key) for key in sorted(unknown))
        )
    missing = expected - set(result)
    if missing:
        raise V2SchemaError(
            f"Partial review synchronization is forbidden; {len(missing)} decisions are missing."
        )
    return result


def _apply_decision(
    row: Mapping[str, Any],
    status: str,
    *,
    reviewer_id: str,
    reviewed_at: str,
) -> dict[str, Any]:
    item = deepcopy(dict(row))
    provenance = item.get("provenance")
    if not isinstance(provenance, dict):
        raise V2SchemaError(f"Candidate {_render_key(_review_key(row))} has no provenance.")
    _assert_existing_review_is_compatible(
        provenance,
        status,
        reviewer_id=reviewer_id,
        reviewed_at=reviewed_at,
        label=f"Candidate {_render_key(_review_key(row))}",
    )
    provenance["review_status"] = status
    provenance["reviewer_id"] = reviewer_id
    provenance["reviewed_at"] = reviewed_at
    return item


def _assert_existing_review_is_compatible(
    metadata: Any,
    status: str,
    *,
    reviewer_id: str,
    reviewed_at: str,
    label: str,
) -> None:
    if not isinstance(metadata, Mapping):
        raise V2SchemaError(f"{label} has no review metadata.")
    existing_status = str(metadata.get("review_status") or "pending")
    existing_reviewer = str(metadata.get("reviewer_id") or "")
    existing_time = str(metadata.get("reviewed_at") or "")
    if existing_status == "pending":
        if existing_reviewer or existing_time:
            raise V2SchemaError(f"{label} has partial pending review metadata.")
        return
    if existing_status != status:
        raise V2SchemaError(f"{label} has conflicting decision {existing_status!r}.")
    if existing_reviewer != reviewer_id or existing_time != reviewed_at:
        raise V2SchemaError(f"{label} has conflicting review metadata.")


def _unique_record_index(
    rows: Iterable[Mapping[str, Any]],
    label: str,
) -> tuple[dict[ReviewKey, dict[str, Any]], list[ReviewKey]]:
    index: dict[ReviewKey, dict[str, Any]] = {}
    order: list[ReviewKey] = []
    for row in rows:
        key = _review_key(row)
        if key in index:
            raise V2SchemaError(f"{label} contains duplicate key {_render_key(key)}.")
        index[key] = dict(row)
        order.append(key)
    if not index:
        raise V2SchemaError(f"{label} is empty.")
    return index, order


def _review_key(row: Mapping[str, Any]) -> ReviewKey:
    trajectory_id = str(row.get("trajectory_id") or "").strip()
    if not trajectory_id:
        raise V2SchemaError("Review record requires trajectory_id.")
    try:
        turn_index = int(row.get("turn_index"))
    except (TypeError, ValueError) as exc:
        raise V2SchemaError(f"Review record {trajectory_id!r} has invalid turn_index.") from exc
    return trajectory_id, turn_index


def _without_review_or_split(row: Mapping[str, Any]) -> dict[str, Any]:
    item = deepcopy(dict(row))
    provenance = item.get("provenance")
    if isinstance(provenance, dict):
        for key in ("review_status", "reviewer_id", "reviewed_at"):
            provenance.pop(key, None)
    item.pop("split", None)
    return item


def _raise_key_difference(
    *,
    expected: set[ReviewKey],
    actual: set[ReviewKey],
    label: str,
) -> None:
    missing = expected - actual
    extra = actual - expected
    parts = []
    if missing:
        parts.append(f"missing {len(missing)} canonical keys")
    if extra:
        parts.append(f"containing {len(extra)} unknown keys")
    raise V2SchemaError(f"{label} is incomplete or stale: " + " and ".join(parts) + ".")


def _normalize_utc_timestamp(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise V2SchemaError("reviewed_at must be non-empty.")
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise V2SchemaError("reviewed_at must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise V2SchemaError("reviewed_at must include the UTC timezone.")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_json_object(path: Path, *, required: bool) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise V2SchemaError(f"Required JSON artifact is missing: {path}")
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V2SchemaError(f"JSON artifact must contain an object: {path}")
    return value


def _render_key(key: ReviewKey) -> str:
    return f"{key[0]}/{key[1]}"
