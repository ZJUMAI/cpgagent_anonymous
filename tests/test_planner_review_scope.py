from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from guideline_planner.io_utils import read_jsonl
from guideline_planner.release_scope import (
    PLANNER_V2_LUNG_ENDOMETRIAL_RELEASE_SCOPE,
    action_target_allows,
    load_release_scope,
    release_scope_hash,
)
from guideline_planner.review_sync import validate_review_decisions_v2
from guideline_planner.schemas_v2 import V2SchemaError, audit_trajectory_dataset_v2
from guideline_planner.trajectory_dataset import require_training_ready_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.private_data
PILOT_ROOT = REPO_ROOT / "datasets" / "planner_v2_pilot"


def test_lung_endometrial_release_scope_is_explicit_and_content_addressed() -> None:
    scope = load_release_scope(PLANNER_V2_LUNG_ENDOMETRIAL_RELEASE_SCOPE)

    assert [item["cancer_family"] for item in scope["action_targets"]] == [
        "endometrial",
        "lung",
    ]
    assert action_target_allows(
        scope,
        cancer_family="lung",
        disease_subtype="nsclc",
        guideline_ids={"NSCLC_2010"},
    )
    assert action_target_allows(
        scope,
        cancer_family="endometrial",
        disease_subtype="ucec",
        guideline_ids={"CSCO子宫内膜癌2023"},
    )
    assert not action_target_allows(
        scope,
        cancer_family="lung",
        disease_subtype="sclc",
        guideline_ids={"SCLC_2010"},
    )
    assert not action_target_allows(
        scope,
        cancer_family="nasopharyngeal",
        disease_subtype="npc",
        guideline_ids={"CSCO鼻咽癌2022"},
    )
    assert len(scope["memory_guidelines"]) == 4
    assert release_scope_hash(scope) == release_scope_hash(
        deepcopy(PLANNER_V2_LUNG_ENDOMETRIAL_RELEASE_SCOPE)
    )


def test_audit_rejects_schema_valid_target_outside_action_release() -> None:
    row = read_jsonl(PILOT_ROOT / "dataset" / "train.jsonl")[0]
    row["state_before"]["disease_subtype"] = "sclc"
    row["state_after"]["disease_subtype"] = "sclc"
    row["state_delta"] = {
        key: value
        for key, value in row["state_delta"].items()
        if key != "disease_subtype"
    }

    report = audit_trajectory_dataset_v2(
        [row],
        release_gates=False,
        release_scope=PLANNER_V2_LUNG_ENDOMETRIAL_RELEASE_SCOPE,
    )

    assert not report.ready
    assert any("outside release scope" in error for error in report.errors)


def test_review_decisions_require_exact_unique_candidate_coverage() -> None:
    expected = [("trajectory-a", 0), ("trajectory-b", 1)]
    complete = [
        {"trajectory_id": "trajectory-a", "turn_index": 0, "review_status": "approved"},
        {"trajectory_id": "trajectory-b", "turn_index": 1, "review_status": "approved"},
    ]
    assert validate_review_decisions_v2(expected, complete) == {
        ("trajectory-a", 0): "approved",
        ("trajectory-b", 1): "approved",
    }

    with pytest.raises(V2SchemaError, match="Partial review"):
        validate_review_decisions_v2(expected, complete[:1])
    with pytest.raises(V2SchemaError, match="unknown candidate"):
        validate_review_decisions_v2(
            expected,
            [
                complete[0],
                {"trajectory_id": "unknown", "turn_index": 0, "review_status": "approved"},
            ],
        )
    with pytest.raises(V2SchemaError, match="Duplicate review"):
        validate_review_decisions_v2(expected[:1], [complete[0], complete[0]])


def test_checked_in_pilot_is_fully_approved_and_manifest_bound() -> None:
    candidates = read_jsonl(PILOT_ROOT / "planner_trajectory.v2.candidates.jsonl")
    queue = read_jsonl(PILOT_ROOT / "review_queue.jsonl")
    split_rows = [
        row
        for split in ("train", "validation", "test")
        for row in read_jsonl(PILOT_ROOT / "dataset" / f"{split}.jsonl")
    ]

    assert len(candidates) == len(split_rows) == 420
    assert len(queue) == 264
    assert {row["provenance"]["review_status"] for row in candidates} == {"approved"}
    assert {row["provenance"]["reviewer_id"] for row in candidates} == {
        "manual-review-team"
    }
    reviewed_at = {row["provenance"]["reviewed_at"] for row in candidates}
    assert len(reviewed_at) == 1
    assert {row["provenance"]["reviewed_at"] for row in split_rows} == reviewed_at
    assert {row["review_status"] for row in queue} == {"approved"}
    assert {row["reviewed_at"] for row in queue} == reviewed_at

    report = require_training_ready_dataset(PILOT_ROOT / "dataset")
    assert report.ready
    assert report.errors == ()
    assert report.release_scope_hash == release_scope_hash(
        PLANNER_V2_LUNG_ENDOMETRIAL_RELEASE_SCOPE
    )
