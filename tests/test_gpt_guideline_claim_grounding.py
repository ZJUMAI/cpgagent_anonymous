from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from guideline_planner.grounding import validate_grounding_assets
from guideline_planner.io_utils import read_jsonl, write_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = REPO_ROOT / "datasets" / "planner_v2_pilot"

pytestmark = pytest.mark.private_data


def test_provenance_only_gpt_claim_grounds_actions_without_templates(
    tmp_path: Path,
) -> None:
    row = deepcopy(read_jsonl(PILOT_ROOT / "planner_trajectory.v2.candidates.jsonl")[0])
    rule_id = "gpt-claim.test.direct-action"
    action_types: set[str] = set()
    memory_ids: set[str] = set()
    source_spans: set[str] = set()
    for actions in row["action_set"].values():
        for action in actions:
            action["guideline_rule_ids"] = [rule_id]
            action_types.add(action["action_type"])
            memory_ids.update(action["supporting_memory_ids"])
            source_spans.update(action["source_spans"])
    for plan in row["accepted_plan_variants"]:
        for action in plan["actions"]:
            for provenance in action["provenance"]:
                provenance["rule_ids"] = [rule_id]

    claim = {
        "schema_version": "gpt_guideline_claim.v2",
        "rule_id": rule_id,
        "claim_id": rule_id,
        "claim_text": "The cited source supports the directly authored action.",
        "provenance_only": True,
        "cancer_family": "lung",
        "guideline_id": "NSCLC_2010",
        "version": "2010",
        "allowed_action_types": sorted(action_types),
        "memory_ids": sorted(memory_ids),
        "source_spans": sorted(source_spans),
    }
    registry = tmp_path / "claims.jsonl"
    write_jsonl(registry, [claim])

    assert (
        validate_grounding_assets(
            [row],
            rule_registry_path=registry,
            memory_dir=PILOT_ROOT / "memory_catalog_mock",
        )
        == []
    )


def test_provenance_only_claim_cannot_smuggle_action_templates(
    tmp_path: Path,
) -> None:
    row = read_jsonl(PILOT_ROOT / "planner_trajectory.v2.candidates.jsonl")[0]
    original_rule = read_jsonl(PILOT_ROOT / "guideline_rules.v2.jsonl")[0]
    claim = {
        "schema_version": "gpt_guideline_claim.v2",
        "rule_id": original_rule["rule_id"],
        "claim_text": "Invalid provenance-only claim.",
        "provenance_only": True,
        "cancer_family": original_rule["cancer_family"],
        "guideline_id": original_rule["guideline_id"],
        "version": original_rule["version"],
        "allowed_action_types": original_rule["allowed_action_types"],
        "memory_ids": original_rule["memory_ids"],
        "source_spans": original_rule["source_spans"],
        "action_templates": original_rule["action_templates"],
    }
    registry = tmp_path / "claims.jsonl"
    write_jsonl(registry, [claim])

    errors = validate_grounding_assets(
        [row],
        rule_registry_path=registry,
        memory_dir=PILOT_ROOT / "memory_catalog_mock",
    )

    assert any("cannot contain action_templates" in error for error in errors)
