"""Adapter helpers for passing planner output to the MedClaw agent layer."""

from __future__ import annotations

from typing import Any, Mapping


def to_medclaw_plan(planner_output: Mapping[str, Any]) -> dict[str, Any]:
    """Expose a parsed plan without applying the runtime V2 validator."""

    parsed = dict(planner_output)
    actions = parsed.get("actions")
    if not isinstance(actions, list):
        actions = []
    return {
        "planner": "latent_guideline_memory_v2",
        "schema_version": parsed.get("schema_version"),
        "output_format": "json",
        "current_phase": parsed.get("current_phase"),
        "proposed_phase": parsed.get("proposed_phase"),
        "actions": actions,
        "suggested_skills": list(
            dict.fromkeys(
                skill
                for action in actions
                if isinstance(action, Mapping)
                for skill in action.get("required_skills", [])
            )
        ),
        "missing_information": parsed.get("missing_information", []),
        "blocked_actions": parsed.get("blocked_actions", []),
        "should_stop": parsed.get("should_stop", False),
        "reason": parsed.get("reason"),
    }
