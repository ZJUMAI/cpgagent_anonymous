"""Shared defaults for the latent guideline-memory planner."""

from __future__ import annotations

DEFAULT_GUIDELINE_DIR = "data/guidelines"
DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-9B"
DEFAULT_MEMORY_TOKENS = 64

MEMORY_TOKEN = "[MEM]"
TASK_TOKENS = {
    "AE": "[AE]",
    "RETRIEVE": "[RETRIEVE]",
    "CONTINUE": "[CONTINUE]",
    "PLAN": "[PLAN]",
}
SPECIAL_TOKENS = [MEMORY_TOKEN, *TASK_TOKENS.values()]

DEFAULT_TASK_RATIOS = {
    "AE": 5 / 13,
    "RETRIEVE": 5 / 13,
    "CONTINUE": 3 / 13,
}

PATIENT_STATE_SCHEMA_VERSION = "patient_state.v2"
PLANNER_ACTION_SCHEMA_VERSION = "planner_action.v2"
PLANNER_TRAJECTORY_SCHEMA_VERSION = "planner_trajectory.v2"

SUPPORTED_PLANNER_CANCER_FAMILIES = (
    "lung",
    "endometrial",
    "nasopharyngeal",
)

PLANNER_OUTPUT_FIELDS = (
    "schema_version",
    "current_phase",
    "proposed_phase",
    "missing_information",
    "actions",
    "blocked_actions",
    "should_stop",
    "reason",
)
