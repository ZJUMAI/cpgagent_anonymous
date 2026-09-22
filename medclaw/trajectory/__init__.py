"""Guideline-grounded patient trajectory construction."""

from medclaw.trajectory.action_set import (
    assemble_action_set,
    canonicalize_guideline_trajectory,
    canonicalize_trajectory_step,
    flatten_visible_state,
    resolve_action_set,
    resolve_verifications,
)
from medclaw.trajectory.builder import build_guideline_trajectory
from medclaw.trajectory.report import extract_patient_events
from medclaw.trajectory.rubric import nsclc_2010_rubric_rules
from medclaw.trajectory.verification import verify_action

__all__ = [
    "assemble_action_set",
    "build_guideline_trajectory",
    "canonicalize_guideline_trajectory",
    "canonicalize_trajectory_step",
    "extract_patient_events",
    "flatten_visible_state",
    "nsclc_2010_rubric_rules",
    "resolve_action_set",
    "resolve_verifications",
    "verify_action",
]
