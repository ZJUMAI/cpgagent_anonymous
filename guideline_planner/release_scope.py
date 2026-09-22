"""Versioned Planner V2 release-scope contracts.

The patient-state schema intentionally supports more diseases than any one
trained Planner release.  A release scope makes that distinction explicit and
content-addressable so admission, training, routing, and runtime artifacts can
all bind to the same action-supervision boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from guideline_planner.constants import SUPPORTED_PLANNER_CANCER_FAMILIES

RELEASE_SCOPE_SCHEMA_VERSION = "planner_release_scope.v2"


PLANNER_V2_LUNG_ENDOMETRIAL_RELEASE_SCOPE: dict[str, Any] = {
    "schema_version": RELEASE_SCOPE_SCHEMA_VERSION,
    "scope_id": "planner_v2_lung_endometrial",
    "schema_cancer_families": list(SUPPORTED_PLANNER_CANCER_FAMILIES),
    "action_targets": [
        {
            "cancer_family": "endometrial",
            "disease_subtypes": ["ucec"],
            "guideline_ids": ["CSCO子宫内膜癌2023"],
        },
        {
            "cancer_family": "lung",
            "disease_subtypes": ["nsclc"],
            "guideline_ids": ["NSCLC_2010"],
        },
    ],
    "memory_guidelines": [
        {
            "cancer_family": "endometrial",
            "disease_subtypes": ["ucec"],
            "guideline_id": "CSCO子宫内膜癌2023",
            "version": "2023",
        },
        {
            "cancer_family": "lung",
            "disease_subtypes": ["nsclc"],
            "guideline_id": "NSCLC_2010",
            "version": "2010",
        },
        {
            "cancer_family": "lung",
            "disease_subtypes": ["sclc"],
            "guideline_id": "SCLC_2010",
            "version": "2010",
        },
        {
            "cancer_family": "nasopharyngeal",
            "disease_subtypes": ["npc"],
            "guideline_id": "CSCO鼻咽癌2022",
            "version": "2022",
        },
    ],
    "deferred_action_targets": [
        {
            "cancer_family": "lung",
            "disease_subtypes": ["sclc"],
            "reason": "No reviewed SCLC Planner trajectories are in this release.",
        },
        {
            "cancer_family": "nasopharyngeal",
            "disease_subtypes": ["npc"],
            "reason": "Patient trajectories are deferred until NPC case data arrives.",
        },
    ],
}


PLANNER_V2_LUNG_ENDOMETRIAL_NPC_RELEASE_SCOPE: dict[str, Any] = {
    "schema_version": RELEASE_SCOPE_SCHEMA_VERSION,
    "scope_id": "planner_v2_lung_endometrial_npc_gpt",
    "schema_cancer_families": list(SUPPORTED_PLANNER_CANCER_FAMILIES),
    "action_targets": [
        {
            "cancer_family": "endometrial",
            "disease_subtypes": ["ucec"],
            "guideline_ids": ["CSCO子宫内膜癌2023"],
        },
        {
            "cancer_family": "lung",
            "disease_subtypes": ["nsclc"],
            "guideline_ids": ["NSCLC_2010"],
        },
        {
            "cancer_family": "nasopharyngeal",
            "disease_subtypes": ["npc"],
            "guideline_ids": ["CSCO鼻咽癌2022"],
        },
    ],
    "memory_guidelines": deepcopy(
        PLANNER_V2_LUNG_ENDOMETRIAL_RELEASE_SCOPE["memory_guidelines"]
    ),
    "deferred_action_targets": [
        {
            "cancer_family": "lung",
            "disease_subtypes": ["sclc"],
            "reason": "No reviewed SCLC Planner trajectories are in this release.",
        }
    ],
    "admission_exceptions": [
        {
            "criterion": "conditional_status",
            "cancer_family": "nasopharyngeal",
            "condition_satisfied": True,
            "reason": (
                "The 39 clinician-reviewed NPC cases contain conditional actions "
                "only while their treatment prerequisites remain unresolved."
            ),
        },
        {
            "criterion": "phase_minimum",
            "cancer_family": "nasopharyngeal",
            "phase": "risk_or_biomarker_stratification",
            "reason": "Observed train/test transitions are 12/3 in the fixed 27/6/6 split.",
        },
        {
            "criterion": "phase_minimum",
            "cancer_family": "nasopharyngeal",
            "phase": "treatment_selection",
            "reason": "Observed train/test transitions are 4/1 in the fixed 27/6/6 split.",
        },
    ],
}


def schema_wide_release_scope() -> dict[str, Any]:
    """Return the backwards-compatible scope used by unscoped generic builds.

    New formal releases should always provide an explicit scope.  Keeping a
    schema-wide default avoids silently changing the semantics of existing
    library callers while still serializing the resolved boundary in their
    manifests.
    """

    return normalize_release_scope(
        {
            "schema_version": RELEASE_SCOPE_SCHEMA_VERSION,
            "scope_id": "schema_wide",
            "schema_cancer_families": list(SUPPORTED_PLANNER_CANCER_FAMILIES),
            "action_targets": [
                {
                    "cancer_family": family,
                    "disease_subtypes": ["*"],
                    "guideline_ids": ["*"],
                }
                for family in SUPPORTED_PLANNER_CANCER_FAMILIES
            ],
            "memory_guidelines": [],
            "deferred_action_targets": [],
        }
    )


def load_release_scope(
    value: str | Path | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Load and canonicalize a release scope.

    ``None`` deliberately resolves to the schema-wide compatibility scope.
    Formal dataset builds bind the returned object and hash into their
    manifest, so training never has to infer the boundary again.
    """

    if value is None:
        return schema_wide_release_scope()
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise ValueError(f"Planner release scope does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("Planner release scope must be a JSON object.")
        return normalize_release_scope(payload)
    return normalize_release_scope(value)


def normalize_release_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a scope and return its deterministic canonical representation."""

    if str(scope.get("schema_version") or "") != RELEASE_SCOPE_SCHEMA_VERSION:
        raise ValueError(
            f"release scope schema_version must be {RELEASE_SCOPE_SCHEMA_VERSION!r}."
        )
    scope_id = str(scope.get("scope_id") or "").strip()
    if not scope_id:
        raise ValueError("release scope requires a non-empty scope_id.")

    schema_families = _unique_strings(
        scope.get("schema_cancer_families"),
        label="schema_cancer_families",
    )
    expected_schema_families = set(SUPPORTED_PLANNER_CANCER_FAMILIES)
    if set(schema_families) != expected_schema_families:
        raise ValueError(
            "release scope schema_cancer_families must exactly match the V2 schema: "
            + ", ".join(sorted(expected_schema_families))
        )

    action_targets = _normalize_targets(
        scope.get("action_targets"),
        label="action_targets",
        schema_families=set(schema_families),
        require_guidelines=True,
    )
    if not action_targets:
        raise ValueError("release scope requires at least one action target.")
    deferred_targets = _normalize_targets(
        scope.get("deferred_action_targets", []),
        label="deferred_action_targets",
        schema_families=set(schema_families),
        require_guidelines=False,
    )
    memory_guidelines = _normalize_memory_guidelines(
        scope.get("memory_guidelines", []),
        schema_families=set(schema_families),
    )
    admission_exceptions = _normalize_admission_exceptions(
        scope.get("admission_exceptions", []),
        schema_families=set(schema_families),
    )

    active_pairs = {
        (target["cancer_family"], subtype)
        for target in action_targets
        for subtype in target["disease_subtypes"]
    }
    deferred_pairs = {
        (target["cancer_family"], subtype)
        for target in deferred_targets
        for subtype in target["disease_subtypes"]
    }
    overlap = active_pairs.intersection(deferred_pairs)
    if overlap:
        raise ValueError(
            "Action and deferred targets overlap: "
            + ", ".join(f"{family}/{subtype}" for family, subtype in sorted(overlap))
        )

    normalized_scope = {
        "schema_version": RELEASE_SCOPE_SCHEMA_VERSION,
        "scope_id": scope_id,
        "schema_cancer_families": sorted(schema_families),
        "action_targets": action_targets,
        "memory_guidelines": memory_guidelines,
        "deferred_action_targets": deferred_targets,
    }
    # Keep hashes stable for existing release scopes that predate explicit
    # admission exceptions. This field is content-bearing and is only emitted
    # when a release actually declares one.
    if admission_exceptions:
        normalized_scope["admission_exceptions"] = admission_exceptions
    return normalized_scope


def _normalize_admission_exceptions(
    value: Any,
    *,
    schema_families: set[str],
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("release scope admission_exceptions must be an array.")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise TypeError(f"admission_exceptions[{index}] must be an object.")
        criterion = str(raw.get("criterion") or "").strip()
        if criterion not in {"conditional_status", "phase_minimum"}:
            raise ValueError(
                f"admission_exceptions[{index}] has unsupported criterion {criterion!r}."
            )
        family = str(raw.get("cancer_family") or "").strip().lower()
        if family not in schema_families:
            raise ValueError(
                f"admission_exceptions[{index}] has unknown cancer_family {family!r}."
            )
        reason = str(raw.get("reason") or "").strip()
        if not reason:
            raise ValueError(f"admission_exceptions[{index}] requires a reason.")
        item: dict[str, Any] = {
            "criterion": criterion,
            "cancer_family": family,
            "reason": reason,
        }
        if criterion == "conditional_status":
            status = raw.get("condition_satisfied")
            if not isinstance(status, bool):
                raise TypeError(
                    f"admission_exceptions[{index}] condition_satisfied must be boolean."
                )
            item["condition_satisfied"] = status
        else:
            phase = str(raw.get("phase") or "").strip()
            if not phase:
                raise ValueError(
                    f"admission_exceptions[{index}] phase_minimum requires phase."
                )
            item["phase"] = phase
        result.append(item)
    return sorted(
        result,
        key=lambda item: (
            item["criterion"],
            item["cancer_family"],
            str(item.get("phase") or ""),
            str(item.get("condition_satisfied") or ""),
        ),
    )


def release_scope_hash(scope: Mapping[str, Any]) -> str:
    normalized = normalize_release_scope(scope)
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def action_target_families(scope: Mapping[str, Any]) -> tuple[str, ...]:
    normalized = normalize_release_scope(scope)
    return tuple(
        sorted({str(item["cancer_family"]) for item in normalized["action_targets"]})
    )


def action_target_allows(
    scope: Mapping[str, Any],
    *,
    cancer_family: str,
    disease_subtype: str | None,
    guideline_ids: set[str] | None = None,
) -> bool:
    normalized = normalize_release_scope(scope)
    subtype = str(disease_subtype or "").strip().lower()
    for target in normalized["action_targets"]:
        if target["cancer_family"] != cancer_family:
            continue
        if "*" not in target["disease_subtypes"] and subtype not in target["disease_subtypes"]:
            continue
        permitted_guides = set(target["guideline_ids"])
        if guideline_ids and "*" not in permitted_guides and not guideline_ids.issubset(permitted_guides):
            continue
        return True
    return False


def _normalize_targets(
    value: Any,
    *,
    label: str,
    schema_families: set[str],
    require_guidelines: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"release scope {label} must be a list.")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise TypeError(f"release scope {label}[{index}] must be an object.")
        family = str(raw.get("cancer_family") or "").strip().lower()
        if family not in schema_families:
            raise ValueError(f"release scope {label}[{index}] has unsupported family {family!r}.")
        subtypes = _unique_strings(
            raw.get("disease_subtypes"),
            label=f"{label}[{index}].disease_subtypes",
            lowercase=True,
        )
        if "*" in subtypes and len(subtypes) != 1:
            raise ValueError(f"release scope {label}[{index}] cannot mix '*' with subtypes.")
        key = (family, tuple(subtypes))
        if key in seen:
            raise ValueError(f"release scope {label} contains duplicate target {key!r}.")
        seen.add(key)
        item: dict[str, Any] = {
            "cancer_family": family,
            "disease_subtypes": subtypes,
        }
        if require_guidelines:
            item["guideline_ids"] = _unique_strings(
                raw.get("guideline_ids"),
                label=f"{label}[{index}].guideline_ids",
            )
        reason = str(raw.get("reason") or "").strip()
        if reason:
            item["reason"] = reason
        normalized.append(item)
    return sorted(
        normalized,
        key=lambda item: (item["cancer_family"], tuple(item["disease_subtypes"])),
    )


def _normalize_memory_guidelines(
    value: Any,
    *,
    schema_families: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("release scope memory_guidelines must be a list.")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise TypeError(f"release scope memory_guidelines[{index}] must be an object.")
        family = str(raw.get("cancer_family") or "").strip().lower()
        if family not in schema_families:
            raise ValueError(
                f"release scope memory_guidelines[{index}] has unsupported family {family!r}."
            )
        guideline_id = str(raw.get("guideline_id") or "").strip()
        version = str(raw.get("version") or "").strip()
        if not guideline_id or not version:
            raise ValueError(
                f"release scope memory_guidelines[{index}] requires guideline_id and version."
            )
        key = (guideline_id, version)
        if key in seen:
            raise ValueError(f"release scope has duplicate memory guideline {key!r}.")
        seen.add(key)
        normalized.append(
            {
                "cancer_family": family,
                "disease_subtypes": _unique_strings(
                    raw.get("disease_subtypes"),
                    label=f"memory_guidelines[{index}].disease_subtypes",
                    lowercase=True,
                ),
                "guideline_id": guideline_id,
                "version": version,
            }
        )
    return sorted(normalized, key=lambda item: (item["guideline_id"], item["version"]))


def _unique_strings(
    value: Any,
    *,
    label: str,
    lowercase: bool = False,
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"release scope {label} must be a non-empty string list.")
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if lowercase:
            text = text.lower()
        if not text:
            raise ValueError(f"release scope {label} contains an empty value.")
        result.append(text)
    if len(set(result)) != len(result):
        raise ValueError(f"release scope {label} contains duplicates.")
    return sorted(result)


def copy_release_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    """Return a defensive canonical copy for artifact serialization."""

    return deepcopy(normalize_release_scope(scope))
