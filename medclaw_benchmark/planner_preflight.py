"""Fast, model-free checks for Planner V2 benchmark cases."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from guideline_planner.release import ResolvedPlannerRelease
from medclaw.utils import read_json, read_yaml
from medclaw_benchmark.batch_runner import _resolve_rubric_path


def preflight_planner_case(
    case_dir: str | Path,
    release: ResolvedPlannerRelease,
) -> dict[str, Any]:
    """Validate action scope and rubric guideline version before model loading."""

    root = Path(case_dir).resolve()
    family, subtype = _infer_case_scope(root)
    release.require_supported_action(family, subtype)
    expected = release.guideline_default(family, subtype)
    rubric_path = _resolve_rubric_path(root, None)
    rubric = read_json(rubric_path)
    observed_versions = _rubric_guideline_versions(rubric)
    expected_version = str(expected["version"])
    if observed_versions and expected_version not in observed_versions:
        raise ValueError(
            f"Case {root.name!r} rubric references guideline version(s) "
            f"{sorted(observed_versions)}, but Planner release requires "
            f"{expected['guideline_id']}@{expected_version}. Update the rubric or "
            "use a release whose decision-time guideline matches it."
        )
    return {
        "case_id": root.name,
        "cancer_family": family,
        "disease_subtype": subtype,
        "guideline_id": expected["guideline_id"],
        "version": expected_version,
        "decision_date": expected["decision_date"],
        "rubric_path": str(rubric_path),
        "rubric_guideline_versions": sorted(observed_versions),
        "status": "ready",
    }


def _infer_case_scope(case_dir: Path) -> tuple[str, str]:
    sources: list[str] = [case_dir.name]
    hidden_path = case_dir / "hidden_state.json"
    if hidden_path.is_file():
        hidden = read_json(hidden_path)
        if isinstance(hidden, Mapping):
            sources.append(json.dumps(hidden, ensure_ascii=False))
    yaml_path = case_dir / "case.yaml"
    if yaml_path.is_file():
        value = read_yaml(yaml_path)
        sources.append(json.dumps(value, ensure_ascii=False))
    for path in sorted(case_dir.rglob("*.md"))[:8]:
        sources.append(path.read_text(encoding="utf-8", errors="ignore")[:20000])
    text = " ".join(sources)
    lowered = text.lower()
    if any(token in lowered for token in ("ucec", "endometr", "子宫内膜")):
        return "endometrial", "ucec"
    if any(token in lowered for token in ("nasopharyn", "\"npc\"", "鼻咽")):
        return "nasopharyngeal", "npc"
    if (
        (re.search(r"\bsclc\b", lowered) and "nsclc" not in lowered)
        or "small cell lung" in lowered
        or "小细胞肺" in text
    ):
        return "lung", "sclc"
    if any(
        token in lowered
        for token in ("nsclc", "luad", "lung", "肺癌", "肺腺癌")
    ):
        return "lung", "nsclc"
    raise ValueError(
        f"Cannot infer Planner action scope for case {case_dir.name!r}; add "
        "cancer_type/project_id to hidden_state.json or clinical.cancer to case.yaml."
    )


def _rubric_guideline_versions(value: Any) -> set[str]:
    """Extract only years attached to explicit guideline-family references."""

    texts: list[str] = []

    def visit(item: Any, key: str = "") -> None:
        if isinstance(item, Mapping):
            for child_key, child in item.items():
                visit(child, str(child_key))
        elif isinstance(item, list):
            for child in item:
                visit(child, key)
        elif isinstance(item, str):
            combined = f"{key} {item}"
            lowered = combined.lower()
            if any(
                marker in lowered
                for marker in ("guideline", "nccn", "csco", "指南")
            ):
                texts.append(combined)

    visit(value)
    versions: set[str] = set()
    for text in texts:
        versions.update(re.findall(r"(?<!\d)(20\d{2})(?!\d)", text))
    return versions
