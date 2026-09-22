"""Resolve report / rubric / gold trajectory paths in processed case packs.

Canonical layout is the same for LUNG, UCEC, and NPC::

    processed/{CANCER}/{CASE_ID}/
      evaluation/{CASE_ID}_rubric.json
      evaluation/{CASE_ID}_trajectory.json
      reports/integrated_reports/{CASE_ID}_T0_full_report_zh.md

UCEC / NPC gold trajectories may still be missing; treat that as empty and
skip, do not synthesize one. UCEC currently also has a few compatibility
filenames (``reports/{id}_full_report_zh.md``, ``{id}_T1_rubric.json``).
"""

from __future__ import annotations

from pathlib import Path

KNOWN_CANCERS = ("LUNG", "UCEC", "NPC")
# Gold files are not ready for these packs; do not invent NSCLC trajectories.
PENDING_GOLD_TRAJECTORY = frozenset({"UCEC", "NPC"})


def case_id_of(case_dir: str | Path) -> str:
    return Path(case_dir).name


def infer_cancer_type(case_dir: str | Path) -> str | None:
    """Return LUNG / UCEC / NPC when the parent folder or report layout matches."""

    case_dir = Path(case_dir)
    parent_name = case_dir.parent.name.upper()
    if parent_name in KNOWN_CANCERS:
        return parent_name
    if (
        case_dir
        / "reports"
        / "integrated_reports"
        / f"{case_dir.name}_T0_full_report_zh.md"
    ).is_file():
        return "LUNG"
    if (case_dir / "reports" / f"{case_dir.name}_full_report_zh.md").is_file():
        return "UCEC"
    return None


def evaluation_dir(case_dir: str | Path) -> Path:
    return Path(case_dir) / "evaluation"


def gold_trajectory_write_path(case_dir: str | Path) -> Path:
    case_dir = Path(case_dir)
    return evaluation_dir(case_dir) / f"{case_id_of(case_dir)}_trajectory.json"


def find_gold_trajectory(case_dir: str | Path) -> Path | None:
    """Return ``evaluation/{id}_trajectory.json`` when the file exists."""

    case_dir = Path(case_dir)
    case_id = case_id_of(case_dir)
    for path in (
        evaluation_dir(case_dir) / f"{case_id}_trajectory.json",
        evaluation_dir(case_dir) / "trajectory.json",
        case_dir / "trajectory.json",
    ):
        if path.is_file():
            return path.resolve()
    return None


def find_report_path(case_dir: str | Path, report_path: str | Path | None = None) -> Path:
    if report_path is not None:
        path = Path(report_path)
        if not path.is_file():
            raise FileNotFoundError(f"Case report not found: {path}")
        return path.resolve()

    case_dir = Path(case_dir)
    case_id = case_id_of(case_dir)
    candidates = [
        case_dir / "reports" / "integrated_reports" / f"{case_id}_T0_full_report_zh.md",
        case_dir / "reports" / f"{case_id}_full_report_zh.md",
        case_dir / f"{case_id}_report_zh.md",
    ]
    matches = sorted(
        (case_dir / "reports" / "integrated_reports").glob("*_full_report_zh.md")
    )
    matches.extend(sorted((case_dir / "reports").glob("*_full_report_zh.md")))
    matches.extend(sorted(case_dir.glob("*_report_zh.md")))
    for path in [*candidates, *matches]:
        if path.is_file():
            return path.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No Chinese case report found for {case_id}. Searched: {searched}"
    )


def find_rubric_path(case_dir: str | Path, rubric_path: str | Path | None = None) -> Path:
    """Prefer lung-style ``{id}_rubric.json``; fall back to UCEC ``{id}_T1_rubric.json``."""

    if rubric_path is not None:
        path = Path(rubric_path)
        if not path.is_file():
            raise FileNotFoundError(f"Rubric JSON not found: {path}")
        return path.resolve()

    case_dir = Path(case_dir)
    case_id = case_id_of(case_dir)
    eval_dir = evaluation_dir(case_dir)
    candidates = [
        eval_dir / f"{case_id}_rubric.json",
        eval_dir / f"{case_id}_T1_rubric.json",
        case_dir / f"{case_id}_rubric.json",
    ]
    matches = sorted(
        path
        for path in eval_dir.glob("*_rubric.json")
        if path.is_file() and not path.name.endswith("_R1_rubric.json")
    )
    matches.extend(sorted(case_dir.glob("*_rubric.json")))
    for path in [*candidates, *matches]:
        if path.is_file():
            return path.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No rubric JSON found for {case_id}. Searched: {searched}")


def list_rubric_paths(case_dir: str | Path) -> list[Path]:
    """Primary rubric plus optional UCEC R1 rubric when it exists."""

    primary = find_rubric_path(case_dir)
    extra = evaluation_dir(case_dir) / f"{case_id_of(case_dir)}_R1_rubric.json"
    if extra.is_file() and extra.resolve() != primary:
        return [primary, extra.resolve()]
    return [primary]


def should_generate_gold_trajectory(case_dir: str | Path) -> bool:
    """Do not invent gold trajectories for UCEC / NPC while those files are empty."""

    return infer_cancer_type(case_dir) not in PENDING_GOLD_TRAJECTORY
