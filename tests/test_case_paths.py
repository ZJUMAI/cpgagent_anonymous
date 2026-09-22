from __future__ import annotations

import json
from pathlib import Path

from medclaw_benchmark.case_paths import (
    find_gold_trajectory,
    find_report_path,
    find_rubric_path,
    gold_trajectory_write_path,
    infer_cancer_type,
    list_rubric_paths,
    should_generate_gold_trajectory,
)


def _write(path: Path, text: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _lung_style_pack(root: Path, cancer: str, case_id: str, *, with_trajectory: bool) -> Path:
    case_dir = root / "processed" / cancer / case_id
    _write(
        case_dir / "reports" / "integrated_reports" / f"{case_id}_T0_full_report_zh.md",
        f"# {cancer}\n",
    )
    _write(case_dir / "evaluation" / f"{case_id}_rubric.json", json.dumps({"case_id": case_id}))
    if with_trajectory:
        _write(
            case_dir / "evaluation" / f"{case_id}_trajectory.json",
            json.dumps({"case_id": case_id}),
        )
    return case_dir


def test_lung_paths_use_evaluation_trajectory(tmp_path: Path) -> None:
    case_id = "TCGA-38-4625"
    case_dir = _lung_style_pack(tmp_path, "LUNG", case_id, with_trajectory=True)
    _write(case_dir / "trajectory.json", json.dumps({"legacy": True}))
    report = (
        case_dir / "reports" / "integrated_reports" / f"{case_id}_T0_full_report_zh.md"
    )
    rubric = case_dir / "evaluation" / f"{case_id}_rubric.json"
    gold = case_dir / "evaluation" / f"{case_id}_trajectory.json"

    assert infer_cancer_type(case_dir) == "LUNG"
    assert find_report_path(case_dir) == report.resolve()
    assert find_rubric_path(case_dir) == rubric.resolve()
    assert find_gold_trajectory(case_dir) == gold.resolve()
    assert gold_trajectory_write_path(case_dir) == gold
    assert should_generate_gold_trajectory(case_dir) is True


def test_npc_uses_same_paths_as_lung_and_skips_empty_trajectory(tmp_path: Path) -> None:
    case_id = "M512268"
    case_dir = _lung_style_pack(tmp_path, "NPC", case_id, with_trajectory=False)
    report = (
        case_dir / "reports" / "integrated_reports" / f"{case_id}_T0_full_report_zh.md"
    )
    rubric = case_dir / "evaluation" / f"{case_id}_rubric.json"

    assert infer_cancer_type(case_dir) == "NPC"
    assert find_report_path(case_dir) == report.resolve()
    assert find_rubric_path(case_dir) == rubric.resolve()
    assert find_gold_trajectory(case_dir) is None
    assert gold_trajectory_write_path(case_dir) == (
        case_dir / "evaluation" / f"{case_id}_trajectory.json"
    )
    assert should_generate_gold_trajectory(case_dir) is False


def test_ucec_empty_trajectory_is_skipped_but_same_path(tmp_path: Path) -> None:
    case_id = "01487873"
    case_dir = _lung_style_pack(tmp_path, "UCEC", case_id, with_trajectory=False)

    assert infer_cancer_type(case_dir) == "UCEC"
    assert find_gold_trajectory(case_dir) is None
    assert should_generate_gold_trajectory(case_dir) is False
    assert gold_trajectory_write_path(case_dir) == (
        case_dir / "evaluation" / f"{case_id}_trajectory.json"
    )


def test_ucec_reads_trajectory_when_file_exists(tmp_path: Path) -> None:
    case_id = "03850333"
    case_dir = _lung_style_pack(tmp_path, "UCEC", case_id, with_trajectory=True)
    gold = case_dir / "evaluation" / f"{case_id}_trajectory.json"

    assert find_gold_trajectory(case_dir) == gold.resolve()
    assert should_generate_gold_trajectory(case_dir) is False


def test_ucec_legacy_t1_r1_rubric_still_resolves(tmp_path: Path) -> None:
    case_id = "03850333"
    case_dir = tmp_path / "processed" / "UCEC" / case_id
    report = case_dir / "reports" / f"{case_id}_full_report_zh.md"
    t1 = case_dir / "evaluation" / f"{case_id}_T1_rubric.json"
    r1 = case_dir / "evaluation" / f"{case_id}_R1_rubric.json"
    _write(report, "# ucec\n")
    _write(t1, json.dumps({"view": "T1"}))
    _write(r1, json.dumps({"view": "R1"}))

    assert find_report_path(case_dir) == report.resolve()
    assert find_rubric_path(case_dir) == t1.resolve()
    assert list_rubric_paths(case_dir) == [t1.resolve(), r1.resolve()]
    assert find_gold_trajectory(case_dir) is None
