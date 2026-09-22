from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_corrected_public_lung_reference_pack_when_present() -> None:
    manifest_path = ROOT / "evaluation" / "references" / "manifest.json"
    if not manifest_path.is_file():
        # The source repository does not contain the generated release tree.
        return
    manifest = _read(manifest_path)
    assert manifest["schema_version"] == "portable_judge_references.v2"
    assert manifest["case_count"] == 30
    assert manifest["guideline_target"] == {
        "decision_date": "2010-12-31",
        "evaluation_mode": "fixed_historical_guideline",
        "guideline_id": "NSCLC_2010",
        "version": "2010",
    }

    for item in manifest["cases"]:
        case_id = item["case_id"]
        rubric_path = manifest_path.parent / item["rubric_path"]
        trajectory_path = manifest_path.parent / item["trajectory_path"]
        clinical_path = ROOT / "data" / "LUNG" / case_id / "clinical" / "clinical.json"
        rubric = _read(rubric_path)
        trajectory = _read(trajectory_path)
        clinical = _read(clinical_path)

        assert _sha256(rubric_path) == item["rubric_sha256"]
        assert _sha256(trajectory_path) == item["trajectory_sha256"]
        assert rubric["case_id"] == trajectory["case_id"] == clinical["case_id"] == case_id
        assert trajectory["guideline_id"] == "NSCLC_2010"
        assert trajectory["guideline_version"] == "2010"
        known = trajectory["canonical_state"]["known_facts"]
        assert known["sex"] == clinical["sex"]
        for trajectory_key, clinical_key in (
            ("T", "pathologic_t_stage"),
            ("N", "pathologic_n_stage"),
            ("M", "pathologic_m_stage"),
        ):
            assert known[trajectory_key].upper() == clinical[clinical_key].upper()

        serialized = json.dumps(trajectory, ensure_ascii=False)
        assert "CSCO_NSCLC_" + "2025" not in serialized
        actual_tnm = "".join(known[key] for key in ("T", "N", "M"))
        treatment_actions = [
            action["action"]
            for step in trajectory["trajectory"]
            if step["phase"] == "initial_treatment_decision"
            for action in step["action_set"]["required"]
        ]
        if "X" in actual_tnm.upper():
            assert (
                "complete unknown TNM components before deriving AJCC stage group "
                f"(currently {actual_tnm})"
            ) in treatment_actions
        else:
            assert f"derive AJCC stage group from {actual_tnm}" in treatment_actions


def test_demo_uses_nccn_2010_and_t3n0m0_when_present() -> None:
    demo = ROOT / "examples" / "cases" / "TCGA-38-4626"
    trajectory_path = demo / "evaluation" / "TCGA-38-4626_trajectory.json"
    if not trajectory_path.is_file():
        return
    trajectory = _read(trajectory_path)
    rubric = _read(demo / "evaluation" / "TCGA-38-4626_rubric.json")
    known = trajectory["canonical_state"]["known_facts"]
    assert (known["T"], known["N"], known["M"]) == ("T3", "N0", "M0")
    assert trajectory["guideline_id"] == "NSCLC_2010"
    assert trajectory["guideline_version"] == "2010"
    assert rubric["source_files"]["matched_guideline"] == "NSCLC_2010"
    assert "CSCO 2025" not in json.dumps(rubric, ensure_ascii=False)
