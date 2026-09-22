from __future__ import annotations

from pathlib import Path

from medclaw.stores.case_store import CaseStore


def test_case_store_loads_yaml_metadata(tmp_path: Path) -> None:
    case_dir = tmp_path / "CASE-1"
    case_dir.mkdir(parents=True)
    (case_dir / "case.yaml").write_text(
        "\n".join(
            [
                "case_id: CASE-1",
                "description: explicit metadata",
                "data: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = CaseStore(tmp_path).load("CASE-1")

    assert case["case_id"] == "CASE-1"
    assert case["description"] == "explicit metadata"


def test_case_store_falls_back_to_modality_layout_without_yaml(tmp_path: Path) -> None:
    case_dir = tmp_path / "CASE-2"
    ct = case_dir / "radiology" / "nifti" / "CASE-2_T0_ct_preprocessed.nii.gz"
    report = (
        case_dir
        / "reports"
        / "pathology_reports"
        / "CASE-2_T0_pathology_report.txt"
    )
    ct.parent.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    ct.write_bytes(b"nifti")
    report.write_text("diagnosis", encoding="utf-8")

    case = CaseStore(tmp_path).load("CASE-2")

    assert case["case_id"] == "CASE-2"
    assert case["data"]["ct_preprocessed_uri"].endswith(
        "CASE-2_T0_ct_preprocessed.nii.gz"
    )
    assert case["data"]["pathology_report_uri"].endswith(
        "CASE-2_T0_pathology_report.txt"
    )
    assert case["mock_data"]["wsi_uri"] == "mock://wsi/CASE-2.svs"
