"""Build guideline-grounded trajectory artifacts for local cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medclaw.trajectory import build_guideline_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Path to the patient markdown report. Defaults to 病人数据/patient_md/{case_id}/{case_id}.md.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Destination JSON. Defaults to examples/cases/{case_id}/evaluation/{case_id}_trajectory.json.",
    )
    parser.add_argument("--clinical-path", type=Path)
    parser.add_argument("--guideline-id", default="NSCLC_2010")
    parser.add_argument("--guideline-version", default="2010")
    parser.add_argument("--decision-date", default="2010-12-31")
    parser.add_argument("--diagnosis-year", type=int)
    args = parser.parse_args()

    report_path = args.report_path or Path("..") / "病人数据" / "patient_md" / args.case_id / f"{args.case_id}.md"
    output_path = args.output_path or (
        Path("examples") / "cases" / args.case_id / "evaluation" / f"{args.case_id}_trajectory.json"
    )
    data = build_guideline_trajectory(
        args.case_id,
        report_path,
        guideline_id=args.guideline_id,
        guideline_version=args.guideline_version,
        decision_date=args.decision_date,
        diagnosis_year=args.diagnosis_year,
        clinical_path=args.clinical_path,
        output_path=output_path,
    )
    print(
        f"Wrote {output_path} with {len(data['patient_event_table'])} events "
        f"and {len(data['trajectory'])} trajectory steps."
    )


if __name__ == "__main__":
    main()
