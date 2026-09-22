#!/usr/bin/env python3
"""Run case-level quality control across benchmark cohorts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_ROOT = ROOT / "病人数据" / "both_report_md"
DEFAULT_MOLECULAR_ROOT = ROOT / "病人数据" / "molecule_md"
DEFAULT_CLINICAL_TSV = (
    ROOT / "病人数据" / "raw" / "clinical.project-tcga-luad.2025-10-29" / "clinical.tsv"
)
DEFAULT_CASES_ROOT = Path(__file__).resolve().parents[1] / "examples" / "cases"
DEFAULT_OUTPUT_DIR = ROOT / "病人数据" / "qc_output"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medclaw.validation.case_qc import run_case_qc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run case-level QC / data completeness audit for benchmark cases."
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help="Directory containing per-case markdown reports.",
    )
    parser.add_argument(
        "--molecular-root",
        type=Path,
        default=DEFAULT_MOLECULAR_ROOT,
        help="Directory containing per-case molecular markdown reports.",
    )
    parser.add_argument(
        "--clinical-tsv",
        type=Path,
        default=DEFAULT_CLINICAL_TSV,
        help="Optional structured clinical TSV for field enrichment.",
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=DEFAULT_CASES_ROOT,
        help="Optional benchmark case.yaml root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for QC outputs.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Restrict QC to one or more case IDs.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a compact JSON summary to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_case_qc(
        output_dir=args.output_dir,
        reports_root=args.reports_root if args.reports_root.exists() else None,
        molecular_root=args.molecular_root if args.molecular_root.exists() else None,
        clinical_tsv=args.clinical_tsv if args.clinical_tsv.exists() else None,
        cases_root=args.cases_root if args.cases_root.exists() else None,
        case_ids=args.case_ids,
    )

    counts: dict[str, int] = {}
    by_cancer: dict[str, dict[str, int]] = {}
    for result in results:
        counts[result.qc_class] = counts.get(result.qc_class, 0) + 1
        bucket = by_cancer.setdefault(result.cancer_type, {})
        bucket[result.qc_class] = bucket.get(result.qc_class, 0) + 1

    print(f"QC completed for {len(results)} cases.")
    print(f"Output directory: {args.output_dir}")
    print(
        "Class counts: "
        f"A={counts.get('A_complete_modern', 0)}, "
        f"B={counts.get('B_usable_incomplete', 0)}, "
        f"C={counts.get('C_skeleton_only', 0)}, "
        f"D={counts.get('D_exclude', 0)}"
    )
    for cancer_type, bucket in sorted(by_cancer.items()):
        print(
            f"  {cancer_type}: "
            f"A={bucket.get('A_complete_modern', 0)}, "
            f"B={bucket.get('B_usable_incomplete', 0)}, "
            f"C={bucket.get('C_skeleton_only', 0)}, "
            f"D={bucket.get('D_exclude', 0)}"
        )

    if args.print_summary:
        print(
            json.dumps(
                {
                    "total_cases": len(results),
                    "class_counts": counts,
                    "by_cancer_type": by_cancer,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
