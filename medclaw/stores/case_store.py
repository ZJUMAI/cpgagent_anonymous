"""YAML-backed benchmark case loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medclaw.utils import DataIOError, read_yaml, safe_component


class CaseStore:
    """Load structured benchmark cases from a directory tree."""

    def __init__(self, cases_root: Path) -> None:
        self.cases_root = Path(cases_root).resolve()

    def load(self, case_id: str) -> dict[str, Any]:
        case_id = safe_component(case_id, "case_id")
        case_dir = self.cases_root / case_id
        path = case_dir / "case.yaml"
        if not path.is_file():
            if case_dir.is_dir():
                return self._fallback_case(case_id, case_dir)
            raise DataIOError(
                f"Could not load case {case_id!r}: "
                f"case directory does not exist: {case_dir}"
            )

        try:
            data = read_yaml(path)
        except DataIOError as exc:
            raise DataIOError(f"Could not load case {case_id!r}: {exc}") from exc
        if not isinstance(data, dict):
            raise DataIOError(f"Case file must contain a YAML mapping: {path}")
        if data.get("case_id") != case_id:
            raise DataIOError(
                f"Case ID mismatch in {path}: expected {case_id!r}, found {data.get('case_id')!r}"
            )
        return data

    def _fallback_case(self, case_id: str, case_dir: Path) -> dict[str, Any]:
        data: dict[str, Any] = {}
        ct = self._unique_file(
            case_dir / "radiology" / "nifti",
            "*_ct_preprocessed.nii.gz",
        )
        report = self._unique_file(
            case_dir / "reports" / "pathology_reports",
            "*_pathology_report.txt",
        )
        if ct is not None:
            data["ct_preprocessed_uri"] = self._path_for_case_metadata(ct)
            data["ct_format"] = "NIfTI"
        if report is not None:
            data["pathology_report_uri"] = self._path_for_case_metadata(report)

        return {
            "case_id": case_id,
            "description": f"Local benchmark case {case_id}.",
            "data": data,
            "mock_data": {
                "wsi_uri": f"mock://wsi/{case_id}.svs",
                "pathology_report_uri": f"mock://reports/{case_id}.txt",
            },
            "clinical": {
                "cancer": "lung adenocarcinoma",
                "question": "What additional evidence is needed before treatment selection?",
                "guideline": "NCCN NSCLC",
                "task": "Segment the lung tumor and inspect the largest tumor ROI.",
            },
        }

    @staticmethod
    def _unique_file(root: Path, pattern: str) -> Path | None:
        if not root.is_dir():
            return None
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _path_for_case_metadata(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return str(resolved)
