"""Case-level aggregation of findings produced by skills."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from medclaw.utils import read_json, safe_component, write_json


class EvidenceBoard:
    """Maintain a durable summary of evidence collected for one case."""

    def __init__(
        self,
        case_id: str,
        runs_root: Path,
        *,
        missing_information: Iterable[str] | None = None,
        load_existing: bool = True,
    ) -> None:
        self.case_id = safe_component(case_id, "case_id")
        self.runs_root = Path(runs_root).resolve()
        self.path = self.runs_root / self.case_id / "evidence_board.json"
        self.evidence: list[dict[str, Any]] = []
        self.missing_information: list[str] = list(missing_information or [])

        if load_existing and self.path.is_file():
            self._load()

    def add_result(self, skill_name: str, result: Mapping[str, Any]) -> None:
        if result.get("status") != "success":
            return

        findings = result.get("findings", {})
        if not isinstance(findings, Mapping):
            findings = {"value": findings}
        summary = findings.get("summary")
        if not isinstance(summary, str) or not summary:
            summary = json.dumps(findings, ensure_ascii=False, sort_keys=True)

        artifacts = result.get("artifacts", [])
        self.evidence.append(
            {
                "source": skill_name,
                "summary": summary,
                "findings": dict(findings),
                "artifacts": list(artifacts) if isinstance(artifacts, list) else [],
                "call_id": result.get("call_id"),
            }
        )

        reported_missing = findings.get("missing_information", [])
        if isinstance(reported_missing, list):
            for item in reported_missing:
                if isinstance(item, str) and item not in self.missing_information:
                    self.missing_information.append(item)
        self.save()

    def summary(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "case_id": self.case_id,
                "evidence": self.evidence,
                "missing_information": self.missing_information,
            }
        )

    def save(self) -> Path:
        write_json(self.path, self.summary())
        return self.path

    def _load(self) -> None:
        data = read_json(self.path)
        if not isinstance(data, dict):
            raise ValueError(f"Evidence board must contain a JSON object: {self.path}")
        if data.get("case_id") != self.case_id:
            raise ValueError(f"Evidence board case_id mismatch: {self.path}")
        evidence = data.get("evidence", [])
        missing = data.get("missing_information", [])
        if not isinstance(evidence, list) or not isinstance(missing, list):
            raise ValueError(f"Evidence board has invalid list fields: {self.path}")
        self.evidence = evidence
        self.missing_information = missing
