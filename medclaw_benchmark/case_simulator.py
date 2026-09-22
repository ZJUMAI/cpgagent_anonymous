"""Phase-gated access to benchmark case data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from medclaw.utils import read_json


PHASES = (
    "diagnosis_phase",
    "pathology_text_phase",
    "pathology_image_phase",
    "radiology_phase",
    "staging_phase",
    "treatment_phase",
    "progression_phase",
    "outcome_phase",
)


@dataclass
class CaseSimulator:
    """Expose hidden case information through stage-aware benchmark tools."""

    case_dir: str | Path
    _phase: str = "diagnosis_phase"
    _hidden_state: dict[str, Any] = field(init=False)
    _release_policy: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.case_dir = Path(self.case_dir).resolve()
        self._hidden_state = self._read("hidden_state.json")
        self._release_policy = self._read("release_policy.json")

    @property
    def case_id(self) -> str:
        return str(self._hidden_state["case_id"])

    def get_initial_observation(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "phase": self._phase,
            "visible_information": dict(self._hidden_state.get("initial_prompt", {})),
        }

    def get_allowed_fields(self, phase: str) -> list[str]:
        fields = self._release_policy.get(phase, [])
        return [str(item) for item in fields] if isinstance(fields, list) else []

    def current_phase(self) -> str:
        return self._phase

    def advance_phase(self, new_phase: str) -> None:
        if new_phase not in PHASES:
            raise ValueError(f"Unknown benchmark phase {new_phase!r}")
        self._phase = new_phase

    def query(
        self,
        skill_name: str,
        payload: Mapping[str, Any] | None = None,
        phase: str | None = None,
    ) -> dict[str, Any]:
        """Return a phase-gated benchmark tool result."""

        active_phase = phase or self._phase
        payload = dict(payload or {})
        if skill_name == "clinical.read_summary":
            data = self._clinical_for_phase(active_phase)
            return _success(skill_name, "clinical", "Clinical summary retrieved.", data)
        if skill_name == "clinical.read_treatment":
            return _success(
                skill_name,
                "treatment",
                "Treatment records retrieved.",
                self._read("clinical/treatment.json"),
            )
        if skill_name == "clinical.read_follow_up":
            if active_phase not in {"progression_phase", "outcome_phase"}:
                return _success(
                    skill_name,
                    "follow_up",
                    "Follow-up exists but is not released in the current phase.",
                    {"released": False},
                )
            return _success(
                skill_name,
                "follow_up",
                "Follow-up and outcome records retrieved.",
                self._read("clinical/follow_up.json"),
            )
        if skill_name == "pathology.read_report":
            path = self.case_dir / "pathology" / "pathology_report.txt"
            text = path.read_text(encoding="utf-8") if path.is_file() else ""
            return _success(
                skill_name,
                "pathology_text",
                "Pathology report text retrieved.",
                {"report_text": text or "not_available", "source": str(path)},
            )
        if skill_name == "pathology.read_slide_metadata":
            return _success(
                skill_name,
                "pathology_slide_metadata",
                "Pathology slide metadata retrieved.",
                self._read("pathology/slide_metadata.json"),
            )
        if skill_name == "pathology.read_wsi_manifest":
            return _success(
                skill_name,
                "wsi_manifest",
                "WSI manifest retrieved.",
                self._read("pathology/wsi_manifest.json"),
            )
        if skill_name == "radiology.read_ct_manifest":
            return _success(
                skill_name,
                "ct_manifest",
                "CT manifest retrieved.",
                self._read("radiology/ct_manifest.json"),
            )
        if skill_name == "molecular.query_biomarkers":
            return _success(
                skill_name,
                "molecular",
                "Molecular biomarker results retrieved.",
                self._read("molecular/biomarkers.json"),
            )
        if skill_name == "guideline.retrieve":
            return _success(
                skill_name,
                "guideline",
                "Guideline nodes retrieved.",
                self._read("guideline/relevant_nodes.json"),
            )
        return {
            "status": "missing",
            "skill_name": skill_name,
            "modality": "unknown",
            "findings": {
                "summary": f"Benchmark simulator does not provide {skill_name}.",
                "data": {},
            },
            "artifacts": [],
            "warnings": [f"Unknown simulator skill: {skill_name}"],
        }

    def _clinical_for_phase(self, phase: str) -> dict[str, Any]:
        data = self._read("clinical/clinical.json")
        if phase not in {"progression_phase", "outcome_phase"}:
            data.pop("vital_status_at_last_follow_up", None)
            data.pop("lost_to_follow_up", None)
        return data

    def _read(self, relative_path: str) -> dict[str, Any]:
        path = self.case_dir / relative_path
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"Benchmark case file must contain an object: {path}")
        return data


def _success(
    skill_name: str,
    modality: str,
    summary: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "success",
        "skill_name": skill_name,
        "modality": modality,
        "findings": {
            "summary": summary,
            "data": data,
        },
        "artifacts": [],
        "warnings": [],
    }
