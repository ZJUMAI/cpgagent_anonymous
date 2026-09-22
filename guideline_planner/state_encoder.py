"""Patient-state serialization and embedding for latent memory routing."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

import torch

from guideline_planner.routing_types import PatientState


class PatientStateEncoder(torch.nn.Module):
    """Encode canonical patient state with the trained planner query encoder.

    The query backend remains the source of the semantic representation. A
    zero-initialized residual adapter can be trained by the routing trainer
    without changing the existing query/slot projection checkpoint.
    """

    def __init__(
        self,
        query_encoder: Callable[..., Any],
        state_dim: int,
        *,
        device: str | Any = "cpu",
    ) -> None:
        super().__init__()
        self.query_encoder = query_encoder
        self.state_dim = int(state_dim)
        self.device_name = str(device)
        self.adapter = torch.nn.Linear(self.state_dim, self.state_dim)
        torch.nn.init.zeros_(self.adapter.weight)
        torch.nn.init.zeros_(self.adapter.bias)
        self.to(device)

    def forward(
        self,
        patient_state: PatientState | Mapping[str, Any] | Sequence[Any],
        trajectory_history: list[Any] | None = None,
    ) -> Any:
        if _is_state_batch(patient_state):
            vectors = [
                self._encode_one(item, trajectory_history=None)
                for item in patient_state
            ]
            return torch.stack(vectors, dim=0)
        return self._encode_one(patient_state, trajectory_history)

    def _encode_one(
        self,
        patient_state: PatientState | Mapping[str, Any],
        trajectory_history: list[Any] | None,
    ) -> Any:
        state = (
            patient_state
            if isinstance(patient_state, PatientState)
            else PatientState.from_mapping(
                patient_state,
                trajectory_history=trajectory_history,
            )
        )
        text = serialize_patient_state(state, trajectory_history)
        value = self.query_encoder(text, embedding_dim=self.state_dim)
        tensor = torch.as_tensor(
            value,
            dtype=torch.float32,
            device=self.adapter.weight.device,
        )
        if tensor.ndim != 1 or int(tensor.shape[0]) != self.state_dim:
            raise RuntimeError(
                "Patient state encoder returned shape "
                f"{tuple(tensor.shape)}; expected ({self.state_dim},)."
            )
        adapted = tensor + self.adapter(tensor)
        return torch.nn.functional.normalize(adapted, dim=-1)


def serialize_patient_state(
    patient_state: PatientState | Mapping[str, Any],
    trajectory_history: Sequence[Any] | None = None,
) -> str:
    """Serialize only visible patient state and past actions in stable order."""

    state = (
        patient_state
        if isinstance(patient_state, PatientState)
        else PatientState.from_mapping(
            patient_state,
            trajectory_history=trajectory_history,
        )
    )
    payload = {
        "schema_version": state.raw_state.get("schema_version"),
        "case_id": state.case_id,
        "current_time": state.current_time,
        "cancer_family": state.raw_state.get("cancer_family") or state.cancer_type,
        "disease_subtype": state.raw_state.get("disease_subtype"),
        "current_phase": state.raw_state.get("current_phase"),
        "decision_date": state.raw_state.get("decision_date"),
        "guideline_context": state.raw_state.get("guideline_context"),
        "known_diagnosis": state.raw_state.get("known_diagnosis"),
        "known_stage": state.raw_state.get("known_stage"),
        "known_biomarkers": state.raw_state.get("known_biomarkers"),
        "risk_stratification": state.raw_state.get("risk_stratification"),
        "available_modalities": state.raw_state.get("available_modalities"),
        "confirmed_evidence": state.confirmed_evidence,
        "unresolved_information": state.unresolved_information,
        "completed_skills": state.raw_state.get("completed_skills"),
        "completed_actions": state.completed_actions,
        "treatment_history": state.raw_state.get("treatment_history"),
        "current_treatment_line": state.raw_state.get("current_treatment_line"),
        "evidence_ledger": state.raw_state.get("evidence_ledger"),
        "last_transition": state.raw_state.get("last_transition"),
        "pending_actions": state.pending_actions,
        "blocked_actions": state.raw_state.get("blocked_actions"),
        "current_decision_stage": state.current_decision_stage,
        "previous_actions": state.previous_actions,
        "free_text_summary": state.free_text_summary,
    }
    if trajectory_history:
        payload["trajectory_history"] = list(trajectory_history)
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _is_state_batch(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, Mapping, PatientState),
    )
