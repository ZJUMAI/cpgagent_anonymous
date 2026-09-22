"""Typed records shared by latent guideline routing components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


Tensor = Any


@dataclass
class GuidelineMemory:
    """One chapter-level memory and its audit metadata."""

    memory_id: str
    guideline_name: str
    guideline_version: str
    cancer_type: str
    section_title: str
    section_path: list[str]
    page_start: int | None
    page_end: int | None
    language: str
    source_chunk_ids: list[str]
    slots: Tensor  # [num_slots, hidden_size], kept on CPU in the bank.
    retrieval_key: Tensor  # [retrieval_dim], kept on CPU in the bank.
    source_rule_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_pages(self) -> str | None:
        if self.page_start is None and self.page_end is None:
            return None
        if self.page_start == self.page_end or self.page_end is None:
            return str(self.page_start)
        if self.page_start is None:
            return str(self.page_end)
        return f"{self.page_start}-{self.page_end}"

    def audit_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "guideline_name": self.guideline_name,
            "guideline_version": self.guideline_version,
            "cancer_type": self.cancer_type,
            "section_title": self.section_title,
            "section_path": list(self.section_path),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "source_pages": self.source_pages,
            "language": self.language,
            "source_chunk_ids": list(self.source_chunk_ids),
            "source_rule_ids": list(self.source_rule_ids),
            "slot_shape": _tensor_shape(self.slots),
            "retrieval_key_shape": _tensor_shape(self.retrieval_key),
        }


@dataclass
class PatientState:
    """Routing view over the existing benchmark patient-state mapping."""

    case_id: str
    current_time: str | int
    confirmed_evidence: dict[str, Any]
    unresolved_information: list[str]
    completed_actions: list[Any]
    pending_actions: list[Any]
    current_decision_stage: str | None
    previous_actions: list[Any]
    free_text_summary: str | None
    cancer_type: str | None = None
    guideline_version: str | None = None
    language: str | None = None
    raw_state: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        trajectory_history: Sequence[Any] | None = None,
    ) -> "PatientState":
        raw = dict(value)
        evidence = value.get("confirmed_evidence")
        if not isinstance(evidence, Mapping):
            evidence = {
                key: value.get(key)
                for key in (
                    "known_diagnosis",
                    "known_stage",
                    "known_biomarkers",
                    "radiology_summary",
                    "pathology_summary",
                    "molecular_summary",
                    "guideline_summary",
                    "evidence_refs",
                    "available_modalities",
                )
                if value.get(key) not in (None, "", [], {})
            }
        completed = value.get("completed_actions")
        if not isinstance(completed, list):
            completed = list(value.get("completed_skills") or [])
        previous = value.get("previous_actions")
        if not isinstance(previous, list):
            previous = list(trajectory_history or [])
        summary = value.get("free_text_summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = _state_summary(value)
        return cls(
            case_id=str(value.get("case_id") or "unknown"),
            current_time=value.get("current_time", value.get("round_index", 0)),
            confirmed_evidence=dict(evidence),
            unresolved_information=_string_list(
                value.get("unresolved_information", value.get("missing_information"))
            ),
            completed_actions=list(completed),
            pending_actions=list(value.get("pending_actions") or []),
            current_decision_stage=_optional_text(
                value.get("current_decision_stage", value.get("current_phase"))
            ),
            previous_actions=list(previous),
            free_text_summary=summary,
            cancer_type=_optional_text(
                value.get("cancer_family", value.get("cancer_type"))
            ),
            guideline_version=_optional_text(
                value.get("guideline_version") or _state_guideline_version(value)
            ),
            language=_optional_text(value.get("language")) or "zh",
            raw_state=raw,
        )

    def to_mapping(self) -> dict[str, Any]:
        result = dict(self.raw_state)
        result.update(
            {
                "case_id": self.case_id,
                "current_time": self.current_time,
                "confirmed_evidence": self.confirmed_evidence,
                "unresolved_information": self.unresolved_information,
                "completed_actions": self.completed_actions,
                "pending_actions": self.pending_actions,
                "current_decision_stage": self.current_decision_stage,
                "previous_actions": self.previous_actions,
                "free_text_summary": self.free_text_summary,
            }
        )
        if self.cancer_type:
            result["cancer_type"] = self.cancer_type
        if self.guideline_version:
            result["guideline_version"] = self.guideline_version
        if self.language:
            result["language"] = self.language
        return result


@dataclass
class MemoryCandidate:
    memory: GuidelineMemory
    seed_score: float | None = None
    combined_score: float | None = None

    @property
    def memory_id(self) -> str:
        return self.memory.memory_id

    def audit_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "section_title": self.memory.section_title,
            "cancer_type": self.memory.cancer_type,
            "seed_score": self.seed_score,
            "combined_score": self.combined_score,
            "source_pages": self.memory.source_pages,
        }


@dataclass
class MemoryActivation:
    memory_id: str
    weight: float
    seed_score: float | None = None
    gate_score: float | None = None
    expert_confidence: float | None = None
    progress_score: float | None = None
    selected: bool = False
    section_title: str | None = None
    source_pages: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MemoryActivation":
        return cls(
            memory_id=str(value.get("memory_id") or ""),
            weight=float(value.get("weight") or 0.0),
            seed_score=_optional_float(value.get("seed_score")),
            gate_score=_optional_float(value.get("gate_score")),
            expert_confidence=_optional_float(value.get("expert_confidence")),
            progress_score=_optional_float(value.get("progress_score")),
            selected=bool(value.get("selected", float(value.get("weight") or 0.0) > 0)),
            section_title=_optional_text(value.get("section_title")),
            source_pages=_optional_text(value.get("source_pages")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateOutput:
    activations: list[MemoryActivation]
    selected_activations: list[MemoryActivation]
    logits: Tensor  # [num_candidates] or [batch, num_candidates]
    probabilities: Tensor
    selected_indices: list[int]
    selected_weights: Tensor
    diagnostics: dict[str, Any]


@dataclass
class FusedMemory:
    slots: Tensor  # [fused_slots, hidden_size]
    attention_mask: Tensor  # [fused_slots]
    memory_ids: list[str]
    slot_slices: dict[str, tuple[int, int]]
    segment_ids: Tensor  # [fused_slots], -1 when source slots were resampled.
    diagnostics: dict[str, Any] = field(default_factory=dict)
    memory_attention_mass: Tensor | None = None
    attention_weights: Tensor | None = None
    anchor_slots: Tensor | None = None


@dataclass
class ExpertOutput:
    memory_id: str
    proposal_tokens: Tensor  # [num_proposal_tokens, hidden_size]
    confidence: Tensor  # scalar tensor
    progress_score: Tensor  # scalar tensor
    memory_key: Tensor | None = None

    def audit_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "proposal_shape": _tensor_shape(self.proposal_tokens),
            "confidence": _tensor_scalar(self.confidence),
            "progress_score": _tensor_scalar(self.progress_score),
        }


@dataclass
class CompetitionOutput:
    aggregated_proposals: Tensor
    expert_weights: Tensor
    selected_memory_ids: list[str]
    expert_pair_relations: Tensor | None
    diagnostics: dict[str, Any]


@dataclass
class RoutingResult:
    patient_state: PatientState
    state_repr: Tensor
    seed_candidates: list[MemoryCandidate]
    merged_candidates: list[MemoryCandidate]
    gate_output: GateOutput
    fused_memory: FusedMemory
    expert_outputs: list[ExpertOutput]
    competition_output: CompetitionOutput | None
    decoder_prefix: Tensor
    diagnostics: dict[str, Any]

    @property
    def active_memories(self) -> list[MemoryActivation]:
        return self.gate_output.selected_activations

    def audit_dict(self) -> dict[str, Any]:
        normalized_entropy = self.gate_output.diagnostics.get("normalized_entropy")
        return {
            "seed_candidates": [item.audit_dict() for item in self.seed_candidates],
            "merged_candidates": [item.audit_dict() for item in self.merged_candidates],
            "active_memories": [item.to_dict() for item in self.active_memories],
            "candidate_activations": [
                item.to_dict() for item in self.gate_output.activations
            ],
            "gate_diagnostics": _json_safe(self.gate_output.diagnostics),
            "normalized_gate_entropy": normalized_entropy,
            "fusion_diagnostics": _json_safe(self.fused_memory.diagnostics),
            "decoder_prefix_shape": _tensor_shape(self.decoder_prefix),
            "diagnostics": _json_safe(self.diagnostics),
        }


@dataclass
class PlannerStepResult:
    action: dict[str, Any]
    current_objective: str | None
    expected_state_change: list[str]
    active_memories: list[MemoryActivation]
    expert_outputs: list[dict[str, Any]]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": _json_safe(self.action),
            "current_objective": self.current_objective,
            "expected_state_change": list(self.expected_state_change),
            "active_memories": [item.to_dict() for item in self.active_memories],
            "expert_outputs": _json_safe(self.expert_outputs),
            "diagnostics": _json_safe(self.diagnostics),
        }


def _state_summary(value: Mapping[str, Any]) -> str | None:
    parts: list[str] = []
    for key in (
        "known_diagnosis",
        "known_stage",
        "radiology_summary",
        "pathology_summary",
        "molecular_summary",
    ):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            parts.append(f"{key}: {item.strip()}")
    return "\n".join(parts) or None


def _state_guideline_version(value: Mapping[str, Any]) -> str | None:
    context = value.get("guideline_context")
    if not isinstance(context, Mapping):
        return None
    guidelines = context.get("guidelines")
    if not isinstance(guidelines, list) or len(guidelines) != 1:
        return None
    first = guidelines[0]
    return str(first.get("version") or "") if isinstance(first, Mapping) else None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _tensor_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]


def _tensor_scalar(value: Any) -> float | None:
    try:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().item()
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and hasattr(value, "shape"):
        if getattr(value, "numel", lambda: 2)() == 1:
            return _tensor_scalar(value)
        return {"tensor_shape": _tensor_shape(value)}
    return value
