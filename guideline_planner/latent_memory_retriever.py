"""State-conditioned retrieval over chapter-level latent guideline memories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from guideline_planner.retrieval import (
    load_memory_metadata_records,
    load_memory_slots,
)
from guideline_planner.memory import load_memory_store_meta
from guideline_planner.routing_types import GuidelineMemory, MemoryCandidate


_ALLOWED_FILTERS = {
    "cancer_type",
    "guideline_id",
    "version",
    "guideline_version",
    "language",
}


class GuidelineMemoryBank:
    """In-memory metadata/key index with CPU-resident latent slots."""

    def __init__(
        self,
        memories: Iterable[GuidelineMemory],
        *,
        root: str | Path,
        store_meta: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.store_meta = dict(store_meta or {})
        self.memories = list(memories)
        self.lookup = {memory.memory_id: memory for memory in self.memories}
        if len(self.lookup) != len(self.memories):
            raise ValueError("Guideline memory IDs must be unique within one store.")
        self.fingerprint = memory_store_fingerprint(self.root)
        self.slot_hidden_size = _consistent_dim(self.memories, "slots", axis=-1)
        self.memory_tokens = _consistent_dim(self.memories, "slots", axis=0)
        self.retrieval_dim = _consistent_dim(self.memories, "retrieval_key", axis=-1)

    @classmethod
    def from_directory(cls, memory_dir: str | Path) -> "GuidelineMemoryBank":
        import torch

        memories: list[GuidelineMemory] = []
        for record in load_memory_metadata_records(memory_dir):
            slots_value = load_memory_slots(record["latent_vector_path"])
            slots = torch.as_tensor(slots_value, dtype=torch.float32, device="cpu")
            key = torch.as_tensor(
                np.load(record["retrieval_embedding_path"]).astype("float32"),
                dtype=torch.float32,
                device="cpu",
            )
            if slots.ndim != 2:
                raise RuntimeError(
                    f"Memory {record.get('guideline_memory_id')} slots must be 2D; "
                    f"got {tuple(slots.shape)}."
                )
            if key.ndim != 1:
                raise RuntimeError(
                    f"Memory {record.get('guideline_memory_id')} retrieval key must "
                    f"be 1D; got {tuple(key.shape)}."
                )
            memories.append(_memory_from_record(record, slots, key))
        if not memories:
            raise RuntimeError(f"No guideline memories found in {memory_dir}.")
        return cls(
            memories,
            root=memory_dir,
            store_meta=load_memory_store_meta(memory_dir),
        )

    def __len__(self) -> int:
        return len(self.memories)

    def __iter__(self) -> Any:
        return iter(self.memories)


class LatentMemoryRetriever(torch.nn.Module):
    """Cosine seed retriever with coarse metadata filtering only."""

    def __init__(self, state_dim: int, retrieval_dim: int) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.retrieval_dim = int(retrieval_dim)
        self.state_projection = torch.nn.Linear(
            self.state_dim,
            self.retrieval_dim,
            bias=False,
        )
        if self.state_dim == self.retrieval_dim:
            torch.nn.init.eye_(self.state_projection.weight)
        else:
            torch.nn.init.xavier_uniform_(self.state_projection.weight)

    def search(
        self,
        state_repr: Any,
        memory_bank: GuidelineMemoryBank | Sequence[GuidelineMemory],
        top_k: int,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[MemoryCandidate]:
        if state_repr.ndim != 1:
            raise ValueError(
                "LatentMemoryRetriever.search expects one state vector; "
                f"got shape {tuple(state_repr.shape)}."
            )
        memories = (
            memory_bank.memories
            if isinstance(memory_bank, GuidelineMemoryBank)
            else list(memory_bank)
        )
        filtered = filter_memories(memories, metadata_filter)
        if not filtered or int(top_k) <= 0:
            return []
        projected = torch.nn.functional.normalize(
            self.state_projection(
                state_repr.to(
                    device=self.state_projection.weight.device,
                    dtype=self.state_projection.weight.dtype,
                )
            ),
            dim=-1,
        )
        keys = torch.stack(
            [
                memory.retrieval_key.to(
                    device=projected.device,
                    dtype=projected.dtype,
                )
                for memory in filtered
            ],
            dim=0,
        )
        keys = torch.nn.functional.normalize(keys, dim=-1)
        scores = torch.matmul(keys, projected)
        count = min(max(int(top_k), 0), len(filtered))
        indices = scores.argsort(descending=True)[:count].tolist()
        return [
            MemoryCandidate(
                memory=filtered[index],
                seed_score=float(scores[index].detach().float().cpu()),
                combined_score=float(scores[index].detach().float().cpu()),
            )
            for index in indices
        ]


def filter_memories(
    memories: Sequence[GuidelineMemory],
    metadata_filter: Mapping[str, Any] | None,
) -> list[GuidelineMemory]:
    filters = dict(metadata_filter or {})
    unknown = sorted(set(filters) - _ALLOWED_FILTERS)
    if unknown:
        raise ValueError(
            "Only coarse cancer_type/guideline_id/version/language filters are allowed; "
            f"received {unknown}."
        )
    result = []
    for memory in memories:
        if not _matches_cancer(memory.cancer_type, filters.get("cancer_type")):
            continue
        memory_guideline_id = str(
            memory.metadata.get("guideline_id") or memory.guideline_name
        )
        if not _matches_scalar(memory_guideline_id, filters.get("guideline_id")):
            continue
        expected_version = filters.get("version", filters.get("guideline_version"))
        if not _matches_scalar(memory.guideline_version, expected_version):
            continue
        if not _matches_scalar(memory.language, filters.get("language")):
            continue
        result.append(memory)
    return result


def memory_store_fingerprint(memory_dir: str | Path) -> str:
    root = Path(memory_dir)
    digest = hashlib.sha256()
    for filename in ("memory_store_meta.json", "slot_metadata.jsonl"):
        path = root / filename
        if path.is_file():
            digest.update(filename.encode("utf-8"))
            digest.update(path.read_bytes())
    records = (
        load_memory_metadata_records(root)
        if (root / "slot_metadata.jsonl").is_file()
        else []
    )
    for record in records:
        for field in ("latent_vector_path", "retrieval_embedding_path"):
            path = Path(record[field])
            digest.update(field.encode("utf-8"))
            try:
                logical_path = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                logical_path = path.name
            digest.update(logical_path.encode("utf-8"))
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def canonical_cancer_type(value: Any) -> tuple[str | None, str | None]:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "luad": ("lung", "nsclc"),
        "lusc": ("lung", "nsclc"),
        "nsclc": ("lung", "nsclc"),
        "non_small_cell_lung_cancer": ("lung", "nsclc"),
        "sclc": ("lung", "sclc"),
        "small_cell_lung_cancer": ("lung", "sclc"),
        "lung": ("lung", None),
        "lung_cancer": ("lung", None),
        "ucec": ("endometrial", "ucec"),
        "endometrial": ("endometrial", "ucec"),
        "endometrial_carcinoma": ("endometrial", "ucec"),
        "breast": ("breast", "breast"),
        "brca": ("breast", "breast"),
        "colorectal": ("colorectal", "colorectal"),
        "crc": ("colorectal", "colorectal"),
        "head_neck": ("head_neck", "head_neck"),
        "nasopharyngeal": ("nasopharyngeal", "nasopharyngeal"),
        "nasopharyngeal_carcinoma": ("nasopharyngeal", "nasopharyngeal"),
        "npc": ("nasopharyngeal", "nasopharyngeal"),
        "lymphoma": ("lymphoma", "lymphoma"),
        "unknown": (None, None),
        "": (None, None),
    }
    return aliases.get(text, (text or None, text or None))


def _memory_from_record(record: Mapping[str, Any], slots: Any, key: Any) -> GuidelineMemory:
    section_path = record.get("section_path")
    if not isinstance(section_path, list):
        section_path = []
        for name in ("chapter", "section", "h1_title"):
            value = record.get(name)
            if value is not None and str(value).strip() and str(value) not in section_path:
                section_path.append(str(value))
    source_chunks = record.get("source_chunk_ids", record.get("source_span_ids", []))
    if not isinstance(source_chunks, list):
        source_chunks = [str(source_chunks)] if source_chunks else []
    source_rules = record.get("source_rule_ids", [])
    if not isinstance(source_rules, list):
        source_rules = [str(source_rules)] if source_rules else []
    return GuidelineMemory(
        memory_id=str(record["guideline_memory_id"]),
        guideline_name=str(record.get("guideline_name") or record.get("guideline_id") or "unknown"),
        guideline_version=str(record.get("version") or "unknown"),
        cancer_type=str(record.get("cancer_type") or "unknown"),
        section_title=str(record.get("h1_title") or record.get("section") or "unknown"),
        section_path=[str(item) for item in section_path],
        page_start=_optional_int(record.get("page_start")),
        page_end=_optional_int(record.get("page_end")),
        language=str(record.get("language") or "zh"),
        source_chunk_ids=[str(item) for item in source_chunks],
        slots=slots,
        retrieval_key=key,
        source_rule_ids=[str(item) for item in source_rules],
        metadata=dict(record),
    )


def _matches_cancer(actual: str, expected: Any) -> bool:
    if expected in (None, "", []):
        return True
    if isinstance(expected, Mapping):
        expected_family, _ = canonical_cancer_type(expected.get("family"))
        _, expected_subtype = canonical_cancer_type(expected.get("subtype"))
        actual_family, actual_subtype = canonical_cancer_type(actual)
        if expected_family is not None and actual_family != expected_family:
            return False
        if expected_subtype is None:
            return True
        if actual_subtype == expected_subtype:
            return True
        return bool(expected.get("allow_generic")) and actual_subtype is None
    expected_values = expected if isinstance(expected, (list, tuple, set)) else [expected]
    actual_family, actual_subtype = canonical_cancer_type(actual)
    for item in expected_values:
        family, subtype = canonical_cancer_type(item)
        if family is None:
            return True
        if actual_family != family:
            continue
        if subtype is None or actual_subtype == subtype:
            return True
    return False


def _matches_scalar(actual: str, expected: Any) -> bool:
    if expected in (None, "", []):
        return True
    values = expected if isinstance(expected, (list, tuple, set)) else [expected]
    return str(actual).lower() in {str(item).lower() for item in values}


def _consistent_dim(memories: Sequence[GuidelineMemory], field: str, *, axis: int) -> int:
    dimensions = {int(getattr(memory, field).shape[axis]) for memory in memories}
    if len(dimensions) != 1:
        raise RuntimeError(f"Inconsistent {field} dimensions in memory bank: {dimensions}")
    return dimensions.pop()


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
