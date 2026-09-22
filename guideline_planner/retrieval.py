"""Retrieve latent guideline-memory slots from local embeddings."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from guideline_planner.io_utils import read_jsonl, stable_short_id


class ContrastiveSlotSelector:
    """Lightweight selector interface for query-slot similarity ranking."""

    def __init__(self, *, embedding_dim: int | None = None) -> None:
        self.embedding_dim = embedding_dim

    def encode_query(self, query: str, *, embedding_dim: int | None = None) -> np.ndarray:
        dim = embedding_dim or self.embedding_dim or 128
        vector = np.zeros(dim, dtype="float32")
        tokens = _query_tokens(query)
        if not tokens:
            tokens = [query or "empty"]
        for token in tokens:
            index = int(stable_short_id(token, 8), 16) % dim
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def score(self, query_embedding: np.ndarray, slot_embedding: np.ndarray) -> float:
        return float(np.dot(query_embedding, slot_embedding))


def retrieve_latent_guideline_memory(
    query: str,
    top_k: int,
    *,
    memory_dir: str | Path,
    filters: Mapping[str, Any] | None = None,
    selector: ContrastiveSlotSelector | None = None,
    load_slots: bool = True,
) -> list[dict[str, Any]]:
    """Filter slot metadata, rank by vector similarity, and return slots + sources."""

    records = load_memory_metadata_records(memory_dir)
    filtered = [record for record in records if _matches_filters(record, filters or {})]
    if not filtered:
        return []

    first_embedding = np.load(filtered[0]["retrieval_embedding_path"])
    ranker = selector or ContrastiveSlotSelector(embedding_dim=int(first_embedding.shape[-1]))
    query_embedding = ranker.encode_query(query, embedding_dim=int(first_embedding.shape[-1]))
    scored: list[tuple[float, dict[str, Any], np.ndarray]] = []
    for record in filtered:
        embedding = np.load(record["retrieval_embedding_path"]).astype("float32")
        scored.append((ranker.score(query_embedding, embedding), record, embedding))
    scored.sort(key=lambda item: item[0], reverse=True)

    results: list[dict[str, Any]] = []
    for score, record, embedding in scored[: max(top_k, 0)]:
        payload = {
            "score": score,
            "guideline_memory_id": record["guideline_memory_id"],
            "metadata": record,
            "retrieval_embedding": embedding,
        }
        if load_slots:
            payload["memory_slots"] = load_memory_slots(
                Path(record["latent_vector_path"])
            )
        results.append(payload)
    return results


def load_memory_metadata_records(memory_dir: str | Path) -> list[dict[str, Any]]:
    """Load slot metadata and resolve artifact paths without loading tensors."""

    root = Path(memory_dir)
    metadata_path = root / "slot_metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"slot metadata not found: {metadata_path}")
    return [_normalize_paths(record, root) for record in read_jsonl(metadata_path)]


def load_memory_slots(path: str | Path) -> Any:
    """Load a latent slot tensor as saved by ``extract-memory``."""

    return _load_pt(Path(path))


def _matches_filters(record: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, expected in filters.items():
        if expected is None or expected == "":
            continue
        value = record.get(key)
        if key == "cancer_type" and isinstance(expected, Mapping):
            if not _matches_cancer_filter(value, expected):
                return False
            continue
        if isinstance(expected, (list, tuple, set)):
            if value not in expected:
                return False
        elif value != expected:
            return False
    return True


def _matches_cancer_filter(actual: Any, expected: Mapping[str, Any]) -> bool:
    actual_family, actual_subtype = _canonical_cancer_type(actual)
    expected_family, _ = _canonical_cancer_type(expected.get("family"))
    _, expected_subtype = _canonical_cancer_type(expected.get("subtype"))
    if expected_family is not None and actual_family != expected_family:
        return False
    if expected_subtype is None or actual_subtype == expected_subtype:
        return True
    return bool(expected.get("allow_generic")) and actual_subtype is None


def _canonical_cancer_type(value: Any) -> tuple[str | None, str | None]:
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
        "nasopharyngeal": ("nasopharyngeal", "nasopharyngeal"),
        "nasopharyngeal_carcinoma": ("nasopharyngeal", "nasopharyngeal"),
        "npc": ("nasopharyngeal", "nasopharyngeal"),
        "unknown": (None, None),
        "": (None, None),
    }
    return aliases.get(text, (text or None, text or None))


def _normalize_paths(record: dict[str, Any], root: Path) -> dict[str, Any]:
    normalized = dict(record)
    for key in ("latent_vector_path", "retrieval_embedding_path"):
        normalized[key] = str(_resolve_metadata_path(str(normalized[key]), root))
    return normalized


def _resolve_metadata_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    normalized_value = value.replace("\\", "/")
    normalized_root = root.as_posix().rstrip("/")
    if normalized_value == normalized_root or normalized_value.startswith(
        f"{normalized_root}/"
    ):
        return path
    return root / path


def _query_tokens(query: str) -> list[str]:
    import re

    return re.findall(r"[\u4e00-\u9fffA-Za-z0-9\-+/]{2,}", query)


def _load_pt(path: Path) -> Any:
    try:
        import torch

        value = torch.load(path, map_location="cpu")
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return value
    except Exception:
        with path.open("rb") as handle:
            return pickle.load(handle)
