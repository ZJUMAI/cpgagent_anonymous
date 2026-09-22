"""Strictly separated V2 latent-memory query and planner decoder runtimes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from guideline_planner.artifacts import (
    ArtifactBindingError,
    build_memory_store_fingerprint,
    validate_artifact_hash,
    validate_binding,
)
from guideline_planner.memory import load_memory_store_meta, resolve_memory_store_path
from guideline_planner.modeling import (
    generate_with_memory_slots,
    load_planner_model_bundle,
    projected_query_embedding_with_bundle,
)


class LatentMemoryQueryEncoder:
    """Frozen query encoder bound to the encoder that produced the memory store."""

    def __init__(
        self,
        *,
        base_model: str,
        adapter_path: str | Path,
        tokenizer_path: str | Path,
        retrieval_projection_path: str | Path,
        device: str | None = None,
        model_dtype: str = "auto",
        max_query_tokens: int = 256,
    ) -> None:
        self.base_model = str(base_model)
        self.adapter_path = str(adapter_path)
        self.tokenizer_path = str(tokenizer_path)
        self.retrieval_projection_path = str(retrieval_projection_path)
        self.max_query_tokens = int(max_query_tokens)
        self.bundle = load_planner_model_bundle(
            model_name=base_model,
            adapter_path=adapter_path,
            tokenizer_path=tokenizer_path,
            retrieval_projection_path=retrieval_projection_path,
            device=device,
            model_dtype=model_dtype,
            adapter_trainable=False,
        )
        self.hidden_size = int(self.bundle.hidden_size)
        self.device = self.bundle.device

    @classmethod
    def from_memory_store(
        cls,
        memory_dir: str | Path,
        meta: Mapping[str, Any] | None = None,
        *,
        device: str | None = None,
        model_dtype: str = "auto",
        max_query_tokens: int = 256,
    ) -> "LatentMemoryQueryEncoder":
        root = Path(memory_dir)
        store_meta = dict(meta or load_memory_store_meta(root))
        _validate_memory_store_artifacts(root, store_meta)
        base_model = store_meta.get("base_model_snapshot_path") or store_meta.get("base_model")
        if not isinstance(base_model, str) or not base_model:
            raise ArtifactBindingError("Memory Store does not identify its base model.")
        return cls(
            base_model=base_model,
            adapter_path=_existing_meta_path(root, store_meta, "memory_encoder_adapter_path"),
            tokenizer_path=_existing_meta_path(root, store_meta, "tokenizer_path"),
            retrieval_projection_path=_existing_meta_path(
                root,
                store_meta,
                "retrieval_projection_path",
            ),
            device=device,
            model_dtype=model_dtype,
            max_query_tokens=max_query_tokens,
        )

    def encode_query(self, query: str, *, embedding_dim: int | None = None) -> np.ndarray:
        embedding = projected_query_embedding_with_bundle(
            self.bundle,
            query,
            max_query_tokens=self.max_query_tokens,
        )
        if embedding_dim is not None and int(embedding.shape[-1]) != int(embedding_dim):
            raise RuntimeError(
                f"Query embedding dim {embedding.shape[-1]} does not match "
                f"slot embedding dim {embedding_dim}."
            )
        return embedding

    def score(self, query_embedding: np.ndarray, slot_embedding: np.ndarray) -> float:
        return float(np.dot(query_embedding, slot_embedding))


class LatentPlannerDecoder:
    """Planner Decoder LoRA plus its explicit memory-to-decoder bridge."""

    META_FILENAME = "planner_decoder_meta.json"

    def __init__(
        self,
        *,
        base_model: str,
        adapter_path: str | Path,
        tokenizer_path: str | Path,
        retrieval_projection_path: str | Path,
        bridge_path: str | Path,
        device: str | None = None,
        model_dtype: str = "auto",
        adapter_trainable: bool = False,
    ) -> None:
        self.base_model = str(base_model)
        self.adapter_path = str(adapter_path)
        self.tokenizer_path = str(tokenizer_path)
        self.retrieval_projection_path = str(retrieval_projection_path)
        self.bridge_path = str(bridge_path)
        self.bundle = load_planner_model_bundle(
            model_name=base_model,
            adapter_path=adapter_path,
            tokenizer_path=tokenizer_path,
            retrieval_projection_path=retrieval_projection_path,
            device=device,
            model_dtype=model_dtype,
            adapter_trainable=adapter_trainable,
        )
        self.hidden_size = int(self.bundle.hidden_size)
        self.device = self.bundle.device
        self.memory_to_decoder_bridge = _load_bridge(
            self.bundle.torch,
            bridge_path,
            hidden_size=self.hidden_size,
            device=self.device,
            trainable=adapter_trainable,
        )
        self.last_generation_info: dict[str, Any] = {}
        self.artifact_meta: dict[str, Any] = {}

    @classmethod
    def from_artifacts(
        cls,
        memory_dir: str | Path,
        decoder_artifact_dir: str | Path,
        *,
        device: str | None = None,
        model_dtype: str = "auto",
        adapter_trainable: bool = False,
    ) -> "LatentPlannerDecoder":
        memory_root = Path(memory_dir)
        decoder_root = Path(decoder_artifact_dir)
        memory_meta = load_memory_store_meta(memory_root)
        _validate_memory_store_artifacts(memory_root, memory_meta)
        decoder_meta = _read_meta(decoder_root / cls.META_FILENAME)
        if decoder_meta.get("artifact_role") != "planner_decoder":
            raise ArtifactBindingError("planner_decoder_meta.json has the wrong artifact_role.")
        validate_binding(
            memory_meta,
            decoder_meta,
            fields=("memory_store_fingerprint", "base_model_revision", "tokenizer_hash"),
            label="Planner Decoder / Memory Store",
        )
        adapter = _artifact_path(decoder_root, decoder_meta, "planner_decoder_adapter_path")
        bridge = _artifact_path(decoder_root, decoder_meta, "memory_to_decoder_bridge_path")
        tokenizer = _artifact_path(decoder_root, decoder_meta, "tokenizer_path")
        projection = _artifact_path(memory_root, memory_meta, "retrieval_projection_path")
        validate_artifact_hash(
            label="Planner Decoder adapter",
            path=adapter,
            expected_hash=decoder_meta.get("planner_decoder_adapter_hash"),
        )
        validate_artifact_hash(
            label="memory_to_decoder_bridge",
            path=bridge,
            expected_hash=decoder_meta.get("memory_to_decoder_bridge_hash"),
        )
        validate_artifact_hash(
            label="Planner Decoder tokenizer",
            path=tokenizer,
            expected_hash=decoder_meta.get("tokenizer_hash"),
        )
        base_model = decoder_meta.get("base_model_snapshot_path") or decoder_meta.get("base_model")
        if not isinstance(base_model, str) or not base_model:
            raise ArtifactBindingError("Planner Decoder does not identify its base model.")
        instance = cls(
            base_model=base_model,
            adapter_path=adapter,
            tokenizer_path=tokenizer,
            retrieval_projection_path=projection,
            bridge_path=bridge,
            device=device,
            model_dtype=model_dtype,
            adapter_trainable=adapter_trainable,
        )
        instance.artifact_meta = decoder_meta
        return instance

    @classmethod
    def from_memory_store(cls, *_: Any, **__: Any) -> "LatentPlannerDecoder":
        raise ArtifactBindingError(
            "Planner V2 cannot load a Decoder from a Memory Store adapter. "
            "Provide an explicit planner_decoder artifact directory."
        )

    def generate_plan(
        self,
        memory_slots: Any,
        prompt: str,
        *,
        max_new_tokens: int | str | None = 512,
    ) -> str:
        slots = self.transform_memory_slots(memory_slots)
        diagnostics: dict[str, Any] = {}
        text = generate_with_memory_slots(
            self.bundle,
            slots,
            prompt,
            max_new_tokens=max_new_tokens,
            generation_diagnostics=diagnostics,
        )
        self.last_generation_info = diagnostics
        return text

    def transform_memory_slots(self, memory_slots: Any) -> np.ndarray:
        torch = self.bundle.torch
        values = torch.as_tensor(
            np.asarray(memory_slots, dtype="float32"),
            device=self.device,
            dtype=next(self.memory_to_decoder_bridge.parameters()).dtype,
        )
        with torch.set_grad_enabled(self.memory_to_decoder_bridge.training):
            transformed = self.memory_to_decoder_bridge(values)
        return transformed.detach().cpu().float().numpy()

    def count_text_tokens(self, text: str) -> int:
        encoded = self.bundle.tokenizer(text, add_special_tokens=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids or [])


def _validate_memory_store_artifacts(memory_dir: Path, meta: Mapping[str, Any]) -> None:
    if meta.get("artifact_role") != "memory_store" or int(meta.get("format_version") or 0) != 2:
        raise ArtifactBindingError("Planner V2 requires a format_version=2 Memory Store.")
    expected_fingerprint = meta.get("memory_store_fingerprint")
    actual_fingerprint = build_memory_store_fingerprint(meta)
    if expected_fingerprint != actual_fingerprint:
        raise ArtifactBindingError(
            "Memory Store lineage fingerprint does not match its metadata."
        )
    for path_field, hash_field, label in (
        ("memory_encoder_adapter_path", "memory_encoder_adapter_hash", "Memory Encoder adapter"),
        ("tokenizer_path", "tokenizer_hash", "Memory Encoder tokenizer"),
        ("retrieval_projection_path", "retrieval_projection_hash", "retrieval projection"),
    ):
        path = _existing_meta_path(memory_dir, meta, path_field)
        validate_artifact_hash(label=label, path=path, expected_hash=meta.get(hash_field))


def _load_bridge(torch: Any, path: str | Path, *, hidden_size: int, device: str, trainable: bool) -> Any:
    bridge = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    payload = torch.load(str(path), map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    bridge.load_state_dict(state, strict=True)
    bridge.to(device)
    for parameter in bridge.parameters():
        parameter.requires_grad = bool(trainable)
    bridge.train(mode=bool(trainable))
    return bridge


def _read_meta(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Planner Decoder metadata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactBindingError(f"{path} must contain a JSON object.")
    return payload


def _artifact_path(root: Path, meta: Mapping[str, Any], field_name: str) -> Path:
    value = meta.get(field_name)
    if not isinstance(value, str) or not value:
        raise ArtifactBindingError(f"Artifact metadata is missing {field_name!r}.")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path


def _existing_meta_path(memory_dir: str | Path, meta: Mapping[str, Any], field_name: str) -> Path:
    path = resolve_memory_store_path(memory_dir, meta.get(field_name), field_name)
    if not path.exists():
        raise FileNotFoundError(
            f"{field_name} from memory_store_meta.json does not exist: {path}"
        )
    return path
