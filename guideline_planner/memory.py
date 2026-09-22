"""Latent guideline-memory extraction and local slot artifact writing."""

from __future__ import annotations

import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from guideline_planner.artifacts import (
    build_memory_store_fingerprint,
    sha256_json,
    sha256_path,
    validate_artifact_hash,
)
from guideline_planner.constants import DEFAULT_MEMORY_TOKENS, DEFAULT_MODEL_NAME
from guideline_planner.io_utils import (
    read_jsonl,
    safe_filename,
    stable_short_id,
    write_json,
    write_jsonl,
)
from guideline_planner.modeling import (
    encode_memory_slots_with_bundle,
    load_planner_model_bundle,
    projected_slot_embedding_with_bundle,
)
from guideline_planner.progress import ProgressReporter


MEMORY_STORE_META_FILENAME = "memory_store_meta.json"


def extract_memory_slots(
    chunks: str | Path | Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    model_revision: str | None = None,
    model_snapshot_path: str | Path | None = None,
    memory_tokens: int | None = None,
    mock: bool = False,
    training_run_dir: str | Path | None = None,
    adapter_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    retrieval_projection_path: str | Path | None = None,
    device: str | None = None,
    model_dtype: str = "auto",
    max_encoder_tokens: int = 4096,
) -> dict[str, Any]:
    """Save one latent slot tensor and retrieval embedding per first-level section."""

    records = _load_chunks(chunks)
    progress = ProgressReporter("export-memory", len(records), unit="chunk")
    output = Path(output_dir)
    memory_root = output / "memory"
    metadata_records: list[dict[str, Any]] = []
    used_names: dict[tuple[str, str], int] = defaultdict(int)
    training_config = _load_training_config(training_run_dir)
    original_model_name = model_name
    resolved_snapshot_path: str | None = None
    if training_config:
        original_model_name = str(training_config.get("model_name") or model_name)
        model_revision = model_revision or _optional_string(training_config.get("model_revision"))
        resolved_snapshot_path = _optional_string(
            model_snapshot_path
            or training_config.get("model_snapshot_path")
            or training_config.get("model_load_path")
        )
        if not resolved_snapshot_path and not mock:
            raise RuntimeError(
                "This training run does not record a fixed model snapshot. "
                "Re-train with the fixed-snapshot code path or pass "
                "--model-snapshot-path explicitly."
            )
        model_name = resolved_snapshot_path or original_model_name
        if memory_tokens is None:
            memory_tokens = int(training_config.get("memory_tokens") or DEFAULT_MEMORY_TOKENS)
    elif model_snapshot_path is not None:
        resolved_snapshot_path = str(Path(model_snapshot_path).expanduser().resolve())
        model_name = resolved_snapshot_path
    if memory_tokens is None:
        memory_tokens = DEFAULT_MEMORY_TOKENS
    if training_run_dir is not None:
        run_dir = Path(training_run_dir)
        encoder_meta = _load_memory_encoder_meta(run_dir)
        if encoder_meta.get("artifact_role") != "memory_encoder":
            raise RuntimeError("training_run_dir is not a V2 Memory Encoder artifact.")
        adapter_path = adapter_path or run_dir / "memory_encoder_adapter"
        tokenizer_path = tokenizer_path or run_dir / "tokenizer"
        retrieval_projection_path = retrieval_projection_path or run_dir / "retrieval_projection.pt"
        for label, path_value, hash_field in (
            ("Memory Encoder adapter", adapter_path, "memory_encoder_adapter_hash"),
            ("Memory Encoder tokenizer", tokenizer_path, "tokenizer_hash"),
            ("Memory Encoder projection", retrieval_projection_path, "retrieval_projection_hash"),
        ):
            validate_artifact_hash(
                label=label,
                path=path_value,
                expected_hash=encoder_meta.get(hash_field),
            )
    bundle = None
    if not mock:
        adapter_path = _require_path(adapter_path, "adapter_path")
        tokenizer_path = _require_path(tokenizer_path, "tokenizer_path")
        retrieval_projection_path = _require_path(
            retrieval_projection_path,
            "retrieval_projection_path",
        )
        progress.message("loading frozen Memory Encoder and projection artifacts")
        bundle = load_planner_model_bundle(
            model_name=model_name,
            model_revision=model_revision,
            adapter_path=adapter_path,
            tokenizer_path=tokenizer_path,
            retrieval_projection_path=retrieval_projection_path,
            device=device,
            model_dtype=model_dtype,
        )

    progress.start(status="exporting")
    for index, chunk in enumerate(records):
        guideline_id = str(chunk["guideline_id"])
        h1_title = str(chunk["h1_title"])
        source_span_id = str(chunk["source_span_id"])
        guideline_dir = memory_root / safe_filename(guideline_id, max_length=120)
        guideline_dir.mkdir(parents=True, exist_ok=True)
        base_name = safe_filename(h1_title)
        key = (guideline_id, base_name)
        used_names[key] += 1
        if used_names[key] > 1:
            base_name = f"{base_name}--{stable_short_id(source_span_id, 8)}"

        slots = (
            _mock_slots(str(chunk["text"]), memory_tokens)
            if mock
            else encode_memory_slots_with_bundle(
                bundle,
                str(chunk["text"]),
                memory_tokens=memory_tokens,
                max_encoder_tokens=max_encoder_tokens,
            )
        )
        embedding = slot_embedding(slots) if mock else projected_slot_embedding_with_bundle(bundle, slots)
        latent_path = guideline_dir / f"{base_name}.pt"
        embedding_path = guideline_dir / f"{base_name}.npy"
        _save_pt(latent_path, slots)
        np.save(embedding_path, embedding.astype("float32"))

        guideline_memory_id = source_span_id
        metadata_records.append(
            {
                "guideline_memory_id": guideline_memory_id,
                "guideline_id": guideline_id,
                "guideline_name": chunk.get("guideline_name") or guideline_id,
                "version": chunk.get("version"),
                "cancer_type": chunk.get("cancer_type"),
                "chapter": chunk.get("chapter"),
                "section": chunk.get("section"),
                "h1_title": h1_title,
                "section_path": _section_path(chunk),
                "language": chunk.get("language") or _infer_language(str(chunk.get("text") or "")),
                "source_rule_ids": chunk.get("source_rule_ids", []),
                "source_span_ids": chunk.get("source_span_ids", [source_span_id]),
                "source_chunk_ids": chunk.get("source_span_ids", [source_span_id]),
                "slot_type": "h1_section",
                "latent_vector_path": str(latent_path.relative_to(output)),
                "retrieval_embedding_path": str(embedding_path.relative_to(output)),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
            }
        )
        progress.update(
            index + 1,
            metrics={"guideline": guideline_id, "section": h1_title},
        )

    progress.message("writing metadata, copying artifacts, and hashing the Memory Store")
    metadata_path = output / "slot_metadata.jsonl"
    write_jsonl(metadata_path, metadata_records)
    slot_hidden_size = _infer_slot_hidden_size(output, metadata_records)
    if not mock:
        adapter_path, tokenizer_path, retrieval_projection_path = (
            _materialize_encoder_artifacts(
                output,
                adapter_path=adapter_path,
                tokenizer_path=tokenizer_path,
                retrieval_projection_path=retrieval_projection_path,
            )
        )
    meta = _memory_store_meta(
        output=output,
        model_name=model_name,
        original_model_name=original_model_name,
        model_revision=model_revision,
        model_snapshot_path=resolved_snapshot_path,
        memory_tokens=memory_tokens,
        slot_hidden_size=slot_hidden_size,
        mock=mock,
        training_run_dir=training_run_dir,
        adapter_path=adapter_path,
        tokenizer_path=tokenizer_path,
        retrieval_projection_path=retrieval_projection_path,
        records=records,
    )
    write_json(output / MEMORY_STORE_META_FILENAME, meta)
    progress.finish(metrics={"slots": len(metadata_records)})
    return {
        "memory_dir": str(output),
        "metadata_path": str(metadata_path),
        "memory_store_meta_path": str(output / MEMORY_STORE_META_FILENAME),
        "slot_count": len(metadata_records),
        "model_name": model_name,
        "memory_tokens": memory_tokens,
        "mock": mock,
        "trained": not mock,
    }


def slot_embedding(slots: np.ndarray) -> np.ndarray:
    vector = slots.mean(axis=0)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.astype("float32")


def _load_chunks(chunks: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(chunks, (str, Path)):
        return read_jsonl(Path(chunks))
    return [dict(chunk) for chunk in chunks]


def _mock_slots(text: str, memory_tokens: int, hidden_size: int = 128) -> np.ndarray:
    seed = int(stable_short_id(text, 8), 16)
    rng = np.random.default_rng(seed)
    slots = rng.normal(size=(memory_tokens, hidden_size)).astype("float32")
    return slots


def _save_pt(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        torch.save(torch.from_numpy(array), path)
    except Exception:
        with path.open("wb") as handle:
            pickle.dump(array, handle)


def load_memory_store_meta(memory_dir: str | Path) -> dict[str, Any]:
    root = Path(memory_dir)
    path = root / MEMORY_STORE_META_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {MEMORY_STORE_META_FILENAME} in {root}. "
            "Re-run extract-memory with --training-run-dir so the planner knows "
            "which base model, adapter, tokenizer, and projection can decode the slots."
        )
    import json

    meta = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise RuntimeError(f"{path} must contain a JSON object.")
    return meta


def resolve_memory_store_path(memory_dir: str | Path, value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"memory_store_meta.json is missing required field {field_name!r}.")
    root = Path(memory_dir)
    path = Path(value)
    if path.is_absolute():
        return path
    root_relative = root / path
    if root_relative.exists():
        return root_relative
    return path


def _load_training_config(training_run_dir: str | Path | None) -> dict[str, Any]:
    if training_run_dir is None:
        return {}
    import json

    path = Path(training_run_dir) / "training_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"training_config.json not found in {training_run_dir}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object.")
    return payload


def _require_path(value: str | Path | None, field_name: str) -> Path:
    if value is None:
        raise ValueError(
            f"{field_name} is required for trained latent memory export. "
            "Pass --training-run-dir or provide explicit adapter/tokenizer/projection paths."
        )
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path


def _memory_store_meta(
    *,
    output: Path,
    model_name: str,
    original_model_name: str,
    model_revision: str | None,
    model_snapshot_path: str | None,
    memory_tokens: int,
    slot_hidden_size: int,
    mock: bool,
    training_run_dir: str | Path | None,
    adapter_path: str | Path | None,
    tokenizer_path: str | Path | None,
    retrieval_projection_path: str | Path | None,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    resolved_revision = (
        model_revision
        or (Path(model_snapshot_path).name if model_snapshot_path else None)
        or ("mock" if mock else "unversioned")
    )
    meta = {
        "format_version": 2,
        "artifact_role": "memory_store",
        "base_model": model_name,
        "base_model_name": original_model_name,
        "base_model_revision": resolved_revision,
        "base_model_snapshot_path": model_snapshot_path,
        "memory_encoder_adapter_path": _artifact_meta_path(output, adapter_path),
        "memory_encoder_adapter_hash": (
            sha256_path(adapter_path) if adapter_path is not None else "mock"
        ),
        "tokenizer_path": _artifact_meta_path(output, tokenizer_path),
        "tokenizer_hash": sha256_path(tokenizer_path) if tokenizer_path is not None else "mock",
        "retrieval_projection_path": _artifact_meta_path(
            output, retrieval_projection_path
        ),
        "retrieval_projection_hash": (
            sha256_path(retrieval_projection_path)
            if retrieval_projection_path is not None
            else "mock"
        ),
        "chunk_corpus_hash": sha256_json(records),
        "schema_hash": sha256_json(
            {
                "format_version": 2,
                "slot_metadata": sorted(
                    (
                        "guideline_memory_id",
                        "guideline_id",
                        "version",
                        "cancer_type",
                        "source_rule_ids",
                        "source_span_ids",
                        "latent_vector_path",
                        "retrieval_embedding_path",
                    )
                ),
            }
        ),
        "memory_tokens": int(memory_tokens),
        "slot_hidden_size": int(slot_hidden_size),
        "created_from_run": _string_path(training_run_dir),
        "trained": not mock,
        "mock": bool(mock),
        "metadata_path": str((output / "slot_metadata.jsonl").relative_to(output)),
    }
    meta["memory_store_fingerprint"] = build_memory_store_fingerprint(meta)
    return meta


def _string_path(value: str | Path | None) -> str | None:
    return str(value) if value is not None else None


def _materialize_encoder_artifacts(
    output: Path,
    *,
    adapter_path: str | Path | None,
    tokenizer_path: str | Path | None,
    retrieval_projection_path: str | Path | None,
) -> tuple[Path, Path, Path]:
    artifact_root = output / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    adapter = _copy_artifact(
        _require_path(adapter_path, "adapter_path"),
        artifact_root / "memory_encoder_adapter",
    )
    tokenizer = _copy_artifact(
        _require_path(tokenizer_path, "tokenizer_path"),
        artifact_root / "tokenizer",
    )
    projection = _copy_artifact(
        _require_path(retrieval_projection_path, "retrieval_projection_path"),
        artifact_root / "retrieval_projection.pt",
    )
    return adapter, tokenizer, projection


def _copy_artifact(source: Path, destination: Path) -> Path:
    source = source.resolve()
    if destination.exists():
        if sha256_path(source) != sha256_path(destination):
            raise RuntimeError(
                f"Refusing to overwrite a different exported artifact: {destination}"
            )
        return destination
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return destination


def _artifact_meta_path(output: Path, value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    try:
        return path.resolve().relative_to(output.resolve()).as_posix()
    except ValueError:
        return str(path)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _load_memory_encoder_meta(run_dir: Path) -> dict[str, Any]:
    import json

    path = run_dir / "memory_encoder_meta.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"V2 memory_encoder_meta.json not found in training run: {run_dir}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object.")
    return payload


def _section_path(chunk: Mapping[str, Any]) -> list[str]:
    explicit = chunk.get("section_path")
    if isinstance(explicit, list):
        return [str(item) for item in explicit if str(item).strip()]
    result: list[str] = []
    for key in ("chapter", "section", "h1_title"):
        value = chunk.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _infer_language(text: str) -> str:
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "zh"
    return "en"


def _infer_slot_hidden_size(output: Path, metadata_records: list[dict[str, Any]]) -> int:
    if not metadata_records:
        return 0
    first_path = output / metadata_records[0]["latent_vector_path"]
    array = _load_pt_array(first_path)
    if array.ndim != 2:
        raise RuntimeError(f"Latent vector must be a 2D array, got shape {array.shape}.")
    return int(array.shape[-1])


def _load_pt_array(path: Path) -> np.ndarray:
    try:
        import torch

        value = torch.load(path, map_location="cpu")
        if hasattr(value, "detach"):
            return value.detach().cpu().float().numpy()
    except Exception:
        pass
    with path.open("rb") as handle:
        return np.asarray(pickle.load(handle), dtype="float32")
