"""Content hashes and explicit cross-stage artifact bindings for Planner V2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class ArtifactBindingError(RuntimeError):
    pass


def sha256_path(value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Artifact path does not exist: {path}")
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.name.encode("utf-8"))
        _update_file_digest(digest, path)
        return digest.hexdigest()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        _update_file_digest(digest, item)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_memory_store_fingerprint(meta: Mapping[str, Any]) -> str:
    required = (
        "base_model_name",
        "base_model_revision",
        "tokenizer_hash",
        "memory_encoder_adapter_hash",
        "retrieval_projection_hash",
        "chunk_corpus_hash",
        "memory_tokens",
        "slot_hidden_size",
        "schema_hash",
    )
    missing = [key for key in required if meta.get(key) in (None, "")]
    if missing:
        raise ArtifactBindingError(
            "Cannot fingerprint memory store; missing fields: " + ", ".join(missing)
        )
    return sha256_json({key: meta[key] for key in required})


def validate_artifact_hash(
    *,
    label: str,
    path: str | Path,
    expected_hash: str | None,
) -> str:
    if not expected_hash:
        raise ArtifactBindingError(f"{label} does not record an expected content hash.")
    actual = sha256_path(path)
    if actual != expected_hash:
        raise ArtifactBindingError(
            f"{label} content hash mismatch: expected={expected_hash}, actual={actual}."
        )
    return str(actual)


def validate_binding(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    fields: tuple[str, ...],
    label: str,
) -> None:
    mismatches = []
    for field in fields:
        left = actual.get(field)
        right = expected.get(field)
        if left in (None, "") or right in (None, ""):
            mismatches.append(f"{field}=missing")
        elif str(left) != str(right):
            mismatches.append(f"{field}: actual={left!r}, expected={right!r}")
    if mismatches:
        raise ArtifactBindingError(f"{label} binding mismatch: " + "; ".join(mismatches))


def _update_file_digest(digest: Any, path: Path) -> None:
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
