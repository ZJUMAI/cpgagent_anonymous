"""Portable, hash-bound Planner V2 release manifests.

The release manifest is the only supported production entrypoint for combining
the independently trained Memory Encoder, Memory Store, Planner Decoder and
Router artifacts.  Individual artifact paths remain available as explicit
debug overrides, but an overridden runtime no longer represents the immutable
release recorded by the manifest.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from guideline_planner.artifacts import (
    ArtifactBindingError,
    build_memory_store_fingerprint,
    sha256_json,
    sha256_path,
)
from guideline_planner.release_scope import (
    PLANNER_V2_LUNG_ENDOMETRIAL_RELEASE_SCOPE,
    load_release_scope,
    release_scope_hash,
)

RELEASE_SCHEMA_VERSION = "planner_release.v2"
RELEASE_MANIFEST_FILENAMES = ("planner_release.json", "planner_release.v2.json")
RELEASE_MODES = ("latent_topk", "daa_full")

DEFAULT_ACTION_SCOPE: tuple[dict[str, Any], ...] = (
    {"cancer_family": "lung", "disease_subtypes": ["nsclc"]},
    {"cancer_family": "endometrial", "disease_subtypes": ["ucec"]},
    {"cancer_family": "nasopharyngeal", "disease_subtypes": ["npc"]},
)
DEFAULT_MEMORY_SCOPE: tuple[dict[str, Any], ...] = (
    {"cancer_family": "lung", "disease_subtypes": ["nsclc", "sclc"]},
    {"cancer_family": "endometrial", "disease_subtypes": ["ucec"]},
    {
        "cancer_family": "nasopharyngeal",
        "disease_subtypes": ["npc"],
    },
)
DEFAULT_DEFERRED_ACTION_SCOPE: tuple[dict[str, Any], ...] = (
    {"cancer_family": "lung", "disease_subtypes": ["sclc"]},
)
DEFAULT_GUIDELINE_CONTEXTS: dict[str, dict[str, str]] = {
    "lung/nsclc": {
        "guideline_id": "NSCLC_2010",
        "version": "2010",
        "decision_date": "2010-12-31",
    },
    "endometrial/ucec": {
        "guideline_id": "CSCO子宫内膜癌2023",
        "version": "2023",
        "decision_date": "2023-12-31",
    },
    "nasopharyngeal/npc": {
        "guideline_id": "CSCO鼻咽癌2022",
        "version": "2022",
        "decision_date": "2022-12-31",
    },
}


@dataclass(frozen=True)
class ResolvedPlannerRelease:
    """Validated runtime view of a ``planner_release.v2`` manifest."""

    manifest_path: Path
    release_id: str
    default_mode: str
    mode: str
    dataset_dir: Path
    memory_encoder_dir: Path
    memory_dir: Path
    baseline_decoder_artifact_dir: Path
    decoder_artifact_dir: Path
    routing_config: Path | None
    routing_checkpoint: Path | None
    top_k: int
    action_scope: tuple[dict[str, Any], ...]
    memory_scope: tuple[dict[str, Any], ...]
    deferred_action_scope: tuple[dict[str, Any], ...]
    guideline_defaults: dict[str, dict[str, str]]
    base_model_name: str
    base_model_revision: str
    base_model_snapshot_path: Path
    manifest: dict[str, Any]
    debug_overrides: tuple[str, ...] = ()

    def guideline_default(
        self,
        cancer_family: str,
        disease_subtype: str | None,
    ) -> dict[str, str]:
        key = _scope_key(cancer_family, disease_subtype)
        value = self.guideline_defaults.get(key)
        if value is None:
            raise ArtifactBindingError(
                "Planner release does not define a guideline default for "
                f"{key!r}; this cancer/subtype is not in the action scope."
            )
        return dict(value)

    def supports_action(
        self,
        cancer_family: str,
        disease_subtype: str | None,
    ) -> bool:
        return _scope_contains(self.action_scope, cancer_family, disease_subtype)

    def require_supported_action(
        self,
        cancer_family: str,
        disease_subtype: str | None,
    ) -> None:
        if self.supports_action(cancer_family, disease_subtype):
            return
        key = _scope_key(cancer_family, disease_subtype)
        if _scope_contains(
            self.deferred_action_scope,
            cancer_family,
            disease_subtype,
        ):
            reason = "is deferred and currently participates in Memory Encoder training only"
        else:
            reason = "is outside the release action scope"
        raise ArtifactBindingError(f"Planner action invocation for {key!r} {reason}.")

    def with_debug_overrides(
        self,
        **overrides: Any,
    ) -> ResolvedPlannerRelease:
        allowed = {
            "memory_dir",
            "decoder_artifact_dir",
            "routing_config",
            "routing_checkpoint",
            "top_k",
        }
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise TypeError(f"Unknown Planner release debug overrides: {unknown}")
        if self.mode == "latent_topk" and any(
            overrides.get(key) is not None
            for key in ("routing_config", "routing_checkpoint")
        ):
            raise ValueError(
                "Routing debug overrides require mode='daa_full'; latent_topk must "
                "remain router-free."
            )
        values: dict[str, Any] = {}
        used: list[str] = []
        for key, value in overrides.items():
            if value is None:
                continue
            used.append(key)
            if key == "top_k":
                value = int(value)
                if value < 1:
                    raise ValueError("Planner top_k must be positive.")
            elif key in {
                "memory_dir",
                "decoder_artifact_dir",
                "routing_config",
                "routing_checkpoint",
            }:
                value = Path(value).expanduser().resolve()
                if not value.exists():
                    raise FileNotFoundError(f"Planner debug override does not exist: {value}")
            values[key] = value
        values["debug_overrides"] = tuple(sorted(set(self.debug_overrides).union(used)))
        return replace(self, **values)

    def public_config(self) -> dict[str, Any]:
        return {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": self.release_id,
            "manifest_path": str(self.manifest_path),
            "default_mode": self.default_mode,
            "mode": self.mode,
            "top_k": self.top_k,
            "base_model_name": self.base_model_name,
            "base_model_revision": self.base_model_revision,
            "base_model_snapshot_path": str(self.base_model_snapshot_path),
            "action_scope": [dict(item) for item in self.action_scope],
            "memory_scope": [dict(item) for item in self.memory_scope],
            "deferred_action_scope": [dict(item) for item in self.deferred_action_scope],
            "guideline_defaults": {
                key: dict(value) for key, value in self.guideline_defaults.items()
            },
            "debug_overrides": list(self.debug_overrides),
            "immutable_release": not self.debug_overrides,
        }


def resolve_planner_release(
    release_dir_or_manifest: str | Path,
    *,
    mode: str | None = None,
    validate_hashes: bool = True,
    overrides: Mapping[str, Any] | None = None,
) -> ResolvedPlannerRelease:
    """Load, validate and resolve a Planner release into runtime paths.

    This intentionally has no Torch dependency, so fixed-test prediction,
    MedClaw and packaging checks can share exactly the same loader.
    """

    manifest_path = _find_manifest(release_dir_or_manifest)
    payload = _read_object(manifest_path)
    if payload.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise ArtifactBindingError(
            f"Planner release must use schema_version={RELEASE_SCHEMA_VERSION!r}."
        )
    root = manifest_path.parent
    default_mode = _release_mode(payload.get("default_mode"), field="default_mode")
    selected_mode = _release_mode(mode or default_mode, field="mode")
    top_k = int(payload.get("top_k") or 0)
    if top_k < 1:
        raise ArtifactBindingError("Planner release top_k must be positive.")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ArtifactBindingError("Planner release artifacts must be an object.")
    resolved_refs: dict[str, Path | None] = {}
    for name in (
        "dataset",
        "memory_encoder",
        "memory_store",
        "baseline_decoder",
        "runtime_decoder",
        "routing_config",
        "routing_checkpoint",
    ):
        required = name in {
            "dataset",
            "memory_encoder",
            "memory_store",
            "baseline_decoder",
            "runtime_decoder",
        }
        resolved_refs[name] = _resolve_artifact_ref(
            root,
            artifacts.get(name),
            name=name,
            required=required,
            validate_hashes=validate_hashes,
        )
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ArtifactBindingError("Planner release bindings must be an object.")
    for artifact_name, binding_name in (
        ("dataset", "dataset_hash"),
        ("memory_encoder", "memory_encoder_hash"),
        ("memory_store", "memory_store_hash"),
        ("baseline_decoder", "baseline_decoder_hash"),
        ("runtime_decoder", "runtime_decoder_hash"),
        ("routing_config", "routing_config_hash"),
        ("routing_checkpoint", "routing_checkpoint_hash"),
    ):
        reference = artifacts.get(artifact_name)
        expected = reference.get("sha256") if isinstance(reference, Mapping) else None
        recorded = bindings.get(binding_name)
        if expected is None and recorded is None:
            continue
        _equal(recorded, expected, f"release binding {binding_name}")

    if selected_mode == "daa_full":
        if resolved_refs["routing_config"] is None or resolved_refs["routing_checkpoint"] is None:
            raise ArtifactBindingError(
                "daa_full release mode requires routing_config and routing_checkpoint."
            )
        decoder_dir = resolved_refs["runtime_decoder"]
    else:
        decoder_dir = resolved_refs["baseline_decoder"]

    raw_scope = payload.get("scope")
    if not isinstance(raw_scope, Mapping):
        raise ArtifactBindingError("Planner release scope must be an object.")
    try:
        canonical_scope = load_release_scope(raw_scope)
    except ValueError as exc:
        raise ArtifactBindingError(f"Invalid Planner release scope: {exc}") from exc
    actual_scope_hash = release_scope_hash(canonical_scope)
    if payload.get("release_scope_hash") != actual_scope_hash:
        raise ArtifactBindingError("Planner release scope hash mismatch.")
    _equal(
        bindings.get("release_scope_hash"),
        actual_scope_hash,
        "release binding release_scope_hash",
    )
    trajectory_dataset_hash = _validate_dataset_release(
        _required_path(resolved_refs["dataset"]),
        release_scope_hash_value=actual_scope_hash,
    )
    _equal(
        bindings.get("trajectory_dataset_hash"),
        trajectory_dataset_hash,
        "release binding trajectory_dataset_hash",
    )
    action_scope = _normalize_scope(
        canonical_scope.get("action_targets"),
        "scope.action_targets",
    )
    memory_scope = _normalize_scope(
        _memory_scope_targets(canonical_scope.get("memory_guidelines")),
        "scope.memory_guidelines",
    )
    deferred = _normalize_scope(
        canonical_scope.get("deferred_action_targets", []),
        "scope.deferred_action_targets",
        allow_empty=True,
    )
    defaults = _normalize_guideline_defaults(payload.get("guideline_defaults"))
    for entry in action_scope:
        for subtype in entry["disease_subtypes"]:
            key = _scope_key(entry["cancer_family"], subtype)
            if key not in defaults:
                raise ArtifactBindingError(
                    f"Planner release action scope {key!r} has no guideline default."
                )
            default = defaults[key]
            permitted_ids = _action_guideline_ids(
                canonical_scope,
                entry["cancer_family"],
                subtype,
            )
            if "*" not in permitted_ids and default["guideline_id"] not in permitted_ids:
                raise ArtifactBindingError(
                    f"Guideline default for {key!r} is not allowed by release scope."
                )
            if not _memory_scope_has_guideline(
                canonical_scope,
                default["guideline_id"],
                default["version"],
            ):
                raise ArtifactBindingError(
                    f"Guideline default for {key!r} is absent from memory scope: "
                    f"{default['guideline_id']}@{default['version']}."
                )

    base_model = payload.get("base_model")
    if not isinstance(base_model, Mapping):
        raise ArtifactBindingError("Planner release base_model must be an object.")
    model_name = _nonempty(base_model.get("name"), "base_model.name")
    model_revision = _nonempty(base_model.get("revision"), "base_model.revision")
    snapshot_value = _nonempty(
        base_model.get("snapshot_path"),
        "base_model.snapshot_path",
    )
    snapshot_path = Path(snapshot_value).expanduser().resolve()
    if not snapshot_path.is_dir():
        raise FileNotFoundError(
            f"Planner release base-model snapshot is missing: {snapshot_path}"
        )
    identity = sha256_json({"name": model_name, "revision": model_revision})
    if base_model.get("identity_hash") != identity:
        raise ArtifactBindingError("Planner release base-model identity hash mismatch.")
    _equal(
        bindings.get("base_model_identity_hash"),
        identity,
        "release binding base_model_identity_hash",
    )

    _validate_cross_stage_bindings(
        payload,
        memory_encoder_dir=_required_path(resolved_refs["memory_encoder"]),
        memory_dir=_required_path(resolved_refs["memory_store"]),
        baseline_decoder_dir=_required_path(resolved_refs["baseline_decoder"]),
        runtime_decoder_dir=_required_path(resolved_refs["runtime_decoder"]),
        model_name=model_name,
        model_revision=model_revision,
        model_snapshot_path=snapshot_path,
        trajectory_dataset_hash=trajectory_dataset_hash,
    )

    result = ResolvedPlannerRelease(
        manifest_path=manifest_path,
        release_id=_nonempty(payload.get("release_id"), "release_id"),
        default_mode=default_mode,
        mode=selected_mode,
        dataset_dir=_required_path(resolved_refs["dataset"]),
        memory_encoder_dir=_required_path(resolved_refs["memory_encoder"]),
        memory_dir=_required_path(resolved_refs["memory_store"]),
        baseline_decoder_artifact_dir=_required_path(
            resolved_refs["baseline_decoder"]
        ),
        decoder_artifact_dir=_required_path(decoder_dir),
        routing_config=(
            _required_path(resolved_refs["routing_config"])
            if selected_mode == "daa_full"
            else None
        ),
        routing_checkpoint=(
            _required_path(resolved_refs["routing_checkpoint"])
            if selected_mode == "daa_full"
            else None
        ),
        top_k=top_k,
        action_scope=action_scope,
        memory_scope=memory_scope,
        deferred_action_scope=deferred,
        guideline_defaults=defaults,
        base_model_name=model_name,
        base_model_revision=model_revision,
        base_model_snapshot_path=snapshot_path,
        manifest=dict(payload),
    )
    if overrides:
        result = result.with_debug_overrides(**dict(overrides))
    return result


def build_planner_release_manifest(
    output_dir: str | Path,
    *,
    release_id: str,
    dataset_dir: str | Path,
    memory_encoder_dir: str | Path,
    memory_dir: str | Path,
    baseline_decoder_dir: str | Path,
    runtime_decoder_dir: str | Path | None = None,
    routing_config: str | Path | None = None,
    routing_checkpoint: str | Path | None = None,
    default_mode: str = "latent_topk",
    top_k: int = 4,
    release_scope: str | Path | Mapping[str, Any] | None = None,
    guideline_defaults: Mapping[str, Mapping[str, Any]] | None = None,
    quality_gate_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a fully bound release manifest and immediately validate it."""

    mode = _release_mode(default_mode, field="default_mode")
    if int(top_k) < 1:
        raise ValueError("Planner release top_k must be positive.")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / RELEASE_MANIFEST_FILENAMES[0]
    if manifest_path.exists():
        raise FileExistsError(
            f"Planner release manifest already exists and will not be overwritten: "
            f"{manifest_path}"
        )
    paths = {
        "dataset": Path(dataset_dir).expanduser().resolve(),
        "memory_encoder": Path(memory_encoder_dir).expanduser().resolve(),
        "memory_store": Path(memory_dir).expanduser().resolve(),
        "baseline_decoder": Path(baseline_decoder_dir).expanduser().resolve(),
        "runtime_decoder": Path(
            runtime_decoder_dir or baseline_decoder_dir
        ).expanduser().resolve(),
        "routing_config": (
            Path(routing_config).expanduser().resolve() if routing_config else None
        ),
        "routing_checkpoint": (
            Path(routing_checkpoint).expanduser().resolve() if routing_checkpoint else None
        ),
    }
    if mode == "daa_full" and (
        paths["routing_config"] is None or paths["routing_checkpoint"] is None
    ):
        raise ArtifactBindingError(
            "Cannot package a daa_full release without routing config and checkpoint."
        )
    for name, path in paths.items():
        if path is not None and not path.exists():
            raise FileNotFoundError(f"Planner release artifact {name!r} does not exist: {path}")

    memory_meta = _read_object(paths["memory_store"] / "memory_store_meta.json")
    model_name = _nonempty(
        memory_meta.get("base_model_name") or memory_meta.get("base_model"),
        "memory_store.base_model_name",
    )
    model_revision = _nonempty(
        memory_meta.get("base_model_revision"),
        "memory_store.base_model_revision",
    )
    model_snapshot_path = Path(
        _nonempty(
            memory_meta.get("base_model_snapshot_path"),
            "memory_store.base_model_snapshot_path",
        )
    ).expanduser().resolve()
    if not model_snapshot_path.is_dir():
        raise FileNotFoundError(
            f"Base-model snapshot recorded by Memory Store is missing: "
            f"{model_snapshot_path}"
        )
    artifacts = {
        name: _artifact_ref(output, path)
        for name, path in paths.items()
    }
    baseline_meta = _read_object(
        paths["baseline_decoder"] / "planner_decoder_meta.json"
    )
    runtime_meta = _read_object(
        paths["runtime_decoder"] / "planner_decoder_meta.json"
    )
    encoder_meta = _read_object(
        paths["memory_encoder"] / "memory_encoder_meta.json"
    )
    try:
        canonical_scope = load_release_scope(
            release_scope
            if release_scope is not None
            else PLANNER_V2_LUNG_ENDOMETRIAL_RELEASE_SCOPE
        )
    except ValueError as exc:
        raise ArtifactBindingError(f"Invalid Planner release scope: {exc}") from exc
    scope_hash = release_scope_hash(canonical_scope)
    if guideline_defaults is None:
        guideline_defaults = _guideline_defaults_for_action_scope(canonical_scope)
    trajectory_dataset_hash = _validate_dataset_release(
        paths["dataset"],
        release_scope_hash_value=scope_hash,
    )
    model_identity_hash = sha256_json(
        {"name": model_name, "revision": model_revision}
    )
    bindings = {
        "dataset_hash": artifacts["dataset"]["sha256"],
        "memory_encoder_hash": artifacts["memory_encoder"]["sha256"],
        "memory_encoder_adapter_hash": memory_meta.get("memory_encoder_adapter_hash"),
        "memory_store_hash": artifacts["memory_store"]["sha256"],
        "memory_store_fingerprint": memory_meta.get("memory_store_fingerprint"),
        "tokenizer_hash": memory_meta.get("tokenizer_hash"),
        "retrieval_projection_hash": memory_meta.get("retrieval_projection_hash"),
        "baseline_decoder_hash": artifacts["baseline_decoder"]["sha256"],
        "baseline_decoder_adapter_hash": baseline_meta.get(
            "planner_decoder_adapter_hash"
        ),
        "baseline_bridge_hash": baseline_meta.get("memory_to_decoder_bridge_hash"),
        "runtime_decoder_hash": artifacts["runtime_decoder"]["sha256"],
        "runtime_decoder_adapter_hash": runtime_meta.get(
            "planner_decoder_adapter_hash"
        ),
        "runtime_bridge_hash": runtime_meta.get("memory_to_decoder_bridge_hash"),
        "routing_config_hash": (
            artifacts["routing_config"]["sha256"]
            if artifacts["routing_config"] is not None
            else None
        ),
        "routing_checkpoint_hash": (
            artifacts["routing_checkpoint"]["sha256"]
            if artifacts["routing_checkpoint"] is not None
            else None
        ),
        "release_scope_hash": scope_hash,
        "base_model_identity_hash": model_identity_hash,
        "trajectory_dataset_hash": trajectory_dataset_hash,
    }
    # Fail before writing if any of the essential independently recorded hashes
    # are missing. This prevents a legacy adapter_path-only artifact from being
    # promoted into a V2 release.
    for key in (
        "memory_encoder_adapter_hash",
        "memory_store_fingerprint",
        "tokenizer_hash",
        "retrieval_projection_hash",
        "baseline_decoder_adapter_hash",
        "baseline_bridge_hash",
        "runtime_decoder_adapter_hash",
        "runtime_bridge_hash",
    ):
        _nonempty(bindings.get(key), f"bindings.{key}")
    _nonempty(encoder_meta.get("memory_encoder_adapter_hash"), "encoder adapter hash")

    payload = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_id": _nonempty(release_id, "release_id"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "default_mode": mode,
        "top_k": int(top_k),
        "base_model": {
            "name": model_name,
            "revision": model_revision,
            "snapshot_path": str(model_snapshot_path),
            "identity_hash": model_identity_hash,
        },
        "artifacts": artifacts,
        "bindings": bindings,
        "scope": canonical_scope,
        "release_scope_hash": scope_hash,
        "guideline_defaults": {
            str(key): dict(value) for key, value in guideline_defaults.items()
        },
    }
    if quality_gate_report is not None:
        payload["quality_gates"] = dict(quality_gate_report)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resolve_planner_release(manifest_path, validate_hashes=True)
    return {**payload, "manifest_path": str(manifest_path)}


def _validate_cross_stage_bindings(
    payload: Mapping[str, Any],
    *,
    memory_encoder_dir: Path,
    memory_dir: Path,
    baseline_decoder_dir: Path,
    runtime_decoder_dir: Path,
    model_name: str,
    model_revision: str,
    model_snapshot_path: Path,
    trajectory_dataset_hash: str,
) -> None:
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ArtifactBindingError("Planner release bindings must be an object.")
    memory_meta = _read_object(memory_dir / "memory_store_meta.json")
    if memory_meta.get("artifact_role") != "memory_store" or not memory_meta.get("trained"):
        raise ArtifactBindingError("Planner release requires a trained V2 Memory Store.")
    if memory_meta.get("memory_store_fingerprint") != build_memory_store_fingerprint(
        memory_meta
    ):
        raise ArtifactBindingError("Memory Store lineage fingerprint is invalid.")
    encoder_meta = _read_object(memory_encoder_dir / "memory_encoder_meta.json")
    if encoder_meta.get("artifact_role") != "memory_encoder":
        raise ArtifactBindingError("Planner release memory_encoder has the wrong role.")
    baseline_meta = _read_object(baseline_decoder_dir / "planner_decoder_meta.json")
    runtime_meta = _read_object(runtime_decoder_dir / "planner_decoder_meta.json")
    for label, meta in (
        ("baseline decoder", baseline_meta),
        ("runtime decoder", runtime_meta),
    ):
        if meta.get("artifact_role") != "planner_decoder":
            raise ArtifactBindingError(f"Planner release {label} has the wrong role.")
        _equal(
            meta.get("memory_store_fingerprint"),
            memory_meta.get("memory_store_fingerprint"),
            f"{label} memory-store fingerprint",
        )
        _equal(
            meta.get("memory_encoder_adapter_hash"),
            memory_meta.get("memory_encoder_adapter_hash"),
            f"{label} encoder adapter hash",
        )
        _equal(
            meta.get("base_model_revision"),
            model_revision,
            f"{label} base-model revision",
        )
        _equal(
            meta.get("trajectory_dataset_hash"),
            trajectory_dataset_hash,
            f"{label} trajectory dataset hash",
        )
        _equal(
            Path(str(meta.get("base_model_snapshot_path") or "")).expanduser().resolve(),
            model_snapshot_path,
            f"{label} base-model snapshot",
        )
    _equal(
        encoder_meta.get("memory_encoder_adapter_hash"),
        memory_meta.get("memory_encoder_adapter_hash"),
        "Memory Encoder adapter hash",
    )
    _equal(
        encoder_meta.get("tokenizer_hash"),
        memory_meta.get("tokenizer_hash"),
        "Memory Encoder tokenizer hash",
    )
    _equal(
        encoder_meta.get("retrieval_projection_hash"),
        memory_meta.get("retrieval_projection_hash"),
        "Memory Encoder projection hash",
    )
    encoder_name = encoder_meta.get("base_model_name")
    if encoder_name not in (None, ""):
        _equal(encoder_name, model_name, "Memory Encoder base-model name")
    _equal(
        encoder_meta.get("base_model_revision"),
        model_revision,
        "Memory Encoder base-model revision",
    )
    _equal(
        Path(
            str(encoder_meta.get("base_model_snapshot_path") or "")
        ).expanduser().resolve(),
        model_snapshot_path,
        "Memory Encoder base-model snapshot",
    )
    recorded = {
        "memory_encoder_adapter_hash": memory_meta.get("memory_encoder_adapter_hash"),
        "memory_store_fingerprint": memory_meta.get("memory_store_fingerprint"),
        "tokenizer_hash": memory_meta.get("tokenizer_hash"),
        "retrieval_projection_hash": memory_meta.get("retrieval_projection_hash"),
        "baseline_decoder_adapter_hash": baseline_meta.get(
            "planner_decoder_adapter_hash"
        ),
        "baseline_bridge_hash": baseline_meta.get("memory_to_decoder_bridge_hash"),
        "runtime_decoder_adapter_hash": runtime_meta.get(
            "planner_decoder_adapter_hash"
        ),
        "runtime_bridge_hash": runtime_meta.get("memory_to_decoder_bridge_hash"),
    }
    for key, actual in recorded.items():
        _equal(bindings.get(key), actual, f"release binding {key}")


def _validate_dataset_release(
    dataset_dir: Path,
    *,
    release_scope_hash_value: str,
) -> str:
    manifest = _read_object(dataset_dir / "manifest.json")
    admission = _read_object(dataset_dir / "admission_report.json")
    if not bool(manifest.get("ready")) or not bool(admission.get("ready")):
        raise ArtifactBindingError(
            "Planner release requires a training-ready trajectory dataset."
        )
    errors = admission.get("errors")
    if errors not in (None, []):
        raise ArtifactBindingError(
            "Planner release dataset admission report contains errors."
        )
    _equal(
        manifest.get("release_scope_hash"),
        release_scope_hash_value,
        "trajectory dataset release-scope hash",
    )
    _equal(
        admission.get("release_scope_hash"),
        release_scope_hash_value,
        "trajectory admission release-scope hash",
    )
    dataset_hash = _nonempty(
        manifest.get("dataset_hash"),
        "trajectory dataset manifest dataset_hash",
    )
    return dataset_hash


def _find_manifest(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Planner release does not exist: {path}")
    matches = [path / name for name in RELEASE_MANIFEST_FILENAMES if (path / name).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Planner release directory must contain exactly one of "
            f"{RELEASE_MANIFEST_FILENAMES}: {path}"
        )
    return matches[0]


def _resolve_artifact_ref(
    root: Path,
    value: Any,
    *,
    name: str,
    required: bool,
    validate_hashes: bool,
) -> Path | None:
    if value is None:
        if required:
            raise ArtifactBindingError(f"Planner release is missing artifact {name!r}.")
        return None
    if not isinstance(value, Mapping):
        raise ArtifactBindingError(f"Planner release artifact {name!r} must be an object.")
    raw_path = _nonempty(value.get("path"), f"artifacts.{name}.path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = (root / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Planner release artifact {name!r} is missing: {path}")
    expected = _nonempty(value.get("sha256"), f"artifacts.{name}.sha256")
    if validate_hashes:
        actual = sha256_path(path)
        if actual != expected:
            raise ArtifactBindingError(
                f"Planner release artifact {name!r} hash mismatch: "
                f"expected={expected}, actual={actual}."
            )
    return path


def _artifact_ref(root: Path, path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    return {
        "path": Path(os.path.relpath(path, root)).as_posix(),
        "sha256": _nonempty(sha256_path(path), f"artifact hash for {path}"),
    }


def _normalize_scope(
    value: Any,
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ArtifactBindingError(f"{label} must be a non-empty list.")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ArtifactBindingError(f"{label}[{index}] must be an object.")
        family = _canonical_family(item.get("cancer_family"))
        raw_subtypes = item.get("disease_subtypes")
        if not isinstance(raw_subtypes, list) or not raw_subtypes:
            raise ArtifactBindingError(
                f"{label}[{index}].disease_subtypes must be non-empty."
            )
        subtypes = sorted({_canonical_subtype(family, value) for value in raw_subtypes})
        for subtype in subtypes:
            pair = (family, subtype)
            if pair in seen:
                raise ArtifactBindingError(f"Duplicate {label} entry: {_scope_key(*pair)}")
            seen.add(pair)
        result.append({"cancer_family": family, "disease_subtypes": subtypes})
    return tuple(result)


def _memory_scope_targets(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ArtifactBindingError("scope.memory_guidelines must be a non-empty list.")
    grouped: dict[str, set[str]] = {}
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ArtifactBindingError(
                f"scope.memory_guidelines[{index}] must be an object."
            )
        family = _canonical_family(item.get("cancer_family"))
        raw_subtypes = item.get("disease_subtypes")
        if not isinstance(raw_subtypes, list) or not raw_subtypes:
            raise ArtifactBindingError(
                f"scope.memory_guidelines[{index}].disease_subtypes must be non-empty."
            )
        grouped.setdefault(family, set()).update(
            _canonical_subtype(family, subtype) for subtype in raw_subtypes
        )
    return [
        {"cancer_family": family, "disease_subtypes": sorted(subtypes)}
        for family, subtypes in sorted(grouped.items())
    ]


def _guideline_defaults_for_action_scope(
    scope: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Select decision-time defaults for every action-enabled cancer subtype."""

    result: dict[str, dict[str, str]] = {}
    for item in scope.get("action_targets", []):
        if not isinstance(item, Mapping):
            continue
        family = _canonical_family(item.get("cancer_family"))
        for raw_subtype in item.get("disease_subtypes", []):
            subtype = _canonical_subtype(family, raw_subtype)
            key = _scope_key(family, subtype)
            context = DEFAULT_GUIDELINE_CONTEXTS.get(key)
            if context is None:
                raise ArtifactBindingError(
                    f"Planner release action scope {key!r} has no known decision-time "
                    "guideline default."
                )
            result[key] = dict(context)
    return result


def _action_guideline_ids(
    scope: Mapping[str, Any],
    family: str,
    subtype: str,
) -> set[str]:
    result: set[str] = set()
    for item in scope.get("action_targets", []):
        if not isinstance(item, Mapping) or item.get("cancer_family") != family:
            continue
        subtypes = item.get("disease_subtypes", [])
        if subtype in subtypes or "*" in subtypes:
            result.update(str(value) for value in item.get("guideline_ids", []))
    return result


def _memory_scope_has_guideline(
    scope: Mapping[str, Any],
    guideline_id: str,
    version: str,
) -> bool:
    return any(
        isinstance(item, Mapping)
        and str(item.get("guideline_id") or "") == guideline_id
        and str(item.get("version") or "") == version
        for item in scope.get("memory_guidelines", [])
    )


def _normalize_guideline_defaults(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or not value:
        raise ArtifactBindingError("Planner release guideline_defaults must be non-empty.")
    result: dict[str, dict[str, str]] = {}
    for raw_key, raw_context in value.items():
        if not isinstance(raw_context, Mapping):
            raise ArtifactBindingError(f"Guideline default {raw_key!r} must be an object.")
        parts = str(raw_key).split("/", 1)
        if len(parts) != 2:
            raise ArtifactBindingError(
                f"Guideline default key {raw_key!r} must be family/subtype."
            )
        key = _scope_key(parts[0], parts[1])
        result[key] = {
            "guideline_id": _nonempty(
                raw_context.get("guideline_id"), f"guideline_defaults.{key}.guideline_id"
            ),
            "version": _nonempty(
                raw_context.get("version"), f"guideline_defaults.{key}.version"
            ),
            "decision_date": _nonempty(
                raw_context.get("decision_date"),
                f"guideline_defaults.{key}.decision_date",
            ),
        }
    return result


def _scope_contains(
    scope: Sequence[Mapping[str, Any]],
    family: str,
    subtype: str | None,
) -> bool:
    canonical_family = _canonical_family(family)
    canonical_subtype = _canonical_subtype(canonical_family, subtype)
    return any(
        item.get("cancer_family") == canonical_family
        and canonical_subtype in item.get("disease_subtypes", [])
        for item in scope
    )


def _scope_key(family: Any, subtype: Any) -> str:
    canonical_family = _canonical_family(family)
    return f"{canonical_family}/{_canonical_subtype(canonical_family, subtype)}"


def _canonical_family(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {"ucec": "endometrial", "npc": "nasopharyngeal"}
    text = aliases.get(text, text)
    if text not in {"lung", "endometrial", "nasopharyngeal"}:
        raise ArtifactBindingError(f"Unsupported Planner cancer family: {value!r}.")
    return text


def _canonical_subtype(family: str, value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    aliases = {
        "endometrial_carcinoma": "ucec",
        "nasopharyngeal": "npc",
        "nasopharyngeal_carcinoma": "npc",
    }
    text = aliases.get(text, text)
    if not text:
        text = "unknown"
    return text


def _release_mode(value: Any, *, field: str) -> str:
    mode = str(value or "").strip().lower()
    if mode not in RELEASE_MODES:
        raise ArtifactBindingError(f"{field} must be one of {RELEASE_MODES}, got {value!r}.")
    return mode


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Planner release metadata is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactBindingError(f"Planner metadata must be an object: {path}")
    return payload


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual in (None, "") or expected in (None, "") or str(actual) != str(expected):
        raise ArtifactBindingError(
            f"{label} mismatch: actual={actual!r}, expected={expected!r}."
        )


def _nonempty(value: Any, label: str) -> str:
    if value in (None, ""):
        raise ArtifactBindingError(f"Planner release is missing {label}.")
    return str(value)


def _required_path(value: Path | None) -> Path:
    if value is None:
        raise AssertionError("Required Planner release path was not resolved.")
    return value
