"""Configuration and ablation profiles for latent guideline routing."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RetrievalRoutingConfig:
    seed_top_k: int = 4
    max_candidates: int = 24
    similarity: str = "cosine"
    filter_cancer_type: bool = True
    filter_guideline_id: bool = True
    filter_version: bool = True
    filter_language: bool = True


@dataclass(frozen=True)
class GateRoutingConfig:
    enabled: bool = True
    active_top_k: int = 4
    temperature: float = 1.0
    routing_dim: int = 256
    use_seed_score: bool = True


@dataclass(frozen=True)
class DynamicAnchorAttentionConfig:
    enabled: bool = True
    num_heads: int = 8
    ffn_multiplier: int = 4
    dropout: float = 0.0
    attention_dropout: float = 0.0
    anchor_strategy: str = "highest_gate_weight"
    use_patient_state_condition: bool = True
    use_gate_attention_bias: bool = True
    use_rank_embedding: bool = True
    use_memory_identity_embedding: bool = False
    max_num_memories: int = 4
    train_with_variable_k: bool = True
    min_memories: int = 1
    max_memories: int = 4
    memory_dropout: float = 0.1
    bypass_single_memory: bool = False
    return_attention_weights: bool = False
    log_memory_attention_mass: bool = True
    eps: float = 1e-8


@dataclass(frozen=True)
class MemoryFusionConfig:
    strategy: str = "dynamic_anchor_attention"
    output_num_emb: int | None = None
    add_memory_identity_embedding: bool = True
    max_active_memories: int = 32
    dynamic_anchor_attention: DynamicAnchorAttentionConfig = field(
        default_factory=DynamicAnchorAttentionConfig
    )


@dataclass(frozen=True)
class ExpertRoutingConfig:
    """Deprecated, disabled compatibility view for historical imports."""

    enabled: bool = False
    num_proposal_tokens: int = 0
    expert_top_k: int = 0
    confidence_weight: float = 0.0
    progress_weight: float = 0.0
    transformer_layers: int = 0
    relation_head_enabled: bool = False


@dataclass(frozen=True)
class RoutingTrainingConfig:
    use_utility_distillation: bool = False
    utility_masks_per_sample: int = 1
    utility_temperature: float = 1.0
    train_decoder_lora: bool = True
    lambda_action: float = 1.0
    lambda_retrieval: float = 0.3
    lambda_utility: float = 0.0
    lambda_provenance: float = 0.2
    lambda_gate_margin: float = 0.1
    lambda_sparse: float = 0.01
    gate_margin: float = 0.1
    retrieval_pretrain_steps: int = 100
    identity_warmup_steps: int = 100
    distillation_steps: int = 0
    lambda_identity: float = 1.0
    lambda_identity_cosine: float = 0.1
    lambda_distillation: float = 1.0
    lambda_attention_entropy: float = 0.0
    lambda_anchor: float = 0.05
    distillation_temperature: float = 1.0
    joint_calibration_fraction: float = 0.10
    joint_decoder_lr_multiplier: float = 0.10


@dataclass(frozen=True)
class RoutingRuntimeConfig:
    mode: str = "test"
    checkpoint_path: str | None = None
    allow_untrained: bool = False
    save_daa_attention_tensors: bool = False


@dataclass(frozen=True)
class LatentGuidelineRoutingConfig:
    enabled: bool = False
    profile: str = "dynamic_anchor"
    retrieval: RetrievalRoutingConfig = field(default_factory=RetrievalRoutingConfig)
    gate: GateRoutingConfig = field(default_factory=GateRoutingConfig)
    memory_fusion: MemoryFusionConfig = field(default_factory=MemoryFusionConfig)
    training: RoutingTrainingConfig = field(default_factory=RoutingTrainingConfig)
    runtime: RoutingRuntimeConfig = field(default_factory=RoutingRuntimeConfig)

    @property
    def expert(self) -> ExpertRoutingConfig:
        """Expose an always-disabled legacy view without rejoining the main path."""

        return ExpertRoutingConfig()

    @classmethod
    def from_value(
        cls,
        value: "LatentGuidelineRoutingConfig | Mapping[str, Any] | str | Path | None",
    ) -> "LatentGuidelineRoutingConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            payload = _read_yaml_or_json(Path(value))
        elif isinstance(value, Mapping):
            payload = dict(value)
        else:
            raise TypeError(f"Unsupported routing config type: {type(value).__name__}")
        if "latent_guideline_routing" in payload:
            nested = payload["latent_guideline_routing"]
            if not isinstance(nested, Mapping):
                raise ValueError("latent_guideline_routing must be a mapping.")
            payload = dict(nested)
        if "expert" in payload:
            warnings.warn(
                "The routing 'expert' section is deprecated and ignored because "
                "Expert Competition was removed from the planner path.",
                DeprecationWarning,
                stacklevel=2,
            )
        removed = sorted(set(payload).intersection({"pathway", "transition_graph"}))
        if removed:
            raise ValueError(
                "Transition-graph/pathway routing was removed; delete config sections: "
                + ", ".join(removed)
            )
        config = cls(
            enabled=bool(payload.get("enabled", False)),
            profile=str(payload.get("profile") or "dynamic_anchor"),
            retrieval=_section(RetrievalRoutingConfig, payload, "retrieval"),
            gate=_section(GateRoutingConfig, payload, "gate"),
            memory_fusion=_memory_fusion_section(payload),
            training=_section(RoutingTrainingConfig, payload, "training"),
            runtime=_section(RoutingRuntimeConfig, payload, "runtime"),
        )
        return config.apply_profile(config.profile)

    def apply_profile(self, profile: str | None) -> "LatentGuidelineRoutingConfig":
        name = (profile or self.profile).strip().lower().replace("-", "_")
        if name in {"soft_expert", "soft_expert_pathway", "dynamic_anchor_pathway"}:
            warnings.warn(
                f"Routing profile {name!r} is no longer supported because pathway "
                "routing was removed; use 'dynamic_anchor'.",
                DeprecationWarning,
                stacklevel=2,
            )
            name = "dynamic_anchor"
        if name in {"default", "dynamic_anchor"}:
            return replace(
                self,
                profile="dynamic_anchor",
                memory_fusion=replace(
                    self.memory_fusion,
                    strategy="dynamic_anchor_attention",
                ),
            )
        if name == "text_rag":
            return replace(self, enabled=False, profile=name)
        if name == "latent_top1":
            return replace(
                self,
                enabled=True,
                profile=name,
                gate=replace(self.gate, enabled=False, active_top_k=1),
                memory_fusion=replace(self.memory_fusion, strategy="top1_only"),
            )
        if name == "latent_topk_concatenation":
            return replace(
                self,
                enabled=True,
                profile=name,
                gate=replace(self.gate, enabled=False),
                memory_fusion=replace(
                    self.memory_fusion,
                    strategy="weighted_concatenation",
                ),
            )
        if name == "soft_memory_mixture":
            return replace(
                self,
                enabled=True,
                profile=name,
                gate=replace(self.gate, enabled=True),
                memory_fusion=replace(
                    self.memory_fusion,
                    strategy="dynamic_anchor_attention",
                ),
            )
        raise ValueError(
            "Unknown routing profile. Expected one of text_rag, latent_top1, "
            "latent_topk_concatenation, soft_memory_mixture, or dynamic_anchor."
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _section(cls: Any, payload: Mapping[str, Any], key: str) -> Any:
    value = payload.get(key, {})
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Routing config section {key!r} must be a mapping.")
    value = dict(value)
    if cls is RoutingTrainingConfig and "lambda_confidence" in value:
        warnings.warn(
            "training.lambda_confidence is ignored because Expert Competition "
            "was removed from the planner path.",
            DeprecationWarning,
            stacklevel=2,
        )
        value.pop("lambda_confidence")
    if cls is RoutingRuntimeConfig and "save_expert_tensors" in value:
        warnings.warn(
            "runtime.save_expert_tensors is deprecated; use "
            "save_daa_attention_tensors.",
            DeprecationWarning,
            stacklevel=2,
        )
        value.setdefault("save_daa_attention_tensors", value.pop("save_expert_tensors"))
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in routing config section {key!r}: {unknown}")
    return cls(**dict(value))


def _memory_fusion_section(payload: Mapping[str, Any]) -> MemoryFusionConfig:
    value = payload.get("memory_fusion", {}) or {}
    if not isinstance(value, Mapping):
        raise ValueError("Routing config section 'memory_fusion' must be a mapping.")
    data = dict(value)
    daa_value = data.pop("dynamic_anchor_attention", {}) or {}
    if not isinstance(daa_value, Mapping):
        raise ValueError("memory_fusion.dynamic_anchor_attention must be a mapping.")
    daa_allowed = set(DynamicAnchorAttentionConfig.__dataclass_fields__)
    daa_unknown = sorted(set(daa_value) - daa_allowed)
    if daa_unknown:
        raise ValueError(
            "Unknown keys in memory_fusion.dynamic_anchor_attention: "
            f"{daa_unknown}"
        )
    allowed = set(MemoryFusionConfig.__dataclass_fields__) - {"dynamic_anchor_attention"}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in routing config section 'memory_fusion': {unknown}")
    return MemoryFusionConfig(
        **data,
        dynamic_anchor_attention=DynamicAnchorAttentionConfig(**dict(daa_value)),
    )


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Routing config not found: {path}")
    if path.suffix.lower() == ".json":
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read routing YAML config.") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Routing config must contain a mapping: {path}")
    return dict(data)
