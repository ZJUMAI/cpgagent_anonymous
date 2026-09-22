"""Latent guideline-memory planner package."""

from guideline_planner.chunking import GuidelineChunk, chunk_guidelines
from guideline_planner.dataset import build_training_data
from guideline_planner.routing_config import LatentGuidelineRoutingConfig
from guideline_planner.routing_dataset import build_routing_training_data
from guideline_planner.routing_types import (
    GuidelineMemory,
    MemoryActivation,
    PatientState,
    PlannerStepResult,
)

__all__ = [
    "GuidelineChunk",
    "GuidelineMemory",
    "DynamicAnchorAttentionResampler",
    "LatentGuidelinePlanner",
    "LatentGuidelineRoutingConfig",
    "LatentPlannerDecoder",
    "LatentPlannerError",
    "MemoryActivation",
    "MultiHeadCrossAttentionWithRoutingBias",
    "PatientState",
    "PlannerStepResult",
    "ResolvedPlannerRelease",
    "build_planner_release_manifest",
    "build_routing_training_data",
    "build_training_data",
    "chunk_guidelines",
    "extract_memory_slots",
    "retrieve_latent_guideline_memory",
    "resolve_planner_release",
]


def __getattr__(name: str):
    if name in {
        "DynamicAnchorAttentionResampler",
        "MultiHeadCrossAttentionWithRoutingBias",
    }:
        from guideline_planner.dynamic_anchor_attention import (
            DynamicAnchorAttentionResampler,
            MultiHeadCrossAttentionWithRoutingBias,
        )

        return {
            "DynamicAnchorAttentionResampler": DynamicAnchorAttentionResampler,
            "MultiHeadCrossAttentionWithRoutingBias": (
                MultiHeadCrossAttentionWithRoutingBias
            ),
        }[name]
    if name == "LatentPlannerDecoder":
        from guideline_planner.latent_decoder import LatentPlannerDecoder

        return LatentPlannerDecoder
    if name in {"LatentGuidelinePlanner", "LatentPlannerError"}:
        from guideline_planner.planner import LatentGuidelinePlanner, LatentPlannerError

        return {
            "LatentGuidelinePlanner": LatentGuidelinePlanner,
            "LatentPlannerError": LatentPlannerError,
        }[name]
    if name == "extract_memory_slots":
        from guideline_planner.memory import extract_memory_slots

        return extract_memory_slots
    if name == "retrieve_latent_guideline_memory":
        from guideline_planner.retrieval import retrieve_latent_guideline_memory

        return retrieve_latent_guideline_memory
    if name in {
        "ResolvedPlannerRelease",
        "build_planner_release_manifest",
        "resolve_planner_release",
    }:
        from guideline_planner.release import (
            ResolvedPlannerRelease,
            build_planner_release_manifest,
            resolve_planner_release,
        )

        return {
            "ResolvedPlannerRelease": ResolvedPlannerRelease,
            "build_planner_release_manifest": build_planner_release_manifest,
            "resolve_planner_release": resolve_planner_release,
        }[name]
    raise AttributeError(f"module 'guideline_planner' has no attribute {name!r}")
