"""Resolve benchmark modality capabilities to existing MedClaw skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from medclaw.registry.skill_registry import SkillRegistry, SkillSpec


DEFAULT_CAPABILITY_NAMES = {
    "guideline.retrieve": "guideline.retrieve",
    "guideline.csco_retrieve": "guideline.retrieve",
    "radiology.ct_roi": "radiology.lung_tumor_roi",
    "radiology.lung_tumor_roi": "radiology.lung_tumor_roi",
    "pathology.wsi_roi": "pathology.conch_patch_roi",
    "pathology.conch_v1_5_roi": "pathology.conch_patch_roi",
    "pathology.conch_patch_roi": "pathology.conch_patch_roi",
    "pathology.ucec_conch_patch_roi": "pathology.ucec_conch_patch_roi",
    "pathology.npc_conch_patch_roi": "pathology.npc_conch_patch_roi",
    "radiology.ucec_mri_roi": "radiology.ucec_mri_roi",
    "radiology.npc_mri_roi": "radiology.npc_mri_roi",
}


@dataclass
class SkillResolver:
    """Resolve benchmark capabilities without duplicating existing skills."""

    registry: SkillRegistry | Any
    overrides: Mapping[str, str] | None = None
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if hasattr(self.registry, "registry"):
            self.registry = self.registry.registry
        self.overrides = dict(self.overrides or {})

    def resolve(self, capability: str) -> str | None:
        if capability in self.overrides:
            return self._existing_or_warn(self.overrides[capability], capability)

        preferred = DEFAULT_CAPABILITY_NAMES.get(capability)
        if preferred:
            found = self._existing_or_warn(preferred, capability, warn=False)
            if found:
                return found

        for skill in self._skills():
            tags = _capability_tags(skill)
            if capability in tags:
                return skill.name

        keywords = _keywords_for(capability)
        for skill in self._skills():
            haystack = " ".join(
                [
                    skill.name.lower(),
                    skill.description.lower(),
                    str(skill.modalities).lower(),
                ]
            )
            if all(keyword in haystack for keyword in keywords):
                return skill.name

        self.warnings.append(f"No skill found for capability {capability!r}.")
        return None

    def require(self, capability: str) -> str:
        skill = self.resolve(capability)
        if not skill:
            raise LookupError(f"No skill found for capability {capability!r}.")
        return skill

    def _existing_or_warn(
        self,
        skill_name: str,
        capability: str,
        *,
        warn: bool = True,
    ) -> str | None:
        try:
            self.registry.get_skill(skill_name)
            return skill_name
        except KeyError:
            if warn:
                self.warnings.append(
                    f"Configured skill {skill_name!r} for {capability!r} is not registered."
                )
            return None

    def _skills(self) -> list[SkillSpec]:
        names = [card["name"] for card in self.registry.list_tools_for_agent()]
        return [self.registry.get_skill(str(name)) for name in names]


def _capability_tags(skill: SkillSpec) -> set[str]:
    tags: set[str] = set()
    for container in (skill.modalities, skill.visibility):
        value = container.get("capabilities") if isinstance(container, Mapping) else None
        if isinstance(value, list):
            tags.update(str(item) for item in value)
    return tags


def _keywords_for(capability: str) -> tuple[str, ...]:
    if capability.startswith("radiology"):
        return ("lung", "roi")
    if capability.startswith("pathology"):
        return ("conch", "roi")
    if capability.startswith("guideline"):
        return ("guideline",)
    return tuple(part for part in capability.lower().split(".") if part)
