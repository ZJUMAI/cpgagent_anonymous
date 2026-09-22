"""Recursive discovery of executable MedClaw skills."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from medclaw.utils import DataIOError, read_yaml, safe_component


class SkillManifestError(ValueError):
    """Raised when a skill manifest is missing or malformed."""


@dataclass(frozen=True)
class SkillSpec:
    """Normalized metadata for one registered skill."""

    name: str
    version: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    runtime: Mapping[str, Any]
    resources: Mapping[str, Any]
    skill_dir: Path
    manifest_path: Path
    modalities: Mapping[str, Any]
    visibility: Mapping[str, Any]

    @property
    def expose_to_agent(self) -> bool:
        return bool(self.visibility.get("expose_to_agent", True))


class SkillRegistry:
    """Discover and expose skill manifests beneath a skills directory."""

    REQUIRED_FIELDS = {
        "name",
        "version",
        "description",
        "input_schema",
        "output_schema",
        "runtime",
        "resources",
    }
    REQUIRED_RUNTIME_FIELDS = {"entrypoint"}

    def __init__(self, skills_root: Path) -> None:
        self.skills_root = Path(skills_root).resolve()
        self._skills: dict[str, SkillSpec] = {}
        self.refresh()

    def refresh(self) -> None:
        if not self.skills_root.exists():
            raise SkillManifestError(f"Skills directory does not exist: {self.skills_root}")
        if not self.skills_root.is_dir():
            raise SkillManifestError(f"Skills root is not a directory: {self.skills_root}")

        discovered: dict[str, SkillSpec] = {}
        for manifest_path in sorted(self.skills_root.rglob("skill.yaml")):
            spec = self._load_manifest(manifest_path)
            if spec.name in discovered:
                previous = discovered[spec.name].manifest_path
                raise SkillManifestError(
                    f"Duplicate skill name {spec.name!r} in {previous} and {manifest_path}"
                )
            discovered[spec.name] = spec
        self._skills = discovered

    def get_skill(self, name: str) -> SkillSpec:
        try:
            return self._skills[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._skills)) or "<none>"
            raise KeyError(f"Unknown skill {name!r}. Available skills: {available}") from exc

    def list_tools_for_agent(self) -> list[dict[str, Any]]:
        """Return compact tool cards suitable for an agent prompt."""

        return [
            {
                "name": skill.name,
                "version": skill.version,
                "description": skill.description,
                "modalities": dict(skill.modalities),
                "input_schema": dict(skill.input_schema),
            }
            for skill in sorted(self._skills.values(), key=lambda item: item.name)
            if skill.expose_to_agent
        ]

    def __len__(self) -> int:
        return len(self._skills)

    def _load_manifest(self, manifest_path: Path) -> SkillSpec:
        try:
            data = read_yaml(manifest_path)
        except DataIOError as exc:
            raise SkillManifestError(str(exc)) from exc

        if not isinstance(data, dict):
            raise SkillManifestError(f"Skill manifest must be a YAML mapping: {manifest_path}")

        missing = sorted(self.REQUIRED_FIELDS - data.keys())
        if missing:
            raise SkillManifestError(
                f"Skill manifest {manifest_path} is missing required fields: {', '.join(missing)}"
            )

        for field in ("name", "version", "description"):
            if not isinstance(data[field], str) or not data[field].strip():
                raise SkillManifestError(
                    f"Skill manifest {manifest_path} field {field!r} must be a non-empty string"
                )

        for field in ("input_schema", "output_schema", "runtime", "resources"):
            if not isinstance(data[field], dict):
                raise SkillManifestError(
                    f"Skill manifest {manifest_path} field {field!r} must be a mapping"
                )

        runtime = data["runtime"]
        missing_runtime = sorted(self.REQUIRED_RUNTIME_FIELDS - runtime.keys())
        if missing_runtime:
            raise SkillManifestError(
                f"Skill manifest {manifest_path} runtime is missing required fields: "
                f"{', '.join(missing_runtime)}"
            )
        entrypoint = runtime["entrypoint"]
        if not isinstance(entrypoint, str) or not entrypoint.strip():
            raise SkillManifestError(
                f"Skill manifest {manifest_path} runtime field 'entrypoint' "
                "must be a non-empty string"
            )
        name = data["name"].strip()
        try:
            safe_component(name, "skill name")
        except ValueError as exc:
            raise SkillManifestError(f"{manifest_path}: {exc}") from exc

        skill_dir = manifest_path.parent.resolve()
        visibility = data.get("visibility", {})
        if not isinstance(visibility, dict):
            raise SkillManifestError(
                f"Skill manifest {manifest_path} field 'visibility' must be a mapping"
            )
        agent_card = visibility.get("agent_card")
        if agent_card and not (skill_dir / str(agent_card)).is_file():
            raise SkillManifestError(
                f"Skill manifest {manifest_path} references missing agent_card: {agent_card}"
            )

        modalities = data.get("modalities", {})
        if not isinstance(modalities, dict):
            raise SkillManifestError(
                f"Skill manifest {manifest_path} field 'modalities' must be a mapping"
            )

        return SkillSpec(
            name=name,
            version=data["version"].strip(),
            description=data["description"].strip(),
            input_schema=data["input_schema"],
            output_schema=data["output_schema"],
            runtime=runtime,
            resources=data["resources"],
            skill_dir=skill_dir,
            manifest_path=manifest_path.resolve(),
            modalities=modalities,
            visibility=visibility,
        )
