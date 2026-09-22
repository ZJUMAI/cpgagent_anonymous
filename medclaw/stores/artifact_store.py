"""Content registration for files emitted by skills."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from medclaw.utils import safe_component


class ArtifactStore:
    """Copy skill output files into a stable artifact namespace."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def register_file(
        self,
        local_path: Path,
        case_id: str,
        skill_name: str,
        call_id: str,
        artifact_type: str,
    ) -> str:
        source = Path(local_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Artifact file does not exist: {source}")

        case_id = safe_component(case_id, "case_id")
        skill_name = safe_component(skill_name, "skill_name")
        call_id = safe_component(call_id, "call_id")
        if not isinstance(artifact_type, str) or not artifact_type:
            raise ValueError("artifact_type must be a non-empty string")

        destination_dir = self.root / case_id / skill_name / call_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        if source != destination.resolve():
            shutil.copy2(source, destination)

        return (
            f"artifact://{quote(case_id, safe='')}/"
            f"{quote(skill_name, safe='')}/{quote(call_id, safe='')}/"
            f"{quote(source.name, safe='')}"
        )

    def resolve_uri(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "artifact":
            raise ValueError(f"Unsupported artifact URI scheme: {uri}")

        case_id = unquote(parsed.netloc)
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) != 3:
            raise ValueError(f"Malformed artifact URI: {uri}")

        skill_name, call_id, filename = parts
        safe_component(case_id, "case_id")
        safe_component(skill_name, "skill_name")
        safe_component(call_id, "call_id")
        if Path(filename).name != filename or filename in {"", ".", ".."}:
            raise ValueError(f"Malformed artifact filename in URI: {uri}")

        path = (self.root / case_id / skill_name / call_id / filename).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Artifact URI escapes the store root: {uri}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Artifact URI does not resolve to a file: {uri}")
        return path
