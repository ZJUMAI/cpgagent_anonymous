"""Content-addressed teacher outputs for auditable Planner V2 data generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from guideline_planner.artifacts import sha256_json
from guideline_planner.schemas_v2 import teacher_cache_key


class TeacherResponseCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(
        self,
        *,
        model: str,
        model_version: str,
        prompt: str,
        rule_compiler_version: str,
    ) -> dict[str, Any] | None:
        key = teacher_cache_key(
            model=f"{model}@{model_version}",
            prompt=prompt,
            rule_compiler_version=rule_compiler_version,
        )
        path = self.root / f"{key}.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("cache_key") != key:
            raise RuntimeError(f"Invalid teacher cache record: {path}")
        return payload

    def put(
        self,
        *,
        model: str,
        model_version: str,
        prompt: str,
        rule_compiler_version: str,
        output: Mapping[str, Any],
    ) -> dict[str, Any]:
        key = teacher_cache_key(
            model=f"{model}@{model_version}",
            prompt=prompt,
            rule_compiler_version=rule_compiler_version,
        )
        record = {
            "schema_version": "teacher_response_cache.v2",
            "cache_key": key,
            "teacher_model": model,
            "teacher_model_version": model_version,
            "rule_compiler_version": rule_compiler_version,
            "prompt_hash": sha256_json(prompt),
            "output_hash": sha256_json(output),
            "output": dict(output),
        }
        path = self.root / f"{key}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != record:
                raise RuntimeError(
                    "Teacher cache key collision or non-deterministic overwrite: "
                    f"{path}"
                )
            return record
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return record
