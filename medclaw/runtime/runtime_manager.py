"""End-to-end orchestration for one skill invocation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from medclaw.registry.schema_validator import SchemaValidationError, SchemaValidator
from medclaw.registry.skill_registry import SkillRegistry, SkillSpec
from medclaw.runtime.skill_runner import RunnerResult, SkillRunner
from medclaw.stores.artifact_store import ArtifactStore
from medclaw.stores.audit_log import AuditLog
from medclaw.utils import DataIOError, hash_json, read_json, safe_component, utc_now, write_json


class RuntimeManager:
    """Validate, execute, register, audit, and return a skill result."""

    def __init__(
        self,
        registry: SkillRegistry,
        runner: SkillRunner,
        artifact_store: ArtifactStore,
        audit_log: AuditLog,
        runs_root: Path,
        *,
        schema_validator: SchemaValidator | None = None,
    ) -> None:
        self.registry = registry
        self.runner = runner
        self.artifact_store = artifact_store
        self.audit_log = audit_log
        self.runs_root = Path(runs_root).resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.schema_validator = schema_validator or SchemaValidator()

    def invoke(
        self,
        skill_name: str,
        arguments: Mapping[str, Any],
        *,
        call_id: str | None = None,
    ) -> dict[str, Any]:
        """Invoke one skill and always return a structured result."""

        started_at = utc_now()
        call_id = call_id or f"call_{uuid4().hex}"
        try:
            safe_component(call_id, "call_id")
        except ValueError:
            call_id = f"call_invalid_{uuid4().hex}"

        if isinstance(arguments, Mapping):
            audit_arguments: dict[str, Any] = dict(arguments)
        else:
            audit_arguments = {"invalid_arguments": repr(arguments)}

        case_id = self._audit_case_id(audit_arguments)
        skill: SkillSpec | None = None
        runner_result: RunnerResult | None = None
        output_path: Path | None = None
        input_hash = self._best_effort_hash(audit_arguments)

        try:
            skill = self.registry.get_skill(skill_name)
            self.schema_validator.validate(
                audit_arguments,
                skill.input_schema,
                context=f"{skill.name} input",
            )

            raw_case_id = audit_arguments.get("case_id")
            if not isinstance(raw_case_id, str):
                raise SchemaValidationError(
                    f"{skill.name} input validation failed: $.case_id must be a string"
                )
            case_id = safe_component(raw_case_id, "case_id")

            work_dir = self.runs_root / case_id / call_id
            work_dir.mkdir(parents=True, exist_ok=False)
            input_path = work_dir / "input.json"
            output_path = work_dir / "output.json"

            payload = {
                "call_id": call_id,
                "case_id": case_id,
                "arguments": audit_arguments,
            }
            input_hash = hash_json(payload)
            write_json(input_path, payload)

            runner_result = self.runner.run(
                skill,
                work_dir=work_dir,
                input_path=input_path,
                output_path=output_path,
            )
            if not runner_result.succeeded:
                message = runner_result.stderr or "Skill process returned a non-zero exit code."
                result = self._failure_result(
                    skill_name=skill.name,
                    skill_version=skill.version,
                    call_id=call_id,
                    input_hash=input_hash,
                    started_at=started_at,
                    message=message,
                )
                return self._finalize(
                    case_id=case_id,
                    skill_name=skill.name,
                    arguments=audit_arguments,
                    result=result,
                    runner_result=runner_result,
                    output_path=output_path,
                )

            raw_output = read_json(output_path)
            if not isinstance(raw_output, dict):
                raise DataIOError(f"Skill output must contain a JSON object: {output_path}")

            result = self._normalize_output(
                raw_output,
                skill=skill,
                case_id=case_id,
                call_id=call_id,
                input_hash=input_hash,
                started_at=started_at,
                work_dir=work_dir,
            )
            self.schema_validator.validate(
                result,
                skill.output_schema,
                context=f"{skill.name} output",
            )
            return self._finalize(
                case_id=case_id,
                skill_name=skill.name,
                arguments=audit_arguments,
                result=result,
                runner_result=runner_result,
                output_path=output_path,
            )
        except Exception as exc:
            result = self._failure_result(
                skill_name=skill.name if skill else skill_name,
                skill_version=skill.version if skill else "unknown",
                call_id=call_id,
                input_hash=input_hash,
                started_at=started_at,
                message=str(exc),
            )
            return self._finalize(
                case_id=case_id,
                skill_name=skill.name if skill else skill_name,
                arguments=audit_arguments,
                result=result,
                runner_result=runner_result,
                output_path=output_path,
            )

    def call_skill(
        self,
        skill_name: str,
        arguments: Mapping[str, Any],
        *,
        call_id: str | None = None,
    ) -> dict[str, Any]:
        """Alias for callers that prefer an explicit skill-oriented method name."""

        return self.invoke(skill_name, arguments, call_id=call_id)

    def _normalize_output(
        self,
        raw_output: dict[str, Any],
        *,
        skill: SkillSpec,
        case_id: str,
        call_id: str,
        input_hash: str,
        started_at: str,
        work_dir: Path,
    ) -> dict[str, Any]:
        result = dict(raw_output)
        status = result.get("status")
        if status == "success":
            result["artifacts"] = self._register_artifacts(
                result.get("artifacts", []),
                work_dir=work_dir,
                case_id=case_id,
                skill_name=skill.name,
                call_id=call_id,
            )

        result["call_id"] = call_id
        result["provenance"] = {
            "skill_name": skill.name,
            "skill_version": skill.version,
            "input_hash": input_hash,
            "started_at": started_at,
            "finished_at": utc_now(),
        }
        return result

    def _register_artifacts(
        self,
        artifacts: Any,
        *,
        work_dir: Path,
        case_id: str,
        skill_name: str,
        call_id: str,
    ) -> list[dict[str, Any]]:
        if not isinstance(artifacts, list):
            raise ValueError("Skill output field 'artifacts' must be a list")

        registered = []
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, Mapping):
                raise ValueError(f"Artifact at index {index} must be an object")

            artifact_type = artifact.get("type")
            if not isinstance(artifact_type, str) or not artifact_type:
                raise ValueError(f"Artifact at index {index} is missing a non-empty 'type'")

            existing_uri = artifact.get("uri")
            if isinstance(existing_uri, str) and existing_uri.startswith("artifact://"):
                registered_path = self.artifact_store.resolve_uri(existing_uri)
                normalized = dict(artifact)
                self._add_artifact_content_metadata(normalized, registered_path)
                registered.append(normalized)
                continue

            path_value = artifact.get("path")
            if not isinstance(path_value, str) or not path_value:
                raise ValueError(
                    f"Artifact at index {index} must provide a local 'path' or artifact:// 'uri'"
                )
            local_path = Path(path_value)
            if not local_path.is_absolute():
                local_path = work_dir / local_path

            uri = self.artifact_store.register_file(
                local_path,
                case_id,
                skill_name,
                call_id,
                artifact_type,
            )
            normalized = {key: value for key, value in artifact.items() if key != "path"}
            normalized["uri"] = uri
            self._add_artifact_content_metadata(
                normalized,
                self.artifact_store.resolve_uri(uri),
            )
            registered.append(normalized)
        return registered

    @staticmethod
    def _add_artifact_content_metadata(artifact: dict[str, Any], path: Path) -> None:
        artifact["size_bytes"] = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        artifact["sha256"] = digest.hexdigest()

    def _finalize(
        self,
        *,
        case_id: str,
        skill_name: str,
        arguments: Mapping[str, Any],
        result: dict[str, Any],
        runner_result: RunnerResult | None,
        output_path: Path | None,
    ) -> dict[str, Any]:
        if output_path is not None:
            write_json(output_path, result)

        provenance = result["provenance"]
        record = {
            "call_id": result["call_id"],
            "case_id": case_id,
            "skill_name": skill_name,
            "arguments": dict(arguments),
            "status": result["status"],
            "started_at": provenance["started_at"],
            "finished_at": provenance["finished_at"],
            "stdout": runner_result.stdout if runner_result else "",
            "stderr": runner_result.stderr if runner_result else "",
            "returncode": runner_result.returncode if runner_result else None,
            "input_hash": provenance["input_hash"],
            "output_hash": hash_json(result),
            "command": runner_result.command if runner_result else [],
        }
        self.audit_log.append(case_id, record)
        return result

    @staticmethod
    def _failure_result(
        *,
        skill_name: str,
        skill_version: str,
        call_id: str,
        input_hash: str,
        started_at: str,
        message: str,
    ) -> dict[str, Any]:
        finished_at = utc_now()
        return {
            "status": "failed",
            "call_id": call_id,
            "findings": {"error": message},
            "artifacts": [],
            "provenance": {
                "skill_name": skill_name,
                "skill_version": skill_version,
                "input_hash": input_hash,
                "started_at": started_at,
                "finished_at": finished_at,
            },
            "warnings": [message],
        }

    @staticmethod
    def _audit_case_id(arguments: Mapping[str, Any]) -> str:
        value = arguments.get("case_id")
        if isinstance(value, str):
            try:
                return safe_component(value, "case_id")
            except ValueError:
                return "invalid-case"
        return "unknown-case"

    @staticmethod
    def _best_effort_hash(value: Any) -> str:
        try:
            return hash_json(value)
        except DataIOError:
            return hash_json({"unhashable_value": repr(value)})
