"""Subprocess runner for MedClaw skill entrypoints."""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from medclaw.registry.skill_registry import SkillSpec


class SkillRunnerConfigError(ValueError):
    """Raised when a skill runner configuration is invalid."""


@dataclass(frozen=True)
class RunnerResult:
    """Captured subprocess outcome."""

    status: str
    command: list[str]
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return self.status == "success" and self.returncode == 0


class SkillRunner:
    """Execute a skill entrypoint with the active Python interpreter."""

    def run(
        self,
        skill: SkillSpec,
        *,
        work_dir: Path,
        input_path: Path,
        output_path: Path,
        timeout_sec: float | None = None,
    ) -> RunnerResult:
        work_dir = Path(work_dir).resolve()
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        try:
            command = self._build_command(
                skill,
                input_path=input_path,
                output_path=output_path,
            )
        except (SkillRunnerConfigError, ValueError) as exc:
            return RunnerResult(
                status="failed",
                command=[],
                stdout="",
                stderr=str(exc),
                returncode=None,
            )

        if timeout_sec is None:
            timeout_sec = skill.resources.get("timeout_sec", 300)

        try:
            completed = subprocess.run(
                command,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=float(timeout_sec),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return RunnerResult(
                status="failed",
                command=command,
                stdout=self._as_text(exc.stdout),
                stderr=self._as_text(exc.stderr)
                or f"Skill execution timed out after {timeout_sec} seconds.",
                returncode=124,
                timed_out=True,
            )
        except FileNotFoundError as exc:
            return RunnerResult(
                status="failed",
                command=command,
                stdout="",
                stderr=f"Could not start skill process: {exc}",
                returncode=None,
            )
        except OSError as exc:
            return RunnerResult(
                status="failed",
                command=command,
                stdout="",
                stderr=f"Skill process failed to start: {exc}",
                returncode=None,
            )
        except ValueError as exc:
            return RunnerResult(
                status="failed",
                command=command,
                stdout="",
                stderr=f"Invalid skill process configuration: {exc}",
                returncode=None,
            )

        return RunnerResult(
            status="success" if completed.returncode == 0 else "failed",
            command=command,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )

    def _build_command(
        self,
        skill: SkillSpec,
        *,
        input_path: Path,
        output_path: Path,
    ) -> list[str]:
        entrypoint = skill.runtime.get("entrypoint")
        if not isinstance(entrypoint, str) or not entrypoint.strip():
            raise SkillRunnerConfigError(f"Skill {skill.name!r} has an invalid entrypoint")

        tokens = shlex.split(entrypoint, posix=True)
        if not tokens:
            raise SkillRunnerConfigError(f"Skill {skill.name!r} has an empty entrypoint")

        executable_name = Path(tokens[0]).name.lower()
        if executable_name not in {"python", "python.exe", "python3", "python3.exe"}:
            raise SkillRunnerConfigError(
                f"Skill {skill.name!r} entrypoint must start with python: {entrypoint!r}"
            )
        tokens[0] = sys.executable

        for index in range(1, len(tokens)):
            candidate = Path(tokens[index])
            if not candidate.is_absolute():
                skill_relative = skill.skill_dir / candidate
                if skill_relative.exists():
                    tokens[index] = str(skill_relative.resolve())

        tokens.extend(["--input", str(input_path), "--output", str(output_path)])
        return tokens

    @staticmethod
    def _as_text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value
