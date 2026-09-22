"""Thin agent-facing routing layer over the runtime manager."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping

from medclaw.registry.skill_registry import SkillRegistry
from medclaw.runtime.runtime_manager import RuntimeManager


class ToolRoutingError(ValueError):
    """Raised when an LLM function name cannot be routed to a skill."""


class ToolRouter:
    """Expose registered skills without leaking runtime implementation details."""

    def __init__(self, registry: SkillRegistry, runtime_manager: RuntimeManager) -> None:
        self.registry = registry
        self.runtime_manager = runtime_manager
        self._function_to_skill = self._build_function_mapping()

    def list_tools(self) -> list[dict[str, Any]]:
        return self.registry.list_tools_for_agent()

    def list_function_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function definitions for the core model."""

        tools = []
        for card in self.registry.list_tools_for_agent():
            skill_name = card["name"]
            function_name = self.function_name_for_skill(skill_name)
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": (
                            f"{card['description']} "
                            f"This function executes the MedClaw skill {skill_name!r}."
                        ),
                        "parameters": deepcopy(card["input_schema"]),
                    },
                }
            )
        return tools

    def call(self, skill_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self.runtime_manager.invoke(skill_name, arguments)

    def call_function(
        self,
        function_name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Route an LLM function call to its registered MedClaw skill."""

        return self.call(self.skill_name_for_function(function_name), arguments)

    def skill_name_for_function(self, function_name: str) -> str:
        try:
            return self._function_to_skill[function_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._function_to_skill)) or "<none>"
            raise ToolRoutingError(
                f"Unknown tool function {function_name!r}. Available functions: {available}"
            ) from exc

    @staticmethod
    def function_name_for_skill(skill_name: str) -> str:
        """Convert dotted skill names into function-call-safe names."""

        normalized = re.sub(r"[^A-Za-z0-9_-]", "__", skill_name)
        return f"medclaw__{normalized}"

    def _build_function_mapping(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for card in self.registry.list_tools_for_agent():
            skill_name = card["name"]
            function_name = self.function_name_for_skill(skill_name)
            if function_name in mapping:
                raise ToolRoutingError(
                    f"Skills {mapping[function_name]!r} and {skill_name!r} map to the "
                    f"same tool function {function_name!r}."
                )
            mapping[function_name] = skill_name
        return mapping
