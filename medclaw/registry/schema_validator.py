"""JSON Schema validation for skill inputs and outputs."""

from __future__ import annotations

from typing import Any, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


class SchemaValidationError(ValueError):
    """Raised when a value does not conform to a skill schema."""


class SchemaValidator:
    """Validate JSON-compatible values against JSON Schema."""

    @staticmethod
    def validate(instance: Any, schema: Mapping[str, Any], *, context: str = "value") -> None:
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise SchemaValidationError(f"Invalid JSON Schema for {context}: {exc.message}") from exc

        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
        if not errors:
            return

        details = []
        for error in errors[:5]:
            location = "$"
            for part in error.absolute_path:
                location += f"[{part}]" if isinstance(part, int) else f".{part}"
            details.append(f"{location}: {error.message}")

        suffix = "" if len(errors) <= 5 else f" ({len(errors) - 5} more errors)"
        raise SchemaValidationError(f"{context} validation failed: {'; '.join(details)}{suffix}")
