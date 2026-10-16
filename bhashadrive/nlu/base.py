"""Replaceable interfaces for the language-understanding component."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMError(RuntimeError):
    """Base class for local language-model failures."""


class LLMUnavailableError(LLMError):
    """Model runtime not reachable or model not installed. No cloud fallback exists."""


class LLMTimeoutError(LLMError):
    pass


class LLMOutputError(LLMError):
    """Model returned output that is not valid JSON / does not match the schema."""


class LanguageModel(ABC):
    """Minimal contract: return a JSON object for a system+user prompt."""

    name: str = "abstract"

    @abstractmethod
    def generate_json(self, system: str, user: str, json_schema: dict[str, Any] | None = None) -> dict[str, Any]: ...

    def health_check(self) -> None:
        """Raise LLMUnavailableError if the model cannot be used."""
