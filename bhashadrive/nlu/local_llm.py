"""Local LLM backend served by Ollama on the loopback interface."""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import httpx

from ..config import LLMConfig
from .base import LanguageModel, LLMOutputError, LLMTimeoutError, LLMUnavailableError

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class OllamaLLM(LanguageModel):
    def __init__(self, config: LLMConfig | None = None):
        self.config = c = config or LLMConfig()
        host = urlparse(c.endpoint).hostname or ""
        if host not in _LOOPBACK and not c.allow_remote_endpoint:
            raise LLMUnavailableError(
                f"Refusing non-local LLM endpoint {c.endpoint}. BhashaDrive runs inference locally; "
                "set BD_LLM_ALLOW_REMOTE=true only for a trusted on-premise server."
            )
        self.name = f"ollama:{c.model}"
        self._client = httpx.Client(base_url=c.endpoint, timeout=c.timeout_s, trust_env=False)

    def health_check(self) -> None:
        try:
            r = self._client.get("/api/tags", timeout=5)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"Ollama is not reachable at {self.config.endpoint} ({exc}). Start it with `ollama serve`."
            ) from exc
        names = {m.get("name") for m in r.json().get("models", [])}
        if self.config.model not in names and f"{self.config.model}:latest" not in names:
            raise LLMUnavailableError(
                f"Model '{self.config.model}' is not installed in Ollama (found: {sorted(names)}). "
                f"Run `ollama pull {self.config.model}` once with internet access."
            )

    def warmup(self, timeout_s: float = 180.0) -> float:
        """Load the model into memory (first load can take far longer than a normal request)."""
        import time

        t0 = time.perf_counter()
        try:
            r = self._client.post("/api/generate", json={"model": self.config.model, "prompt": "",
                                                          "keep_alive": self.config.keep_alive}, timeout=timeout_s)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Could not load '{self.config.model}' in Ollama: {exc}") from exc
        return time.perf_counter() - t0

    def model_info(self) -> dict[str, Any]:
        try:
            r = self._client.post("/api/show", json={"model": self.config.model}, timeout=10)
            r.raise_for_status()
            d = r.json().get("details", {})
            return {"model": self.config.model, **{k: d.get(k) for k in ("family", "parameter_size", "quantization_level")}}
        except httpx.HTTPError:
            return {"model": self.config.model}

    def generate_json(self, system: str, user: str, json_schema: dict[str, Any] | None = None) -> dict[str, Any]:
        c = self.config
        payload = {
            "model": c.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": json_schema or "json",  # constrained decoding to the schema
            "keep_alive": c.keep_alive,
            "options": {"temperature": c.temperature, "num_ctx": c.num_ctx, "seed": 7},
        }
        try:
            r = self._client.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Local LLM timed out after {c.timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Ollama request failed at {c.endpoint}: {exc}. Is `ollama serve` running?") from exc
        if r.status_code == 404:
            raise LLMUnavailableError(f"Model '{c.model}' not found. Run `ollama pull {c.model}`.")
        if r.status_code >= 400:
            raise LLMUnavailableError(f"Ollama error {r.status_code}: {r.text[:300]}")
        content = r.json().get("message", {}).get("content", "")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"Model output is not valid JSON: {content[:200]!r}") from exc
        if not isinstance(data, dict):
            raise LLMOutputError("Model output JSON is not an object")
        return data


class MockLLM(LanguageModel):
    """Returns canned JSON (or raises) for unit tests."""

    name = "mock-llm"

    def __init__(self, responses: list[dict | Exception] | dict | Exception):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.calls: list[tuple[str, str]] = []

    def generate_json(self, system: str, user: str, json_schema: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((system, user))
        item = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item
