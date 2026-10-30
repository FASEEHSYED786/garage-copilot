"""LLM clients over plain HTTP (no vendor SDKs needed).

All clients implement `complete(system, user) -> str`. `build_llm` returns None in offline
mode; callers then fall back to deterministic, extractive behaviour.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from garage_copilot.config import Settings


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAIClient:
    URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self, api_key: str, model: str = "gpt-4o-mini", client: httpx.Client | None = None
    ):
        self._client = client or httpx.Client(timeout=60.0)
        self._api_key = api_key
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.post(
            self.URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class AnthropicClient:
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(
        self, api_key: str, model: str = "claude-sonnet-4-5", client: httpx.Client | None = None
    ):
        self._client = client or httpx.Client(timeout=60.0)
        self._api_key = api_key
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.post(
            self.URL,
            headers={"x-api-key": self._api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "max_tokens": 800,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        resp.raise_for_status()
        blocks = resp.json()["content"]
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()


def build_llm(settings: Settings) -> LLMClient | None:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("GARAGE_LLM_PROVIDER=openai needs OPENAI_API_KEY")
        return OpenAIClient(settings.openai_api_key, settings.llm_model or "gpt-4o-mini")
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("GARAGE_LLM_PROVIDER=anthropic needs ANTHROPIC_API_KEY")
        return AnthropicClient(
            settings.anthropic_api_key, settings.llm_model or "claude-sonnet-4-5"
        )
    if settings.llm_provider == "offline":
        return None
    raise ValueError(f"Unknown GARAGE_LLM_PROVIDER: {settings.llm_provider}")
