"""LLM client wrapper.

Uses LiteLLM for provider abstraction (OpenAI primary, Anthropic/Azure as
failover). In tests, `FakeLLM` returns canned responses so agent flow can be
exercised without API calls.

The interface is intentionally narrow: `complete(messages, ...) → str`. If
you need tool-calling or JSON mode, use `complete_json(...)`.
"""
from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Callable

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseLLM(ABC):
    @abstractmethod
    async def complete(self, messages: list[dict], *, model: str | None = None,
                       temperature: float = 0.2, max_tokens: int = 800) -> str: ...

    async def complete_json(self, messages: list[dict], schema_hint: str = "",
                            model: str | None = None) -> dict:
        """Ask for JSON and parse; retry once on parse failure."""
        if schema_hint and messages and messages[-1]["role"] == "user":
            messages = list(messages)
            messages[-1] = {**messages[-1],
                            "content": messages[-1]["content"] + f"\n\nRespond in JSON matching: {schema_hint}"}
        for attempt in range(2):
            raw = await self.complete(messages, model=model, temperature=0.0)
            try:
                # Strip ```json fences if present.
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("`")
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:].strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                if attempt == 1:
                    raise
        return {}


class FakeLLM(BaseLLM):
    """Deterministic scripted LLM for tests.

    You give it a callable `responder(messages, **kw) -> str` at construction
    time and it returns whatever the callable says.
    """

    def __init__(self, responder: Callable[[list[dict]], str] | None = None):
        self.responder = responder or (lambda msgs, **kw: "OK")
        self.calls: list[dict] = []

    async def complete(self, messages: list[dict], *, model: str | None = None,
                       temperature: float = 0.2, max_tokens: int = 800) -> str:
        self.calls.append({"messages": messages, "model": model, "temperature": temperature})
        return self.responder(messages, model=model, temperature=temperature)


class LiteLLMClient(BaseLLM):
    """Production LLM client. LiteLLM handles the provider routing."""

    async def complete(self, messages: list[dict], *, model: str | None = None,
                       temperature: float = 0.2, max_tokens: int = 800) -> str:
        import litellm
        model = model or settings.primary_llm_model
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: litellm.completion(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
                api_key=settings.openai_api_key,
            ),
        )
        return resp.choices[0].message.content


def get_llm() -> BaseLLM:
    if settings.is_test or not settings.openai_api_key:
        # In test env or without a key, return a stub that returns a
        # keyword-based canned response. Real tests inject their own FakeLLM.
        return FakeLLM()
    return LiteLLMClient()
