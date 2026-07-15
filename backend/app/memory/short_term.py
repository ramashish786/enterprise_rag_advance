"""Short-term memory backed by Redis in production; in-memory in tests.

The interface mirrors what LangGraph checkpointers expose (get/put by
thread_id). We deliberately keep it small: you don't need the full
LangGraph checkpointer API to pass state between turns; you need
serialize/deserialize on a per-session key.

State layout:
    {
      "messages": [ {"role": "...", "content": "...", "created_at": "..."}, ... ],
      "summary": "<rolling summary of turns before the rolling window>",
      "meta":    { "turn_count": int, "last_updated": ISO }
    }

Rolling summarization: once `len(messages) > max_messages_before_summary`,
we summarize the oldest half into `summary` and drop those messages. In
tests, summarization is a deterministic stub; in prod it uses the router LLM.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseSTMStore(ABC):
    @abstractmethod
    async def get(self, thread_id: str) -> Optional[dict]: ...

    @abstractmethod
    async def put(self, thread_id: str, state: dict, ttl_seconds: int = 3600) -> None: ...

    @abstractmethod
    async def delete(self, thread_id: str) -> None: ...


class InMemorySTMStore(BaseSTMStore):
    def __init__(self):
        self._store: dict[str, dict] = {}

    async def get(self, thread_id: str) -> Optional[dict]:
        return self._store.get(thread_id)

    async def put(self, thread_id: str, state: dict, ttl_seconds: int = 3600) -> None:
        self._store[thread_id] = state

    async def delete(self, thread_id: str) -> None:
        self._store.pop(thread_id, None)


class RedisSTMStore(BaseSTMStore):
    def __init__(self, url: str | None = None):
        self.url = url or settings.redis_url
        self._client = None

    async def _get_client(self):
        if self._client is None:
            import redis.asyncio as redis
            self._client = redis.from_url(self.url, decode_responses=True)
        return self._client

    @staticmethod
    def _key(thread_id: str) -> str:
        return f"stm:{thread_id}"

    async def get(self, thread_id: str) -> Optional[dict]:
        client = await self._get_client()
        raw = await client.get(self._key(thread_id))
        return json.loads(raw) if raw else None

    async def put(self, thread_id: str, state: dict, ttl_seconds: int = 3600) -> None:
        client = await self._get_client()
        await client.set(self._key(thread_id), json.dumps(state, default=str), ex=ttl_seconds)

    async def delete(self, thread_id: str) -> None:
        client = await self._get_client()
        await client.delete(self._key(thread_id))


def get_stm_store() -> BaseSTMStore:
    if settings.is_test:
        return InMemorySTMStore()
    return RedisSTMStore()


# --- Session state helpers ---

def new_state() -> dict:
    return {
        "messages": [],
        "summary": "",
        "meta": {"turn_count": 0, "last_updated": datetime.now(timezone.utc).isoformat()},
    }


def append_message(state: dict, role: str, content: str, metadata: Optional[dict] = None) -> dict:
    state["messages"].append({
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    state["meta"]["turn_count"] += 1
    state["meta"]["last_updated"] = datetime.now(timezone.utc).isoformat()
    return state


def summarize_if_needed(
    state: dict,
    *,
    max_messages: int = 20,
    keep_recent: int = 8,
    summarizer=None,
) -> dict:
    """Rolling summarization: once message count exceeds max_messages, roll
    the oldest into the summary and keep only the most recent `keep_recent`.
    """
    messages = state.get("messages", [])
    if len(messages) <= max_messages:
        return state

    to_summarize = messages[:-keep_recent]
    to_keep = messages[-keep_recent:]

    if summarizer:
        summary_snippet = summarizer(to_summarize, existing_summary=state.get("summary", ""))
    else:
        # Deterministic stub: concatenate turn count into the running summary.
        prior = state.get("summary", "")
        summary_snippet = (
            f"{prior}\n[Prior turns condensed: {len(to_summarize)} messages.]"
            if prior
            else f"[Prior turns condensed: {len(to_summarize)} messages.]"
        )

    state["summary"] = summary_snippet.strip()
    state["messages"] = to_keep
    return state
