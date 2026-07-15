"""Embedding client.

Production goes through LiteLLM to OpenAI's `text-embedding-3-large`. Tests
use a deterministic FakeEmbedder that hashes tokens into a fixed vector
space, so retrieval quality tests are reproducible without API calls.
"""
from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

from app.core.config import settings


class BaseEmbedder(ABC):
    """Async-only interface, matching how the ingestion worker calls it."""

    dim: int

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


class FakeEmbedder(BaseEmbedder):
    """Deterministic, dependency-free embedder for tests.

    Encodes texts into a fixed-dim vector via token hashing + L2-normalization,
    which produces reasonable cosine similarity behavior for exact-match and
    near-match retrieval within tests.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [t.lower() for t in text.split() if t.strip()]
        for tok in tokens:
            h = hashlib.md5(tok.encode()).digest()
            for i in range(0, min(len(h), 16)):
                idx = (h[i] * 31 + i) % self.dim
                sign = 1.0 if (h[i] & 1) else -1.0
                vec[idx] += sign
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OpenAIEmbedder(BaseEmbedder):
    """Production OpenAI embedder via LiteLLM.

    Batches up to 100 inputs per call (LiteLLM handles retries/backoff).
    """

    def __init__(self, model: str | None = None, dim: int | None = None):
        self.model = model or settings.embedding_model
        self.dim = dim or settings.embedding_dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import litellm
        # LiteLLM's embedding endpoint is sync; running in a thread keeps the
        # event loop free.
        import asyncio
        loop = asyncio.get_event_loop()
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            resp = await loop.run_in_executor(
                None,
                lambda: litellm.embedding(
                    model=self.model,
                    input=batch,
                    dimensions=self.dim,
                    api_key=settings.openai_api_key,
                ),
            )
            for row in resp.data:
                out.append(row["embedding"])
        return out


def get_embedder() -> BaseEmbedder:
    """Return the embedder configured for the current environment."""
    if settings.is_test or not settings.openai_api_key:
        return FakeEmbedder()
    return OpenAIEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
