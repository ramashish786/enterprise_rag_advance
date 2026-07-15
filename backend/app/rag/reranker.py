"""Cross-encoder reranker.

Production uses a small cross-encoder (BGE-reranker or Cohere). For tests
and offline dev, we use a lexical overlap heuristic that produces reasonable
ordering without requiring any model download.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.core.config import settings
from app.rag.retriever import RetrievedChunk


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]: ...


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _overlap(a: str, b: str) -> float:
    ta = {w.lower() for w in _WORD_RE.findall(a)}
    tb = {w.lower() for w in _WORD_RE.findall(b)}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class LexicalReranker(BaseReranker):
    """Deterministic fallback. Combines original score with Jaccard overlap."""

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]:
        scored = []
        for c in chunks:
            overlap = _overlap(query, c.content)
            scored.append(RetrievedChunk(
                id=c.id, content=c.content,
                score=0.5 * c.score + 0.5 * overlap,
                metadata=c.metadata,
            ))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker (loads `sentence-transformers`).

    Model default: `BAAI/bge-reranker-v2-m3`. Takes a few seconds to load
    on first use; subsequent calls are fast.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        import asyncio
        loop = asyncio.get_event_loop()
        pairs = [(query, c.content) for c in chunks]
        scores = await loop.run_in_executor(None, lambda: self._load().predict(pairs))
        scored = [
            RetrievedChunk(id=c.id, content=c.content, score=float(s), metadata=c.metadata)
            for c, s in zip(chunks, scores)
        ]
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]


def get_reranker() -> BaseReranker:
    """Return the reranker configured for the current environment.

    Tests get the lexical reranker; production gets the cross-encoder.
    """
    if settings.is_test or not settings.reranker_enabled:
        return LexicalReranker()
    return CrossEncoderReranker()
