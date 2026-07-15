"""Hybrid retriever.

`HybridRetriever` is the pure-logic version: combines BM25 + dense scores
and an optional rerank over a list of candidate chunks, used in unit tests.
`PgVectorRetriever` is the production version, running SQL against Postgres
(`ts_rank_cd` for BM25, pgvector cosine for dense) with RBAC enforced in the
WHERE clause before top-k truncation, not after (see the ADR in
docs/ARCHITECTURE.docx for why post-filtering doesn't work). Both return
the same `RetrievedChunk` shape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence
from uuid import UUID

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.embedder import BaseEmbedder, cosine, get_embedder

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    id: UUID | str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)

    def to_source_dict(self) -> dict:
        return {"id": str(self.id), "content": self.content, **self.metadata}


@dataclass
class ChunkRecord:
    """In-memory record used by HybridRetriever in tests."""
    id: UUID | str
    content: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)


# --- Scoring helpers ---

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def bm25_scores(query: str, corpus: Sequence[str]) -> list[float]:
    """BM25 with sensible defaults (k1=1.5, b=0.75).

    Uses `rank_bm25` in production; when unavailable, falls back to a manual
    implementation. Kept dep-light so unit tests don't require the package.
    """
    try:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [_tokenize(d) for d in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        return list(bm25.get_scores(_tokenize(query)))
    except ImportError:
        return _bm25_fallback(query, corpus)


def _bm25_fallback(query: str, corpus: Sequence[str]) -> list[float]:
    k1, b = 1.5, 0.75
    tokenized = [_tokenize(d) for d in corpus]
    q_tokens = _tokenize(query)
    N = max(1, len(corpus))
    avg_len = sum(len(t) for t in tokenized) / N if tokenized else 0

    from collections import Counter
    df: Counter[str] = Counter()
    for toks in tokenized:
        for t in set(toks):
            df[t] += 1

    import math
    scores = []
    for toks in tokenized:
        tf = Counter(toks)
        score = 0.0
        dl = len(toks) or 1
        for q in q_tokens:
            if q not in tf:
                continue
            idf = math.log(1 + (N - df[q] + 0.5) / (df[q] + 0.5))
            num = tf[q] * (k1 + 1)
            den = tf[q] + k1 * (1 - b + b * dl / (avg_len or 1))
            score += idf * (num / den)
        scores.append(score)
    return scores


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


# --- Retriever ---

class HybridRetriever:
    """Pure in-memory hybrid retriever used in tests and as the reference impl."""

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        self.embedder = embedder or get_embedder()
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    async def retrieve(
        self,
        query: str,
        corpus: list[ChunkRecord],
        *,
        top_k_dense: int | None = None,
        top_k_bm25: int | None = None,
        top_k_final: int | None = None,
        filter_fn=None,
    ) -> list[RetrievedChunk]:
        if not corpus:
            return []

        top_k_dense = top_k_dense or settings.retrieval_top_k_dense
        top_k_bm25 = top_k_bm25 or settings.retrieval_top_k_bm25
        top_k_final = top_k_final or settings.rerank_top_k

        # RBAC PRE-filter: happens before search, not after top-k.
        if filter_fn:
            candidates = [c for c in corpus if filter_fn(c)]
        else:
            candidates = corpus
        if not candidates:
            return []

        # BM25
        bm_scores = bm25_scores(query, [c.content for c in candidates])
        # Dense
        q_emb = await self.embedder.embed_one(query)
        dense_scores = [cosine(q_emb, c.embedding) for c in candidates]

        norm_bm = _minmax(bm_scores)
        norm_dense = _minmax(dense_scores)

        # Take top-N from each side, union, then reciprocal-rank fuse.
        top_bm = sorted(range(len(candidates)), key=lambda i: norm_bm[i], reverse=True)[:top_k_bm25]
        top_dense = sorted(range(len(candidates)), key=lambda i: norm_dense[i], reverse=True)[:top_k_dense]
        union = set(top_bm) | set(top_dense)

        combined = [
            (i, self.dense_weight * norm_dense[i] + self.bm25_weight * norm_bm[i])
            for i in union
        ]
        combined.sort(key=lambda x: x[1], reverse=True)
        top = combined[:top_k_final]

        return [
            RetrievedChunk(
                id=candidates[i].id,
                content=candidates[i].content,
                score=score,
                metadata=candidates[i].metadata,
            )
            for i, score in top
        ]
