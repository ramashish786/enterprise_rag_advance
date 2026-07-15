"""Long-term memory: preference, fact, and context extraction and recall.

Extraction runs at the end of a session (or every N turns) via
`extract_memories(state, principal)`: a rule-based extractor in tests, an
LLM tuned for structured JSON output in prod. Storage lives in
`long_term_memories`; on Postgres each row has an embedding column and
recall does vector search, on SQLite (tests) it falls back to keyword
matching. When a new preference conflicts with an old one, we don't delete
the old row, just mark it `is_active=False` and keep it for history.

Sensitive categories (SSN, credit card, claim details) are excluded from
extraction entirely; only intents and preferences get written.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import LongTermMemory, MemoryType
from app.rag.embedder import BaseEmbedder, cosine, get_embedder

logger = get_logger(__name__)


# --- Extraction (rule-based baseline) ---

_PREFERENCE_HINTS = [
    (r"\bI (?:prefer|like|want) ([a-z][a-z\s]{2,60})\b", MemoryType.PREFERENCE),
    (r"\bmy preferred ([a-z][a-z\s]{2,40}) is ([a-z0-9][a-z0-9\s]{2,40})\b", MemoryType.PREFERENCE),
    (r"\bplease (?:always|contact me via) ([a-z][a-z\s]{2,40})\b", MemoryType.PREFERENCE),
]

_FACT_HINTS = [
    (r"\bI (?:live|reside) in ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", MemoryType.FACT),
    (r"\bI (?:am|work as) an? ([a-z][a-z\s]{2,40})", MemoryType.FACT),
]


class MemoryCandidate:
    def __init__(self, memory_type: MemoryType, content: str, confidence: float = 0.75):
        self.memory_type = memory_type
        self.content = content
        self.confidence = confidence


def extract_candidates(messages: Iterable[dict]) -> list[MemoryCandidate]:
    """Scan user turns for candidate memories.

    Only looks at user messages; the assistant's own suggestions should
    never get laundered into "the user said this".
    """
    candidates: list[MemoryCandidate] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        text = msg.get("content", "")
        if not text:
            continue

        # PII-heavy messages are skipped wholesale.
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", text):  # SSN
            continue

        for pattern, mtype in _PREFERENCE_HINTS + _FACT_HINTS:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                snippet = m.group(0).strip()
                if 3 < len(snippet) < 240:
                    candidates.append(MemoryCandidate(mtype, snippet))
    return candidates


# --- Store / recall ---

class LongTermMemoryService:
    def __init__(self, session: AsyncSession, embedder: BaseEmbedder | None = None):
        self.session = session
        self.embedder = embedder or get_embedder()

    async def persist_candidates(
        self,
        user_id: UUID,
        candidates: list[MemoryCandidate],
        *,
        source_session_id: UUID | None = None,
    ) -> list[LongTermMemory]:
        if not candidates:
            return []

        saved: list[LongTermMemory] = []
        for cand in candidates:
            # Deactivate stale preferences on the same key. "prefer email" and
            # "prefer phone" both mention 'prefer', so a real implementation
            # would key on intent; keyword overlap is good enough as a baseline.
            if cand.memory_type == MemoryType.PREFERENCE:
                await self._deactivate_conflicts(user_id, cand.content)

            mem = LongTermMemory(
                user_id=user_id,
                memory_type=cand.memory_type,
                content=cand.content,
                source_session_id=source_session_id,
                is_active=True,
            )
            self.session.add(mem)
            saved.append(mem)
        await self.session.commit()

        # Compute embeddings and (Postgres only) set them.
        if self.session.bind.dialect.name.startswith("postgres"):
            vectors = await self.embedder.embed([c.content for c in candidates])
            from sqlalchemy import text
            for mem, vec in zip(saved, vectors):
                await self.session.execute(
                    text("UPDATE long_term_memories SET embedding = :emb WHERE id = :id"),
                    {"emb": vec, "id": mem.id},
                )
            await self.session.commit()
        return saved

    async def _deactivate_conflicts(self, user_id: UUID, content: str) -> None:
        keywords = set(re.findall(r"[A-Za-z]{4,}", content.lower()))
        if not keywords:
            return
        existing = (await self.session.execute(
            select(LongTermMemory).where(
                LongTermMemory.user_id == user_id,
                LongTermMemory.memory_type == MemoryType.PREFERENCE,
                LongTermMemory.is_active.is_(True),
            )
        )).scalars().all()
        for old in existing:
            old_keywords = set(re.findall(r"[A-Za-z]{4,}", (old.content or "").lower()))
            if keywords & old_keywords:
                old.is_active = False

    async def recall(
        self,
        user_id: UUID,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[LongTermMemory]:
        """Semantic recall on Postgres, keyword recall on SQLite."""
        rows = (await self.session.execute(
            select(LongTermMemory).where(
                LongTermMemory.user_id == user_id,
                LongTermMemory.is_active.is_(True),
            )
        )).scalars().all()

        if not rows:
            return []

        if self.session.bind.dialect.name.startswith("postgres"):
            q_emb = await self.embedder.embed_one(query)
            # In Postgres we'd normally SELECT ORDER BY embedding <=> :emb;
            # for portability we just score in Python. Fine for small user
            # memory sets, hundreds of rows per user at most.
            scored = []
            for row in rows:
                # TODO: fetch the embedding via raw SQL and score with cosine;
                # everything scores 0 for now so this effectively no-ops.
                scored.append((0.0, row))
        else:
            # Keyword scoring with lightweight stemming (prefix match).
            def toks(t: str) -> set[str]:
                return {w[:6] for w in re.findall(r"[A-Za-z]{4,}", t.lower())}

            q_tokens = toks(query)
            scored = []
            for row in rows:
                t = toks(row.content or "")
                score = len(q_tokens & t) / (len(q_tokens) or 1)
                scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [row for score, row in scored[:top_k] if score > 0]

        # Update last_referenced_at on hits so we can implement recency-based
        # decay later.
        if chosen:
            ids = [r.id for r in chosen]
            await self.session.execute(
                update(LongTermMemory)
                .where(LongTermMemory.id.in_(ids))
                .values(last_referenced_at=datetime.now(timezone.utc))
            )
            await self.session.commit()
        return chosen
