"""Ingestion service.

Coordinates: parse → chunk → embed → persist. Updates document status
transitions (queued → parsing → embedding → indexed) so admins can see
progress. Deduplication is content-hash based.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import Document, DocumentChunk, DocumentStatus, DocumentVisibility
from app.rag.chunker import chunk_document
from app.rag.embedder import BaseEmbedder, get_embedder
from app.rag.parser import parse

logger = get_logger(__name__)


def compute_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


async def register_document(
    session: AsyncSession,
    *,
    filename: str,
    file_path: str | Path,
    mime_type: str,
    uploaded_by: UUID,
    department_id: Optional[UUID] = None,
    visibility: DocumentVisibility = DocumentVisibility.INTERNAL,
) -> Document:
    """Create a Document row in QUEUED state. Returns existing on hash match."""
    content_hash = compute_hash(file_path)
    existing = (
        await session.execute(
            select(Document).where(Document.content_hash == content_hash)
        )
    ).scalar_one_or_none()
    if existing:
        logger.info("document_dedupe_hit", filename=filename, hash=content_hash[:16])
        return existing

    doc = Document(
        filename=filename,
        content_hash=content_hash,
        mime_type=mime_type,
        uploaded_by=uploaded_by,
        department_id=department_id,
        visibility=visibility,
        status=DocumentStatus.QUEUED,
        version=1,
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    return doc


async def process_document(
    session: AsyncSession,
    *,
    document_id: UUID,
    file_path: str | Path,
    embedder: BaseEmbedder | None = None,
) -> Document:
    """Parse → chunk → embed → write chunks.

    Idempotent by content_hash + document_id. On error, marks document FAILED
    with the exception text. On success, marks INDEXED.
    """
    embedder = embedder or get_embedder()

    doc = (
        await session.execute(select(Document).where(Document.id == document_id))
    ).scalar_one()

    # Transition to parsing
    doc.status = DocumentStatus.PARSING
    await session.commit()

    try:
        parsed = parse(file_path)
        _parents, children = chunk_document(parsed)
        if not children:
            raise ValueError("parsed document produced no chunks")

        # Transition to embedding
        doc.status = DocumentStatus.EMBEDDING
        await session.commit()

        vectors = await embedder.embed([c.content for c in children])

        # Persist chunks.
        for chunk, vec in zip(children, vectors):
            row = DocumentChunk(
                id=chunk.id,
                document_id=doc.id,
                parent_chunk_id=None,  # we store children flat; parents held only in-memory
                chunk_index=chunk.index,
                content=chunk.content,
                token_count=chunk.approx_tokens,
                department_id=doc.department_id,
                visibility=doc.visibility,
            )
            session.add(row)
            # Embeddings are written by the Postgres-native code path; for
            # SQLite tests we skip since there's no vector type.
            if session.bind.dialect.name.startswith("postgres"):
                # pgvector expects list[float]
                await session.execute(
                    text := __import__("sqlalchemy").text(
                        "UPDATE document_chunks SET embedding = :emb WHERE id = :id"
                    ),
                    {"emb": vec, "id": row.id},
                )

        doc.status = DocumentStatus.INDEXED
        await session.commit()
        logger.info("document_indexed", filename=doc.filename, chunk_count=len(children))
        return doc

    except Exception as exc:
        doc.status = DocumentStatus.FAILED
        doc.error_message = str(exc)[:2000]
        await session.commit()
        logger.exception("document_ingestion_failed", filename=doc.filename)
        raise
