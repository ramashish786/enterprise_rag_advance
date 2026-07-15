"""Add pgvector embedding column and BM25 tsvector on document_chunks.

Also adds an embedding column to long_term_memories for semantic recall.
Postgres-only; on other dialects this is a no-op.

Revision ID: 0002
Revises: 0001
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.core.config import settings

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if not bind.dialect.name.startswith("postgres"):
        return

    dims = settings.embedding_dimensions

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        f"ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding vector({dims})"
    )
    op.execute(
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector"
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION document_chunks_tsv_trigger() RETURNS trigger AS $$
        BEGIN
            NEW.content_tsv := to_tsvector('english', coalesce(NEW.content, ''));
            RETURN NEW;
        END; $$ LANGUAGE plpgsql;
        """
    )
    op.execute("DROP TRIGGER IF EXISTS document_chunks_tsv_update ON document_chunks")
    op.execute(
        """
        CREATE TRIGGER document_chunks_tsv_update
        BEFORE INSERT OR UPDATE ON document_chunks
        FOR EACH ROW EXECUTE FUNCTION document_chunks_tsv_trigger();
        """
    )

    # HNSW for dense, GIN for BM25.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_content_tsv "
        "ON document_chunks USING gin (content_tsv)"
    )

    # long_term_memories embedding
    op.execute(
        f"ALTER TABLE long_term_memories ADD COLUMN IF NOT EXISTS embedding vector({dims})"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_long_term_memories_embedding_hnsw "
        "ON long_term_memories USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if not bind.dialect.name.startswith("postgres"):
        return
    op.execute("DROP INDEX IF EXISTS ix_long_term_memories_embedding_hnsw")
    op.execute("ALTER TABLE long_term_memories DROP COLUMN IF EXISTS embedding")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_content_tsv")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
    op.execute("DROP TRIGGER IF EXISTS document_chunks_tsv_update ON document_chunks")
    op.execute("DROP FUNCTION IF EXISTS document_chunks_tsv_trigger()")
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS content_tsv")
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS embedding")
