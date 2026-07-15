"""Initial schema (created from SQLAlchemy models).

Revision ID: 0001
Revises:
Create Date: 2025-01-01

For portability with unit tests, the actual schema is created by
`Base.metadata.create_all` at test time. In production, this migration builds
the schema and the follow-up migration (0002) adds the pgvector column and
BM25 tsvector column.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.db.base import Base
from app.db import models  # noqa: F401  register tables

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create all tables from metadata. This works because the models use
    # portable column types (GUID, JSONType) that render correctly on
    # Postgres.
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
