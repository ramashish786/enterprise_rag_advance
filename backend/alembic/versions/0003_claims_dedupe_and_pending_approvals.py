"""Add Claim.dedupe_key and the pending_approvals table.

Revision ID: 0003
Revises: 0002
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.db.types import GUID, JSONType

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("claims", sa.Column("dedupe_key", sa.String(64), nullable=True))
    op.create_unique_constraint("uq_claims_dedupe_key", "claims", ["dedupe_key"])

    op.create_table(
        "pending_approvals",
        sa.Column("id", GUID, primary_key=True),
        sa.Column("thread_id", GUID, nullable=True),
        sa.Column("action_type", sa.String(64), nullable=False),
        sa.Column("args", JSONType, nullable=False),
        sa.Column("idempotency_key", sa.String(64), nullable=False),
        sa.Column("requested_by", GUID, sa.ForeignKey("users.id"), nullable=False),
        sa.Column(
            # SQLAlchemy's Enum binds by member *name* by default (matches
            # how user_type/policy_status/etc. were auto-created elsewhere
            # in this schema), so the native labels are uppercase even
            # though ApprovalStatus's values are lowercase.
            "status",
            sa.Enum("PENDING", "APPROVED", "REJECTED", name="approval_status"),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column("approver_id", GUID, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution_note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("idempotency_key", name="uq_pending_approvals_idempotency_key"),
    )


def downgrade() -> None:
    op.drop_table("pending_approvals")
    bind = op.get_bind()
    if bind.dialect.name.startswith("postgres"):
        op.execute("DROP TYPE IF EXISTS approval_status")
    op.drop_constraint("uq_claims_dedupe_key", "claims", type_="unique")
    op.drop_column("claims", "dedupe_key")
