"""Domain models: identity/access, policies/claims, chat, memory, docs, audit.

All UUID PKs use the GUID type so unit tests can run on SQLite.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum as PyEnum
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, new_uuid
from app.db.types import GUID, JSONType


# --- Enums ---

class UserType(str, PyEnum):
    CUSTOMER = "customer"
    EMPLOYEE = "employee"
    ADMINISTRATOR = "administrator"


class PolicyType(str, PyEnum):
    LIFE = "life"
    HOME = "home"
    VEHICLE = "vehicle"


class PolicyStatus(str, PyEnum):
    ACTIVE = "active"
    LAPSED = "lapsed"
    CANCELLED = "cancelled"
    PENDING = "pending"


class DocumentStatus(str, PyEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    EMBEDDING = "embedding"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentVisibility(str, PyEnum):
    PUBLIC = "public"       # anyone including customers
    INTERNAL = "internal"   # employees + admins
    RESTRICTED = "restricted"  # admins only


class MemoryType(str, PyEnum):
    PREFERENCE = "preference"
    FACT = "fact"
    CONTEXT = "context"


class ApprovalStatus(str, PyEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# --- Identity & access ---

class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(GUID, primary_key=True, default=new_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    user_type: Mapped[UserType] = mapped_column(
        SAEnum(UserType, name="user_type"), nullable=False
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    mfa_secret: Mapped[Optional[str]] = mapped_column(String(64))

    roles: Mapped[list["UserRole"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    employee_profile: Mapped[Optional["EmployeeProfile"]] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    policies: Mapped[list["InsurancePolicy"]] = relationship(back_populates="customer")


class Role(Base, TimestampMixin):
    __tablename__ = "roles"

    id: Mapped[UUID] = mapped_column(GUID, primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String(255), default="")

    permissions: Mapped[list["RolePermission"]] = relationship(
        back_populates="role", cascade="all, delete-orphan"
    )


class Permission(Base, TimestampMixin):
    """(resource, action) pairs, e.g. ("policy", "read_own") or ("document", "upload")."""
    __tablename__ = "permissions"
    __table_args__ = (UniqueConstraint("resource", "action"),)

    id: Mapped[UUID] = mapped_column(GUID, primary_key=True, default=new_uuid)
    resource: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(String(255), default="")


class RolePermission(Base):
    __tablename__ = "role_permissions"

    role_id: Mapped[UUID] = mapped_column(GUID, ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True)
    permission_id: Mapped[UUID] = mapped_column(GUID, ForeignKey("permissions.id", ondelete="CASCADE"), primary_key=True)

    role: Mapped[Role] = relationship(back_populates="permissions")
    permission: Mapped[Permission] = relationship()


class UserRole(Base):
    __tablename__ = "user_roles"

    user_id: Mapped[UUID] = mapped_column(GUID, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role_id: Mapped[UUID] = mapped_column(GUID, ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True)

    user: Mapped[User] = relationship(back_populates="roles")
    role: Mapped[Role] = relationship()


class Department(Base, TimestampMixin):
    __tablename__ = "departments"

    id: Mapped[UUID] = mapped_column(GUID, primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    policy_type: Mapped[PolicyType] = mapped_column(
        SAEnum(PolicyType, name="policy_type"), nullable=False
    )


class EmployeeProfile(Base, TimestampMixin):
    __tablename__ = "employee_profiles"

    id: Mapped[UUID] = mapped_column(GUID, primary_key=True, default=new_uuid)
    user_id: Mapped[UUID] = mapped_column(GUID, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    department_id: Mapped[UUID] = mapped_column(GUID, ForeignKey("departments.id"), nullable=False)
    employee_number: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)

    user: Mapped[User] = relationship(back_populates="employee_profile")
    department: Mapped[Department] = relationship()


# --- Insurance domain ---

class InsurancePolicy(Base, TimestampMixin):
    __tablename__ = "insurance_policies"
    __table_args__ = (
        Index("ix_insurance_policies_customer_type", "customer_id", "policy_type"),
    )

    id: Mapped[UUID] = mapped_column(GUID, primary_key=True, default=new_uuid)
    policy_number: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    customer_id: Mapped[UUID] = mapped_column(GUID, ForeignKey("users.id"), nullable=False)
    policy_type: Mapped[PolicyType] = mapped_column(
        SAEnum(PolicyType, name="policy_type"), nullable=False
    )
    coverage_amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    premium: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[PolicyStatus] = mapped_column(
        SAEnum(PolicyStatus, name="policy_status"), nullable=False, default=PolicyStatus.ACTIVE
    )
    details: Mapped[dict] = mapped_column(JSONType, default=dict)

    customer: Mapped[User] = relationship(back_populates="policies")
    claims: Mapped[list["Claim"]] = relationship(back_populates="policy", cascade="all, delete-orphan")


class Claim(Base, TimestampMixin):
    __tablename__ = "claims"

    id: Mapped[UUID] = mapped_column(GUID, primary_key=True, default=new_uuid)
    claim_number: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    policy_id: Mapped[UUID] = mapped_column(GUID, ForeignKey("insurance_policies.id", ondelete="CASCADE"), nullable=False)
    filed_date: Mapped[date] = mapped_column(Date, nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)  # open / approved / denied / settled
    description: Mapped[str] = mapped_column(Text, default="")
    # Idempotency guard for file_claim: a deterministic hash of (thread_id,
    # action_type, tool_args), so a node re-executed after a crash-and-resume
    # hits this unique constraint instead of creating a second claim.
    dedupe_key: Mapped[Optional[str]] = mapped_column(String(64), unique=True)

    policy: Mapped[InsurancePolicy] = relationship(back_populates="claims")


# --- Conversations ---

class ChatSession(Base, TimestampMixin):
    __tablename__ = "chat_sessions"

    id: Mapped[UUID] = mapped_column(GUID, primary_key=True, default=new_uuid)
    user_id: Mapped[UUID] = mapped_column(GUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), default="New conversation")
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base, TimestampMixin):
    __tablename__ = "chat_messages"

    id: Mapped[UUID] = mapped_column(GUID, primary_key=True, default=new_uuid)
    session_id: Mapped[UUID] = mapped_column(GUID, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user / assistant / system / tool
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Provenance for assistant messages: retrieved chunk ids, tool calls, model used, etc.
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


# --- Human-in-the-loop approvals ---

class PendingApproval(Base, TimestampMixin):
    """Queryable approval inbox for actions gated behind human sign-off.

    For graph-driven actions (file_claim, request_policy_cancellation,
    request_policy_change), `thread_id` is the ChatSession id the LangGraph
    checkpointer paused on; resolving the row resumes that graph via
    Command(resume=...). Document deletion doesn't go through the graph at
    all (see document_routes.py) but reuses this table so there's one
    approval inbox rather than two.
    """
    __tablename__ = "pending_approvals"

    id: Mapped[UUID] = mapped_column(GUID, primary_key=True, default=new_uuid)
    thread_id: Mapped[Optional[UUID]] = mapped_column(GUID)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    args: Mapped[dict] = mapped_column(JSONType, default=dict)
    # Hash of (thread_id/resource, action_type, args) — prevents two separate
    # pending rows from the same duplicate submission (see app/agent/tools.py).
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    requested_by: Mapped[UUID] = mapped_column(GUID, ForeignKey("users.id"), nullable=False)
    status: Mapped[ApprovalStatus] = mapped_column(
        SAEnum(ApprovalStatus, name="approval_status"), nullable=False, default=ApprovalStatus.PENDING
    )
    approver_id: Mapped[Optional[UUID]] = mapped_column(GUID, ForeignKey("users.id"))
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    resolution_note: Mapped[Optional[str]] = mapped_column(Text)


# --- Long-term memory ---

class LongTermMemory(Base, TimestampMixin):
    __tablename__ = "long_term_memories"
    __table_args__ = (
        Index("ix_long_term_memories_user_type", "user_id", "memory_type"),
    )

    id: Mapped[UUID] = mapped_column(GUID, primary_key=True, default=new_uuid)
    user_id: Mapped[UUID] = mapped_column(GUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    memory_type: Mapped[MemoryType] = mapped_column(
        SAEnum(MemoryType, name="memory_type"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_session_id: Mapped[Optional[UUID]] = mapped_column(GUID, ForeignKey("chat_sessions.id"))
    last_referenced_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    # In Postgres this will additionally have a `pgvector` embedding column,
    # added by an out-of-band migration to keep the base models portable.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


# --- Knowledge base ---

class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(GUID, primary_key=True, default=new_uuid)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    uploaded_by: Mapped[UUID] = mapped_column(GUID, ForeignKey("users.id"), nullable=False)
    department_id: Mapped[Optional[UUID]] = mapped_column(GUID, ForeignKey("departments.id"))
    visibility: Mapped[DocumentVisibility] = mapped_column(
        SAEnum(DocumentVisibility, name="document_visibility"), nullable=False,
        default=DocumentVisibility.INTERNAL,
    )
    status: Mapped[DocumentStatus] = mapped_column(
        SAEnum(DocumentStatus, name="document_status"), nullable=False,
        default=DocumentStatus.QUEUED,
    )
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    effective_date: Mapped[Optional[date]] = mapped_column(Date)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base, TimestampMixin):
    """Chunk of a document ready for retrieval.

    The `embedding` column is added by a Postgres-only migration
    (`vector(embedding_dimensions)`). Kept off the base model so this file
    stays portable for SQLite-based unit tests.
    """
    __tablename__ = "document_chunks"
    __table_args__ = (
        Index("ix_document_chunks_document_id", "document_id"),
    )

    id: Mapped[UUID] = mapped_column(GUID, primary_key=True, default=new_uuid)
    document_id: Mapped[UUID] = mapped_column(GUID, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    parent_chunk_id: Mapped[Optional[UUID]] = mapped_column(GUID, ForeignKey("document_chunks.id"))
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Denormalized RBAC metadata for RLS-style filtering at retrieval time.
    department_id: Mapped[Optional[UUID]] = mapped_column(GUID, ForeignKey("departments.id"))
    visibility: Mapped[DocumentVisibility] = mapped_column(
        SAEnum(DocumentVisibility, name="document_visibility"), nullable=False,
        default=DocumentVisibility.INTERNAL,
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


# --- Audit ---

class AuditLog(Base):
    """Append-only, hash-chained audit log.

    Each row's `chain_hash` = sha256(prev_chain_hash || row_hash). Any tamper
    with historical rows breaks the chain, which the periodic verifier flags.
    """
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_user_id_created", "user_id", "created_at"),
        Index("ix_audit_logs_event_type", "event_type"),
    )

    id: Mapped[UUID] = mapped_column(GUID, primary_key=True, default=new_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    user_id: Mapped[Optional[UUID]] = mapped_column(GUID, ForeignKey("users.id"))
    session_id: Mapped[Optional[UUID]] = mapped_column(GUID, ForeignKey("chat_sessions.id"))
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[Optional[str]] = mapped_column(String(64))
    resource_id: Mapped[Optional[str]] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)  # success | denied | error
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)
    prev_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    chain_hash: Mapped[str] = mapped_column(String(64), nullable=False)
