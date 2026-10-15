"""Agent tools.

Every tool takes a SecurityPrincipal instead of trusting caller-supplied
identity, gates itself with `@authorize_tool`, sticks to parameterized
queries (no LLM-generated SQL), and returns a dict shaped for the
synthesis node to cite.

TOOL_DESCRIPTIONS below is what the LLM router reads to decide which tool
to call, so keep the wording precise; rephrasing it changes routing
behavior.
"""
from __future__ import annotations

from decimal import Decimal
from datetime import date
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.service import SecurityPrincipal
from app.core.logging import get_logger
from app.db.models import Claim, DocumentChunk, InsurancePolicy, PolicyStatus, User
from app.memory.long_term import LongTermMemoryService
from app.rag.retriever import ChunkRecord, HybridRetriever, RetrievedChunk
from app.rag.reranker import get_reranker
from app.rag.embedder import get_embedder
from app.rbac.engine import authorize_tool, can_view_policy, visible_document_visibilities, visible_departments

logger = get_logger(__name__)


TOOL_DESCRIPTIONS = {
    "get_my_policies": (
        "Look up the *authenticated user's own* insurance policies. Returns "
        "policy number, type, coverage amount, premium, dates, and status."
    ),
    "get_policy_by_number": (
        "Look up a specific policy by policy number, subject to RBAC. "
        "Customers can only view their own; employees only within their "
        "department; admins can view any."
    ),
    "get_claims_for_policy": (
        "Look up claim history for a specific policy, subject to RBAC."
    ),
    "search_knowledge_base": (
        "Semantic + keyword search over the insurance knowledge base "
        "(policy documents, FAQs, coverage guides). Results are pre-filtered "
        "by the user's visibility tier and department scope."
    ),
    "recall_long_term_memory": (
        "Recall the user's stored long-term memories (preferences, facts). "
        "Only ever accesses memories belonging to the authenticated user."
    ),
    "file_claim": (
        "File an insurance claim against one of the authenticated user's own "
        "policies. Requires policy_number, amount, and description. Mutating: "
        "claims above the auto-approve threshold pause for employee/admin "
        "approval before the claim is created."
    ),
    "request_policy_cancellation": (
        "Request cancellation of one of the authenticated user's own policies. "
        "Requires policy_number and reason. Mutating: always pauses for "
        "employee/admin approval before the policy is cancelled."
    ),
    "request_policy_change": (
        "Request a coverage-amount change on one of the authenticated user's "
        "own policies. Requires policy_number, new_coverage_amount, and reason. "
        "Mutating: always pauses for employee/admin approval before the change "
        "is applied."
    ),
}

# Tools that mutate data rather than just read it. The graph routes these
# through a prepare/approve/apply sequence instead of straight to synthesis.
MUTATING_TOOLS = frozenset({"file_claim", "request_policy_cancellation", "request_policy_change"})


# --- get_my_policies ---

@authorize_tool(("policy", "read_own"), ("policy", "read_department"), ("policy", "read_any"))
async def get_my_policies(
    principal: SecurityPrincipal, session: AsyncSession
) -> list[dict[str, Any]]:
    if principal.user_type != "customer":
        # Non-customers have no "own policies" concept; return empty rather
        # than error so the router can fall back to another tool.
        return []
    rows = (await session.execute(
        select(InsurancePolicy).where(InsurancePolicy.customer_id == principal.user_id)
    )).scalars().all()
    return [_policy_to_dict(p) for p in rows]


# --- get_policy_by_number ---

@authorize_tool(("policy", "read_own"), ("policy", "read_department"), ("policy", "read_any"))
async def get_policy_by_number(
    principal: SecurityPrincipal, session: AsyncSession, policy_number: str,
) -> Optional[dict[str, Any]]:
    if not policy_number or len(policy_number) > 64:
        return None
    row = (await session.execute(
        select(InsurancePolicy).where(InsurancePolicy.policy_number == policy_number)
    )).scalar_one_or_none()
    if not row:
        return None
    if not can_view_policy(principal, customer_id=row.customer_id, policy_type=row.policy_type):
        logger.info(
            "policy_access_denied",
            user_id=str(principal.user_id),
            policy_number=policy_number,
        )
        return None
    return _policy_to_dict(row)


# --- get_claims_for_policy ---

@authorize_tool(("policy", "read_own"), ("policy", "read_department"), ("policy", "read_any"))
async def get_claims_for_policy(
    principal: SecurityPrincipal, session: AsyncSession, policy_number: str,
) -> list[dict[str, Any]]:
    policy = await get_policy_by_number(principal, session, policy_number)
    if not policy:
        return []
    claims = (await session.execute(
        select(Claim).where(Claim.policy_id == UUID(policy["id"]))
    )).scalars().all()
    return [_claim_to_dict(c) for c in claims]


# --- search_knowledge_base ---

@authorize_tool(("document", "read_public"), ("document", "read_internal"), ("document", "read_restricted"))
async def search_knowledge_base(
    principal: SecurityPrincipal, session: AsyncSession, query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """RBAC-pre-filtered hybrid retrieval + rerank."""
    tiers = visible_document_visibilities(principal)
    depts = visible_departments(principal)

    # Fetch candidate chunks from DB (RBAC WHERE-clause).
    stmt = select(DocumentChunk)
    if tiers:
        stmt = stmt.where(DocumentChunk.visibility.in_(list(tiers)))
    else:
        return []
    if depts is not None:
        if not depts:
            # Customer has no department scope: allow public docs regardless
            # of department (department_id is null for global docs).
            stmt = stmt.where(DocumentChunk.department_id.is_(None))
        else:
            # Employee/admin department scope.
            from sqlalchemy import or_
            stmt = stmt.where(or_(
                DocumentChunk.department_id.in_([UUID(d) for d in depts]),
                DocumentChunk.department_id.is_(None),
            ))

    rows = (await session.execute(stmt)).scalars().all()
    if not rows:
        return []

    # Build the in-memory corpus for the reference retriever. In the Postgres
    # deployment this instead runs as pgvector cosine + tsvector BM25 directly
    # in SQL (see PgVectorRetriever).
    embedder = get_embedder()
    corpus: list[ChunkRecord] = []
    contents = [r.content for r in rows]
    vectors = await embedder.embed(contents)
    for row, vec in zip(rows, vectors):
        corpus.append(ChunkRecord(
            id=str(row.id),
            content=row.content,
            embedding=vec,
            metadata={
                "document_id": str(row.document_id),
                "chunk_index": row.chunk_index,
                "visibility": row.visibility.value if hasattr(row.visibility, "value") else str(row.visibility),
            },
        ))

    retriever = HybridRetriever(embedder=embedder)
    initial = await retriever.retrieve(query, corpus, top_k_final=max(top_k * 3, 10))

    reranker = get_reranker()
    final = await reranker.rerank(query, initial, top_k=top_k)
    return [_chunk_to_dict(c) for c in final]


# --- recall_long_term_memory ---

async def recall_long_term_memory(
    principal: SecurityPrincipal, session: AsyncSession, query: str, top_k: int = 5,
) -> list[dict[str, Any]]:
    svc = LongTermMemoryService(session)
    memories = await svc.recall(principal.user_id, query, top_k=top_k)
    return [{
        "id": f"memory:{m.id}",
        "content": m.content,
        "memory_type": m.memory_type.value,
    } for m in memories]


# --- file_claim ---
#
# Mutating. `idempotency_key` is a deterministic hash computed once by
# node_prepare_mutating_action from state that's already checkpointed (thread
# id, action type, tool args) — so if this node re-executes after a crash and
# resume, it recomputes the same key rather than a fresh one. The unique
# constraint on Claim.dedupe_key is the actual guard: a second insert attempt
# with the same key hits IntegrityError and we return the row that already
# exists instead of creating a duplicate.

@authorize_tool(("claim", "file"))
async def file_claim(
    principal: SecurityPrincipal, session: AsyncSession, *,
    policy_number: str, amount: float, description: str, idempotency_key: str,
) -> dict[str, Any]:
    policy = await get_policy_by_number(principal, session, policy_number)
    if not policy:
        return {"error": f"no accessible policy {policy_number!r}"}

    existing = (await session.execute(
        select(Claim).where(Claim.dedupe_key == idempotency_key)
    )).scalar_one_or_none()
    if existing:
        return _claim_to_dict(existing)

    claim = Claim(
        claim_number=f"CLM-{uuid4().hex[:10].upper()}",
        policy_id=UUID(policy["id"]),
        filed_date=date.today(),
        amount=Decimal(str(amount)),
        status="open",
        description=description,
        dedupe_key=idempotency_key,
    )
    session.add(claim)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        existing = (await session.execute(
            select(Claim).where(Claim.dedupe_key == idempotency_key)
        )).scalar_one()
        return _claim_to_dict(existing)
    await session.refresh(claim)
    return _claim_to_dict(claim)


# --- request_policy_cancellation / request_policy_change ---
#
# Unlike file_claim these are UPDATEs, not INSERTs: replaying either one after
# a crash-and-resume lands on the same end state, so there's no duplicate-row
# risk and no dedupe_key check needed here. idempotency_key is still accepted
# (for a uniform call signature from apply_mutating_action) and is what
# PendingApproval.idempotency_key dedupes on, to avoid two separate approval
# requests for the same duplicate submission.

@authorize_tool(("policy", "request_cancellation"))
async def request_policy_cancellation(
    principal: SecurityPrincipal, session: AsyncSession, *,
    policy_number: str, reason: str, idempotency_key: str,
) -> dict[str, Any]:
    policy = await get_policy_by_number(principal, session, policy_number)
    if not policy:
        return {"error": f"no accessible policy {policy_number!r}"}

    row = (await session.execute(
        select(InsurancePolicy).where(InsurancePolicy.id == UUID(policy["id"]))
    )).scalar_one()
    row.status = PolicyStatus.CANCELLED
    await session.commit()
    await session.refresh(row)
    return _policy_to_dict(row)


@authorize_tool(("policy", "request_change"))
async def request_policy_change(
    principal: SecurityPrincipal, session: AsyncSession, *,
    policy_number: str, new_coverage_amount: float, reason: str, idempotency_key: str,
) -> dict[str, Any]:
    policy = await get_policy_by_number(principal, session, policy_number)
    if not policy:
        return {"error": f"no accessible policy {policy_number!r}"}

    row = (await session.execute(
        select(InsurancePolicy).where(InsurancePolicy.id == UUID(policy["id"]))
    )).scalar_one()
    row.coverage_amount = Decimal(str(new_coverage_amount))
    await session.commit()
    await session.refresh(row)
    return _policy_to_dict(row)


MUTATING_TOOL_REGISTRY = {
    "file_claim": file_claim,
    "request_policy_cancellation": request_policy_cancellation,
    "request_policy_change": request_policy_change,
}


# --- Serialization helpers ---

def _policy_to_dict(p: InsurancePolicy) -> dict[str, Any]:
    return {
        "id": str(p.id),
        "policy_number": p.policy_number,
        "policy_type": p.policy_type.value if hasattr(p.policy_type, "value") else str(p.policy_type),
        "coverage_amount": float(p.coverage_amount),
        "premium": float(p.premium),
        "start_date": p.start_date.isoformat(),
        "end_date": p.end_date.isoformat(),
        "status": p.status.value if hasattr(p.status, "value") else str(p.status),
    }


def _claim_to_dict(c: Claim) -> dict[str, Any]:
    return {
        "id": str(c.id),
        "claim_number": c.claim_number,
        "filed_date": c.filed_date.isoformat(),
        "amount": float(c.amount),
        "status": c.status,
        "description": c.description,
    }


def _chunk_to_dict(c: RetrievedChunk) -> dict[str, Any]:
    return {
        "id": str(c.id),
        "content": c.content,
        "score": c.score,
        "metadata": c.metadata,
    }
