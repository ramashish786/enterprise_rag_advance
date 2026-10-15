"""Approval routes: the inbox employees/admins use to resolve actions the
agent graph (or document deletion) paused for human sign-off.

Claim/cancellation/change approvals resume the paused LangGraph thread via
Command(resume=...); document-deletion approvals aren't graph-driven at all
(see document_routes.py) and are applied directly here, gated by a
four-eyes check (the approver can't be the admin who requested it).
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import InsuranceAgent
from app.api.deps import _db_session, current_principal
from app.audit.service import AuditLogService
from app.auth.service import SecurityPrincipal
from app.core.logging import get_logger
from app.db.models import ApprovalStatus, ChatMessage, Document, PendingApproval
from app.memory.short_term import append_message, get_stm_store, new_state, summarize_if_needed
from app.schemas.api import ApprovalDecisionRequest, PendingApprovalOut

logger = get_logger(__name__)

router = APIRouter(prefix="/api/approvals", tags=["approvals"])

# Which permission gates deciding each action type. Checked per-row (rather
# than a single require_permission dependency) since the required
# permission depends on the pending approval's action_type.
_APPROVE_PERMISSION = {
    "file_claim": ("claim", "approve"),
    "request_policy_cancellation": ("policy", "approve_cancellation"),
    "request_policy_change": ("policy", "approve_change"),
    "document_deletion": ("document", "approve_deletion"),
}


@router.get("", response_model=list[PendingApprovalOut])
async def list_approvals(
    status_filter: str = "pending",
    principal: SecurityPrincipal = Depends(current_principal),
    session: AsyncSession = Depends(_db_session),
):
    if principal.user_type not in ("employee", "administrator"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="not permitted")

    stmt = select(PendingApproval)
    if status_filter:
        stmt = stmt.where(PendingApproval.status == ApprovalStatus(status_filter))
    if principal.user_type != "administrator":
        # Employees don't get document-deletion four-eyes confirmations —
        # those are admin-only.
        stmt = stmt.where(PendingApproval.action_type != "document_deletion")
    stmt = stmt.order_by(PendingApproval.created_at.asc())

    rows = (await session.execute(stmt)).scalars().all()
    return [_to_out(r) for r in rows]


@router.post("/{approval_id}/decision", response_model=PendingApprovalOut)
async def decide_approval(
    approval_id: UUID,
    payload: ApprovalDecisionRequest,
    principal: SecurityPrincipal = Depends(current_principal),
    session: AsyncSession = Depends(_db_session),
):
    approval = (await session.execute(
        select(PendingApproval).where(PendingApproval.id == approval_id)
    )).scalar_one_or_none()
    if not approval:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="approval not found")
    if approval.status != ApprovalStatus.PENDING:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="already resolved")

    resource, action = _APPROVE_PERMISSION[approval.action_type]
    if not principal.has_permission(resource, action):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"missing permission {resource}:{action}",
        )

    if approval.action_type == "document_deletion" and approval.requested_by == principal.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="document deletion requires a different administrator to confirm",
        )

    audit = AuditLogService(session)

    if approval.action_type == "document_deletion":
        if payload.decision == "approve":
            doc = (await session.execute(
                select(Document).where(Document.id == UUID(approval.args["document_id"]))
            )).scalar_one_or_none()
            if doc:
                await session.delete(doc)
    else:
        agent = InsuranceAgent(session)
        result = await agent.resume(
            approval.thread_id,
            decision=payload.decision,
            approver_id=principal.user_id,
            note=payload.note,
        )
        if approval.thread_id:
            msg = ChatMessage(
                session_id=approval.thread_id, role="assistant", content=result.answer,
                metadata_={"resolved_approval_id": str(approval.id)},
            )
            session.add(msg)

            stm = get_stm_store()
            state = await stm.get(str(approval.thread_id)) or new_state()
            state = append_message(state, "assistant", result.answer)
            state = summarize_if_needed(state)
            await stm.put(str(approval.thread_id), state)

    approval.status = ApprovalStatus.APPROVED if payload.decision == "approve" else ApprovalStatus.REJECTED
    approval.approver_id = principal.user_id
    approval.resolution_note = payload.note
    approval.resolved_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(approval)

    await audit.log(
        event_type="approval.decided",
        action="decide",
        outcome="success",
        user_id=principal.user_id,
        session_id=approval.thread_id,
        resource_type="pending_approval",
        resource_id=str(approval.id),
        payload={"action_type": approval.action_type, "decision": payload.decision},
    )
    return _to_out(approval)


def _to_out(a: PendingApproval) -> PendingApprovalOut:
    return PendingApprovalOut(
        id=a.id, thread_id=a.thread_id, action_type=a.action_type, args=a.args,
        status=a.status.value, requested_by=a.requested_by, approver_id=a.approver_id,
        created_at=a.created_at, resolved_at=a.resolved_at, resolution_note=a.resolution_note,
    )
