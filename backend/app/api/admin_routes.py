"""Admin routes: audit log inspection, user listing, chain verification."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import _db_session, require_permission, require_user_type
from app.audit.service import AuditLogService
from app.auth.service import SecurityPrincipal
from app.db.models import AuditLog, User

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/audit-log")
async def list_audit(
    limit: int = 100,
    principal: SecurityPrincipal = Depends(require_permission("audit_log", "read")),
    session: AsyncSession = Depends(_db_session),
):
    rows = (await session.execute(
        select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
    )).scalars().all()
    return [
        {
            "id": str(r.id),
            "created_at": r.created_at.isoformat(),
            "user_id": str(r.user_id) if r.user_id else None,
            "event_type": r.event_type,
            "action": r.action,
            "outcome": r.outcome,
            "resource_type": r.resource_type,
            "resource_id": r.resource_id,
            "payload": r.payload,
        }
        for r in rows
    ]


@router.post("/audit-log/verify")
async def verify_audit(
    principal: SecurityPrincipal = Depends(require_permission("audit_log", "read")),
    session: AsyncSession = Depends(_db_session),
):
    ok, broken = await AuditLogService(session).verify_chain()
    return {"ok": ok, "first_broken_id": str(broken) if broken else None}


@router.get("/users")
async def list_users(
    principal: SecurityPrincipal = Depends(require_user_type("administrator")),
    session: AsyncSession = Depends(_db_session),
):
    rows = (await session.execute(select(User))).scalars().all()
    return [
        {
            "id": str(u.id),
            "email": u.email,
            "full_name": u.full_name,
            "user_type": u.user_type.value,
            "is_active": u.is_active,
        }
        for u in rows
    ]
