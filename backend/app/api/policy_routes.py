"""Policy routes: customer's own view, employee department view."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import _db_session, current_principal
from app.auth.service import SecurityPrincipal
from app.db.models import InsurancePolicy
from app.rbac.engine import can_view_policy
from app.schemas.api import PolicyOut

router = APIRouter(prefix="/api/policies", tags=["policies"])


@router.get("", response_model=list[PolicyOut])
async def list_policies(
    principal: SecurityPrincipal = Depends(current_principal),
    session: AsyncSession = Depends(_db_session),
):
    rows = (await session.execute(select(InsurancePolicy))).scalars().all()
    visible = [
        p for p in rows
        if can_view_policy(principal, customer_id=p.customer_id, policy_type=p.policy_type)
    ]
    return [
        PolicyOut(
            id=p.id, policy_number=p.policy_number, policy_type=p.policy_type.value,
            coverage_amount=float(p.coverage_amount), premium=float(p.premium),
            start_date=p.start_date.isoformat(), end_date=p.end_date.isoformat(),
            status=p.status.value,
        ) for p in visible
    ]
