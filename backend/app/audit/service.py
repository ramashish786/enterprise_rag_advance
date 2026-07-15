"""Audit log service.

Every mutating event, authorization decision, LLM call, tool call, and RBAC
denial is appended here. Rows are hash-chained: each row's `chain_hash` =
sha256(prev_chain_hash || row_hash), so any tamper (row edit, insertion,
deletion) breaks the chain. `verify_chain` walks the log and finds the first
broken link.

Append-only enforcement:
    - `AuditLogService.log()` is the only write path.
    - Postgres grants revoke UPDATE/DELETE on `audit_logs` (see rls.py).
    - RLS restricts SELECT to administrators.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import AuditLog

logger = get_logger(__name__)

GENESIS_HASH = "0" * 64


def _canonical_json(payload: dict) -> str:
    """Stable serialization for hashing: sorted keys, no whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _iso(dt: datetime) -> str:
    """Timezone-normalized ISO string.

    SQLite loses tzinfo on roundtrip; Postgres keeps it. To make the hash
    deterministic across both, we drop tzinfo (assuming UTC) and format
    without offset.
    """
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.isoformat()


def _row_hash(row_fields: dict) -> str:
    return hashlib.sha256(_canonical_json(row_fields).encode()).hexdigest()


def _chain_hash(prev: str, row: str) -> str:
    return hashlib.sha256((prev + row).encode()).hexdigest()


class AuditLogService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def _latest_hash(self) -> str:
        stmt = select(AuditLog.chain_hash).order_by(AuditLog.created_at.desc()).limit(1)
        result = await self.session.execute(stmt)
        return result.scalar() or GENESIS_HASH

    async def log(
        self,
        *,
        event_type: str,
        action: str,
        outcome: str,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> AuditLog:
        payload = payload or {}
        now = datetime.now(timezone.utc)
        prev = await self._latest_hash()

        row_fields = {
            "created_at": _iso(now),
            "user_id": str(user_id) if user_id else None,
            "session_id": str(session_id) if session_id else None,
            "event_type": event_type,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "action": action,
            "outcome": outcome,
            "payload": payload,
        }
        row_h = _row_hash(row_fields)
        chain_h = _chain_hash(prev, row_h)

        entry = AuditLog(
            created_at=now,
            user_id=user_id,
            session_id=session_id,
            event_type=event_type,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            payload=payload,
            prev_hash=prev,
            chain_hash=chain_h,
        )
        self.session.add(entry)
        await self.session.commit()
        await self.session.refresh(entry)
        return entry

    async def verify_chain(self) -> tuple[bool, Optional[UUID]]:
        """Walk the chain in order and return (ok, first_broken_id).

        `first_broken_id` is None on success. On failure, points at the row
        whose chain_hash disagrees with the recomputed hash: the first
        tampered position, or the position just after a deletion.
        """
        stmt = select(AuditLog).order_by(AuditLog.created_at.asc(), AuditLog.id.asc())
        rows = (await self.session.execute(stmt)).scalars().all()

        expected_prev = GENESIS_HASH
        for row in rows:
            row_fields = {
                "created_at": _iso(row.created_at),
                "user_id": str(row.user_id) if row.user_id else None,
                "session_id": str(row.session_id) if row.session_id else None,
                "event_type": row.event_type,
                "resource_type": row.resource_type,
                "resource_id": row.resource_id,
                "action": row.action,
                "outcome": row.outcome,
                "payload": row.payload,
            }
            expected_row = _row_hash(row_fields)
            expected_chain = _chain_hash(expected_prev, expected_row)

            if row.prev_hash != expected_prev or row.chain_hash != expected_chain:
                return False, row.id
            expected_prev = row.chain_hash

        return True, None
