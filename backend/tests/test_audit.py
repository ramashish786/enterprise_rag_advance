"""Audit log tests including tamper-evidence."""
from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select

from app.audit.service import AuditLogService, GENESIS_HASH
from app.db.models import AuditLog


async def test_first_entry_uses_genesis_prev(db_session):
    svc = AuditLogService(db_session)
    entry = await svc.log(
        event_type="auth.login", action="login", outcome="success",
        payload={"email": "a@b.com"},
    )
    assert entry.prev_hash == GENESIS_HASH
    assert len(entry.chain_hash) == 64


async def test_chain_links_across_entries(db_session):
    svc = AuditLogService(db_session)
    e1 = await svc.log(event_type="a", action="x", outcome="success")
    e2 = await svc.log(event_type="b", action="y", outcome="success")
    e3 = await svc.log(event_type="c", action="z", outcome="denied")

    assert e2.prev_hash == e1.chain_hash
    assert e3.prev_hash == e2.chain_hash


async def test_verify_chain_passes_when_untouched(db_session):
    svc = AuditLogService(db_session)
    for i in range(5):
        await svc.log(event_type=f"e{i}", action="x", outcome="success",
                      payload={"i": i})
    ok, broken = await svc.verify_chain()
    assert ok
    assert broken is None


async def test_verify_chain_detects_payload_tamper(db_session):
    svc = AuditLogService(db_session)
    await svc.log(event_type="a", action="x", outcome="success")
    await svc.log(event_type="b", action="y", outcome="success", payload={"amount": 100})
    await svc.log(event_type="c", action="z", outcome="success")

    # simulate a tamper by modifying the middle row's payload directly
    victim = (await db_session.execute(
        select(AuditLog).where(AuditLog.event_type == "b")
    )).scalar_one()
    victim.payload = {"amount": 999999}
    await db_session.commit()

    ok, broken = await svc.verify_chain()
    assert not ok
    assert broken == victim.id


async def test_verify_chain_detects_deletion(db_session):
    svc = AuditLogService(db_session)
    await svc.log(event_type="a", action="x", outcome="success")
    e2 = await svc.log(event_type="b", action="y", outcome="success")
    await svc.log(event_type="c", action="z", outcome="success")

    # Delete the middle row directly.
    await db_session.delete(e2)
    await db_session.commit()

    ok, broken = await svc.verify_chain()
    assert not ok  # the row after the deletion becomes the broken link


async def test_rbac_denials_are_logged(db_session):
    svc = AuditLogService(db_session)
    uid = uuid4()
    entry = await svc.log(
        event_type="rbac.denied",
        action="read",
        outcome="denied",
        user_id=uid,
        resource_type="policy",
        resource_id="POL-123",
        payload={"required_permission": "policy:read_any"},
    )
    assert entry.outcome == "denied"
    assert entry.user_id == uid
