"""Human-in-the-loop approval flow: interrupt/resume and idempotency.

Runs against the hermetic SQLite + InMemorySaver path (see
app/agent/checkpointer.py) — no real Postgres needed to exercise the graph's
checkpoint/interrupt/resume mechanics.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import select

from app.agent.graph import InsuranceAgent
from app.agent.llm import FakeLLM
from app.agent.tools import file_claim
from app.auth.service import load_principal
from app.db.models import Claim, InsurancePolicy, PolicyStatus, User


async def _alice(seeded_session):
    row = (await seeded_session.execute(
        select(User).where(User.email == "alice@example.com")
    )).scalar_one()
    return await load_principal(seeded_session, row.id)


@pytest.mark.asyncio
async def test_low_value_claim_auto_approves_no_interrupt(seeded_session):
    principal = await _alice(seeded_session)
    agent = InsuranceAgent(seeded_session, llm=FakeLLM())

    result = await agent.run(
        "file a claim for $500 on LIFE-100001 for a cracked windshield", principal,
    )
    assert result.interrupted is False
    assert "CLM-" in result.answer

    claims = (await seeded_session.execute(select(Claim))).scalars().all()
    assert len(claims) == 1
    assert float(claims[0].amount) == 500.0


@pytest.mark.asyncio
async def test_high_value_claim_interrupts_then_approve_creates_claim(seeded_session):
    principal = await _alice(seeded_session)
    session_id = uuid4()
    agent = InsuranceAgent(seeded_session, llm=FakeLLM())

    result = await agent.run(
        "file a claim for $60,000 on LIFE-100001 for fire damage",
        principal, session_id=session_id,
    )
    assert result.interrupted is True
    assert result.pending_action["type"] == "file_claim"
    assert result.pending_action["args"]["policy_number"] == "LIFE-100001"

    claims = (await seeded_session.execute(select(Claim))).scalars().all()
    assert len(claims) == 0

    # Resuming via a brand-new InsuranceAgent instance mirrors production
    # (approval_routes.py constructs a fresh agent per request); it still
    # resolves the same paused thread because the checkpointer is shared.
    resumer = InsuranceAgent(seeded_session, llm=FakeLLM())
    resumed = await resumer.resume(session_id, decision="approve", approver_id=principal.user_id)

    assert resumed.interrupted is False
    assert "CLM-" in resumed.answer

    claims = (await seeded_session.execute(select(Claim))).scalars().all()
    assert len(claims) == 1
    assert float(claims[0].amount) == 60000.0


@pytest.mark.asyncio
async def test_reject_does_not_create_claim(seeded_session):
    principal = await _alice(seeded_session)
    session_id = uuid4()
    agent = InsuranceAgent(seeded_session, llm=FakeLLM())

    await agent.run(
        "file a claim for $60,000 on LIFE-100001 for fire damage",
        principal, session_id=session_id,
    )
    resumed = await agent.resume(
        session_id, decision="reject", approver_id=principal.user_id, note="not covered",
    )
    assert "rejected" in resumed.answer.lower()

    claims = (await seeded_session.execute(select(Claim))).scalars().all()
    assert len(claims) == 0


@pytest.mark.asyncio
async def test_cancellation_always_interrupts_and_applies_on_approve(seeded_session):
    principal = await _alice(seeded_session)
    session_id = uuid4()
    agent = InsuranceAgent(seeded_session, llm=FakeLLM())

    result = await agent.run(
        "please cancel my policy LIFE-100001, found cheaper coverage elsewhere",
        principal, session_id=session_id,
    )
    assert result.interrupted is True
    assert result.pending_action["type"] == "request_policy_cancellation"

    await agent.resume(session_id, decision="approve", approver_id=principal.user_id)

    policy = (await seeded_session.execute(
        select(InsurancePolicy).where(InsurancePolicy.policy_number == "LIFE-100001")
    )).scalar_one()
    assert policy.status == PolicyStatus.CANCELLED


@pytest.mark.asyncio
async def test_change_coverage_always_interrupts_and_applies_on_approve(seeded_session):
    principal = await _alice(seeded_session)
    session_id = uuid4()
    agent = InsuranceAgent(seeded_session, llm=FakeLLM())

    result = await agent.run(
        "please increase coverage on LIFE-100001 to $750,000",
        principal, session_id=session_id,
    )
    assert result.interrupted is True
    assert result.pending_action["type"] == "request_policy_change"

    await agent.resume(session_id, decision="approve", approver_id=principal.user_id)

    policy = (await seeded_session.execute(
        select(InsurancePolicy).where(InsurancePolicy.policy_number == "LIFE-100001")
    )).scalar_one()
    assert float(policy.coverage_amount) == 750000.0


@pytest.mark.asyncio
async def test_idempotency_key_prevents_duplicate_claim_on_replay(seeded_session):
    """The crown-jewel test: simulates a node re-executing after a
    crash-and-resume by calling file_claim twice with the same
    deterministic idempotency_key — exactly one Claim row must result."""
    principal = await _alice(seeded_session)
    key = "fixed-test-idempotency-key"

    first = await file_claim(
        principal, seeded_session,
        policy_number="LIFE-100001", amount=1000, description="test claim",
        idempotency_key=key,
    )
    second = await file_claim(
        principal, seeded_session,
        policy_number="LIFE-100001", amount=1000, description="test claim",
        idempotency_key=key,
    )
    assert first["id"] == second["id"]

    claims = (await seeded_session.execute(
        select(Claim).where(Claim.dedupe_key == key)
    )).scalars().all()
    assert len(claims) == 1
