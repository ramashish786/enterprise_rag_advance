"""RBAC boundary tests.

These are the highest-value tests in the suite: they encode the security
posture. If any of these break, we have a data-leakage regression.
"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from app.auth.service import load_principal
from app.core.exceptions import PermissionDenied
from app.db.models import InsurancePolicy, User
from app.rbac.engine import (
    assert_permission,
    authorize_tool,
    can_view_policy,
    visible_departments,
    visible_document_visibilities,
)


# --- Basic permission assertions ---

async def test_customer_cannot_upload_documents(seeded_session):
    alice = (await seeded_session.execute(
        select(User).where(User.email == "alice@example.com")
    )).scalar_one()
    principal = await load_principal(seeded_session, alice.id)

    with pytest.raises(PermissionDenied):
        assert_permission(principal, "document", "upload")


async def test_admin_can_do_everything(seeded_session):
    admin = (await seeded_session.execute(
        select(User).where(User.email == "admin@example.com")
    )).scalar_one()
    principal = await load_principal(seeded_session, admin.id)

    for pair in [("document", "upload"), ("user", "manage"), ("audit_log", "read")]:
        assert_permission(principal, *pair)


# --- Policy scope: the key data-leakage boundary ---

async def test_customer_can_view_own_policy_only(seeded_session):
    alice = (await seeded_session.execute(
        select(User).where(User.email == "alice@example.com")
    )).scalar_one()
    bob = (await seeded_session.execute(
        select(User).where(User.email == "bob@example.com")
    )).scalar_one()

    alice_p = await load_principal(seeded_session, alice.id)

    alice_policy = (await seeded_session.execute(
        select(InsurancePolicy).where(InsurancePolicy.customer_id == alice.id).limit(1)
    )).scalar_one()
    bob_policy = (await seeded_session.execute(
        select(InsurancePolicy).where(InsurancePolicy.customer_id == bob.id).limit(1)
    )).scalar_one()

    assert can_view_policy(alice_p, customer_id=alice.id, policy_type=alice_policy.policy_type)
    assert not can_view_policy(alice_p, customer_id=bob.id, policy_type=bob_policy.policy_type)


async def test_life_employee_sees_only_life_policies(seeded_session):
    emp = (await seeded_session.execute(
        select(User).where(User.email == "life.employee@example.com")
    )).scalar_one()
    principal = await load_principal(seeded_session, emp.id)

    # Life policies visible regardless of customer identity.
    life = (await seeded_session.execute(
        select(InsurancePolicy).where(InsurancePolicy.policy_type == "life").limit(1)
    )).scalar_one()
    home = (await seeded_session.execute(
        select(InsurancePolicy).where(InsurancePolicy.policy_type == "home").limit(1)
    )).scalar_one()
    vehicle = (await seeded_session.execute(
        select(InsurancePolicy).where(InsurancePolicy.policy_type == "vehicle").limit(1)
    )).scalar_one()

    assert can_view_policy(
        principal, customer_id=life.customer_id, policy_type=life.policy_type
    )
    # But home and vehicle should be denied.
    assert not can_view_policy(
        principal, customer_id=home.customer_id, policy_type=home.policy_type
    )
    assert not can_view_policy(
        principal, customer_id=vehicle.customer_id, policy_type=vehicle.policy_type
    )


async def test_customer_cannot_see_any_internal_documents(seeded_session):
    alice = (await seeded_session.execute(
        select(User).where(User.email == "alice@example.com")
    )).scalar_one()
    principal = await load_principal(seeded_session, alice.id)

    tiers = visible_document_visibilities(principal)
    assert tiers == {"public"}


async def test_employee_sees_public_and_internal_documents(seeded_session):
    emp = (await seeded_session.execute(
        select(User).where(User.email == "life.employee@example.com")
    )).scalar_one()
    principal = await load_principal(seeded_session, emp.id)

    tiers = visible_document_visibilities(principal)
    assert tiers == {"public", "internal"}


async def test_admin_sees_all_document_tiers(seeded_session):
    admin = (await seeded_session.execute(
        select(User).where(User.email == "admin@example.com")
    )).scalar_one()
    principal = await load_principal(seeded_session, admin.id)

    tiers = visible_document_visibilities(principal)
    assert tiers == {"public", "internal", "restricted"}


async def test_visible_departments_by_role(seeded_session):
    admin = (await seeded_session.execute(
        select(User).where(User.email == "admin@example.com")
    )).scalar_one()
    admin_p = await load_principal(seeded_session, admin.id)
    assert visible_departments(admin_p) is None  # unrestricted

    emp = (await seeded_session.execute(
        select(User).where(User.email == "life.employee@example.com")
    )).scalar_one()
    emp_p = await load_principal(seeded_session, emp.id)
    depts = visible_departments(emp_p)
    assert depts is not None and len(depts) == 1

    alice = (await seeded_session.execute(
        select(User).where(User.email == "alice@example.com")
    )).scalar_one()
    alice_p = await load_principal(seeded_session, alice.id)
    assert visible_departments(alice_p) == set()


# --- Tool decorator ---

async def test_authorize_tool_decorator_blocks_unauthorized(seeded_session):
    alice = (await seeded_session.execute(
        select(User).where(User.email == "alice@example.com")
    )).scalar_one()
    principal = await load_principal(seeded_session, alice.id)

    @authorize_tool(("document", "upload"))
    async def upload_thing(p): return "ok"

    with pytest.raises(PermissionDenied):
        await upload_thing(principal)


async def test_authorize_tool_decorator_allows_authorized(seeded_session):
    admin = (await seeded_session.execute(
        select(User).where(User.email == "admin@example.com")
    )).scalar_one()
    principal = await load_principal(seeded_session, admin.id)

    @authorize_tool(("document", "upload"))
    async def upload_thing(p): return "ok"

    assert await upload_thing(principal) == "ok"
