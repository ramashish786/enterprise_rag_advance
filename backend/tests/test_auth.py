"""Tests for the auth service and JWT primitives."""
from __future__ import annotations

import pytest
from sqlalchemy import select

from app.auth.service import (
    authenticate,
    issue_tokens,
    load_principal,
    revoke_refresh_token,
    _is_revoked,
)
from app.core.exceptions import AuthenticationError, InvalidTokenError
from app.core.security import decode_token
from app.db.models import User


async def test_authenticate_success(seeded_session):
    user = await authenticate(seeded_session, "alice@example.com", "customer-pass")
    assert user.email == "alice@example.com"


async def test_authenticate_wrong_password_is_uniform_error(seeded_session):
    with pytest.raises(AuthenticationError):
        await authenticate(seeded_session, "alice@example.com", "wrong")


async def test_authenticate_unknown_email_is_uniform_error(seeded_session):
    # Same error class as wrong-password to avoid user enumeration.
    with pytest.raises(AuthenticationError):
        await authenticate(seeded_session, "nobody@example.com", "anything")


async def test_issue_and_decode_access_token(seeded_session):
    alice = (await seeded_session.execute(
        select(User).where(User.email == "alice@example.com")
    )).scalar_one()
    access, refresh, ttl = issue_tokens(alice.id, "customer")

    payload = decode_token(access, expected_type="access")
    assert payload["sub"] == str(alice.id)
    assert payload["user_type"] == "customer"
    assert ttl > 0

    # Decoding as wrong type must fail (defense against token confusion).
    with pytest.raises(InvalidTokenError):
        decode_token(access, expected_type="refresh")


async def test_refresh_token_revocation(seeded_session):
    alice = (await seeded_session.execute(
        select(User).where(User.email == "alice@example.com")
    )).scalar_one()
    _access, refresh, _ttl = issue_tokens(alice.id, "customer")

    payload = decode_token(refresh, expected_type="refresh")
    assert not _is_revoked(payload["jti"])

    revoke_refresh_token(refresh)
    assert _is_revoked(payload["jti"])


async def test_load_principal_populates_roles_and_perms(seeded_session):
    alice = (await seeded_session.execute(
        select(User).where(User.email == "alice@example.com")
    )).scalar_one()
    p = await load_principal(seeded_session, alice.id)

    assert p.email == "alice@example.com"
    assert p.user_type == "customer"
    assert "customer" in p.roles

    # Customer permissions: read own policies, but not read_any.
    assert p.has_permission("policy", "read_own")
    assert not p.has_permission("policy", "read_any")


async def test_load_principal_admin_has_all_perms(seeded_session):
    admin = (await seeded_session.execute(
        select(User).where(User.email == "admin@example.com")
    )).scalar_one()
    p = await load_principal(seeded_session, admin.id)

    assert p.user_type == "administrator"
    assert p.has_permission("user", "manage")
    assert p.has_permission("document", "upload")
    assert p.has_permission("audit_log", "read")


async def test_load_principal_employee_has_department(seeded_session):
    emp = (await seeded_session.execute(
        select(User).where(User.email == "life.employee@example.com")
    )).scalar_one()
    p = await load_principal(seeded_session, emp.id)

    assert p.user_type == "employee"
    assert p.department_name == "Life Insurance"
    assert p.has_permission("policy", "read_department")
