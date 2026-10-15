"""HTTP integration tests using httpx AsyncClient in-process.

Exercises the full stack: auth, RBAC, agent, guardrails, chat SSE.
Uses the same test SQLite DB via dependency override.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import _db_session


@pytest_asyncio.fixture
async def app_client(test_engine, monkeypatch):
    """FastAPI app wired to the test SQLite engine."""
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    # Patch get_session in every module that uses it BEFORE importing the app.
    @asynccontextmanager
    async def _test_get_session(**_kw):
        async with factory() as s:
            try:
                yield s
            except Exception:
                await s.rollback()
                raise

    from app.db import session as session_module, seed as seed_module
    monkeypatch.setattr(session_module, "get_session", _test_get_session)
    monkeypatch.setattr(seed_module, "get_session", _test_get_session)

    # Now seed.
    from app.db.seed import seed_all
    await seed_all()

    # Import app and override the DB dependency.
    from app.main import app

    async def _override_db_session():
        async with factory() as s:
            yield s

    app.dependency_overrides[_db_session] = _override_db_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
class TestAuthEndpoints:
    async def test_login_success(self, app_client):
        r = await app_client.post("/api/auth/login", json={
            "email": "alice@example.com", "password": "customer-pass",
        })
        assert r.status_code == 200
        body = r.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"

    async def test_login_wrong_password(self, app_client):
        r = await app_client.post("/api/auth/login", json={
            "email": "alice@example.com", "password": "wrong",
        })
        assert r.status_code == 401

    async def test_me_requires_auth(self, app_client):
        r = await app_client.get("/api/auth/me")
        assert r.status_code == 401

    async def test_me_returns_current_user(self, app_client):
        login = await app_client.post("/api/auth/login", json={
            "email": "alice@example.com", "password": "customer-pass",
        })
        token = login.json()["access_token"]
        r = await app_client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["email"] == "alice@example.com"
        assert r.json()["user_type"] == "customer"


@pytest.mark.asyncio
class TestPolicyEndpoint:
    async def _login(self, app_client, email, password):
        r = await app_client.post("/api/auth/login", json={"email": email, "password": password})
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    async def test_customer_sees_only_own_policies(self, app_client):
        headers = await self._login(app_client, "alice@example.com", "customer-pass")
        r = await app_client.get("/api/policies", headers=headers)
        assert r.status_code == 200
        policies = r.json()
        # Alice has 2 policies
        assert len(policies) == 2
        # Confirm none are bob's vehicle policy
        assert not any(p["policy_number"] == "VEHICLE-300001" for p in policies)

    async def test_life_employee_sees_only_life_policies(self, app_client):
        headers = await self._login(app_client, "life.employee@example.com", "employee-pass")
        r = await app_client.get("/api/policies", headers=headers)
        assert r.status_code == 200
        policies = r.json()
        # All returned policies must be life
        assert all(p["policy_type"] == "life" for p in policies)
        # Should not include VEHICLE or HOME
        assert not any(p["policy_type"] in ("vehicle", "home") for p in policies)

    async def test_admin_sees_all_policies(self, app_client):
        headers = await self._login(app_client, "admin@example.com", "admin-change-me")
        r = await app_client.get("/api/policies", headers=headers)
        assert r.status_code == 200
        policies = r.json()
        types = {p["policy_type"] for p in policies}
        assert types == {"life", "home", "vehicle"}


@pytest.mark.asyncio
class TestChatEndpoints:
    async def _login(self, app_client, email="alice@example.com", password="customer-pass"):
        r = await app_client.post("/api/auth/login", json={"email": email, "password": password})
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    async def test_create_and_list_session(self, app_client):
        headers = await self._login(app_client)
        r = await app_client.post("/api/chat/sessions", json={"title": "Q about policy"}, headers=headers)
        assert r.status_code == 200
        session_id = r.json()["id"]
        assert r.json()["title"] == "Q about policy"

        r = await app_client.get("/api/chat/sessions", headers=headers)
        assert r.status_code == 200
        sessions = r.json()
        assert any(s["id"] == session_id for s in sessions)

    async def test_customer_cannot_see_other_customer_sessions(self, app_client):
        # Alice creates a session
        alice_h = await self._login(app_client, "alice@example.com", "customer-pass")
        r = await app_client.post("/api/chat/sessions", json={}, headers=alice_h)
        alice_session = r.json()["id"]

        # Bob logs in, should not see Alice's session
        bob_h = await self._login(app_client, "bob@example.com", "customer-pass")
        r = await app_client.get("/api/chat/sessions", headers=bob_h)
        assert r.status_code == 200
        assert not any(s["id"] == alice_session for s in r.json())

        # Bob cannot fetch messages of Alice's session
        r = await app_client.get(f"/api/chat/sessions/{alice_session}/messages", headers=bob_h)
        assert r.status_code == 404


@pytest.mark.asyncio
class TestAdminEndpoints:
    async def _login(self, app_client, email, password):
        r = await app_client.post("/api/auth/login", json={"email": email, "password": password})
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    async def test_customer_cannot_read_audit_log(self, app_client):
        headers = await self._login(app_client, "alice@example.com", "customer-pass")
        r = await app_client.get("/api/admin/audit-log", headers=headers)
        assert r.status_code == 403

    async def test_admin_can_read_audit_log(self, app_client):
        headers = await self._login(app_client, "admin@example.com", "admin-change-me")
        r = await app_client.get("/api/admin/audit-log", headers=headers)
        assert r.status_code == 200
        # After the login we should have at least one entry (auth.login)
        entries = r.json()
        assert any(e["event_type"] == "auth.login" for e in entries)

    async def test_admin_can_verify_audit_chain(self, app_client):
        # Ensure at least one entry so chain isn't empty
        await self._login(app_client, "alice@example.com", "customer-pass")

        headers = await self._login(app_client, "admin@example.com", "admin-change-me")
        r = await app_client.post("/api/admin/audit-log/verify", headers=headers)
        assert r.status_code == 200
        assert r.json()["ok"] is True

    async def test_customer_cannot_upload_documents(self, app_client):
        headers = await self._login(app_client, "alice@example.com", "customer-pass")
        r = await app_client.post(
            "/api/documents/upload",
            headers=headers,
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert r.status_code == 403


@pytest.mark.asyncio
class TestApprovalEndpoints:
    async def _login(self, app_client, email, password):
        r = await app_client.post("/api/auth/login", json={"email": email, "password": password})
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    async def test_customer_cannot_list_approvals(self, app_client):
        headers = await self._login(app_client, "alice@example.com", "customer-pass")
        r = await app_client.get("/api/approvals", headers=headers)
        assert r.status_code == 403

    async def test_employee_can_approve_high_value_claim(self, app_client):
        alice_h = await self._login(app_client, "alice@example.com", "customer-pass")
        r = await app_client.post("/api/chat/sessions", json={}, headers=alice_h)
        session_id = r.json()["id"]

        r = await app_client.post(
            f"/api/chat/sessions/{session_id}/messages",
            json={"content": "file a claim for $60,000 on LIFE-100001 for fire damage"},
            headers=alice_h,
        )
        assert r.status_code == 200
        assert "awaiting_approval" in r.text

        # A second message on the same session while it's paused is rejected
        # rather than interleaved with the unresolved turn.
        r = await app_client.post(
            f"/api/chat/sessions/{session_id}/messages",
            json={"content": "hello?"},
            headers=alice_h,
        )
        assert r.status_code == 409

        emp_h = await self._login(app_client, "life.employee@example.com", "employee-pass")
        r = await app_client.get("/api/approvals?status_filter=pending", headers=emp_h)
        assert r.status_code == 200
        pending = [a for a in r.json() if a["thread_id"] == session_id]
        assert len(pending) == 1
        approval_id = pending[0]["id"]

        r = await app_client.post(
            f"/api/approvals/{approval_id}/decision",
            json={"decision": "approve"},
            headers=emp_h,
        )
        assert r.status_code == 200
        assert r.json()["status"] == "approved"

    async def test_document_deletion_requires_a_different_admin(self, app_client, test_engine):
        admin1_h = await self._login(app_client, "admin@example.com", "admin-change-me")

        r = await app_client.post(
            "/api/documents/upload", headers=admin1_h,
            files={"file": ("delete_me.txt", b"hello world", "text/plain")},
        )
        assert r.status_code == 200
        doc_id = r.json()["id"]

        r = await app_client.post(
            f"/api/documents/{doc_id}/request-deletion", headers=admin1_h, data={"reason": "test"},
        )
        assert r.status_code == 200
        approval_id = r.json()["approval_id"]

        # The requesting admin can't confirm their own request.
        r = await app_client.post(
            f"/api/approvals/{approval_id}/decision",
            json={"decision": "approve"},
            headers=admin1_h,
        )
        assert r.status_code == 403

        # A second administrator can.
        from sqlalchemy import select
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
        from app.core.security import hash_password
        from app.db.models import Role, User, UserRole, UserType

        factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as s:
            admin_role = (await s.execute(select(Role).where(Role.name == "administrator"))).scalar_one()
            second_admin = User(
                email="admin2@example.com", hashed_password=hash_password("admin2-pass"),
                user_type=UserType.ADMINISTRATOR, full_name="Second Admin",
            )
            s.add(second_admin)
            await s.flush()
            s.add(UserRole(user_id=second_admin.id, role_id=admin_role.id))
            await s.commit()

        admin2_h = await self._login(app_client, "admin2@example.com", "admin2-pass")
        r = await app_client.post(
            f"/api/approvals/{approval_id}/decision",
            json={"decision": "approve"},
            headers=admin2_h,
        )
        assert r.status_code == 200
        assert r.json()["status"] == "approved"


@pytest.mark.asyncio
async def test_health_endpoint(app_client):
    r = await app_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}
