"""Shared fixtures.

The default test path is fully hermetic:
    - SQLite in-memory database (no Postgres required)
    - No Redis (short-term memory uses in-memory fallback)
    - No LLM calls (agent/router use a deterministic FakeLLM)

Integration tests that require Postgres or Redis are marked with
`@pytest.mark.integration` and skipped unless the services are up.
"""
from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

import pytest
import pytest_asyncio

# Test env before any app imports
os.environ["APP_ENV"] = "test"
os.environ["JWT_SECRET_KEY"] = "test-secret-32-bytes-long-xxxxxxxxx"

import tempfile
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db import models  # noqa: F401  register tables


@pytest_asyncio.fixture
async def test_engine():
    """Fresh file-backed SQLite per test.

    File-backed rather than :memory:, so every session opened on the engine
    sees the same database. Plain :memory: gives each connection its own DB.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        try:
            yield engine
        finally:
            await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session


@pytest_asyncio.fixture
async def seeded_session(test_engine, monkeypatch) -> AsyncIterator[AsyncSession]:
    """Session with seed data applied.

    Patches app.db.session.get_session so `seed_all()` uses our test engine.
    """
    from contextlib import asynccontextmanager
    from app.db import session as session_module

    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    @asynccontextmanager
    async def _test_get_session(**_kwargs):
        async with factory() as s:
            try:
                yield s
            except Exception:
                await s.rollback()
                raise

    monkeypatch.setattr(session_module, "get_session", _test_get_session)
    # seed.py imports get_session by name, so we must patch it there too.
    from app.db import seed as seed_module
    monkeypatch.setattr(seed_module, "get_session", _test_get_session)

    from app.db.seed import seed_all
    await seed_all()

    async with factory() as session:
        yield session
