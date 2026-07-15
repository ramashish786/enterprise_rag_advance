"""Async engine and session factories.

Two engines: the main RW engine for application endpoints, and a read-only
one for agent tools, running under a Postgres role with SELECT-only grants
so a jailbroken agent physically can't write anything.

Engines are built lazily so importing this module doesn't blow up just
because asyncpg isn't installed in a hermetic test environment.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings


_engine = None
_readonly_engine = None
_session_factory = None
_readonly_session_factory = None


def _get_engine():
    global _engine, _session_factory
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=10,
        )
        _session_factory = async_sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False
        )
    return _engine, _session_factory


def _get_readonly_engine():
    global _readonly_engine, _readonly_session_factory
    if _readonly_engine is None:
        _readonly_engine = create_async_engine(
            settings.readonly_database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
        )
        _readonly_session_factory = async_sessionmaker(
            _readonly_engine, class_=AsyncSession, expire_on_commit=False
        )
    return _readonly_engine, _readonly_session_factory


async def _set_rls_vars(
    session: AsyncSession,
    user_id: Optional[UUID],
    user_type: Optional[str],
    department_id: Optional[UUID],
) -> None:
    if not session.bind.dialect.name.startswith("postgres"):
        return
    await session.execute(
        text("SELECT set_config('app.current_user_id', :v, true)"),
        {"v": str(user_id) if user_id else ""},
    )
    await session.execute(
        text("SELECT set_config('app.current_user_type', :v, true)"),
        {"v": user_type or ""},
    )
    await session.execute(
        text("SELECT set_config('app.current_department_id', :v, true)"),
        {"v": str(department_id) if department_id else ""},
    )


@asynccontextmanager
async def get_session(
    user_id: Optional[UUID] = None,
    user_type: Optional[str] = None,
    department_id: Optional[UUID] = None,
) -> AsyncIterator[AsyncSession]:
    _, factory = _get_engine()
    async with factory() as session:
        await _set_rls_vars(session, user_id, user_type, department_id)
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_readonly_session(
    user_id: Optional[UUID] = None,
    user_type: Optional[str] = None,
    department_id: Optional[UUID] = None,
) -> AsyncIterator[AsyncSession]:
    _, factory = _get_readonly_engine()
    async with factory() as session:
        await _set_rls_vars(session, user_id, user_type, department_id)
        yield session


async def dispose_engines() -> None:
    global _engine, _readonly_engine
    if _engine is not None:
        await _engine.dispose()
    if _readonly_engine is not None:
        await _readonly_engine.dispose()
