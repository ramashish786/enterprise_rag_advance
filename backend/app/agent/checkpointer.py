"""Checkpointer factory: Postgres in production, in-memory for tests.

Mirrors the Fake/Real split used elsewhere in this codebase (llm.get_llm,
embedder.get_embedder, short_term.get_stm_store) and the lazy-singleton
pattern in db/session.py's _get_engine(). The checkpointer (and its
connection pool, for Postgres) is process-wide: a fresh InsuranceAgent is
built per request, but it must resolve to the same underlying storage so a
later request can resume a thread a different request started.
"""
from __future__ import annotations

from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.core.config import settings

_checkpointer: Optional[BaseCheckpointSaver] = None
_pool = None


def get_checkpointer() -> BaseCheckpointSaver:
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    if settings.is_test:
        from langgraph.checkpoint.memory import InMemorySaver
        _checkpointer = InMemorySaver()
        return _checkpointer

    global _pool
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    _pool = AsyncConnectionPool(
        conninfo=settings.checkpoint_database_url,
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    _checkpointer = AsyncPostgresSaver(conn=_pool)
    return _checkpointer


async def setup_checkpointer() -> None:
    """Open the pool and ensure the checkpoint schema exists. Call once at
    app startup. No-op for the in-memory test checkpointer."""
    checkpointer = get_checkpointer()
    if settings.is_test:
        return
    await _pool.open()
    await checkpointer.setup()


async def dispose_checkpointer() -> None:
    if _pool is not None:
        await _pool.close()
