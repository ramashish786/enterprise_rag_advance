"""
db.py
-----
Postgres connection pool and PII audit log writer.

Schema (run once on DB setup):

    CREATE TABLE pii_audit_log (
        id              SERIAL PRIMARY KEY,
        event_id        UUID NOT NULL UNIQUE,
        session_id      TEXT NOT NULL,
        user_token      TEXT NOT NULL,        -- hashed user ID, never raw
        query_hash      TEXT NOT NULL,        -- SHA-256 of original query
        entities_found  TEXT[] NOT NULL,      -- e.g. {UK_NIN, EMAIL_ADDRESS}
        entity_count    INT NOT NULL,
        action_taken    TEXT NOT NULL,        -- always REDACTED
        detected_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
        region          TEXT,                 -- user's region e.g. UK, FR, DE
        source          TEXT DEFAULT 'insurance_assistant'
    );

    -- Index for compliance queries: "all events for user X in date range"
    CREATE INDEX idx_pii_audit_user_token
        ON pii_audit_log(user_token, detected_at DESC);

    -- Index for ops queries: "all events in session"
    CREATE INDEX idx_pii_audit_session
        ON pii_audit_log(session_id, detected_at DESC);
"""

import asyncpg
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Connection pool (initialised once at application startup)
# ------------------------------------------------------------------

_pool: asyncpg.Pool | None = None


async def init_db_pool(dsn: str) -> None:
    """
    Call once at FastAPI startup.
    dsn example: postgresql://user:pass@host:5432/dbname
    """
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=2,
        max_size=10,
        command_timeout=10
    )
    logger.info("Postgres connection pool initialised")


async def close_db_pool() -> None:
    """Call at FastAPI shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        logger.info("Postgres connection pool closed")


@asynccontextmanager
async def get_connection():
    """Acquire a connection from the pool."""
    if not _pool:
        raise RuntimeError(
            "DB pool not initialised. Call init_db_pool() at startup."
        )
    async with _pool.acquire() as conn:
        yield conn


# ------------------------------------------------------------------
# DDL — run once to create audit table
# ------------------------------------------------------------------

CREATE_AUDIT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pii_audit_log (
    id              SERIAL PRIMARY KEY,
    event_id        UUID NOT NULL UNIQUE,
    session_id      TEXT NOT NULL,
    user_token      TEXT NOT NULL,
    query_hash      TEXT NOT NULL,
    entities_found  TEXT[] NOT NULL,
    entity_count    INT NOT NULL,
    action_taken    TEXT NOT NULL DEFAULT 'REDACTED',
    detected_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    region          TEXT,
    source          TEXT DEFAULT 'insurance_assistant'
);

CREATE INDEX IF NOT EXISTS idx_pii_audit_user_token
    ON pii_audit_log(user_token, detected_at DESC);

CREATE INDEX IF NOT EXISTS idx_pii_audit_session
    ON pii_audit_log(session_id, detected_at DESC);
"""


async def ensure_schema() -> None:
    """
    Idempotent — safe to call on every startup.
    Creates table and indexes if they don't exist.
    """
    async with get_connection() as conn:
        await conn.execute(CREATE_AUDIT_TABLE_SQL)
    logger.info("PII audit schema verified")


# ------------------------------------------------------------------
# Audit log writer
# ------------------------------------------------------------------

async def log_pii_event(
    event_id: str,
    session_id: str,
    user_token: str,
    query_hash: str,
    entities_found: list[str],
    region: str | None = None,
) -> None:
    """
    Write one PII detection event to Postgres.

    Parameters
    ----------
    event_id      : UUID string — unique per scrub call
    session_id    : user's session identifier
    user_token    : SHA-256 hash of user_id — NEVER raw user ID
    query_hash    : SHA-256 hash of original query — NEVER raw text
    entities_found: list of entity type strings detected
    region        : user's region code e.g. "UK", "FR", "DE"

    Notes
    -----
    - Raw user IDs and raw query text are NEVER stored.
    - Hashes allow compliance teams to correlate events
      without re-exposing the original sensitive data.
    - Fire-and-forget pattern used in production — audit
      failure must never block the user's query response.
    """
    INSERT_SQL = """
        INSERT INTO pii_audit_log (
            event_id,
            session_id,
            user_token,
            query_hash,
            entities_found,
            entity_count,
            action_taken,
            region
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (event_id) DO NOTHING;
    """
    try:
        async with get_connection() as conn:
            await conn.execute(
                INSERT_SQL,
                event_id,
                session_id,
                user_token,
                query_hash,
                entities_found,
                len(entities_found),
                "REDACTED",
                region,
            )
        logger.debug(
            "PII audit logged",
            extra={
                "event_id": event_id,
                "entity_count": len(entities_found),
                "region": region
            }
        )
    except Exception as exc:
        # Audit failure must NEVER block the user response.
        # Log the failure and continue — compliance team
        # can reconcile from structured logs.
        logger.error(
            "PII audit log write failed",
            extra={
                "event_id": event_id,
                "error": str(exc)
            }
        )
