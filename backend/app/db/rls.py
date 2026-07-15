"""Row-level security policies.

Applied after `alembic upgrade head` (either by a post-migration hook or a
management command). Postgres-only; skipped on SQLite.

Customers only see their own policies; employees see policies matching their
department; admins see everything. Claims inherit visibility from their
policy. Chat sessions/messages/long-term memories are owner-only, with an
admin override for review. Audit logs are admin-read-only and append-only
for everyone (no UPDATE/DELETE, enforced by grants).
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

RLS_POLICIES_SQL = """
-- Enable RLS on protected tables.
ALTER TABLE insurance_policies ENABLE ROW LEVEL SECURITY;
ALTER TABLE claims             ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions      ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages      ENABLE ROW LEVEL SECURITY;
ALTER TABLE long_term_memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs         ENABLE ROW LEVEL SECURITY;

-- Helper: current user id from session var, cast safely.
CREATE OR REPLACE FUNCTION current_app_user_id() RETURNS uuid AS $$
DECLARE v text;
BEGIN
    v := current_setting('app.current_user_id', true);
    IF v IS NULL OR v = '' THEN RETURN NULL; END IF;
    RETURN v::uuid;
END; $$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION current_app_user_type() RETURNS text AS $$
BEGIN
    RETURN current_setting('app.current_user_type', true);
END; $$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION current_app_department_id() RETURNS uuid AS $$
DECLARE v text;
BEGIN
    v := current_setting('app.current_department_id', true);
    IF v IS NULL OR v = '' THEN RETURN NULL; END IF;
    RETURN v::uuid;
END; $$ LANGUAGE plpgsql STABLE;

-- insurance_policies
DROP POLICY IF EXISTS policy_read ON insurance_policies;
CREATE POLICY policy_read ON insurance_policies FOR SELECT USING (
    current_app_user_type() = 'administrator'
    OR (current_app_user_type() = 'customer' AND customer_id = current_app_user_id())
    OR (
        current_app_user_type() = 'employee'
        AND policy_type::text = (
            SELECT d.policy_type::text FROM departments d
            WHERE d.id = current_app_department_id()
        )
    )
);

-- claims: inherit from the underlying policy
DROP POLICY IF EXISTS claim_read ON claims;
CREATE POLICY claim_read ON claims FOR SELECT USING (
    EXISTS (
        SELECT 1 FROM insurance_policies p
        WHERE p.id = claims.policy_id
        -- Reusing insurance_policies RLS by way of a subquery makes this
        -- automatically consistent.
    )
);

-- chat_sessions / chat_messages / long_term_memories: owner-only (admin override for review)
DROP POLICY IF EXISTS chat_session_owner ON chat_sessions;
CREATE POLICY chat_session_owner ON chat_sessions FOR ALL USING (
    current_app_user_type() = 'administrator'
    OR user_id = current_app_user_id()
);

DROP POLICY IF EXISTS chat_message_owner ON chat_messages;
CREATE POLICY chat_message_owner ON chat_messages FOR ALL USING (
    current_app_user_type() = 'administrator'
    OR EXISTS (
        SELECT 1 FROM chat_sessions s
        WHERE s.id = chat_messages.session_id AND s.user_id = current_app_user_id()
    )
);

DROP POLICY IF EXISTS long_term_memory_owner ON long_term_memories;
CREATE POLICY long_term_memory_owner ON long_term_memories FOR ALL USING (
    current_app_user_type() = 'administrator'
    OR user_id = current_app_user_id()
);

-- audit_logs: admin-only read; append-only enforced by grants
DROP POLICY IF EXISTS audit_admin_read ON audit_logs;
CREATE POLICY audit_admin_read ON audit_logs FOR SELECT USING (
    current_app_user_type() = 'administrator'
);
"""


READONLY_GRANTS_SQL = """
-- Read-only role: SELECT only, no INSERT/UPDATE/DELETE, no sequence usage.
GRANT USAGE ON SCHEMA public TO insurance_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO insurance_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO insurance_ro;

-- audit_logs must be append-only for the application role: no UPDATE/DELETE.
REVOKE UPDATE, DELETE ON audit_logs FROM PUBLIC;
"""


async def apply_rls(session: AsyncSession) -> None:
    """Apply RLS policies and read-only grants. Postgres only."""
    if not session.bind.dialect.name.startswith("postgres"):
        return
    for statement in RLS_POLICIES_SQL.split(";"):
        stmt = statement.strip()
        if stmt:
            await session.execute(text(stmt))
    for statement in READONLY_GRANTS_SQL.split(";"):
        stmt = statement.strip()
        if stmt:
            await session.execute(text(stmt))
    await session.commit()
