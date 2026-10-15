-- ============================================================
-- Hash-chained audit log for Insurance Assistant
-- ============================================================
-- Every agent decision is written here.
-- Each row contains the SHA-256 hash of the previous row
-- (previous_hash), making the chain tamper-evident.
--
-- Tamper detection:
-- If any row is modified or deleted after the fact,
-- the hash chain breaks — detectable by the verify function.
--
-- Compliance use:
-- IRDAI requires every agent decision to be reconstructable.
-- This table is the authoritative record.
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_audit_log (
    -- Primary key
    id                  BIGSERIAL PRIMARY KEY,

    -- Chain fields
    event_id            UUID NOT NULL UNIQUE,
    previous_hash       TEXT NOT NULL,
    -- SHA-256 of the previous row's current_hash
    -- First row uses '0' * 64 as genesis sentinel

    current_hash        TEXT NOT NULL UNIQUE,
    -- SHA-256 of this row's canonical content
    -- Computed at insert time — never updated

    -- Who
    user_token          TEXT NOT NULL,
    -- SHA-256 of user_id — never raw user ID
    session_id          TEXT NOT NULL,
    region              TEXT NOT NULL,
    role                TEXT NOT NULL,

    -- What the agent decided
    decision_type       TEXT NOT NULL,
    -- e.g. TOOL_CALL, LLM_RESPONSE, RETRIEVAL,
    --      CLARIFICATION, ACCESS_DENIED, ERROR

    tool_name           TEXT,
    -- NULL if decision_type is not TOOL_CALL

    tool_input          JSONB,
    -- Whitelisted tool inputs — no raw PII
    -- e.g. {"policy_number": "POL-00000234"}
    -- NOT {"national_id": "AB 12 34 56 C"}

    tool_output_summary TEXT,
    -- Brief outcome — not full raw output
    -- e.g. "claim status: In Review"

    llm_model           TEXT,
    -- Which Bedrock model made this decision
    -- e.g. "anthropic.claude-3-haiku-20240307-v1:0"

    prompt_tokens       INT,
    completion_tokens   INT,

    latency_ms          INT,
    -- How long this decision took

    outcome             TEXT NOT NULL,
    -- SUCCESS, FAILURE, PARTIAL, DENIED

    error_code          TEXT,
    -- NULL on success

    -- When
    decided_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Immutability enforcement
    -- Row must never be updated after insert
    CONSTRAINT audit_log_no_update
        CHECK (decided_at IS NOT NULL)
);

-- ============================================================
-- Indexes for compliance queries
-- ============================================================

-- "All decisions in session X" — incident reconstruction
CREATE INDEX IF NOT EXISTS idx_audit_session
    ON agent_audit_log(session_id, decided_at ASC);

-- "All decisions by user token in date range" — user audit
CREATE INDEX IF NOT EXISTS idx_audit_user_token
    ON agent_audit_log(user_token, decided_at ASC);

-- "All decisions of type X" — pattern analysis
CREATE INDEX IF NOT EXISTS idx_audit_decision_type
    ON agent_audit_log(decision_type, decided_at DESC);

-- Chain verification — walk chain by id order
CREATE INDEX IF NOT EXISTS idx_audit_chain
    ON agent_audit_log(id ASC);

-- ============================================================
-- Revoke UPDATE and DELETE — audit log is append-only
-- Run as superuser after table creation
-- ============================================================
-- REVOKE UPDATE, DELETE ON agent_audit_log FROM app_user;
-- Only INSERT and SELECT allowed for application role

-- ============================================================
-- Postgres trigger — prevent UPDATE and DELETE at DB level
-- ============================================================

CREATE OR REPLACE FUNCTION prevent_audit_modification()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION
        'agent_audit_log is immutable — % not permitted',
        TG_OP;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_log_immutable
    BEFORE UPDATE OR DELETE ON agent_audit_log
    FOR EACH ROW
    EXECUTE FUNCTION prevent_audit_modification();
