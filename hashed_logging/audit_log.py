"""
audit_log.py
------------
Hash-chained audit log for the Enterprise Insurance Assistant.

Every agent decision — tool call, LLM response, retrieval,
access denial, error — is written here as an immutable entry.

Hash chain mechanics
--------------------
Each row contains:
- previous_hash : current_hash of the immediately preceding row
- current_hash  : SHA-256 of this row's canonical content

To verify the chain:
    Row N's previous_hash must equal Row N-1's current_hash.
    Row N's current_hash must equal SHA-256(Row N's canonical content).

If any row was modified or deleted after insert:
    The chain breaks at that point — detectable immediately.

What is hashed (canonical content)
------------------------------------
A deterministic JSON string of the row's core fields:
    event_id, previous_hash, user_token, session_id,
    decision_type, tool_name, tool_input, outcome, decided_at

Fields excluded from hash:
    id (assigned by DB), current_hash (is the hash),
    latency_ms, token counts (operational metadata)

Genesis row
-----------
The very first row in the table has no predecessor.
previous_hash = '0' * 64 (64 zero characters).
This is the known genesis sentinel — verifier checks for it.
"""

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Decision types — every agent action maps to one
# ------------------------------------------------------------------

class DecisionType(str, Enum):
    TOOL_CALL       = "TOOL_CALL"
    LLM_RESPONSE    = "LLM_RESPONSE"
    RETRIEVAL       = "RETRIEVAL"
    CLARIFICATION   = "CLARIFICATION"
    ACCESS_DENIED   = "ACCESS_DENIED"
    PII_DETECTED    = "PII_DETECTED"
    ERROR           = "ERROR"


class Outcome(str, Enum):
    SUCCESS  = "SUCCESS"
    FAILURE  = "FAILURE"
    PARTIAL  = "PARTIAL"
    DENIED   = "DENIED"


# Genesis sentinel — previous_hash for the very first row
GENESIS_HASH = "0" * 64


# ------------------------------------------------------------------
# Data contract
# ------------------------------------------------------------------

@dataclass
class AuditEvent:
    """
    One agent decision to be written to the audit log.
    Caller constructs this — audit_log writes it.
    """
    # Who
    user_token:          str        # SHA-256 of user_id — never raw
    session_id:          str
    region:              str
    role:                str

    # What
    decision_type:       DecisionType
    outcome:             Outcome

    # Tool call fields — None if not a tool call
    tool_name:           str | None = None
    tool_input:          dict | None = None   # Whitelisted — no PII
    tool_output_summary: str | None = None

    # LLM fields
    llm_model:           str | None = None
    prompt_tokens:       int | None = None
    completion_tokens:   int | None = None

    # Timing
    latency_ms:          int | None = None

    # Error
    error_code:          str | None = None

    # Generated at write time — not by caller
    event_id:            str = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    decided_at:          datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class AuditEntry:
    """
    Written row — includes chain fields computed at write time.
    """
    event:          AuditEvent
    previous_hash:  str
    current_hash:   str


# ------------------------------------------------------------------
# Hash computation
# ------------------------------------------------------------------

def _canonical_content(
    event: AuditEvent,
    previous_hash: str
) -> str:
    """
    Deterministic JSON string of the fields that form the hash.
    Must be identical at write time and verify time.

    Rules:
    - Keys sorted alphabetically — dict ordering is unreliable
    - Datetime as ISO string — consistent format
    - None values included — omitting changes the hash
    - tool_input sorted if present — nested dict ordering
    """
    content = {
        "decided_at":    event.decided_at.isoformat(),
        "decision_type": event.decision_type.value,
        "event_id":      event.event_id,
        "outcome":       event.outcome.value,
        "previous_hash": previous_hash,
        "session_id":    event.session_id,
        "tool_input":    event.tool_input,
        "tool_name":     event.tool_name,
        "user_token":    event.user_token,
    }

    # sort_keys=True ensures determinism regardless
    # of Python dict insertion order
    return json.dumps(content, sort_keys=True)


def compute_hash(event: AuditEvent, previous_hash: str) -> str:
    """
    SHA-256 of the canonical content string.
    Returns 64-character hex digest.
    """
    canonical = _canonical_content(event, previous_hash)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------
# Audit log writer
# ------------------------------------------------------------------

class AuditLog:
    """
    Append-only hash-chained audit log writer.

    Usage
    -----
    audit_log = AuditLog(pool)
    await audit_log.write(event)

    Thread safety
    -------------
    _get_last_hash uses SELECT FOR UPDATE to prevent
    concurrent writes from racing on the previous_hash.
    Two concurrent writes cannot both read the same
    last hash and produce duplicate chain links.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def write(self, event: AuditEvent) -> AuditEntry:
        """
        Write one audit event to the chain.

        Steps:
        1. Acquire DB connection
        2. Lock last row (SELECT FOR UPDATE)
        3. Compute current_hash using last row's hash
        4. Insert new row
        5. Return AuditEntry with chain fields

        The SELECT FOR UPDATE in step 2 ensures that
        concurrent writes serialise correctly — each
        new row gets the correct previous_hash.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():

                # Step 1 — Get last hash (locked)
                previous_hash = await self._get_last_hash(conn)

                # Step 2 — Compute this row's hash
                current_hash = compute_hash(event, previous_hash)

                # Step 3 — Insert
                await conn.execute("""
                    INSERT INTO agent_audit_log (
                        event_id,
                        previous_hash,
                        current_hash,
                        user_token,
                        session_id,
                        region,
                        role,
                        decision_type,
                        tool_name,
                        tool_input,
                        tool_output_summary,
                        llm_model,
                        prompt_tokens,
                        completion_tokens,
                        latency_ms,
                        outcome,
                        error_code,
                        decided_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7,
                        $8, $9, $10, $11, $12, $13,
                        $14, $15, $16, $17, $18
                    )
                """,
                    event.event_id,
                    previous_hash,
                    current_hash,
                    event.user_token,
                    event.session_id,
                    event.region,
                    event.role,
                    event.decision_type.value,
                    event.tool_name,
                    json.dumps(event.tool_input) if event.tool_input else None,
                    event.tool_output_summary,
                    event.llm_model,
                    event.prompt_tokens,
                    event.completion_tokens,
                    event.latency_ms,
                    event.outcome.value,
                    event.error_code,
                    event.decided_at,
                )

                logger.debug(
                    "Audit entry written",
                    extra={
                        "event_id":      event.event_id,
                        "decision_type": event.decision_type,
                        "outcome":       event.outcome,
                        "current_hash":  current_hash[:16] + "...",
                    }
                )

                return AuditEntry(
                    event=event,
                    previous_hash=previous_hash,
                    current_hash=current_hash,
                )

    async def _get_last_hash(
        self,
        conn: asyncpg.Connection
    ) -> str:
        """
        Fetch current_hash of the most recent row.
        SELECT FOR UPDATE locks the row — prevents
        concurrent writes from reading the same
        previous_hash and breaking the chain.

        Returns GENESIS_HASH if table is empty.
        """
        row = await conn.fetchrow("""
            SELECT current_hash
            FROM agent_audit_log
            ORDER BY id DESC
            LIMIT 1
            FOR UPDATE
        """)

        return row["current_hash"] if row else GENESIS_HASH


# ------------------------------------------------------------------
# Chain verifier
# ------------------------------------------------------------------

class ChainVerificationResult:
    def __init__(self):
        self.total_rows:    int = 0
        self.valid_rows:    int = 0
        self.broken_at:     int | None = None  # Row ID where chain breaks
        self.break_reason:  str | None = None
        self.is_valid:      bool = True

    def __repr__(self):
        if self.is_valid:
            return f"ChainValid(rows={self.total_rows})"
        return (
            f"ChainBROKEN("
            f"broken_at_id={self.broken_at}, "
            f"reason={self.break_reason})"
        )


async def verify_chain(
    pool: asyncpg.Pool,
    from_id: int = 1,
    to_id:   int | None = None
) -> ChainVerificationResult:
    """
    Walk the audit chain and verify every link.

    For each row N:
    1. Recompute expected hash from row N's content
       + row N's stored previous_hash
    2. Compare to row N's stored current_hash
       → Mismatch = row N was modified after insert

    3. Verify row N's previous_hash equals
       row N-1's current_hash
       → Mismatch = row was deleted or inserted
         between N-1 and N

    Runs in a read-only transaction.
    Streams rows to avoid loading full table in memory.

    Returns ChainVerificationResult with break point
    if chain is invalid.
    """
    result = ChainVerificationResult()
    previous_hash = GENESIS_HASH

    async with pool.acquire() as conn:
        # Stream rows in chain order
        query = """
            SELECT
                id,
                event_id,
                previous_hash,
                current_hash,
                user_token,
                session_id,
                region,
                role,
                decision_type,
                tool_name,
                tool_input,
                outcome,
                decided_at
            FROM agent_audit_log
            WHERE id >= $1
            {}
            ORDER BY id ASC
        """.format("AND id <= $2" if to_id else "")

        params = [from_id] if not to_id else [from_id, to_id]

        async for row in conn.cursor(query, *params):
            result.total_rows += 1
            row_id = row["id"]

            # --- Check 1: previous_hash continuity ---
            if row["previous_hash"] != previous_hash:
                result.is_valid = False
                result.broken_at = row_id
                result.break_reason = (
                    f"previous_hash mismatch at id={row_id}. "
                    f"Expected={previous_hash[:16]}... "
                    f"Got={row['previous_hash'][:16]}... "
                    f"A row may have been inserted, deleted, "
                    f"or modified before this one."
                )
                logger.error(result.break_reason)
                return result

            # --- Check 2: current_hash integrity ---
            # Reconstruct the event to recompute hash
            reconstructed_event = AuditEvent(
                event_id      = str(row["event_id"]),
                user_token    = row["user_token"],
                session_id    = row["session_id"],
                region        = row["region"],
                role          = row["role"],
                decision_type = DecisionType(row["decision_type"]),
                outcome       = Outcome(row["outcome"]),
                tool_name     = row["tool_name"],
                tool_input    = (
                    json.loads(row["tool_input"])
                    if row["tool_input"] else None
                ),
                decided_at    = row["decided_at"],
            )

            expected_hash = compute_hash(
                reconstructed_event,
                row["previous_hash"]
            )

            if expected_hash != row["current_hash"]:
                result.is_valid = False
                result.broken_at = row_id
                result.break_reason = (
                    f"current_hash mismatch at id={row_id}. "
                    f"Stored hash does not match recomputed hash. "
                    f"Row content was modified after insert."
                )
                logger.error(result.break_reason)
                return result

            # Row is valid — advance chain
            previous_hash = row["current_hash"]
            result.valid_rows += 1

    return result
