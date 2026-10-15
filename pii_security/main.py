"""
main.py
-------
FastAPI application wiring the PII scrubber into the
Insurance Assistant query endpoint.

Startup sequence
----------------
1. init_db_pool() — creates asyncpg connection pool
2. ensure_schema() — creates pii_audit_log table if absent
3. Request arrives → JWT decoded → UserContext built
4. scrub_query() → ScrubResult
5. Scrubbed query forwarded to LangGraph agent
"""

import logging
import os

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from db import close_db_pool, ensure_schema, init_db_pool
from scrubber import ScrubResult, UserContext, scrub_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Insurance Assistant — PII Scrubber")
security = HTTPBearer()

# ------------------------------------------------------------------
# Startup / shutdown
# ------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    dsn = os.environ["DATABASE_URL"]
    await init_db_pool(dsn)
    await ensure_schema()
    logger.info("Application started")


@app.on_event("shutdown")
async def shutdown():
    await close_db_pool()
    logger.info("Application shut down")


# ------------------------------------------------------------------
# JWT dependency — extracts UserContext from token
# ------------------------------------------------------------------

async def get_user_context(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UserContext:
    """
    Decode JWT and build UserContext.
    In production: validate signature, expiry, and claims.
    Here: simplified extraction for illustration.
    """
    token = credentials.credentials

    # Production: use python-jose or authlib to decode + verify
    # Simplified for illustration:
    try:
        import base64, json
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)  # pad
        payload = json.loads(base64.b64decode(payload_b64))

        return UserContext(
            user_id=payload["sub"],
            session_id=payload.get("session_id", "unknown"),
            region=payload.get("region", "UK"),
            role=payload.get("role", "policyholder"),
            policy_ids=payload.get("policy_ids", [])
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str

    class Config:
        json_schema_extra = {
            "example": {
                "query": (
                    "My NI number is AB 12 34 56 C. "
                    "Can you check my policy POL-00012345 coverage?"
                )
            }
        }


class QueryResponse(BaseModel):
    # What the agent processed — safe for logging
    scrubbed_query: str

    # Audit reference — returned so client can correlate if needed
    event_id: str
    pii_detected: bool
    entities_detected: list[str]

    # Placeholder — in production this would be
    # the LangGraph agent's actual response
    agent_response: str


# ------------------------------------------------------------------
# Query endpoint
# ------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def handle_query(
    request: QueryRequest,
    user_context: UserContext = Depends(get_user_context),
) -> QueryResponse:
    """
    Main entry point for user queries.

    1. Scrub PII from raw query
    2. Log detection event to Postgres
    3. Forward scrubbed query to LangGraph agent
    4. Return agent response
    """

    # Step 1 — Scrub
    result: ScrubResult = await scrub_query(
        raw_query=request.query,
        user_context=user_context,
    )

    # Step 2 — Forward scrubbed query to LangGraph agent
    # In production: invoke your LangGraph graph here
    # agent_response = await insurance_agent.invoke(
    #     query=result.scrubbed_query,
    #     user_context=user_context
    # )
    agent_response = f"[LangGraph agent would process: {result.scrubbed_query}]"

    return QueryResponse(
        scrubbed_query=result.scrubbed_query,
        event_id=result.event_id,
        pii_detected=result.pii_detected,
        entities_detected=result.entities_detected,
        agent_response=agent_response,
    )
