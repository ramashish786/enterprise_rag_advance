"""
langgraph_integration.py
------------------------
Shows how the audit log integrates into LangGraph nodes.

Every agent decision point calls audit_log.write():
- Tool calls (before and after)
- LLM responses
- Retrieval events
- Access denied decisions
- Errors

The audit_log instance is passed through LangGraph state
so every node has access without global state.
"""

import hashlib
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from audit_log import AuditEvent, AuditLog, DecisionType, Outcome

logger = logging.getLogger(__name__)


def hash_user_id(user_id: str) -> str:
    """One-way hash — audit log never stores raw user ID."""
    return hashlib.sha256(user_id.encode()).hexdigest()


# ------------------------------------------------------------------
# LangGraph node — Tool call with audit logging
# ------------------------------------------------------------------

async def tool_call_node(
    state: dict,
    audit_log: AuditLog
) -> dict:
    """
    Example LangGraph node that makes a tool call
    and writes the decision to the audit log.

    Every tool call is one audit entry:
    - What tool was called
    - What inputs (whitelisted — no PII)
    - What outcome
    - How long it took
    """
    start = time.monotonic()

    # --- Execute tool ---
    tool_name = "get_claim_status"

    # Whitelisted inputs — no national ID, no diagnosis
    tool_input = {
        "policy_number": state["policy_number"],
        "claim_id":      state.get("claim_id"),
        # NOT including: national_id, bank_account
    }

    try:
        # Tool execution
        result = await get_claim_status(
            policy_number=state["policy_number"],
            user_id=state["user_id"]
        )

        latency_ms = int((time.monotonic() - start) * 1000)

        # --- Write success to audit log ---
        await audit_log.write(AuditEvent(
            user_token    = hash_user_id(state["user_id"]),
            session_id    = state["session_id"],
            region        = state["region"],
            role          = state["role"],
            decision_type = DecisionType.TOOL_CALL,
            outcome       = Outcome.SUCCESS,
            tool_name     = tool_name,
            tool_input    = tool_input,
            tool_output_summary = f"claim status: {result['status']}",
            latency_ms    = latency_ms,
        ))

        return {**state, "claim_result": result}

    except PermissionError as e:
        latency_ms = int((time.monotonic() - start) * 1000)

        # --- Write access denial to audit log ---
        await audit_log.write(AuditEvent(
            user_token    = hash_user_id(state["user_id"]),
            session_id    = state["session_id"],
            region        = state["region"],
            role          = state["role"],
            decision_type = DecisionType.ACCESS_DENIED,
            outcome       = Outcome.DENIED,
            tool_name     = tool_name,
            tool_input    = tool_input,
            error_code    = "ACCESS_DENIED",
            latency_ms    = latency_ms,
        ))

        raise

    except Exception as e:
        latency_ms = int((time.monotonic() - start) * 1000)

        # --- Write failure to audit log ---
        await audit_log.write(AuditEvent(
            user_token    = hash_user_id(state["user_id"]),
            session_id    = state["session_id"],
            region        = state["region"],
            role          = state["role"],
            decision_type = DecisionType.ERROR,
            outcome       = Outcome.FAILURE,
            tool_name     = tool_name,
            tool_input    = tool_input,
            error_code    = type(e).__name__,
            latency_ms    = latency_ms,
        ))

        raise


# ------------------------------------------------------------------
# LangGraph node — LLM decision with audit logging
# ------------------------------------------------------------------

async def llm_decision_node(
    state: dict,
    audit_log: AuditLog,
    llm: Any
) -> dict:
    """
    Example LangGraph node that makes an LLM call
    and writes the decision to the audit log.

    LLM decisions are audited because:
    - Regulators need to know what the model decided
    - Which model version made the decision
    - How many tokens were consumed
    - How long it took
    """
    start = time.monotonic()

    try:
        response = await llm.ainvoke(state["messages"])
        latency_ms = int((time.monotonic() - start) * 1000)

        # Extract token usage from response metadata
        usage = response.response_metadata.get("usage", {})

        # --- Write LLM decision to audit log ---
        await audit_log.write(AuditEvent(
            user_token        = hash_user_id(state["user_id"]),
            session_id        = state["session_id"],
            region            = state["region"],
            role              = state["role"],
            decision_type     = DecisionType.LLM_RESPONSE,
            outcome           = Outcome.SUCCESS,
            llm_model         = "anthropic.claude-3-haiku-20240307-v1:0",
            prompt_tokens     = usage.get("input_tokens"),
            completion_tokens = usage.get("output_tokens"),
            latency_ms        = latency_ms,
            # Note: actual response content NOT logged
            # Only that a decision was made, by which model,
            # with what token count — content stays in session
        ))

        return {
            **state,
            "messages": state["messages"] + [response]
        }

    except Exception as e:
        latency_ms = int((time.monotonic() - start) * 1000)

        await audit_log.write(AuditEvent(
            user_token    = hash_user_id(state["user_id"]),
            session_id    = state["session_id"],
            region        = state["region"],
            role          = state["role"],
            decision_type = DecisionType.LLM_RESPONSE,
            outcome       = Outcome.FAILURE,
            error_code    = type(e).__name__,
            latency_ms    = latency_ms,
        ))

        raise


# ------------------------------------------------------------------
# Stub — represents actual tool implementation
# ------------------------------------------------------------------

async def get_claim_status(
    policy_number: str,
    user_id: str
) -> dict:
    """Stub — real implementation queries Postgres with RLS."""
    return {
        "claim_id":      "CLM-789012",
        "status":        "In Review",
        "claim_type":    "Hospitalisation",
        "submitted_date": "2024-03-01",
        "next_action":   "Awaiting discharge summary"
    }
