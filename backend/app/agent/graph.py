"""LangGraph state machine.

A turn flows through input_guard (PII redaction + prompt-injection check),
router (pick tools), execute_tools (bounded parallel calls), synthesize
(turn tool output into an answer), output_guard (grounding + citation
check), and finally a background extract_memory step. Each node takes the
state and returns partial updates.

LangGraph isn't available in the test env, so we also run the same nodes
as a plain imperative sequence there. Same logic either way, just no
graph runtime underneath.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.llm import BaseLLM, get_llm
from app.agent.router import Router, RoutingDecision
from app.agent.tools import (
    get_claims_for_policy,
    get_my_policies,
    get_policy_by_number,
    recall_long_term_memory,
    search_knowledge_base,
)
from app.audit.service import AuditLogService
from app.auth.service import SecurityPrincipal
from app.core.config import settings
from app.core.exceptions import GuardrailViolation
from app.core.logging import get_logger
from app.guardrails.orchestrator import InputGuardrails, OutputGuardrails
from app.guardrails.output import SourceChunk

logger = get_logger(__name__)


TOOL_REGISTRY = {
    "get_my_policies":         get_my_policies,
    "get_policy_by_number":    get_policy_by_number,
    "get_claims_for_policy":   get_claims_for_policy,
    "search_knowledge_base":   search_knowledge_base,
    "recall_long_term_memory": recall_long_term_memory,
}


@dataclass
class AgentState:
    query: str
    principal: Any  # SecurityPrincipal
    session_id: Optional[UUID] = None
    prior_summary: str = ""
    prior_messages: list[dict] = field(default_factory=list)

    # Populated by nodes
    redacted_query: str = ""
    routing: Optional[RoutingDecision] = None
    tool_outputs: dict[str, Any] = field(default_factory=dict)
    sources: list[SourceChunk] = field(default_factory=list)
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# --- Individual nodes ---

async def node_input_guard(state: AgentState, guardrails: InputGuardrails) -> AgentState:
    try:
        report = guardrails.check_and_raise(state.query, user_id=str(state.principal.user_id))
        state.redacted_query = report.redacted_text
        state.metadata["input_guard"] = {
            "pii_matches": [
                {"category": m.category, "confidence": m.confidence}
                for m in report.pii_matches
            ],
        }
    except GuardrailViolation as exc:
        state.blocked_reason = f"input_guard:{exc.category}"
        raise
    return state


async def node_router(state: AgentState, router: Router) -> AgentState:
    state.routing = await router.route(state.redacted_query, state.prior_summary)
    logger.info(
        "routing_decision",
        tools=state.routing.tools,
        clarify=bool(state.routing.clarify),
    )
    return state


async def node_execute_tools(state: AgentState, session: AsyncSession) -> AgentState:
    if not state.routing or state.routing.clarify:
        return state
    principal = state.principal
    query = state.routing.query_rewrite or state.redacted_query

    async def _run(name: str):
        fn = TOOL_REGISTRY[name]
        try:
            if name in ("search_knowledge_base", "recall_long_term_memory"):
                return name, await fn(principal, session, query)
            elif name == "get_my_policies":
                return name, await fn(principal, session)
            elif name == "get_policy_by_number":
                pol_num = _extract_policy_number(query)
                return name, await fn(principal, session, pol_num) if pol_num else (name, None)
            elif name == "get_claims_for_policy":
                pol_num = _extract_policy_number(query)
                return name, (await fn(principal, session, pol_num)) if pol_num else (name, [])
        except Exception as exc:
            logger.exception("tool_error", tool=name)
            return name, {"error": str(exc)}

    # Bound parallelism at 3 concurrent tool calls.
    sem = asyncio.Semaphore(3)

    async def _bounded(name: str):
        async with sem:
            return await _run(name)

    results = await asyncio.gather(*[_bounded(n) for n in state.routing.tools])
    state.tool_outputs = dict(results)

    # Build sources list from tool outputs for the output guard.
    for name, output in state.tool_outputs.items():
        if name == "search_knowledge_base" and isinstance(output, list):
            for chunk in output:
                state.sources.append(SourceChunk(id=chunk["id"], content=chunk["content"]))
        elif name in ("get_my_policies", "get_policy_by_number", "get_claims_for_policy"):
            items = output if isinstance(output, list) else ([output] if output else [])
            for item in items:
                if isinstance(item, dict) and "id" in item:
                    state.sources.append(SourceChunk(
                        id=f"policy:{item.get('policy_number', item['id'])}",
                        content=json.dumps(item, default=str),
                    ))
    return state


async def node_synthesize(state: AgentState, llm: BaseLLM) -> AgentState:
    if state.routing and state.routing.clarify:
        state.answer = state.routing.clarify
        return state

    system = (
        "You are an insurance assistant. Answer the user's question using ONLY the "
        "provided tool outputs and knowledge-base excerpts. Every factual claim must "
        "include an inline citation in the form [source: ID] where ID is the id of a "
        "provided source. If you cannot answer with the provided sources, say so."
    )

    sources_block = "\n\n".join(
        f"[source: {s.id}] {s.content[:800]}" for s in state.sources[:10]
    ) or "(no sources returned by tools)"

    tool_block = json.dumps({
        name: (output if not isinstance(output, list) else output[:5])
        for name, output in state.tool_outputs.items()
    }, indent=2, default=str)[:4000]

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": (
            f"User question: {state.redacted_query}\n\n"
            f"Tool outputs:\n{tool_block}\n\n"
            f"Sources:\n{sources_block}\n\n"
            "Answer concisely and cite sources inline as [source: ID]."
        )},
    ]
    state.answer = await llm.complete(messages, temperature=0.2)
    return state


async def node_output_guard(state: AgentState, guardrails: OutputGuardrails) -> AgentState:
    report = guardrails.check(state.answer, state.sources)
    state.metadata["output_guard"] = {
        "ok": report.validation.ok,
        "reason": report.block_reason,
    }
    if report.blocked:
        # Rather than fail hard, we downgrade to a safe fallback answer.
        state.answer = (
            "I don't have enough grounded information to answer that reliably. "
            "Could you rephrase or provide more context?"
        )
        state.blocked_reason = f"output_guard:{report.block_reason}"
    return state


# --- Orchestrator (LangGraph-free reference) ---

class InsuranceAgent:
    """Reference implementation of the agent flow.

    In production this is a StateGraph; here we run the same node sequence
    imperatively so tests exercise the identical logic.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        llm: BaseLLM | None = None,
        router: Router | None = None,
        input_guardrails: InputGuardrails | None = None,
        output_guardrails: OutputGuardrails | None = None,
    ):
        self.session = session
        self.llm = llm or get_llm()
        self.router = router or Router(self.llm)
        self.input_guardrails = input_guardrails or InputGuardrails()
        self.output_guardrails = output_guardrails or OutputGuardrails()

    async def run(
        self,
        query: str,
        principal: SecurityPrincipal,
        *,
        session_id: UUID | None = None,
        prior_summary: str = "",
        prior_messages: list[dict] | None = None,
    ) -> AgentState:
        state = AgentState(
            query=query,
            principal=principal,
            session_id=session_id,
            prior_summary=prior_summary,
            prior_messages=prior_messages or [],
        )
        audit = AuditLogService(self.session)

        try:
            await node_input_guard(state, self.input_guardrails)
            await audit.log(
                event_type="agent.turn_started",
                action="chat",
                outcome="success",
                user_id=principal.user_id,
                session_id=session_id,
                payload={"query_length": len(query)},
            )

            await node_router(state, self.router)
            await node_execute_tools(state, self.session)
            await node_synthesize(state, self.llm)
            await node_output_guard(state, self.output_guardrails)

            await audit.log(
                event_type="agent.turn_completed",
                action="chat",
                outcome="success" if not state.blocked_reason else "denied",
                user_id=principal.user_id,
                session_id=session_id,
                payload={
                    "tools_used": state.routing.tools if state.routing else [],
                    "source_count": len(state.sources),
                    "blocked_reason": state.blocked_reason,
                },
            )
        except GuardrailViolation as exc:
            state.blocked_reason = f"guardrail:{exc.category}"
            state.answer = "That request was blocked by a safety check."
            await audit.log(
                event_type="agent.turn_blocked",
                action="chat",
                outcome="denied",
                user_id=principal.user_id,
                session_id=session_id,
                payload={"category": exc.category},
            )
        return state


def _extract_policy_number(text: str) -> str | None:
    import re
    m = re.search(r"\b((?:LIFE|HOME|VEHICLE)-\d{4,10})\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None
