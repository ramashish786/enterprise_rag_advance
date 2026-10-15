"""LangGraph state machine.

A turn flows through input_guard (PII redaction + prompt-injection check),
router (pick tools), then either the read-only path (execute_tools) or,
for a mutating tool (file_claim, request_policy_cancellation,
request_policy_change), a prepare/approve/apply sequence gated by
interrupt(). Either path rejoins at synthesize (turn results into an
answer) and output_guard (grounding + citation check).

Every node reads from and writes to the AgentState TypedDict below — nodes
return partial-update dicts, merged into the running state by the graph
runtime. The compiled graph is bound to a checkpointer (Postgres in
production, in-memory in tests; see checkpointer.py) that snapshots state
after each node transition, keyed by thread_id (the chat session id). If a
worker dies mid-turn, the next request for that thread_id resumes from the
last completed node instead of starting over — at-least-once execution per
node, which is why file_claim's insert is guarded by a deterministic
idempotency key rather than assumed to run exactly once (see tools.py).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, TypedDict
from uuid import UUID, uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.checkpointer import get_checkpointer
from app.agent.llm import BaseLLM, get_llm
from app.agent.router import Router, RoutingDecision
from app.agent.tools import (
    MUTATING_TOOL_REGISTRY,
    MUTATING_TOOLS,
    get_claims_for_policy,
    get_my_policies,
    get_policy_by_number,
    recall_long_term_memory,
    search_knowledge_base,
)
from app.audit.service import AuditLogService
from app.auth.service import SecurityPrincipal, load_principal
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


class AgentState(TypedDict, total=False):
    query: str
    # Only the id, never the SecurityPrincipal object: nodes that need the
    # full principal reload it fresh via load_principal(session, ...), so
    # permissions are re-checked against live DB state on every resume —
    # relevant because the approver on a resumed turn is a different
    # principal than the one who started it.
    principal_id: str
    session_id: Optional[str]
    prior_summary: str
    prior_messages: list[dict]

    redacted_query: str
    # RoutingDecision/SourceChunk are stored as plain dicts, not dataclass
    # instances — the checkpoint serializer warns on (and may someday
    # reject) unregistered types, so nodes convert at the boundary.
    routing: Optional[dict]
    tool_outputs: dict[str, Any]
    sources: list[dict]
    answer: str
    citations: list[str]
    blocked_reason: Optional[str]
    metadata: dict[str, Any]

    # Human-in-the-loop / mutating-action fields.
    pending_action: Optional[dict]
    approval: Optional[dict]
    mutating_result: Optional[dict]


@dataclass
class AgentResult:
    """External-facing shape callers (chat_routes.py, tests) use — isolates
    the dict-vs-dataclass churn of the real graph state to this module."""
    query: str
    session_id: Optional[UUID] = None
    redacted_query: str = ""
    routing: Optional[RoutingDecision] = None
    tool_outputs: dict[str, Any] = field(default_factory=dict)
    sources: list[SourceChunk] = field(default_factory=list)
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    interrupted: bool = False
    pending_action: Optional[dict] = None


def _to_result(session_id: Optional[UUID], raw: dict) -> AgentResult:
    routing_dict = raw.get("routing")
    interrupted = "__interrupt__" in raw
    return AgentResult(
        query=raw.get("query", ""),
        session_id=session_id,
        redacted_query=raw.get("redacted_query", ""),
        routing=RoutingDecision(**routing_dict) if routing_dict else None,
        tool_outputs=raw.get("tool_outputs", {}),
        sources=[SourceChunk(**s) for s in raw.get("sources", [])],
        answer=raw.get("answer", ""),
        citations=raw.get("citations", []),
        blocked_reason=raw.get("blocked_reason"),
        metadata=raw.get("metadata", {}),
        interrupted=interrupted,
        pending_action=raw.get("pending_action") if interrupted else None,
    )


# --- Individual nodes ---

async def node_input_guard(state: AgentState, config: RunnableConfig) -> dict:
    guardrails: InputGuardrails = config["configurable"]["input_guardrails"]
    report = guardrails.check_and_raise(state["query"], user_id=state["principal_id"])
    metadata = {
        **state.get("metadata", {}),
        "input_guard": {
            "pii_matches": [
                {"category": m.category, "confidence": m.confidence}
                for m in report.pii_matches
            ],
        },
    }
    return {"redacted_query": report.redacted_text, "metadata": metadata}


_REQUIRED_ARGS = {
    "file_claim": ("policy_number", "amount", "description"),
    "request_policy_cancellation": ("policy_number", "reason"),
    "request_policy_change": ("policy_number", "new_coverage_amount", "reason"),
}


def _missing_args(action_type: str, args: dict) -> list[str]:
    return [f for f in _REQUIRED_ARGS.get(action_type, ()) if not args.get(f)]


async def node_router(state: AgentState, config: RunnableConfig) -> dict:
    router: Router = config["configurable"]["router"]
    # Both the LLM and keyword routers only ever see redacted_query, so any
    # policy number in it already reads as "[REDACTED:POLICY_NUMBER]" (see
    # guardrails/pii.py) — neither router can extract a real one from that.
    # Patch it back in from the original, unredacted query before validating
    # required args; this doesn't defeat the redaction, since the LLM/logs
    # still only ever see the redacted version, only the actual mutation
    # (applied later, against a DB row the user's RBAC already scopes) gets
    # the real identifier.
    decision = await router.route(state["redacted_query"], state.get("prior_summary", ""))
    real_policy_number = _extract_policy_number(state["query"])
    if real_policy_number and "policy_number" in decision.tool_args:
        decision.tool_args["policy_number"] = real_policy_number

    # The LLM router doesn't always follow instructions perfectly — it can
    # set clarify *and* pick a mutating tool, or pick one without all the
    # args it needs. Rather than let an incomplete mutating call reach
    # file_claim/etc. and blow up on a missing required argument, downgrade
    # to a clarifying question here so synthesize can ask for what's missing.
    mutating = next((t for t in decision.tools if t in MUTATING_TOOLS), None)
    if mutating and not decision.clarify:
        missing = _missing_args(mutating, decision.tool_args)
        if missing:
            decision.clarify = (
                f"I can help with that, but I still need: {', '.join(missing)}. "
                "Could you provide those details?"
            )
            decision.tools = []

    logger.info("routing_decision", tools=decision.tools, clarify=bool(decision.clarify))
    return {"routing": asdict(decision)}


def route_after_router(state: AgentState) -> str:
    routing = state.get("routing") or {}
    if routing.get("clarify"):
        # node_execute_tools no-ops when clarify is set; synthesize renders it.
        return "execute_tools"
    tools = routing.get("tools") or []
    if any(t in MUTATING_TOOLS for t in tools):
        return "prepare_mutating_action"
    return "execute_tools"


async def node_execute_tools(state: AgentState, config: RunnableConfig) -> dict:
    routing_dict = state.get("routing")
    if not routing_dict or routing_dict.get("clarify"):
        return {}
    session: AsyncSession = config["configurable"]["session"]
    principal = await load_principal(session, UUID(state["principal_id"]))
    routing = RoutingDecision(**routing_dict)
    query = routing.query_rewrite or state["redacted_query"]

    async def _run(name: str):
        fn = TOOL_REGISTRY[name]
        try:
            if name in ("search_knowledge_base", "recall_long_term_memory"):
                return name, await fn(principal, session, query)
            elif name == "get_my_policies":
                return name, await fn(principal, session)
            elif name == "get_policy_by_number":
                pol_num = _extract_policy_number(query)
                return name, (await fn(principal, session, pol_num) if pol_num else None)
            elif name == "get_claims_for_policy":
                pol_num = _extract_policy_number(query)
                return name, ((await fn(principal, session, pol_num)) if pol_num else [])
        except Exception as exc:
            logger.exception("tool_error", tool=name)
            return name, {"error": str(exc)}

    # Bound parallelism at 3 concurrent tool calls.
    sem = asyncio.Semaphore(3)

    async def _bounded(name: str):
        async with sem:
            return await _run(name)

    results = await asyncio.gather(*[_bounded(n) for n in routing.tools])
    tool_outputs = dict(results)

    # Build sources list from tool outputs for the output guard.
    sources: list[dict] = []
    for name, output in tool_outputs.items():
        if name == "search_knowledge_base" and isinstance(output, list):
            for chunk in output:
                sources.append({"id": chunk["id"], "content": chunk["content"]})
        elif name in ("get_my_policies", "get_policy_by_number", "get_claims_for_policy"):
            items = output if isinstance(output, list) else ([output] if output else [])
            for item in items:
                if isinstance(item, dict) and "id" in item:
                    sources.append({
                        "id": f"policy:{item.get('policy_number', item['id'])}",
                        "content": json.dumps(item, default=str),
                    })
    return {"tool_outputs": tool_outputs, "sources": sources}


# --- Mutating-action path: prepare -> (approval_gate) -> apply ---

def _compute_idempotency_key(thread_id: str, action_type: str, args: dict) -> str:
    """Hashes only checkpointed state (thread id, action type, tool args) —
    no timestamp/nonce — so a node re-executed after a crash-and-resume
    recomputes the exact same key rather than a fresh one."""
    payload = json.dumps(
        {"thread_id": thread_id, "action_type": action_type, "args": args},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _summarize_action(action_type: str, args: dict) -> str:
    if action_type == "file_claim":
        return (
            f"File a ${args.get('amount', 0):,.2f} claim on "
            f"{args.get('policy_number', '?')}: {args.get('description', '')}"
        )
    if action_type == "request_policy_cancellation":
        return f"Cancel policy {args.get('policy_number', '?')}: {args.get('reason', '')}"
    if action_type == "request_policy_change":
        return (
            f"Change coverage on {args.get('policy_number', '?')} to "
            f"${args.get('new_coverage_amount', 0):,.2f}: {args.get('reason', '')}"
        )
    return f"{action_type}: {args}"


async def node_prepare_mutating_action(state: AgentState, config: RunnableConfig) -> dict:
    routing_dict = state["routing"]
    action_type = next(t for t in routing_dict["tools"] if t in MUTATING_TOOLS)
    # node_router already patched tool_args["policy_number"] back to the
    # real value (see there for why), so no fixup needed here.
    args = dict(routing_dict.get("tool_args") or {})

    thread_id = state.get("session_id") or ""
    pending = {
        "type": action_type,
        "args": args,
        "summary": _summarize_action(action_type, args),
        "idempotency_key": _compute_idempotency_key(thread_id, action_type, args),
        "requested_by": state["principal_id"],
    }
    return {"pending_action": pending}


def route_needs_approval(state: AgentState) -> str:
    pending = state["pending_action"]
    if pending["type"] == "file_claim":
        amount = pending["args"].get("amount") or 0
        if amount <= settings.claim_auto_approve_threshold:
            return "apply_mutating_action"
    return "approval_gate"


async def node_approval_gate(state: AgentState) -> dict:
    # This node does nothing but interrupt() — on resume, the whole node
    # body re-executes from the top and interrupt() short-circuits straight
    # to the resume value. Anything placed *before* the interrupt() call
    # here would need to be idempotent too; keep it that way.
    decision = interrupt(state["pending_action"])
    return {"approval": decision}


async def node_apply_mutating_action(state: AgentState, config: RunnableConfig) -> dict:
    pending = state["pending_action"]
    approval = state.get("approval")

    if approval is not None and approval.get("decision") != "approve":
        return {"mutating_result": {
            "status": "rejected",
            "action_type": pending["type"],
            "detail": {"note": approval.get("note")},
        }}

    session: AsyncSession = config["configurable"]["session"]
    # Apply as the original requester, not the approver — the mutation is
    # "customer X files a claim," and the tool's own RBAC gate expects that
    # identity. Reloading fresh (rather than trusting a stale principal)
    # re-validates the requester is still active at execution time.
    principal = await load_principal(session, UUID(pending["requested_by"]))
    fn = MUTATING_TOOL_REGISTRY[pending["type"]]
    result = await fn(principal, session, idempotency_key=pending["idempotency_key"], **pending["args"])
    return {"mutating_result": {"status": "applied", "action_type": pending["type"], "detail": result}}


def _render_mutating_answer(mutating_result: dict) -> str:
    detail = mutating_result.get("detail") or {}
    if mutating_result["status"] == "rejected":
        note = detail.get("note")
        return "That request was rejected" + (f": {note}." if note else ".")
    if "error" in detail:
        return f"I couldn't complete that: {detail['error']}"

    action_type = mutating_result["action_type"]
    if action_type == "file_claim":
        return (
            f"Claim {detail.get('claim_number', '')} filed for "
            f"${detail.get('amount', 0):,.2f} [source: claim:{detail.get('id', '')}]."
        )
    if action_type == "request_policy_cancellation":
        pn = detail.get("policy_number", "")
        return f"Policy {pn} has been cancelled [source: policy:{pn}]."
    if action_type == "request_policy_change":
        pn = detail.get("policy_number", "")
        return (
            f"Policy {pn} coverage updated to ${detail.get('coverage_amount', 0):,.2f} "
            f"[source: policy:{pn}]."
        )
    return "Done."


async def node_synthesize(state: AgentState, config: RunnableConfig) -> dict:
    routing_dict = state.get("routing") or {}
    if routing_dict.get("clarify"):
        return {"answer": routing_dict["clarify"]}

    mutating_result = state.get("mutating_result")
    if mutating_result is not None:
        return {"answer": _render_mutating_answer(mutating_result)}

    llm: BaseLLM = config["configurable"]["llm"]
    sources = [SourceChunk(**s) for s in state.get("sources", [])]

    system = (
        "You are an insurance assistant. Answer the user's question using ONLY the "
        "provided tool outputs and knowledge-base excerpts. Every factual claim must "
        "include an inline citation in the form [source: ID] where ID is the id of a "
        "provided source. If you cannot answer with the provided sources, say so."
    )

    sources_block = "\n\n".join(
        f"[source: {s.id}] {s.content[:800]}" for s in sources[:10]
    ) or "(no sources returned by tools)"

    tool_block = json.dumps({
        name: (output if not isinstance(output, list) else output[:5])
        for name, output in state.get("tool_outputs", {}).items()
    }, indent=2, default=str)[:4000]

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": (
            f"User question: {state.get('redacted_query', '')}\n\n"
            f"Tool outputs:\n{tool_block}\n\n"
            f"Sources:\n{sources_block}\n\n"
            "Answer concisely and cite sources inline as [source: ID]."
        )},
    ]
    answer = await llm.complete(messages, temperature=0.2)
    return {"answer": answer}


async def node_output_guard(state: AgentState, config: RunnableConfig) -> dict:
    guardrails: OutputGuardrails = config["configurable"]["output_guardrails"]
    sources = [SourceChunk(**s) for s in state.get("sources", [])]
    report = guardrails.check(state.get("answer", ""), sources)
    metadata = {
        **state.get("metadata", {}),
        "output_guard": {"ok": report.validation.ok, "reason": report.block_reason},
    }
    updates: dict[str, Any] = {"metadata": metadata}
    if report.blocked:
        # Rather than fail hard, we downgrade to a safe fallback answer.
        updates["answer"] = (
            "I don't have enough grounded information to answer that reliably. "
            "Could you rephrase or provide more context?"
        )
        updates["blocked_reason"] = f"output_guard:{report.block_reason}"
    return updates


# --- Graph construction ---

def _build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("input_guard", node_input_guard)
    g.add_node("router", node_router)
    g.add_node("execute_tools", node_execute_tools)
    g.add_node("prepare_mutating_action", node_prepare_mutating_action)
    g.add_node("approval_gate", node_approval_gate)
    g.add_node("apply_mutating_action", node_apply_mutating_action)
    g.add_node("synthesize", node_synthesize)
    g.add_node("output_guard", node_output_guard)

    g.add_edge(START, "input_guard")
    g.add_edge("input_guard", "router")
    g.add_conditional_edges("router", route_after_router, {
        "execute_tools": "execute_tools",
        "prepare_mutating_action": "prepare_mutating_action",
    })
    g.add_edge("execute_tools", "synthesize")
    g.add_conditional_edges("prepare_mutating_action", route_needs_approval, {
        "approval_gate": "approval_gate",
        "apply_mutating_action": "apply_mutating_action",
    })
    g.add_edge("approval_gate", "apply_mutating_action")
    g.add_edge("apply_mutating_action", "synthesize")
    g.add_edge("synthesize", "output_guard")
    g.add_edge("output_guard", END)
    return g


class InsuranceAgent:
    """Compiled LangGraph state machine for a chat turn, bound to a
    checkpointer keyed by thread_id (the chat session id)."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        llm: BaseLLM | None = None,
        router: Router | None = None,
        input_guardrails: InputGuardrails | None = None,
        output_guardrails: OutputGuardrails | None = None,
        checkpointer=None,
    ):
        self.session = session
        self.llm = llm or get_llm()
        self.router = router or Router(self.llm)
        self.input_guardrails = input_guardrails or InputGuardrails()
        self.output_guardrails = output_guardrails or OutputGuardrails()
        self.checkpointer = checkpointer or get_checkpointer()
        self._graph = _build_graph().compile(checkpointer=self.checkpointer)

    def _config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {
            "thread_id": thread_id,
            "session": self.session,
            "llm": self.llm,
            "router": self.router,
            "input_guardrails": self.input_guardrails,
            "output_guardrails": self.output_guardrails,
        }}

    async def has_pending_turn(self, session_id: UUID) -> bool:
        """True if this thread is paused at the approval_gate interrupt (or,
        in principle, any other unfinished node) — used to reject a new
        message rather than interleave it with an unresolved turn."""
        snap = await self._graph.aget_state(self._config(str(session_id)))
        return bool(snap.next)

    async def run(
        self,
        query: str,
        principal: SecurityPrincipal,
        *,
        session_id: UUID | None = None,
        prior_summary: str = "",
        prior_messages: list[dict] | None = None,
    ) -> AgentResult:
        thread_id = str(session_id) if session_id else str(uuid4())
        config = self._config(thread_id)
        audit = AuditLogService(self.session)

        initial: AgentState = {
            "query": query,
            "principal_id": str(principal.user_id),
            "session_id": str(session_id) if session_id else None,
            "prior_summary": prior_summary,
            "prior_messages": prior_messages or [],
            "metadata": {},
        }

        await audit.log(
            event_type="agent.turn_started",
            action="chat",
            outcome="success",
            user_id=principal.user_id,
            session_id=session_id,
            payload={"query_length": len(query)},
        )

        try:
            raw = await self._graph.ainvoke(initial, config)
        except GuardrailViolation as exc:
            await audit.log(
                event_type="agent.turn_blocked",
                action="chat",
                outcome="denied",
                user_id=principal.user_id,
                session_id=session_id,
                payload={"category": exc.category},
            )
            return AgentResult(
                query=query,
                session_id=session_id,
                answer="That request was blocked by a safety check.",
                blocked_reason=f"guardrail:{exc.category}",
            )

        result = _to_result(session_id, raw)

        if result.interrupted:
            await audit.log(
                event_type="agent.turn_paused_for_approval",
                action="chat",
                outcome="success",
                user_id=principal.user_id,
                session_id=session_id,
                payload={"pending_action": result.pending_action},
            )
        else:
            await audit.log(
                event_type="agent.turn_completed",
                action="chat",
                outcome="success" if not result.blocked_reason else "denied",
                user_id=principal.user_id,
                session_id=session_id,
                payload={
                    "tools_used": result.routing.tools if result.routing else [],
                    "source_count": len(result.sources),
                    "blocked_reason": result.blocked_reason,
                },
            )
        return result

    async def resume(
        self,
        session_id: UUID,
        *,
        decision: str,
        approver_id: UUID,
        note: str | None = None,
    ) -> AgentResult:
        """Resume a graph paused at the approval_gate interrupt(), e.g. from
        the approval decision endpoint."""
        config = self._config(str(session_id))
        raw = await self._graph.ainvoke(
            Command(resume={"decision": decision, "approver_id": str(approver_id), "note": note}),
            config,
        )
        return _to_result(session_id, raw)


def _extract_policy_number(text: str) -> str | None:
    import re
    m = re.search(r"\b((?:LIFE|HOME|VEHICLE)-\d{4,10})\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None
