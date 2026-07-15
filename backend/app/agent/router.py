"""Router node.

The router classifies a user query into one or more tool calls. In tests we
use a keyword-based fallback that produces deterministic routing; in prod
the router is an LLM call with structured JSON output.

Route shape:
    {
      "tools": ["get_my_policies", "search_knowledge_base"],
      "reasoning": "user asked what policies they have and what life insurance covers",
      "query_rewrite": "list my policies and explain life insurance coverage",
      "clarify": null   # or a question if the query is ambiguous
    }
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.agent.llm import BaseLLM, get_llm
from app.agent.tools import TOOL_DESCRIPTIONS
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RoutingDecision:
    tools: list[str] = field(default_factory=list)
    query_rewrite: str = ""
    reasoning: str = ""
    clarify: str | None = None


_ROUTER_SYSTEM_PROMPT = """\
You route insurance-related user queries to one or more tools. Available tools:

{tool_catalog}

Rules:
- Return only strict JSON matching the schema.
- Never invent tools that aren't listed.
- If the user asks about "my" policies/claims: choose get_my_policies or get_policy_by_number.
- For general questions ("what does home insurance cover?"): use search_knowledge_base.
- Combined questions may use multiple tools.
- If the query is ambiguous (e.g., "tell me about it" with no prior context), set
  clarify to a short question and leave tools empty.
- Rewrite the query into a standalone question suitable for retrieval.

Schema: {{
  "tools":         ["tool_name", ...],
  "query_rewrite": "...",
  "reasoning":     "...",
  "clarify":       null | "question"
}}"""


def _tool_catalog() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items())


# --- Keyword fallback used in tests ---

_MY_HINTS = re.compile(r"\b(my|mine|i have|what.*do i have|list.*my|show me my)\b", re.IGNORECASE)
_POLICY_NUM_HINTS = re.compile(r"\b(?:LIFE|HOME|VEHICLE)-\d{4,10}\b", re.IGNORECASE)
_CLAIM_HINTS = re.compile(r"\b(claim|claims|filed|payout)\b", re.IGNORECASE)
_GENERAL_HINTS = re.compile(r"\b(what|how|when|why|explain|difference|cover(?:age)?)\b", re.IGNORECASE)
_MEMORY_HINTS = re.compile(r"\b(remember|last time|previously|my preferences?)\b", re.IGNORECASE)


def keyword_route(query: str) -> RoutingDecision:
    tools: list[str] = []
    if _POLICY_NUM_HINTS.search(query):
        tools.append("get_policy_by_number")
        if _CLAIM_HINTS.search(query):
            tools.append("get_claims_for_policy")
    elif _MY_HINTS.search(query):
        tools.append("get_my_policies")
    if _GENERAL_HINTS.search(query) and not _POLICY_NUM_HINTS.search(query):
        tools.append("search_knowledge_base")
    if _MEMORY_HINTS.search(query):
        tools.append("recall_long_term_memory")

    # Fall back: if we picked nothing, do a KB search.
    if not tools:
        tools.append("search_knowledge_base")

    return RoutingDecision(
        tools=list(dict.fromkeys(tools)),  # dedupe preserving order
        query_rewrite=query.strip(),
        reasoning="keyword-based routing (test path)",
    )


class Router:
    def __init__(self, llm: BaseLLM | None = None):
        self.llm = llm or get_llm()

    async def route(self, query: str, prior_summary: str = "") -> RoutingDecision:
        # Test / no-API path: keyword router.
        from app.core.config import settings
        if settings.is_test or not settings.openai_api_key:
            return keyword_route(query)

        system = _ROUTER_SYSTEM_PROMPT.format(tool_catalog=_tool_catalog())
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Prior session summary: {prior_summary or 'none'}\n\nQuery: {query}"},
        ]
        try:
            data = await self.llm.complete_json(
                messages,
                schema_hint='{"tools": [...], "query_rewrite": "...", "reasoning": "...", "clarify": null}',
                model=None,
            )
        except Exception:
            logger.exception("router_llm_failed_falling_back_to_keywords")
            return keyword_route(query)

        # Whitelist tools the LLM might hallucinate.
        allowed = set(TOOL_DESCRIPTIONS.keys())
        tools = [t for t in data.get("tools", []) if t in allowed]
        return RoutingDecision(
            tools=tools,
            query_rewrite=data.get("query_rewrite") or query,
            reasoning=data.get("reasoning") or "",
            clarify=data.get("clarify"),
        )
