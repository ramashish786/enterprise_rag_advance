"""Output-side guardrails.

Two checks run on assistant outputs. Grounding: for retrieval-augmented
answers, every factual sentence needs support from at least one retrieved
chunk. Production uses an NLI model; here it's a lightweight overlap
heuristic that catches the obvious ungrounded cases and scores each
sentence's confidence. Citation enforcement: factual sentences must
reference a source (chunk id or policy id), or the response gets rejected
and the agent is asked to regenerate with citations.

`OutputValidator.validate(answer, sources)` runs both and returns a
`ValidationResult`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


CITATION_RE = re.compile(r"\[(?:source|src|chunk|policy):\s*([A-Za-z0-9_\-:]+)\]", re.IGNORECASE)


@dataclass
class SourceChunk:
    id: str
    content: str


@dataclass
class SentenceCheck:
    text: str
    is_factual: bool
    is_supported: bool
    cited_source_ids: list[str] = field(default_factory=list)
    support_score: float = 0.0


@dataclass
class ValidationResult:
    ok: bool
    sentences: list[SentenceCheck]
    reason: str = ""


# --- Utilities ---

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

_FACTUAL_HINTS = (
    "is", "are", "was", "were", "has", "have", "had",
    "covers", "includes", "excludes", "requires", "provides",
    "amounts", "costs", "premium", "deductible", "policy",
    "on ", "in ", "as of", "%", "$",
)

_HEDGE_HINTS = (
    "i can help", "sure,", "certainly", "of course", "please", "let me know",
    "would you like", "here's how", "here is how", "the steps",
)


def split_sentences(text: str) -> list[str]:
    sents = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sents if s.strip()]


def looks_factual(sentence: str) -> bool:
    low = sentence.lower()
    if any(h in low for h in _HEDGE_HINTS):
        return False
    # Sentences with numbers, currency, or "policy"-ish tokens are almost
    # certainly factual claims.
    if re.search(r"[$%]\s*\d|\bpolicy\b|\bcoverage\b|\bpremium\b|\bclaim\b", low):
        return True
    return any(w in low.split() for w in _FACTUAL_HINTS)


def _keyword_overlap(a: str, b: str) -> float:
    """Quick-and-dirty support score in [0, 1]."""
    tokens_a = {w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", a)}
    tokens_b = {w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", b)}
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


class OutputValidator:
    def __init__(
        self,
        require_citations: bool = True,
        support_threshold: float = 0.25,
    ):
        self.require_citations = require_citations
        self.support_threshold = support_threshold

    def validate(self, answer: str, sources: list[SourceChunk]) -> ValidationResult:
        sents = split_sentences(answer)
        source_by_id = {s.id: s for s in sources}
        checks: list[SentenceCheck] = []

        for sent in sents:
            factual = looks_factual(sent)
            cited = CITATION_RE.findall(sent)
            best_score = 0.0
            if cited:
                for cid in cited:
                    chunk = source_by_id.get(cid)
                    if chunk:
                        best_score = max(best_score, _keyword_overlap(sent, chunk.content))
            else:
                # no explicit citation, so check against all sources
                for src in sources:
                    best_score = max(best_score, _keyword_overlap(sent, src.content))

            supported = best_score >= self.support_threshold
            checks.append(SentenceCheck(
                text=sent, is_factual=factual, is_supported=supported,
                cited_source_ids=cited, support_score=best_score,
            ))

        if not sources:
            # a general-reasoning answer with no retrieval isn't subject to
            # the citation rule at all; the agent skips this validator for those
            return ValidationResult(ok=True, sentences=checks)

        problems = []
        for c in checks:
            if c.is_factual:
                if self.require_citations and not c.cited_source_ids:
                    problems.append(f"missing citation: {c.text[:80]}")
                elif not c.is_supported:
                    problems.append(f"unsupported by sources: {c.text[:80]}")

        return ValidationResult(
            ok=not problems,
            sentences=checks,
            reason="; ".join(problems[:3]),
        )


_default_validator: OutputValidator | None = None


def default_validator() -> OutputValidator:
    global _default_validator
    if _default_validator is None:
        _default_validator = OutputValidator()
    return _default_validator
