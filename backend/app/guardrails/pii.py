"""PII detection and redaction.

Uses Presidio in production. For tests and when Presidio isn't available, we
fall back to a rule-based detector that catches the common categories plus
insurance-specific patterns (policy numbers, claim IDs, US SSNs).

The interface is intentionally narrow: `PIIDetector.scan(text)` returns a list
of `PIIMatch`, and `redact(text)` returns the text with matches replaced by
`[REDACTED:CATEGORY]` placeholders.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class PIIMatch:
    category: str
    start: int
    end: int
    text: str
    confidence: float


# --- Rule-based fallback ---

_PATTERNS = [
    # Order matters: put longer/more specific patterns first.
    ("SSN",           re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD",   re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("EMAIL",         re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE_US",      re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("POLICY_NUMBER", re.compile(r"\b(?:LIFE|HOME|VEHICLE)-\d{4,10}\b", re.IGNORECASE)),
    ("CLAIM_NUMBER",  re.compile(r"\bCLM-\d{4,10}\b", re.IGNORECASE)),
    ("DATE_OF_BIRTH", re.compile(r"\bDOB[:\s]+\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", re.IGNORECASE)),
    ("US_ZIP",        re.compile(r"\b\d{5}(?:-\d{4})?\b")),
]


class PIIDetector:
    """Presidio if available, rule-based fallback otherwise."""

    def __init__(self, use_presidio: bool = True):
        self._analyzer = None
        if use_presidio:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_analyzer.nlp_engine import NlpEngineProvider
                # spaCy model must be installed (`python -m spacy download en_core_web_sm`)
                provider = NlpEngineProvider(nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                })
                self._analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
            except Exception:
                self._analyzer = None

    def scan(self, text: str) -> List[PIIMatch]:
        if self._analyzer is not None:
            results = self._analyzer.analyze(text=text, language="en")
            matches = [
                PIIMatch(
                    category=r.entity_type,
                    start=r.start,
                    end=r.end,
                    text=text[r.start:r.end],
                    confidence=r.score,
                )
                for r in results
            ]
            # Also apply our insurance-specific rules.
            matches.extend(self._rule_based(text, categories={"POLICY_NUMBER", "CLAIM_NUMBER"}))
        else:
            matches = self._rule_based(text)
        return _dedupe(matches)

    @staticmethod
    def _rule_based(text: str, categories: set[str] | None = None) -> List[PIIMatch]:
        out: List[PIIMatch] = []
        for cat, pattern in _PATTERNS:
            if categories and cat not in categories:
                continue
            for m in pattern.finditer(text):
                out.append(PIIMatch(
                    category=cat, start=m.start(), end=m.end(),
                    text=m.group(0), confidence=0.85,
                ))
        return out

    def redact(self, text: str) -> tuple[str, List[PIIMatch]]:
        matches = self.scan(text)
        # Redact from the end so offsets stay valid.
        redacted = text
        for m in sorted(matches, key=lambda x: x.start, reverse=True):
            redacted = redacted[:m.start] + f"[REDACTED:{m.category}]" + redacted[m.end:]
        return redacted, matches


def _dedupe(matches: List[PIIMatch]) -> List[PIIMatch]:
    """Prefer higher-confidence match when spans overlap."""
    matches.sort(key=lambda m: (m.start, -m.confidence))
    result: List[PIIMatch] = []
    last_end = -1
    for m in matches:
        if m.start >= last_end:
            result.append(m)
            last_end = m.end
    return result


_default_detector: PIIDetector | None = None


def default_detector() -> PIIDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = PIIDetector(use_presidio=False)  # tests use fallback
    return _default_detector
