"""Prompt-injection and jailbreak detection.

Baseline is rule-based (fast, deterministic, no model). In production this is
augmented with a small classifier (Prompt Guard 86M or equivalent) whose
score is combined with the rule signals. Threshold is tunable via settings.

Categories we look for: instruction_override ("ignore previous/above/all
instructions"), role_hijack ("you are now DAN / developer mode / jailbroken"),
system_leak ("reveal your system prompt"), encoding_smuggle (base64/hex
payloads with an obvious wrapper), and classic SQL-injection indicators.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class InjectionCategory(str, Enum):
    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_HIJACK = "role_hijack"
    SYSTEM_LEAK = "system_leak"
    ENCODING_SMUGGLE = "encoding_smuggle"
    SQL_INJECTION = "sql_injection"
    TOOL_ABUSE = "tool_abuse"


@dataclass
class InjectionSignal:
    category: InjectionCategory
    matched_text: str
    confidence: float


_INSTRUCTION_OVERRIDE = [
    r"ignore (?:all |the |your )?(?:previous|above|prior|earlier) instructions?",
    r"disregard (?:all |the |your )?(?:previous|above|prior) instructions?",
    r"forget (?:all |the |your )?(?:previous|above|prior) instructions?",
    r"new instructions?[:\s]",
    r"instead,? (?:do|answer|say|respond)",
    r"override (?:your |the |all )?(?:system|safety|previous|above)",
]

_ROLE_HIJACK = [
    r"\bDAN\b(?:\s+mode)?",
    r"\bdeveloper mode\b",
    r"\bjailbroken?\b",
    r"\byou are now\b",
    r"\bact as (?:if you (?:are|were)|a jailbroken)",
    r"\bpretend (?:to be|you are) (?:an? )?(?:jailbroken|unrestricted|uncensored)",
    r"\bignore your (?:safety )?(?:rules|guidelines|policies|training)",
]

_SYSTEM_LEAK = [
    r"reveal (?:your |the )?(?:system|initial|original) (?:prompt|instructions?)",
    r"(?:show|tell|give) me (?:your |the )?(?:system|initial|original) (?:prompt|instructions?)",
    r"what (?:are|were) your (?:original|initial) instructions?",
    r"repeat (?:everything|all) (?:above|before)",
    r"print (?:everything|all) (?:above|before)",
]

_ENCODING_SMUGGLE = [
    r"base64[:\s]+[A-Za-z0-9+/=]{40,}",
    r"decode this[:\s]",
    r"execute (?:this )?(?:code|script)[:\s]",
]

_SQL_INJECTION = [
    r"(?:'|\")\s*(?:or|and)\s+['\"]?\d",
    r"';\s*(?:drop|delete|update|insert|alter|truncate)",
    r"union\s+select",
    r"--\s*$",
    r"/\*.*?\*/",
    r"xp_cmdshell",
]

_TOOL_ABUSE = [
    r"(?:call|invoke|execute) tool[:\s]",
    r"send (?:a |the )?http request",
    r"read (?:the |a )?(?:file|env|environment)",
]


def _match(patterns: list[str], text: str) -> tuple[bool, str]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return True, m.group(0)
    return False, ""


class PromptInjectionDetector:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def scan(self, text: str) -> list[InjectionSignal]:
        signals: list[InjectionSignal] = []
        checks = [
            (_INSTRUCTION_OVERRIDE, InjectionCategory.INSTRUCTION_OVERRIDE, 0.85),
            (_ROLE_HIJACK,          InjectionCategory.ROLE_HIJACK,          0.9),
            (_SYSTEM_LEAK,          InjectionCategory.SYSTEM_LEAK,          0.9),
            (_ENCODING_SMUGGLE,     InjectionCategory.ENCODING_SMUGGLE,     0.7),
            (_SQL_INJECTION,        InjectionCategory.SQL_INJECTION,        0.95),
            (_TOOL_ABUSE,           InjectionCategory.TOOL_ABUSE,           0.75),
        ]
        for patterns, category, conf in checks:
            hit, matched = _match(patterns, text)
            if hit:
                signals.append(InjectionSignal(
                    category=category, matched_text=matched, confidence=conf,
                ))
        return signals

    def is_injection(self, text: str) -> tuple[bool, list[InjectionSignal]]:
        signals = self.scan(text)
        max_conf = max((s.confidence for s in signals), default=0.0)
        return max_conf >= self.threshold, signals


_default_detector: PromptInjectionDetector | None = None


def default_detector() -> PromptInjectionDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = PromptInjectionDetector()
    return _default_detector
