"""Guardrail orchestrator.

`InputGuardrails.check(text, principal)` runs input-side checks in order,
returns a `GuardrailReport`, and (optionally) raises `GuardrailViolation`.
`OutputGuardrails.check(answer, sources)` runs output-side checks.

Callers choose whether to enforce (raise) or observe (log + count).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.core.config import settings
from app.core.exceptions import GuardrailViolation
from app.core.logging import get_logger
from app.guardrails.injection import (
    InjectionSignal,
    PromptInjectionDetector,
    default_detector as default_injection,
)
from app.guardrails.output import (
    OutputValidator,
    SourceChunk,
    ValidationResult,
    default_validator,
)
from app.guardrails.pii import PIIDetector, PIIMatch, default_detector as default_pii

logger = get_logger(__name__)


@dataclass
class InputReport:
    original_text: str
    redacted_text: str
    pii_matches: list[PIIMatch] = field(default_factory=list)
    injection_signals: list[InjectionSignal] = field(default_factory=list)
    blocked: bool = False
    block_reason: Optional[str] = None


@dataclass
class OutputReport:
    validation: ValidationResult
    blocked: bool = False
    block_reason: Optional[str] = None


class InputGuardrails:
    def __init__(
        self,
        pii: PIIDetector | None = None,
        injection: PromptInjectionDetector | None = None,
        pii_redact: Optional[bool] = None,
        block_on_injection: Optional[bool] = None,
    ):
        self.pii = pii or default_pii()
        self.injection = injection or default_injection()
        self.pii_redact = (
            pii_redact if pii_redact is not None else settings.enable_pii_redaction
        )
        self.block_on_injection = (
            block_on_injection
            if block_on_injection is not None
            else settings.enable_prompt_injection_check
        )

    def check(self, text: str, *, user_id: str | None = None) -> InputReport:
        report = InputReport(original_text=text, redacted_text=text)

        # prompt injection check runs first and can block outright
        is_injection, signals = self.injection.is_injection(text)
        report.injection_signals = signals
        if is_injection and self.block_on_injection:
            report.blocked = True
            report.block_reason = (
                "prompt_injection: "
                + ", ".join(s.category.value for s in signals[:3])
            )
            logger.warning(
                "guardrail_input_blocked",
                reason="prompt_injection",
                categories=[s.category.value for s in signals],
                user_id=user_id,
            )
            return report

        # PII redaction doesn't block; we redact for downstream use but the
        # original text still gets stored in the audit log
        if self.pii_redact:
            redacted, matches = self.pii.redact(text)
            report.pii_matches = matches
            report.redacted_text = redacted
            if matches:
                logger.info(
                    "guardrail_pii_redacted",
                    count=len(matches),
                    categories=list({m.category for m in matches}),
                    user_id=user_id,
                )
        return report

    def check_and_raise(self, text: str, *, user_id: str | None = None) -> InputReport:
        report = self.check(text, user_id=user_id)
        if report.blocked:
            raise GuardrailViolation(
                "prompt_injection",
                report.block_reason,
                user_id=user_id,
            )
        return report


class OutputGuardrails:
    def __init__(self, validator: OutputValidator | None = None):
        self.validator = validator or default_validator()
        self.enabled = settings.enable_output_validation

    def check(self, answer: str, sources: list[SourceChunk]) -> OutputReport:
        if not self.enabled:
            return OutputReport(validation=ValidationResult(ok=True, sentences=[]))

        validation = self.validator.validate(answer, sources)
        blocked = not validation.ok
        reason = validation.reason if blocked else None

        if blocked:
            logger.warning(
                "guardrail_output_blocked",
                reason=reason,
                factual_sentences=sum(1 for s in validation.sentences if s.is_factual),
                unsupported=sum(
                    1 for s in validation.sentences if s.is_factual and not s.is_supported
                ),
            )
        return OutputReport(validation=validation, blocked=blocked, block_reason=reason)
