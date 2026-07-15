"""Domain exceptions.

Keeping these as a small taxonomy makes API-layer error mapping trivial and
gives audit/logging code stable labels to key off (as opposed to grepping
strings).
"""
from __future__ import annotations


class InsuranceAgentError(Exception):
    """Base for all app-raised errors."""

    default_message = "An error occurred"
    http_status = 500

    def __init__(self, message: str | None = None, **context):
        super().__init__(message or self.default_message)
        self.context = context


# --- Auth / RBAC ---

class AuthenticationError(InsuranceAgentError):
    default_message = "Authentication failed"
    http_status = 401


class InvalidTokenError(AuthenticationError):
    default_message = "Invalid or expired token"


class PermissionDenied(InsuranceAgentError):
    default_message = "Permission denied"
    http_status = 403


# --- Guardrails ---

class GuardrailViolation(InsuranceAgentError):
    """Raised when input or output violates a guardrail.

    The `category` distinguishes prompt injection vs PII vs toxicity vs
    hallucination, so we can log and count them separately.
    """

    default_message = "Request blocked by guardrails"
    http_status = 400

    def __init__(self, category: str, message: str | None = None, **context):
        super().__init__(message, category=category, **context)
        self.category = category


# --- Rate / quota ---

class RateLimitExceeded(InsuranceAgentError):
    default_message = "Rate limit exceeded"
    http_status = 429


class QuotaExceeded(InsuranceAgentError):
    default_message = "Usage quota exceeded"
    http_status = 429


# --- Data ---

class NotFoundError(InsuranceAgentError):
    default_message = "Resource not found"
    http_status = 404


class ValidationError(InsuranceAgentError):
    default_message = "Validation failed"
    http_status = 422
