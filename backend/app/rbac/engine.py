"""RBAC engine.

Enforcement happens at three levels: the API layer (`require_permission`
FastAPI dependency) gates HTTP endpoints; the tool layer (`authorize_tool`)
gates agent tools before they touch the DB or vector store, so even a
jailbroken agent has to pass this gate to call anything; and the retrieval
layer (visibility filters in the retriever) plus RLS at the DB level are
the last line of defense.
"""
from __future__ import annotations

from functools import wraps
from typing import Callable

from app.auth.service import SecurityPrincipal
from app.core.exceptions import PermissionDenied
from app.core.logging import get_logger

logger = get_logger(__name__)


# --- Assertion helpers ---

def assert_permission(
    principal: SecurityPrincipal, resource: str, action: str
) -> None:
    if not principal.has_permission(resource, action):
        logger.info(
            "rbac_denied",
            user_id=str(principal.user_id),
            user_type=principal.user_type,
            resource=resource,
            action=action,
        )
        raise PermissionDenied(
            f"missing permission {resource}:{action}",
            resource=resource,
            action=action,
        )


def assert_any_permission(
    principal: SecurityPrincipal, *pairs: tuple[str, str]
) -> None:
    if not principal.has_any(*pairs):
        logger.info(
            "rbac_denied_any",
            user_id=str(principal.user_id),
            user_type=principal.user_type,
            required_any=[f"{r}:{a}" for r, a in pairs],
        )
        raise PermissionDenied(
            "missing at least one of the required permissions",
            required_any=[f"{r}:{a}" for r, a in pairs],
        )


# --- Domain-specific policy checks (RBAC + scope) ---

def can_view_policy(principal: SecurityPrincipal, *, customer_id, policy_type) -> bool:
    """Check whether the principal can view a specific policy record.

    This is the *scope* check on top of the permission check. RBAC says
    'employee can read policies in their department'; the scope check makes
    sure the specific row matches.
    """
    if principal.has_permission("policy", "read_any"):
        return True

    if principal.user_type == "customer":
        return principal.has_permission("policy", "read_own") and (
            str(customer_id) == str(principal.user_id)
        )

    if principal.user_type == "employee":
        if not principal.has_permission("policy", "read_department"):
            return False
        # department name to policy type mapping is fixed in seed data
        dept_map = {
            "Life Insurance": "life",
            "Home Insurance": "home",
            "Vehicle Insurance": "vehicle",
        }
        expected = dept_map.get(principal.department_name or "")
        return expected == (
            policy_type.value if hasattr(policy_type, "value") else str(policy_type)
        )

    return False


def visible_document_visibilities(principal: SecurityPrincipal) -> set[str]:
    """Return the set of document visibility levels the principal may see.

    Used by the retriever to filter chunks *before* vector search: RBAC at
    retrieval time, not after.
    """
    tiers: set[str] = set()
    if principal.has_permission("document", "read_public"):
        tiers.add("public")
    if principal.has_permission("document", "read_internal"):
        tiers.add("internal")
    if principal.has_permission("document", "read_restricted"):
        tiers.add("restricted")
    return tiers


def visible_departments(principal: SecurityPrincipal) -> set[str] | None:
    """None means 'no department restriction'; a set means 'only these'."""
    if principal.user_type == "administrator":
        return None
    if principal.user_type == "employee" and principal.department_id:
        return {str(principal.department_id)}
    return set()  # customers see no department-scoped documents


# --- Tool-level authz decorator ---

def authorize_tool(*required: tuple[str, str]) -> Callable:
    """Wrap an agent tool coroutine so it checks permissions first.

    Usage:
        @authorize_tool(("policy", "read_own"))
        async def get_my_policies(principal, ...): ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(principal: SecurityPrincipal, *args, **kwargs):
            assert_any_permission(principal, *required)
            return await func(principal, *args, **kwargs)
        return wrapper
    return decorator
