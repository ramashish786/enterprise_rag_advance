"""FastAPI dependencies for authentication and authorization.

Endpoints depend on `current_principal` to get a resolved SecurityPrincipal;
`require_permission("resource", "action")` gates endpoints declaratively.
"""
from __future__ import annotations

from typing import Callable
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.service import SecurityPrincipal, load_principal
from app.core.exceptions import (
    AuthenticationError,
    InsuranceAgentError,
    InvalidTokenError,
    PermissionDenied,
)
from app.core.security import decode_token
from app.db.session import get_session

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


async def _db_session():
    """Yields a plain RW session (RLS vars set later by dependency chain)."""
    async with get_session() as session:
        yield session


async def current_principal(
    token: str | None = Depends(oauth2_scheme),
    session: AsyncSession = Depends(_db_session),
) -> SecurityPrincipal:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_token(token, expected_type="access")
        user_id = UUID(payload["sub"])
        principal = await load_principal(session, user_id)
    except (InvalidTokenError, AuthenticationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal


def require_permission(resource: str, action: str) -> Callable:
    async def _dep(principal: SecurityPrincipal = Depends(current_principal)) -> SecurityPrincipal:
        if not principal.has_permission(resource, action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"missing permission {resource}:{action}",
            )
        return principal
    return _dep


def require_user_type(*allowed: str) -> Callable:
    async def _dep(principal: SecurityPrincipal = Depends(current_principal)) -> SecurityPrincipal:
        if principal.user_type not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="user type not permitted",
            )
        return principal
    return _dep


def domain_error_to_http(exc: InsuranceAgentError) -> HTTPException:
    return HTTPException(status_code=exc.http_status, detail=str(exc))
