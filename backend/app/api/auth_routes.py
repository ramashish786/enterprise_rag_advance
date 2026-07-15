"""Auth routes: login, refresh, me, logout."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import _db_session, current_principal
from app.audit.service import AuditLogService
from app.auth.service import (
    SecurityPrincipal,
    authenticate,
    issue_tokens,
    load_principal,
    revoke_refresh_token,
)
from app.core.config import settings
from app.core.exceptions import AuthenticationError, InvalidTokenError
from app.core.security import decode_token
from app.schemas.api import LoginRequest, RefreshRequest, TokenPair, UserOut

router = APIRouter(prefix="/api/auth", tags=["auth"])

REFRESH_COOKIE_NAME = "refresh_token"


def _set_refresh_cookie(response: Response, token: str) -> None:
    # httpOnly so JS can never read it (XSS-resistant); scoped to /api/auth
    # so it's only ever sent to the endpoints that need it. `secure` is
    # relaxed outside production since local dev runs over plain HTTP.
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=settings.app_env == "production",
        samesite="lax",
        path="/api/auth",
        max_age=settings.refresh_token_ttl_days * 24 * 3600,
    )


def _extract_refresh_token(payload: RefreshRequest, request: Request) -> str | None:
    return payload.refresh_token or request.cookies.get(REFRESH_COOKIE_NAME)


@router.post("/login", response_model=TokenPair)
async def login(payload: LoginRequest, response: Response, session: AsyncSession = Depends(_db_session)):
    try:
        user = await authenticate(session, str(payload.email), payload.password)
    except AuthenticationError as exc:
        # Log auth attempt (even failures) to the audit log.
        await AuditLogService(session).log(
            event_type="auth.login_failed",
            action="login",
            outcome="denied",
            payload={"email": str(payload.email)},
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    access, refresh, ttl = issue_tokens(user.id, user.user_type.value)
    _set_refresh_cookie(response, refresh)
    await AuditLogService(session).log(
        event_type="auth.login",
        action="login",
        outcome="success",
        user_id=user.id,
        payload={"email": user.email},
    )
    return TokenPair(access_token=access, refresh_token=refresh, expires_in=ttl)


@router.post("/refresh", response_model=TokenPair)
async def refresh(
    request: Request,
    response: Response,
    payload: RefreshRequest = RefreshRequest(),
    session: AsyncSession = Depends(_db_session),
):
    from uuid import UUID

    token = _extract_refresh_token(payload, request)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No refresh token provided")
    try:
        decoded = decode_token(token, expected_type="refresh")
    except InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    user_id = UUID(decoded["sub"])
    principal = await load_principal(session, user_id)

    revoke_refresh_token(token)  # rotate on use
    access, new_refresh, ttl = issue_tokens(user_id, principal.user_type)
    _set_refresh_cookie(response, new_refresh)
    return TokenPair(access_token=access, refresh_token=new_refresh, expires_in=ttl)


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    payload: RefreshRequest = RefreshRequest(),
    principal: SecurityPrincipal = Depends(current_principal),
    session: AsyncSession = Depends(_db_session),
):
    token = _extract_refresh_token(payload, request)
    try:
        if token:
            revoke_refresh_token(token)
    except InvalidTokenError:
        pass
    response.delete_cookie(key=REFRESH_COOKIE_NAME, path="/api/auth")
    await AuditLogService(session).log(
        event_type="auth.logout",
        action="logout",
        outcome="success",
        user_id=principal.user_id,
    )
    return {"ok": True}


@router.get("/me", response_model=UserOut)
async def me(principal: SecurityPrincipal = Depends(current_principal)):
    return UserOut(
        id=principal.user_id,
        email=principal.email,
        full_name=principal.full_name,
        user_type=principal.user_type,
        roles=principal.roles,
        department=principal.department_name,
    )
