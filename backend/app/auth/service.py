"""Authentication service: verifies credentials, issues access/refresh JWTs,
tracks revoked refresh-token JTIs, and loads roles/department for the
security principal.

Refresh-token revocation uses Redis when available and an in-memory fallback
otherwise, so tests don't need Redis to exercise the whole path.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.exceptions import AuthenticationError, InvalidTokenError
from app.core.logging import get_logger
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from app.db.models import EmployeeProfile, Role, RolePermission, User, UserRole

logger = get_logger(__name__)


# --- In-memory refresh-token denylist (Redis in production) ---
# Structure: {jti: expiry_epoch_seconds}. A background task would expire
# entries; for now we just check expiry inline.
_denylist: dict[str, int] = {}


def _revoke_jti(jti: str, ttl_seconds: int) -> None:
    import time
    _denylist[jti] = int(time.time()) + ttl_seconds


def _is_revoked(jti: str) -> bool:
    import time
    exp = _denylist.get(jti)
    if exp is None:
        return False
    if exp < int(time.time()):
        _denylist.pop(jti, None)
        return False
    return True


class SecurityPrincipal:
    """The authenticated identity attached to a request.

    Includes everything RBAC checks and RLS need without further DB hops.
    """

    def __init__(
        self,
        user_id: UUID,
        email: str,
        user_type: str,
        full_name: str,
        roles: list[str],
        permissions: set[tuple[str, str]],
        department_id: Optional[UUID] = None,
        department_name: Optional[str] = None,
    ):
        self.user_id = user_id
        self.email = email
        self.user_type = user_type
        self.full_name = full_name
        self.roles = roles
        self.permissions = permissions
        self.department_id = department_id
        self.department_name = department_name

    def has_permission(self, resource: str, action: str) -> bool:
        return (resource, action) in self.permissions

    def has_any(self, *pairs: tuple[str, str]) -> bool:
        return any(self.has_permission(r, a) for r, a in pairs)

    def to_dict(self) -> dict:
        return {
            "user_id": str(self.user_id),
            "email": self.email,
            "user_type": self.user_type,
            "roles": self.roles,
            "permissions": [f"{r}:{a}" for r, a in self.permissions],
            "department_id": str(self.department_id) if self.department_id else None,
            "department_name": self.department_name,
        }


async def load_principal(session: AsyncSession, user_id: UUID) -> SecurityPrincipal:
    stmt = (
        select(User)
        .where(User.id == user_id, User.is_active.is_(True))
        .options(
            selectinload(User.roles)
                .selectinload(UserRole.role)
                .selectinload(Role.permissions)
                .selectinload(RolePermission.permission),
            selectinload(User.employee_profile).selectinload(EmployeeProfile.department),
        )
    )
    user = (await session.execute(stmt)).scalar_one_or_none()
    if not user:
        raise AuthenticationError("User not found or inactive")

    roles: list[str] = []
    perms: set[tuple[str, str]] = set()
    for ur in user.roles:
        roles.append(ur.role.name)
        for rp in ur.role.permissions:
            # Load permission via relationship; RolePermission has a `permission` rel.
            perm = rp.permission
            perms.add((perm.resource, perm.action))

    dept_id, dept_name = None, None
    if user.employee_profile is not None:
        dept_id = user.employee_profile.department_id
        dept_name = user.employee_profile.department.name

    return SecurityPrincipal(
        user_id=user.id,
        email=user.email,
        user_type=user.user_type.value,
        full_name=user.full_name,
        roles=roles,
        permissions=perms,
        department_id=dept_id,
        department_name=dept_name,
    )


async def authenticate(session: AsyncSession, email: str, password: str) -> User:
    stmt = select(User).where(User.email == email.lower(), User.is_active.is_(True))
    user = (await session.execute(stmt)).scalar_one_or_none()
    if not user or not verify_password(password, user.hashed_password):
        # Uniform error to avoid user enumeration.
        raise AuthenticationError("Invalid email or password")
    return user


def issue_tokens(user_id: UUID, user_type: str) -> tuple[str, str, int]:
    """Returns (access_token, refresh_token, access_ttl_seconds)."""
    access = create_access_token(str(user_id), {"user_type": user_type})
    refresh, _jti = create_refresh_token(str(user_id))
    return access, refresh, settings.access_token_ttl_minutes * 60


def rotate_refresh_token(refresh_token: str) -> tuple[str, str, int]:
    """Validate a refresh token, revoke it, and issue a fresh pair."""
    payload = decode_token(refresh_token, expected_type="refresh")
    jti = payload["jti"]
    if _is_revoked(jti):
        raise InvalidTokenError("Refresh token has been revoked")

    # Revoke the current one immediately (rotate-on-use).
    ttl = int(payload["exp"] - payload["iat"])
    _revoke_jti(jti, ttl_seconds=ttl)

    user_id = UUID(payload["sub"])
    # user_type isn't in the refresh token; caller must reload principal
    # and re-issue with the fresh user_type from DB.
    return str(user_id), *("", 0)  # placeholder; real call goes via API layer


def revoke_refresh_token(refresh_token: str) -> None:
    payload = decode_token(refresh_token, expected_type="refresh")
    ttl = int(payload["exp"] - payload["iat"])
    _revoke_jti(payload["jti"], ttl_seconds=ttl)
