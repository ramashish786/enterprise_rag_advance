"""Password hashing and JWT signing.

Uses argon2-cffi directly (OWASP-recommended for new deployments) and PyJWT
directly. The previous implementation went through passlib and python-jose;
both are unmaintained, so we cut them out.

Argon2id parameters follow OWASP's minimum recommendation as of 2024:
    time_cost=3, memory_cost=64 MB, parallelism=4, hash_len=32, salt_len=16.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import jwt
from argon2 import PasswordHasher
from argon2 import exceptions as argon2_exceptions

from app.core.config import settings
from app.core.exceptions import InvalidTokenError

_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=64 * 1024,  # 64 MiB
    parallelism=4,
    hash_len=32,
    salt_len=16,
)


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _hasher.verify(hashed, plain)
    except (
        argon2_exceptions.VerifyMismatchError,
        argon2_exceptions.VerificationError,
        argon2_exceptions.InvalidHashError,
    ):
        return False
    except Exception:
        # Any other malformed-hash case: fail closed as a mismatch, not a 500.
        return False


def needs_rehash(hashed: str) -> bool:
    """Called after a successful verify: if the stored hash was produced with
    weaker parameters than the current policy, rehash and persist. In prod
    this hooks the login flow so credential-strength upgrades happen silently.
    """
    try:
        return _hasher.check_needs_rehash(hashed)
    except Exception:
        return False


# --- JWT ---

def _base_claims(subject: str, ttl: timedelta, token_type: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
        "jti": str(uuid4()),
        "type": token_type,
    }


def create_access_token(subject: str, extra_claims: dict[str, Any] | None = None) -> str:
    claims = _base_claims(
        subject, timedelta(minutes=settings.access_token_ttl_minutes), "access"
    )
    if extra_claims:
        claims.update(extra_claims)
    return jwt.encode(claims, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(subject: str) -> tuple[str, str]:
    """Returns (token, jti) so the jti can be tracked for revocation."""
    claims = _base_claims(
        subject, timedelta(days=settings.refresh_token_ttl_days), "refresh"
    )
    token = jwt.encode(claims, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token, claims["jti"]


def decode_token(token: str, expected_type: str | None = None) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError as exc:
        raise InvalidTokenError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise InvalidTokenError(str(exc)) from exc

    if expected_type and payload.get("type") != expected_type:
        raise InvalidTokenError(
            f"Expected token type {expected_type!r}, got {payload.get('type')!r}"
        )
    return payload
