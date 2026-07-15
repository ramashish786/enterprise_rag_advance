"""Pydantic schemas for the API layer.

Kept in one file per module (auth/chat/documents/...) and imported by the
respective router.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# --- Auth ---

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int  # seconds


class RefreshRequest(BaseModel):
    # Browser clients rely on the httpOnly refresh_token cookie instead; this
    # field exists for non-browser API clients that can't hold cookies.
    refresh_token: Optional[str] = None


class UserOut(BaseModel):
    id: UUID
    email: EmailStr
    full_name: str
    user_type: str
    roles: list[str] = []
    department: Optional[str] = None


# --- Chat ---

class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class UpdateSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ChatSessionOut(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class ChatMessageOut(BaseModel):
    id: UUID
    role: str
    content: str
    created_at: datetime
    metadata: dict = {}


# --- Documents ---

class DocumentOut(BaseModel):
    id: UUID
    filename: str
    status: str
    visibility: str
    version: int
    created_at: datetime


# --- Policies ---

class PolicyOut(BaseModel):
    id: UUID
    policy_number: str
    policy_type: str
    coverage_amount: float
    premium: float
    start_date: str
    end_date: str
    status: str
