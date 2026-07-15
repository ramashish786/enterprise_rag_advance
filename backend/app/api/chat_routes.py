"""Chat routes: session CRUD + streamed message endpoint."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import InsuranceAgent
from app.api.deps import _db_session, current_principal
from app.auth.service import SecurityPrincipal
from app.core.logging import get_logger
from app.db.models import ChatMessage, ChatSession
from app.memory.long_term import LongTermMemoryService, extract_candidates
from app.memory.short_term import (
    append_message,
    get_stm_store,
    new_state,
    summarize_if_needed,
)
from app.schemas.api import (
    ChatMessageOut,
    ChatSessionOut,
    CreateSessionRequest,
    SendMessageRequest,
    UpdateSessionRequest,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


# --- Session CRUD ---

@router.post("/sessions", response_model=ChatSessionOut)
async def create_session(
    payload: CreateSessionRequest,
    principal: SecurityPrincipal = Depends(current_principal),
    session: AsyncSession = Depends(_db_session),
):
    row = ChatSession(user_id=principal.user_id, title=payload.title or "New conversation")
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return ChatSessionOut(
        id=row.id, title=row.title,
        created_at=row.created_at, updated_at=row.updated_at,
    )


@router.get("/sessions", response_model=list[ChatSessionOut])
async def list_sessions(
    principal: SecurityPrincipal = Depends(current_principal),
    session: AsyncSession = Depends(_db_session),
):
    rows = (await session.execute(
        select(ChatSession)
        .where(ChatSession.user_id == principal.user_id, ChatSession.is_archived.is_(False))
        .order_by(ChatSession.updated_at.desc())
    )).scalars().all()
    return [
        ChatSessionOut(id=r.id, title=r.title, created_at=r.created_at, updated_at=r.updated_at)
        for r in rows
    ]


@router.patch("/sessions/{session_id}", response_model=ChatSessionOut)
async def rename_session(
    session_id: UUID,
    payload: UpdateSessionRequest,
    principal: SecurityPrincipal = Depends(current_principal),
    session: AsyncSession = Depends(_db_session),
):
    chat = (await session.execute(select(ChatSession).where(ChatSession.id == session_id))).scalar_one_or_none()
    if not chat or chat.user_id != principal.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="title cannot be blank")

    chat.title = title
    await session.commit()
    await session.refresh(chat)
    return ChatSessionOut(
        id=chat.id, title=chat.title,
        created_at=chat.created_at, updated_at=chat.updated_at,
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: UUID,
    principal: SecurityPrincipal = Depends(current_principal),
    session: AsyncSession = Depends(_db_session),
):
    chat = (await session.execute(select(ChatSession).where(ChatSession.id == session_id))).scalar_one_or_none()
    if not chat or chat.user_id != principal.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    # Soft delete: keeps message history for audit purposes, just drops it
    # from the user's session list (list_sessions already filters these out).
    chat.is_archived = True
    await session.commit()


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageOut])
async def list_messages(
    session_id: UUID,
    principal: SecurityPrincipal = Depends(current_principal),
    session: AsyncSession = Depends(_db_session),
):
    # Scope check: this session must belong to the principal (admin can peek).
    chat = (await session.execute(select(ChatSession).where(ChatSession.id == session_id))).scalar_one_or_none()
    if not chat or (chat.user_id != principal.user_id and principal.user_type != "administrator"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    msgs = (await session.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc())
    )).scalars().all()
    return [
        ChatMessageOut(
            id=m.id, role=m.role, content=m.content,
            created_at=m.created_at, metadata=m.metadata_,
        ) for m in msgs
    ]


# --- Streaming send message ---

@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: UUID,
    payload: SendMessageRequest,
    principal: SecurityPrincipal = Depends(current_principal),
    session: AsyncSession = Depends(_db_session),
):
    """Runs the agent, persists the exchange, streams SSE events.

    Events, in order: "routing" (tools + reasoning), "sources" (retrieved
    chunks), a run of "answer_chunk" (the final answer sliced up so it
    streams like a token-by-token reply), then "done" with the message id.
    """
    chat = (await session.execute(select(ChatSession).where(ChatSession.id == session_id))).scalar_one_or_none()
    if not chat or chat.user_id != principal.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    # Persist the user message immediately.
    user_msg = ChatMessage(
        session_id=session_id, role="user", content=payload.content,
    )
    session.add(user_msg)
    await session.commit()

    # auto-title the session from its first message
    is_first_message = (await session.execute(
        select(func.count()).select_from(ChatMessage).where(ChatMessage.session_id == session_id)
    )).scalar_one() == 1
    if is_first_message and chat.title == "New conversation":
        chat.title = _derive_title(payload.content)
        await session.commit()

    # Load short-term state
    stm = get_stm_store()
    state = await stm.get(str(session_id)) or new_state()

    async def _event_stream() -> AsyncIterator[dict]:
        agent = InsuranceAgent(session)
        result = await agent.run(
            payload.content,
            principal,
            session_id=session_id,
            prior_summary=state.get("summary", ""),
            prior_messages=state.get("messages", []),
        )

        yield {"event": "routing", "data": json.dumps({
            "tools": result.routing.tools if result.routing else [],
            "reasoning": result.routing.reasoning if result.routing else "",
        })}

        yield {"event": "sources", "data": json.dumps([
            {"id": s.id, "content": s.content[:400]} for s in result.sources[:5]
        ])}

        # Chunk the final answer into pseudo-tokens for streaming UX.
        chunk_size = 60
        for i in range(0, len(result.answer), chunk_size):
            yield {"event": "answer_chunk", "data": json.dumps({
                "delta": result.answer[i:i+chunk_size]
            })}
            await asyncio.sleep(0)

        # Persist the assistant message with provenance metadata.
        assistant_msg = ChatMessage(
            session_id=session_id, role="assistant", content=result.answer,
            metadata_={
                "tools_used": result.routing.tools if result.routing else [],
                "source_count": len(result.sources),
                "blocked_reason": result.blocked_reason,
            },
        )
        session.add(assistant_msg)
        await session.commit()

        # Update short-term memory
        state2 = append_message(state, "user", payload.content)
        state2 = append_message(
            state2, "assistant", result.answer,
            metadata={"tools_used": result.routing.tools if result.routing else []},
        )
        state2 = summarize_if_needed(state2)
        await stm.put(str(session_id), state2)

        # Extract long-term memory candidates from recent turns.
        try:
            svc = LongTermMemoryService(session)
            candidates = extract_candidates(state2["messages"][-4:])
            if candidates:
                await svc.persist_candidates(
                    principal.user_id, candidates, source_session_id=session_id
                )
        except Exception:
            logger.exception("ltm_extraction_failed")

        yield {"event": "done", "data": json.dumps({
            "message_id": str(assistant_msg.id),
            "blocked_reason": result.blocked_reason,
        })}

    return EventSourceResponse(_event_stream())


def _derive_title(text: str, max_len: int = 50) -> str:
    """Title a session from its first message. Could hand this off to an
    LLM-generated summary later; a truncated first line is fine for now."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_len:
        return collapsed
    truncated = collapsed[:max_len].rsplit(" ", 1)[0]
    return f"{truncated}…"
