"""Tests for short-term and long-term memory."""
from __future__ import annotations

import pytest
from sqlalchemy import select

from app.db.models import LongTermMemory, MemoryType, User
from app.memory.long_term import LongTermMemoryService, extract_candidates
from app.memory.short_term import (
    InMemorySTMStore,
    append_message,
    new_state,
    summarize_if_needed,
)


# --- Short-term ---

@pytest.mark.asyncio
async def test_stm_roundtrip():
    store = InMemorySTMStore()
    state = new_state()
    state = append_message(state, "user", "Hi")
    state = append_message(state, "assistant", "Hello!")
    await store.put("thread-1", state)

    fetched = await store.get("thread-1")
    assert fetched is not None
    assert len(fetched["messages"]) == 2
    assert fetched["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_stm_delete():
    store = InMemorySTMStore()
    await store.put("t1", new_state())
    await store.delete("t1")
    assert await store.get("t1") is None


def test_summarization_rolls_old_messages():
    state = new_state()
    for i in range(25):
        state = append_message(state, "user", f"question {i}")
        state = append_message(state, "assistant", f"answer {i}")

    initial_count = len(state["messages"])
    assert initial_count == 50

    state = summarize_if_needed(state, max_messages=20, keep_recent=8)
    assert len(state["messages"]) == 8
    assert "Prior turns condensed" in state["summary"]


def test_summarization_noop_below_threshold():
    state = new_state()
    for i in range(3):
        state = append_message(state, "user", f"q{i}")

    state = summarize_if_needed(state, max_messages=20)
    assert state["summary"] == ""
    assert len(state["messages"]) == 3


# --- Long-term extraction ---

class TestLongTermExtraction:
    def test_extracts_preference_from_user_message(self):
        messages = [
            {"role": "user", "content": "I prefer email for notifications please."},
        ]
        cands = extract_candidates(messages)
        assert any(c.memory_type == MemoryType.PREFERENCE for c in cands)

    def test_does_not_extract_from_assistant_messages(self):
        messages = [
            {"role": "assistant", "content": "I prefer to help you with policies."},
        ]
        cands = extract_candidates(messages)
        assert cands == []

    def test_skips_messages_with_ssn(self):
        messages = [
            {"role": "user", "content": "My SSN is 123-45-6789 and I prefer phone contact."},
        ]
        cands = extract_candidates(messages)
        assert cands == []


# --- Long-term persistence + recall ---

@pytest.mark.asyncio
async def test_persist_and_recall(seeded_session):
    alice = (await seeded_session.execute(
        select(User).where(User.email == "alice@example.com")
    )).scalar_one()
    svc = LongTermMemoryService(seeded_session)

    from app.memory.long_term import MemoryCandidate
    await svc.persist_candidates(alice.id, [
        MemoryCandidate(MemoryType.PREFERENCE, "I prefer email notifications"),
        MemoryCandidate(MemoryType.FACT, "I live in Denver"),
    ])

    recalled = await svc.recall(alice.id, "notification preferences")
    assert any("email" in m.content.lower() for m in recalled)


@pytest.mark.asyncio
async def test_preference_conflict_deactivates_old(seeded_session):
    alice = (await seeded_session.execute(
        select(User).where(User.email == "alice@example.com")
    )).scalar_one()
    svc = LongTermMemoryService(seeded_session)

    from app.memory.long_term import MemoryCandidate
    await svc.persist_candidates(alice.id, [
        MemoryCandidate(MemoryType.PREFERENCE, "I prefer email"),
    ])
    await svc.persist_candidates(alice.id, [
        MemoryCandidate(MemoryType.PREFERENCE, "I prefer phone"),
    ])

    all_prefs = (await seeded_session.execute(
        select(LongTermMemory).where(
            LongTermMemory.user_id == alice.id,
            LongTermMemory.memory_type == MemoryType.PREFERENCE,
        )
    )).scalars().all()
    active = [m for m in all_prefs if m.is_active]
    inactive = [m for m in all_prefs if not m.is_active]
    assert len(active) == 1
    assert "phone" in active[0].content
    assert len(inactive) == 1
    assert "email" in inactive[0].content
