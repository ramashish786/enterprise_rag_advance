"""
test_audit_log.py
-----------------
Tests the hash chain mechanics in isolation —
no live Postgres needed.

Tests:
1. Hash is deterministic — same input = same hash always
2. Chain links correctly — each row's previous_hash
   equals the prior row's current_hash
3. Tamper detection — modifying any field breaks the chain
4. Genesis row — first row uses GENESIS_HASH as previous_hash
5. Different events produce different hashes
"""

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from audit_log import (
    GENESIS_HASH,
    AuditEvent,
    AuditLog,
    ChainVerificationResult,
    DecisionType,
    Outcome,
    compute_hash,
    _canonical_content,
    verify_chain,
)


# ------------------------------------------------------------------
# Helper — build a test event
# ------------------------------------------------------------------

def make_event(
    decision_type: DecisionType = DecisionType.TOOL_CALL,
    outcome: Outcome = Outcome.SUCCESS,
    tool_name: str = "get_claim_status",
    session_id: str = "sess_001",
) -> AuditEvent:
    return AuditEvent(
        user_token    = "a" * 64,  # Simulated hash
        session_id    = session_id,
        region        = "UK",
        role          = "policyholder",
        decision_type = decision_type,
        outcome       = outcome,
        tool_name     = tool_name,
        tool_input    = {"policy_number": "POL-00000234"},
        tool_output_summary = "claim status: In Review",
        llm_model     = None,
        latency_ms    = 95,
        decided_at    = datetime(2024, 3, 15, 10, 23, 45, tzinfo=timezone.utc)
    )


# ------------------------------------------------------------------
# Test 1 — Hash is deterministic
# ------------------------------------------------------------------

def test_hash_deterministic():
    event = make_event()

    hash1 = compute_hash(event, GENESIS_HASH)
    hash2 = compute_hash(event, GENESIS_HASH)

    assert hash1 == hash2, "Same input must produce same hash"
    assert len(hash1) == 64, "SHA-256 must be 64 hex chars"
    print(f"  hash = {hash1[:32]}...")
    print("  PASS ✓ — hash is deterministic")


# ------------------------------------------------------------------
# Test 2 — Changing any field changes the hash
# ------------------------------------------------------------------

def test_field_change_changes_hash():
    event = make_event()
    original_hash = compute_hash(event, GENESIS_HASH)

    # Change outcome
    tampered = make_event(outcome=Outcome.FAILURE)
    tampered_hash = compute_hash(tampered, GENESIS_HASH)

    assert original_hash != tampered_hash, (
        "Changing outcome must change hash"
    )
    print("  PASS ✓ — field change changes hash")


# ------------------------------------------------------------------
# Test 3 — previous_hash change changes current_hash
# ------------------------------------------------------------------

def test_previous_hash_change_changes_current():
    event = make_event()

    hash_with_genesis = compute_hash(event, GENESIS_HASH)
    hash_with_other   = compute_hash(event, "b" * 64)

    assert hash_with_genesis != hash_with_other, (
        "Different previous_hash must produce different current_hash"
    )
    print("  PASS ✓ — previous_hash is part of the hash")


# ------------------------------------------------------------------
# Test 4 — Chain links correctly across multiple events
# ------------------------------------------------------------------

def test_chain_links():
    """
    Simulate writing 3 events and verify the chain.
    No DB — pure in-memory chain simulation.
    """
    events = [
        make_event(decision_type=DecisionType.TOOL_CALL,
                   tool_name="get_claim_status"),
        make_event(decision_type=DecisionType.LLM_RESPONSE,
                   tool_name=None),
        make_event(decision_type=DecisionType.TOOL_CALL,
                   tool_name="get_policy_details"),
    ]

    chain = []
    previous_hash = GENESIS_HASH

    for event in events:
        current_hash = compute_hash(event, previous_hash)
        chain.append({
            "event":         event,
            "previous_hash": previous_hash,
            "current_hash":  current_hash,
        })
        previous_hash = current_hash

    # Verify chain links
    for i, entry in enumerate(chain):
        if i == 0:
            # First row must use genesis hash
            assert entry["previous_hash"] == GENESIS_HASH, (
                f"Row 0 previous_hash must be GENESIS_HASH"
            )
        else:
            # Each row's previous_hash must equal prior row's current_hash
            assert entry["previous_hash"] == chain[i-1]["current_hash"], (
                f"Row {i} previous_hash must equal row {i-1} current_hash"
            )

        # Each row's current_hash must match recomputed hash
        recomputed = compute_hash(entry["event"], entry["previous_hash"])
        assert recomputed == entry["current_hash"], (
            f"Row {i} current_hash mismatch — row was tampered"
        )

    print(f"  Chain of {len(chain)} entries verified")
    print("  PASS ✓ — chain links correctly")


# ------------------------------------------------------------------
# Test 5 — Tamper detection
# ------------------------------------------------------------------

def test_tamper_detection():
    """
    Build a 3-entry chain.
    Tamper with the middle entry.
    Verify the chain detects the break.
    """
    events = [make_event() for _ in range(3)]

    # Build chain
    chain = []
    previous_hash = GENESIS_HASH
    for event in events:
        current_hash = compute_hash(event, previous_hash)
        chain.append({
            "event":         event,
            "previous_hash": previous_hash,
            "current_hash":  current_hash,
        })
        previous_hash = current_hash

    # Tamper with row 1 (middle) — change outcome
    tampered_event = make_event(outcome=Outcome.FAILURE)
    tampered_event.event_id = chain[1]["event"].event_id  # same ID
    chain[1]["event"] = tampered_event

    # Now verify — should detect break at row 2
    # because row 2's previous_hash still points to
    # original row 1's hash, but row 1 now has
    # a different canonical content

    broken = False
    previous_hash = GENESIS_HASH

    for i, entry in enumerate(chain):
        recomputed = compute_hash(entry["event"], entry["previous_hash"])

        if recomputed != entry["current_hash"]:
            broken = True
            print(f"  Tamper detected at row {i} ✓")
            break

        if entry["previous_hash"] != previous_hash:
            broken = True
            print(f"  Chain break detected at row {i} ✓")
            break

        previous_hash = entry["current_hash"]

    assert broken, "Tamper must be detected"
    print("  PASS ✓ — tamper detected correctly")


# ------------------------------------------------------------------
# Test 6 — Canonical content is deterministic JSON
# ------------------------------------------------------------------

def test_canonical_content_deterministic():
    event = make_event()

    content1 = _canonical_content(event, GENESIS_HASH)
    content2 = _canonical_content(event, GENESIS_HASH)

    assert content1 == content2
    # Verify it's valid JSON
    parsed = json.loads(content1)
    assert "event_id" in parsed
    assert "previous_hash" in parsed
    assert "decision_type" in parsed
    print(f"  canonical = {content1[:80]}...")
    print("  PASS ✓ — canonical content is deterministic JSON")


# ------------------------------------------------------------------
# Test runner
# ------------------------------------------------------------------

def run_all_tests():
    tests = [
        ("Hash is deterministic",               test_hash_deterministic),
        ("Field change changes hash",           test_field_change_changes_hash),
        ("Previous hash change changes current",test_previous_hash_change_changes_current),
        ("Chain links correctly",               test_chain_links),
        ("Tamper detection",                    test_tamper_detection),
        ("Canonical content is deterministic",  test_canonical_content_deterministic),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL ✗ — {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR — {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
