"""
test_scrubber.py
----------------
Verify scrubber input → output without a live Postgres connection.
Mocks the DB log call so the core detection + redaction pipeline
can be tested in isolation.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from unittest.mock import AsyncMock, patch
from scrubber import UserContext, scrub_query

# ------------------------------------------------------------------
# Test cases — (description, raw_query, expected_redacted_fragments)
# ------------------------------------------------------------------

TEST_CASES = [
    {
        "desc": "UK National Insurance Number",
        "query": "My NI number is AB 12 34 56 C, can you check my policy?",
        "should_contain": ["[UK_NIN]"],
        "should_not_contain": ["AB 12 34 56 C"],
    },
    {
        "desc": "Email address",
        "query": "Please send the policy document to john.smith@example.com",
        "should_contain": ["[EMAIL]"],
        "should_not_contain": ["john.smith@example.com"],
    },
    {
        "desc": "Phone number",
        "query": "Call me back on +44 7911 123456 about my claim",
        "should_contain": ["[PHONE]"],
        "should_not_contain": ["+44 7911 123456"],
    },
    {
        "desc": "UAE Emirates ID",
        "query": "My Emirates ID is 784-1990-1234567-1, what is my coverage?",
        "should_contain": ["[UAE_EID]"],
        "should_not_contain": ["784-1990-1234567-1"],
    },
    {
        "desc": "Spanish NIE",
        "query": "Mi NIE es X1234567A, quiero saber mi póliza",
        "should_contain": ["[ES_ID]"],
        "should_not_contain": ["X1234567A"],
    },
    {
        "desc": "Credit card number",
        "query": "I paid premium with card 4111 1111 1111 1111",
        "should_contain": ["[CARD_NUMBER]"],
        "should_not_contain": ["4111 1111 1111 1111"],
    },
    {
        "desc": "Clean query — no national ID PII",
        # London triggers LOCATION — that's correct behaviour.
        # Testing that no national IDs or personal identifiers are detected.
        "query": "What is covered under my health plan for knee surgery?",
        "should_contain": ["knee surgery"],
        "should_not_contain": [],
        "expect_pii": False
    },
    {
        "desc": "Multiple PII types in one query",
        "query": (
            "Hi, my name is John Smith, NI number AB 12 34 56 C, "
            "email john@test.com, phone +44 7700 900000. "
            "Check policy POL-00012345 please."
        ),
        "should_contain": ["[UK_NIN]", "[EMAIL]", "[PHONE]"],
        "should_not_contain": [
            "AB 12 34 56 C",
            "john@test.com",
            "+44 7700 900000"
        ],
    },
]

# ------------------------------------------------------------------
# Mock user context
# ------------------------------------------------------------------

MOCK_USER = UserContext(
    user_id="user_london_001",
    session_id="sess_abc123",
    region="UK",
    role="policyholder",
    policy_ids=["POL-00012345"]
)


# ------------------------------------------------------------------
# Test runner
# ------------------------------------------------------------------

async def run_tests():
    passed = 0
    failed = 0

    # Patch DB log so tests run without Postgres
    with patch("scrubber.log_pii_event", new_callable=AsyncMock) as mock_log:

        for i, case in enumerate(TEST_CASES, 1):
            print(f"\n{'='*60}")
            print(f"Test {i}: {case['desc']}")
            print(f"INPUT  : {case['query']}")

            result = await scrub_query(
                raw_query=case["query"],
                user_context=MOCK_USER
            )

            print(f"OUTPUT : {result.scrubbed_query}")
            print(f"PII?   : {result.pii_detected}")
            print(f"TYPES  : {result.entities_detected}")
            print(f"EVENT  : {result.event_id}")

            # Assertions
            test_passed = True

            for fragment in case.get("should_contain", []):
                if fragment not in result.scrubbed_query:
                    print(f"  FAIL — expected '{fragment}' in output")
                    test_passed = False

            for fragment in case.get("should_not_contain", []):
                if fragment in result.scrubbed_query:
                    print(f"  FAIL — '{fragment}' should be redacted but still present")
                    test_passed = False

            expect_pii = case.get("expect_pii", True)
            if expect_pii and not result.pii_detected:
                print("  FAIL — expected PII to be detected but pii_detected=False")
                test_passed = False
            if not expect_pii and result.pii_detected:
                print(f"  FAIL — expected no PII but detected: {result.entities_detected}")
                test_passed = False

            # Check DB log was called when PII detected
            if result.pii_detected:
                mock_log.assert_called()

            if test_passed:
                print("  PASS ✓")
                passed += 1
            else:
                failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
