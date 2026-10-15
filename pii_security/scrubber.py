"""
scrubber.py
-----------
Core PII scrubbing pipeline for the Enterprise Insurance Assistant.

Flow
----
1. Receive raw user query + session context
2. Run Presidio detection across all EMEA + domain recognisers
3. Anonymise — replace detected spans with [ENTITY_TYPE] placeholders
4. Hash original query and user ID for audit (never store raw)
5. Log detection event to Postgres asynchronously
6. Return scrubbed query to caller

What is NOT stored anywhere
---------------------------
- Raw user query
- Raw user ID / national ID
- Detected PII values (only entity types are logged)
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from recognisers import get_all_recognisers
from db import log_pii_event

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data contracts
# ------------------------------------------------------------------

@dataclass
class UserContext:
    """
    Authenticated session context injected by FastAPI dependency.
    Sourced from JWT — never from user-supplied query text.
    """
    user_id: str          # Raw user ID — used only for hashing
    session_id: str       # Active session identifier
    region: str           # e.g. "UK", "FR", "DE", "UAE"
    role: str             # e.g. "policyholder", "claims_adjuster"
    policy_ids: list[str] = field(default_factory=list)


@dataclass
class ScrubResult:
    """
    Output of the scrubbing pipeline.
    scrubbed_query is safe to pass to the LLM.
    All other fields are for internal routing/audit only.
    """
    scrubbed_query: str           # PII-free text → goes to LLM
    original_query_hash: str      # SHA-256 of original → audit log
    user_token: str               # SHA-256 of user_id → audit log
    entities_detected: list[str]  # Entity type names → audit log
    event_id: str                 # UUID → audit log primary key
    pii_detected: bool            # Convenience flag
    detected_at: datetime


# ------------------------------------------------------------------
# Engine initialisation (done once at module load)
# ------------------------------------------------------------------

def _build_analyzer() -> AnalyzerEngine:
    """
    Build Presidio AnalyzerEngine with:
    - en_core_web_sm spaCy NLP backend
    - All default recognisers (email, phone, credit card etc.)
    - All custom EMEA + insurance domain recognisers
    """
    # NLP backend
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": "en_core_web_sm"}
            ]
        }
    )
    nlp_engine = provider.create_engine()

    # Registry — start with defaults, add custom on top
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)

    for recogniser in get_all_recognisers():
        registry.add_recognizer(recogniser)

    return AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=["en"]
    )


# Initialise once — these are expensive to build
_analyzer = _build_analyzer()
_anonymizer = AnonymizerEngine()

# Replacement operators — each entity type → [ENTITY_TYPE] placeholder
# All detected entities are replaced with their type label.
# This makes scrubbed text readable while removing actual values.
_REPLACE_OPERATORS = {
    "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),

    # Override defaults with labelled placeholders per entity
    "UK_NIN":              OperatorConfig("replace", {"new_value": "[UK_NIN]"}),
    "FR_SOCIAL_SECURITY":  OperatorConfig("replace", {"new_value": "[FR_SS_NUMBER]"}),
    "DE_NATIONAL_ID":      OperatorConfig("replace", {"new_value": "[DE_ID]"}),
    "NL_BSN":              OperatorConfig("replace", {"new_value": "[NL_BSN]"}),
    "ES_NATIONAL_ID":      OperatorConfig("replace", {"new_value": "[ES_ID]"}),
    "UAE_EMIRATES_ID":     OperatorConfig("replace", {"new_value": "[UAE_EID]"}),
    "SE_PERSONNUMMER":     OperatorConfig("replace", {"new_value": "[SE_PERSONNUMMER]"}),
    "PL_PESEL":            OperatorConfig("replace", {"new_value": "[PL_PESEL]"}),
    "PHONE_NUMBER":        OperatorConfig("replace", {"new_value": "[PHONE]"}),
    "EMAIL_ADDRESS":       OperatorConfig("replace", {"new_value": "[EMAIL]"}),
    "CREDIT_CARD":         OperatorConfig("replace", {"new_value": "[CARD_NUMBER]"}),
    "IBAN_CODE":           OperatorConfig("replace", {"new_value": "[BANK_ACCOUNT]"}),
    "PERSON":              OperatorConfig("replace", {"new_value": "[PERSON_NAME]"}),
    "LOCATION":            OperatorConfig("replace", {"new_value": "[LOCATION]"}),
    "POLICY_NUMBER":       OperatorConfig("replace", {"new_value": "[POLICY_NUMBER]"}),
    "CLAIM_ID":            OperatorConfig("replace", {"new_value": "[CLAIM_ID]"}),
    "ICD_CODE":            OperatorConfig("replace", {"new_value": "[DIAGNOSIS_CODE]"}),
}

# Minimum confidence threshold — detections below this are ignored
_SCORE_THRESHOLD = 0.7


# ------------------------------------------------------------------
# Hashing helpers
# ------------------------------------------------------------------

def _sha256(value: str) -> str:
    """One-way hash. Used for user_id and query text in audit logs."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------
# Core scrub function
# ------------------------------------------------------------------

async def scrub_query(
    raw_query: str,
    user_context: UserContext,
) -> ScrubResult:
    """
    Detect and redact PII from a user query.

    Parameters
    ----------
    raw_query    : The exact text the user typed
    user_context : Authenticated session context from JWT

    Returns
    -------
    ScrubResult with scrubbed_query safe to pass to LLM.

    Side effects
    ------------
    Writes one row to pii_audit_log in Postgres if PII detected.
    Fire-and-forget — audit failure does not raise.
    """

    # --- 1. Detect ---------------------------------------------------
    analysis_results = _analyzer.analyze(
        text=raw_query,
        language="en",
        score_threshold=_SCORE_THRESHOLD,
        # All entities — both default Presidio and our custom ones
        entities=None
    )

    event_id = str(uuid.uuid4())
    detected_at = datetime.now(timezone.utc)
    user_token = _sha256(user_context.user_id)
    query_hash = _sha256(raw_query)
    entities_detected = list({r.entity_type for r in analysis_results})
    pii_detected = len(analysis_results) > 0

    # --- 2. Anonymise ------------------------------------------------
    if pii_detected:
        anonymized = _anonymizer.anonymize(
            text=raw_query,
            analyzer_results=analysis_results,
            operators=_REPLACE_OPERATORS
        )
        scrubbed_query = anonymized.text

        logger.info(
            "PII detected and redacted",
            extra={
                "event_id": event_id,
                "entity_types": entities_detected,
                "entity_count": len(entities_detected),
                "region": user_context.region,
                "session_id": user_context.session_id
            }
        )
    else:
        scrubbed_query = raw_query
        logger.debug(
            "No PII detected in query",
            extra={
                "event_id": event_id,
                "session_id": user_context.session_id
            }
        )

    # --- 3. Audit log (fire and forget) ------------------------------
    # Only log when PII was actually found — keeps audit table lean
    if pii_detected:
        await log_pii_event(
            event_id=event_id,
            session_id=user_context.session_id,
            user_token=user_token,       # hashed — never raw user_id
            query_hash=query_hash,       # hashed — never raw query
            entities_found=entities_detected,
            region=user_context.region,
        )

    # --- 4. Return ---------------------------------------------------
    return ScrubResult(
        scrubbed_query=scrubbed_query,
        original_query_hash=query_hash,
        user_token=user_token,
        entities_detected=entities_detected,
        event_id=event_id,
        pii_detected=pii_detected,
        detected_at=detected_at,
    )
