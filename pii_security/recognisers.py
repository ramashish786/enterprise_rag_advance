"""
recognisers.py
--------------
Custom Presidio pattern recognisers for EMEA insurance domain.
Each recogniser targets a specific national ID format or
insurance-domain identifier not covered by Presidio's defaults.
"""

from presidio_analyzer import PatternRecognizer, Pattern


# ------------------------------------------------------------------
# EMEA National Identity Recognisers
# ------------------------------------------------------------------

def uk_nin_recogniser() -> PatternRecognizer:
    """
    UK National Insurance Number
    Format: AB 12 34 56 C (with or without spaces)
    """
    return PatternRecognizer(
        supported_entity="UK_NIN",
        patterns=[
            Pattern(
                name="uk_nin",
                # Two letters, 6 digits, one letter (A-D)
                regex=r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b",
                score=0.95
            )
        ],
        context=["national insurance", "NIN", "NI number"]
    )


def fr_social_security_recogniser() -> PatternRecognizer:
    """
    French Numéro de Sécurité Sociale (INSEE number)
    Format: 1 85 12 75 108 001 57 (15 digits)
    """
    return PatternRecognizer(
        supported_entity="FR_SOCIAL_SECURITY",
        patterns=[
            Pattern(
                name="fr_nss",
                regex=r"\b[12]\s?\d{2}\s?(0[1-9]|1[0-2])\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b",
                score=0.95
            )
        ],
        context=["sécurité sociale", "numéro INSEE", "social security"]
    )


def de_national_id_recogniser() -> PatternRecognizer:
    """
    German Personalausweis (National ID card number)
    Format: L01X00T47 (9 alphanumeric characters, mix of letters and digits)
    Score kept very low — context words are required to raise confidence.
    Pure 9-char alphanumeric is too ambiguous without context.
    """
    return PatternRecognizer(
        supported_entity="DE_NATIONAL_ID",
        patterns=[
            Pattern(
                name="de_personalausweis",
                # Must contain at least 2 letters AND 2 digits — pure words excluded
                regex=r"\b(?=[A-Z0-9]*[A-Z][A-Z0-9]*[A-Z][A-Z0-9]*)(?=[A-Z0-9]*\d[A-Z0-9]*\d[A-Z0-9]*)[A-Z0-9]{9}\b",
                score=0.4   # Very low alone — requires context words to cross threshold
            )
        ],
        context=["personalausweis", "ausweis", "german ID", "reisepass", "German national"]
    )


def uk_phone_recogniser() -> PatternRecognizer:
    """
    UK mobile and landline numbers.
    Covers: +44 7xxx xxxxxx, 07xxx xxxxxx, +44 (0)xxx formats.
    Presidio's default PHONE_NUMBER recogniser misses partial UK formats.
    """
    return PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=[
            Pattern(
                name="uk_mobile_intl",
                regex=r"\+44\s?7\d{3}\s?\d{6}",
                score=0.95
            ),
            Pattern(
                name="uk_mobile_local",
                regex=r"\b07\d{3}\s?\d{6}\b",
                score=0.9
            ),
            Pattern(
                name="uk_landline_intl",
                regex=r"\+44\s?\(0\)\s?\d{2,4}\s?\d{6,8}",
                score=0.9
            ),
            Pattern(
                name="uk_landline_local",
                regex=r"\b0[1-9]\d{1,3}\s?\d{6,8}\b",
                score=0.75
            ),
        ],
        context=["call", "phone", "mobile", "contact", "number", "ring"]
    )


def nl_bsn_recogniser() -> PatternRecognizer:
    """
    Dutch Burger Service Nummer (BSN)
    Format: 9 digits, passes 11-proof validation
    """
    return PatternRecognizer(
        supported_entity="NL_BSN",
        patterns=[
            Pattern(
                name="nl_bsn",
                regex=r"\b\d{9}\b",
                score=0.6  # Low alone — context words raise confidence
            )
        ],
        context=["BSN", "burger service", "burgerservicenummer", "sofinummer"]
    )


def es_national_id_recogniser() -> PatternRecognizer:
    """
    Spanish DNI (citizens) and NIE (foreigners)
    DNI: 8 digits + letter  e.g. 12345678Z
    NIE: X/Y/Z + 7 digits + letter  e.g. X1234567A
    """
    return PatternRecognizer(
        supported_entity="ES_NATIONAL_ID",
        patterns=[
            Pattern(
                name="es_dni",
                regex=r"\b\d{8}[A-HJ-NP-TV-Z]\b",
                score=0.9
            ),
            Pattern(
                name="es_nie",
                regex=r"\b[XYZ]\d{7}[A-HJ-NP-TV-Z]\b",
                score=0.95
            )
        ],
        context=["DNI", "NIE", "documento nacional", "número de identidad"]
    )


def uae_emirates_id_recogniser() -> PatternRecognizer:
    """
    UAE Emirates ID
    Format: 784-XXXX-XXXXXXX-X
    """
    return PatternRecognizer(
        supported_entity="UAE_EMIRATES_ID",
        patterns=[
            Pattern(
                name="emirates_id",
                regex=r"\b784-\d{4}-\d{7}-\d\b",
                score=0.99  # Very specific format
            )
        ],
        context=["Emirates ID", "EID", "emirates identity"]
    )


def se_personnummer_recogniser() -> PatternRecognizer:
    """
    Swedish Personnummer
    Format: YYMMDD-XXXX or YYYYMMDD-XXXX
    """
    return PatternRecognizer(
        supported_entity="SE_PERSONNUMMER",
        patterns=[
            Pattern(
                name="se_personnummer",
                regex=r"\b\d{6,8}[-+]\d{4}\b",
                score=0.85
            )
        ],
        context=["personnummer", "person number", "Swedish ID"]
    )


def pl_pesel_recogniser() -> PatternRecognizer:
    """
    Polish PESEL (national identification number)
    Format: 11 digits
    """
    return PatternRecognizer(
        supported_entity="PL_PESEL",
        patterns=[
            Pattern(
                name="pl_pesel",
                regex=r"\b\d{11}\b",
                score=0.6  # Low alone — context words required
            )
        ],
        context=["PESEL", "polish ID", "numer PESEL"]
    )


# ------------------------------------------------------------------
# Insurance Domain Recognisers
# ------------------------------------------------------------------

def policy_number_recogniser() -> PatternRecognizer:
    """
    Insurance policy number — client-specific format.
    Format: POL or INS prefix followed by 8-12 digits.
    Adjust regex to match actual client policy format.
    """
    return PatternRecognizer(
        supported_entity="POLICY_NUMBER",
        patterns=[
            Pattern(
                name="policy_number_prefixed",
                regex=r"\b(POL|INS|CLM)-?\d{8,12}\b",
                score=0.95
            )
        ],
        context=["policy", "insurance number", "policy number", "policy ID"]
    )


def claim_id_recogniser() -> PatternRecognizer:
    """
    Insurance claim ID.
    Format: CLM prefix followed by digits.
    """
    return PatternRecognizer(
        supported_entity="CLAIM_ID",
        patterns=[
            Pattern(
                name="claim_id",
                regex=r"\b(CLM|CLAIM)-?\d{6,10}\b",
                score=0.95
            )
        ],
        context=["claim", "claim ID", "claim number", "reclamation"]
    )


def icd_code_recogniser() -> PatternRecognizer:
    """
    ICD-10 medical diagnosis codes.
    Format: Letter + 2 digits + optional decimal + digits
    e.g. E11.9 (Type 2 diabetes), J45 (Asthma)
    """
    return PatternRecognizer(
        supported_entity="ICD_CODE",
        patterns=[
            Pattern(
                name="icd10_code",
                regex=r"\b[A-Z]\d{2}\.?\d{0,4}\b",
                score=0.75
            )
        ],
        context=["diagnosis", "ICD", "medical code", "condition code"]
    )


# ------------------------------------------------------------------
# Registry — all recognisers in one place
# ------------------------------------------------------------------

def get_all_recognisers() -> list[PatternRecognizer]:
    """
    Returns all custom recognisers for the EMEA
    Insurance Assistant. Add new recognisers here
    as new regions/document types are onboarded.
    """
    return [
        # EMEA national IDs
        uk_nin_recogniser(),
        fr_social_security_recogniser(),
        de_national_id_recogniser(),
        nl_bsn_recogniser(),
        es_national_id_recogniser(),
        uae_emirates_id_recogniser(),
        se_personnummer_recogniser(),
        pl_pesel_recogniser(),
        uk_phone_recogniser(),

        # Insurance domain identifiers
        # Note: POLICY_NUMBER and CLAIM_ID are extracted
        # before scrubbing and stored in session state.
        # They are included here only to redact if they
        # appear in contexts where they shouldn't (e.g.
        # free-text medical notes).
        policy_number_recogniser(),
        claim_id_recogniser(),
        icd_code_recogniser(),
    ]
