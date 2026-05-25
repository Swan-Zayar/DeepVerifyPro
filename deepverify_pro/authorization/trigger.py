"""Financial-language + amount-threshold detection for the out-of-band trigger.

Feature: F4 (Out-of-Band Financial Authorisation Trigger)
ACM: 1.2 (defence-in-depth — fires independent of detectors), 2.5
Scope: in-product.md

Pure: this module never touches detectors, the network, or detector scores
(CODING_STANDARDS §4.3 — the F4 path MUST NOT be gated on detector output).
It evaluates a transcript string for financial signals and, if matched,
returns a typed :class:`FinancialTriggerResult` that the F4 ADK tool turns
into an out-of-band challenge via :class:`OutOfBandChannel`.

Two signal sources, OR-combined (either alone fires):
  1. **Financial language.** A documented keyword list keyed by category
     (wire transfer, account number, payment approval). Case-insensitive,
     whole-word matched. The keyword list is intentionally narrow and
     conservative — false positives are preferred over false negatives
     (ACM 2.5: known false-positive cost risk, see product.md §3.5).
  2. **Amount above threshold.** A regex extracts currency-tagged amounts
     (USD by default) and compares against
     ``Settings.financial_amount_threshold``. Largest amount wins.
"""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Final, Literal

# The five reasons F4 can fire. The first two are produced by the transcript
# path (this module); the last three are produced by the F3+F4 composition in
# ``tools/provenance_financial_trigger.py``. Together they tell an audit
# reader at a glance WHY the out-of-band challenge was dispatched.
TriggerReasonCode = Literal[
    "financial_language",  # transcript keyword match (e.g. "wire transfer")
    "amount_threshold",  # transcript currency amount ≥ threshold
    "no_manifest",  # F3 verify: no C2PA manifest on a financial doc
    "invalid_signature",  # F3 verify: crypto failure (tampered / forged)
    "untrusted_issuer",  # F3 verify: valid sig but signer not in trust list
]

# Keyword categories — small on purpose. New keywords need owner discussion
# (CODING_STANDARDS §0 Scope Lock — false-positive UX cost is a 2.5 risk).
_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "wire_transfer": ("wire transfer", "bank transfer", "wire the funds", "wire payment"),
    "account_number": ("account number", "iban", "swift code", "routing number"),
    "payment_approval": (
        "approve the payment",
        "authorize the payment",
        "authorise the payment",
        "payment approval",
        "approve the transfer",
        "authorize the transfer",
        "authorise the transfer",
    ),
}

# Each category is matched on word boundaries (\b) so a keyword can never fire
# inside a larger word — e.g. "wire transfer" must NOT match "firewire
# transfers" (the phrase is a substring of it across the word break).
# Substring matching would manufacture nonsensical triggers; word boundaries
# keep F4 to real financial phrases (ACM 2.5: the false-positive cost is a
# named residual risk).
_KEYWORD_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    category: re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b")
    for category, phrases in _KEYWORDS.items()
}

# Currency amount pattern. The currency tag must come BEFORE the number —
# either a symbol ("$10,000", "£10k", "$2 million") or a leading currency code
# ("USD 75,000", "AUD 1,500"). Conservative: a trailing code ("10 million USD")
# and untagged numbers are not matched. Note large amounts must be
# group-separated ("100,000") — a bare "100000" is only partially captured.
_AMOUNT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"""
    (?:
        (?P<symbol>[$£€])\s*(?P<amount1>\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?)(?P<suffix1>\s*(?:k|m|million|thousand))?
        |
        (?P<currency>USD|GBP|EUR|AUD)\s+(?P<amount2>\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?)(?P<suffix2>\s*(?:k|m|million|thousand))?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SUFFIX_MULTIPLIERS: Final[dict[str, float]] = {
    "k": 1_000.0,
    "thousand": 1_000.0,
    "m": 1_000_000.0,
    "million": 1_000_000.0,
}


@dataclass(frozen=True)
class FinancialTriggerResult:
    """Outcome of evaluating a transcript for F4 financial signals.

    ``triggered`` is True iff at least one keyword OR amount-above-threshold
    match was found. The result is small and serialisable — it intentionally
    never carries the transcript text (ACM 1.6 / §4.4: audit records hold
    metadata only, not communication content).
    """

    triggered: bool
    matched_categories: tuple[str, ...] = ()
    largest_amount: float | None = None
    amount_above_threshold: bool = False
    threshold: float = 0.0


@dataclass(frozen=True)
class OutOfBandChallenge:
    """A challenge dispatched on an independent channel (separate device).

    Carries opaque metadata only — never the transcript or detection state.
    The recipient identifier is whatever the channel implementation
    understands (e.g. a registered-device handle); no PII assumptions made.

    ``reason_code`` names the F4 trigger source (one of :data:`TriggerReasonCode`).
    Defaults to ``"financial_language"`` so existing transcript-path callers
    keep working without code changes; the F3+F4 composition sets one of the
    provenance reasons.
    """

    challenge_id: str
    recipient: str
    matched_categories: tuple[str, ...]
    amount: float | None
    threshold: float
    reason_code: TriggerReasonCode = "financial_language"


@dataclass(frozen=True)
class ChallengeReceipt:
    """Channel-side acknowledgement that a challenge was dispatched."""

    challenge_id: str
    channel_name: str
    dispatched: bool
    detail: dict[str, str] = field(default_factory=dict)


class OutOfBandChannel(ABC):
    """Independent-channel sender contract.

    Implementations MUST keep dispatch off the call path (ACM 1.2 — defence
    in depth) and on-prem (ACM 1.6 — no third-party egress in the prototype).
    """

    name: str = "abstract-channel"

    @abstractmethod
    def send(self, challenge: OutOfBandChallenge) -> ChallengeReceipt:
        """Dispatch the challenge on an independent channel."""
        raise NotImplementedError


def _normalise_amount(match: re.Match[str]) -> float | None:
    raw = match.group("amount1") or match.group("amount2")
    if raw is None:  # pragma: no cover — regex guarantees one group hits
        return None
    cleaned = raw.replace(",", "").replace(" ", "")
    try:
        value = float(cleaned)
    except ValueError:  # pragma: no cover — regex guarantees numeric format
        return None
    suffix = (match.group("suffix1") or match.group("suffix2") or "").strip().lower()
    if suffix:
        value *= _SUFFIX_MULTIPLIERS.get(suffix, 1.0)
    return value


def _find_keyword_categories(transcript_lower: str) -> tuple[str, ...]:
    return tuple(
        category
        for category, pattern in _KEYWORD_PATTERNS.items()
        if pattern.search(transcript_lower) is not None
    )


def _find_largest_amount(transcript: str) -> float | None:
    largest: float | None = None
    for match in _AMOUNT_PATTERN.finditer(transcript):
        value = _normalise_amount(match)
        if value is None:
            continue
        if largest is None or value > largest:
            largest = value
    return largest


def evaluate_transcript(transcript: str, *, threshold: float) -> FinancialTriggerResult:
    """Pure evaluation of a transcript for F4 financial signals.

    Returns a :class:`FinancialTriggerResult`. Never raises on malformed
    input; an empty / whitespace transcript simply produces ``triggered=False``.
    """
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    lower = transcript.lower()
    categories = _find_keyword_categories(lower)
    largest = _find_largest_amount(transcript)
    amount_above = largest is not None and largest >= threshold
    triggered = bool(categories) or amount_above
    return FinancialTriggerResult(
        triggered=triggered,
        matched_categories=categories,
        largest_amount=largest,
        amount_above_threshold=amount_above,
        threshold=threshold,
    )


def _transcript_reason(result: FinancialTriggerResult) -> TriggerReasonCode:
    """Pick the most-specific transcript-path reason for a fired result.

    Keyword matches win over amount-only triggers because a named category
    (``wire_transfer`` etc.) is a more specific signal than a bare amount.
    Both can be true simultaneously; the audit payload still carries the full
    ``matched_categories`` + ``largest_amount`` for the reviewer.
    """
    if result.matched_categories:
        return "financial_language"
    return "amount_threshold"


def build_provenance_challenge(
    *,
    recipient: str,
    reason_code: TriggerReasonCode,
) -> OutOfBandChallenge:
    """Mint a provenance-driven :class:`OutOfBandChallenge` (F3+F4 composition).

    Used when F4 fires because a financial document's C2PA manifest is
    missing, cryptographically invalid, or signed by an issuer not on the
    deployment's trust list. The challenge carries no transcript-derived
    metadata (no matched categories, no amount, threshold=0.0); the
    ``reason_code`` is the only failure label the audit reviewer needs.

    Raises :class:`ValueError` for an empty ``recipient`` (silent dispatch
    to no recipient defeats the defence-in-depth check — ACM 1.2) or for a
    transcript-only reason code (``financial_language`` / ``amount_threshold``),
    which would mislabel the audit chain.
    """
    if not recipient.strip():
        raise ValueError(
            "recipient must be non-empty — an out-of-band challenge dispatched "
            "to no recipient silently defeats the defence-in-depth check (ACM 1.2)"
        )
    if reason_code in ("financial_language", "amount_threshold"):
        raise ValueError(
            f"reason_code {reason_code!r} is a transcript-path reason; the "
            "provenance challenge must be one of: no_manifest, "
            "invalid_signature, untrusted_issuer"
        )
    return OutOfBandChallenge(
        challenge_id=str(uuid.uuid4()),
        recipient=recipient,
        matched_categories=(),
        amount=None,
        threshold=0.0,
        reason_code=reason_code,
    )


def build_challenge(
    result: FinancialTriggerResult,
    *,
    recipient: str,
    reason_code: TriggerReasonCode | None = None,
) -> OutOfBandChallenge:
    """Mint an :class:`OutOfBandChallenge` from a fired trigger result.

    Raises :class:`ValueError` for a non-triggered ``result`` or an empty
    ``recipient`` — an out-of-band challenge dispatched to no recipient would
    silently defeat the defence-in-depth check (ACM 1.2).

    ``reason_code`` defaults to the most-specific transcript-path reason
    derived from ``result`` (see :func:`_transcript_reason`). Callers from the
    F3+F4 composition pass an explicit provenance reason.
    """
    if not result.triggered:
        raise ValueError("cannot build a challenge from a non-triggered result")
    if not recipient.strip():
        raise ValueError(
            "recipient must be non-empty — an out-of-band challenge dispatched "
            "to no recipient silently defeats the defence-in-depth check (ACM 1.2)"
        )
    return OutOfBandChallenge(
        challenge_id=str(uuid.uuid4()),
        recipient=recipient,
        matched_categories=result.matched_categories,
        amount=result.largest_amount,
        threshold=result.threshold,
        reason_code=reason_code if reason_code is not None else _transcript_reason(result),
    )
