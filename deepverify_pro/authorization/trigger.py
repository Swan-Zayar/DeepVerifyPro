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
from typing import Final

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

# Currency amount pattern. Matches "$10,000", "USD 50000.50", "10 million USD",
# "£10k" (when prefixed by $/USD/GBP/EUR). Conservative — number must be tagged.
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
    """

    challenge_id: str
    recipient: str
    matched_categories: tuple[str, ...]
    amount: float | None
    threshold: float


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
    matched: list[str] = []
    for category, phrases in _KEYWORDS.items():
        if any(phrase in transcript_lower for phrase in phrases):
            matched.append(category)
    return tuple(matched)


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


def build_challenge(
    result: FinancialTriggerResult,
    *,
    recipient: str,
) -> OutOfBandChallenge:
    """Mint an :class:`OutOfBandChallenge` from a fired trigger result."""
    if not result.triggered:
        raise ValueError("cannot build a challenge from a non-triggered result")
    return OutOfBandChallenge(
        challenge_id=str(uuid.uuid4()),
        recipient=recipient,
        matched_categories=result.matched_categories,
        amount=result.largest_amount,
        threshold=result.threshold,
    )
