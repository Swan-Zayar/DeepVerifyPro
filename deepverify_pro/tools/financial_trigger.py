"""ADK tool — evaluate a transcript for F4 signals and dispatch out-of-band.

Feature: F4 (Out-of-Band Financial Authorisation Trigger)
ACM: 1.2 (independent of detector scores), 1.6, 3.1, 3.7
Scope: in-product.md

Thin deterministic wrapper around :func:`evaluate_transcript` and
:meth:`OutOfBandChannel.send`. Critically, **this tool accepts no detector
and consults no detector score** — F4 is defence-in-depth and fires solely
on transcript signals (CODING_STANDARDS §4.3 / ACM 1.2).

Emits exactly one F5 audit event carrying metadata only. The audit payload
deliberately omits the raw transcript (communication content stays out of
the audit log per §4.4 / ACM 1.6); only matched-category names, amount,
threshold, trigger outcome, and channel receipt are recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

from deepverify_pro.audit.log import AuditLog
from deepverify_pro.authorization.trigger import (
    ChallengeReceipt,
    FinancialTriggerResult,
    OutOfBandChannel,
    build_challenge,
    evaluate_transcript,
)

EVENT_NAME = "financial.trigger"


@dataclass(frozen=True)
class FinancialTriggerOutcome:
    """Combined evaluation + dispatch result returned to the orchestrator."""

    result: FinancialTriggerResult
    receipt: ChallengeReceipt | None


def financial_trigger(
    transcript: str,
    *,
    threshold: float,
    recipient: str,
    channel: OutOfBandChannel,
    audit: AuditLog,
) -> FinancialTriggerOutcome:
    """Evaluate ``transcript``; dispatch via ``channel`` iff a signal matches.

    Returns a :class:`FinancialTriggerOutcome`. The audit event is appended
    on every call (triggered or not) so the F5 log shows that F4 ran — a
    silently-skipped evaluation would be indistinguishable from a missed
    trigger (§7: fail loudly, never silently).
    """
    result = evaluate_transcript(transcript, threshold=threshold)
    receipt: ChallengeReceipt | None = None
    reason_code: str | None = None
    if result.triggered:
        challenge = build_challenge(result, recipient=recipient)
        reason_code = challenge.reason_code
        receipt = channel.send(challenge)

    payload: dict[str, object] = {
        "triggered": result.triggered,
        "reason_code": reason_code,
        "matched_categories": list(result.matched_categories),
        "largest_amount": result.largest_amount,
        "amount_above_threshold": result.amount_above_threshold,
        "threshold": result.threshold,
        "channel_name": channel.name,
        "challenge_id": receipt.challenge_id if receipt is not None else None,
        "dispatched": receipt.dispatched if receipt is not None else False,
    }
    audit.append(EVENT_NAME, payload)
    return FinancialTriggerOutcome(result=result, receipt=receipt)
