"""ADK tool — F3+F4 composition: fire OOB challenge on bad financial provenance.

Feature: F3 (Cryptographic Content Provenance Signing) + F4 (Out-of-Band Financial
    Authorisation Trigger)
ACM: 1.2 (defence in depth — independent of detector scores), 1.3 (honest distinction
    between cryptographic validity and deployment-trusted issuer), 2.5, 3.1, 3.7
Scope: in-product.md  | composition of two product.md features (not a new feature) —
    owner-approved widening of F4's trigger surface from {transcript} to
    {transcript ∪ untrusted-or-unsigned financial doc}

Pure: this tool takes a :class:`ProvenanceResult` and a channel; it never touches
detectors, the network beyond the channel, or detector scores. F4's §4.3 / ACM 1.2
independence guarantee is preserved because the trigger consults *only* the F3
verifier output, never the F1/F2 results.

The challenge fires iff the financial document fails ANY of:
  * has a valid C2PA manifest at all → ``no_manifest``
  * passes cryptographic integrity → ``invalid_signature``
  * is signed by an issuer on the deployment's allow-list → ``untrusted_issuer``

Both has_valid_signature AND is_trusted_issuer must be True for the document to
pass without dispatch — the §5 anti-pattern this tool exists to prevent is
"valid signature alone authorises a financial action" (an attacker can self-sign
their own forgery with a cryptographically-valid manifest using their own cert).

Emits exactly one F5 audit event carrying metadata only: the verdict, issuer
common name, reason code, channel receipt. The audit log never carries the
document bytes (re-asserts ACM 1.6 via :class:`AuditLog`'s media guard).
"""

from __future__ import annotations

from dataclasses import dataclass

from deepverify_pro.audit.log import AuditLog
from deepverify_pro.authorization.trigger import (
    ChallengeReceipt,
    OutOfBandChannel,
    TriggerReasonCode,
    build_provenance_challenge,
)
from deepverify_pro.provenance import ProvenanceResult

EVENT_NAME = "provenance.financial_trigger"


@dataclass(frozen=True)
class ProvenanceFinancialOutcome:
    """Combined verdict + dispatch result returned to the orchestrator.

    ``triggered`` is True iff the F4 challenge was dispatched. ``reason_code``
    is the F4 trigger label when fired, or ``None`` when the document passed
    both checks (valid + trusted).
    """

    triggered: bool
    reason_code: TriggerReasonCode | None
    receipt: ChallengeReceipt | None


def _classify(result: ProvenanceResult) -> TriggerReasonCode | None:
    """Pick the F4 reason code for a provenance verdict, or None if it passes.

    Precedence: no_manifest > invalid_signature > untrusted_issuer. A document
    with a valid signature from a trusted issuer is the only pass condition.
    """
    if not result.has_valid_signature:
        # ``issuer is None`` when c2patool found no claim at all (no manifest);
        # any non-None issuer with an invalid signature means a sig was
        # attempted and the crypto check failed.
        if result.issuer is None:
            return "no_manifest"
        return "invalid_signature"
    if not result.is_trusted_issuer:
        return "untrusted_issuer"
    return None


def provenance_financial_trigger(
    result: ProvenanceResult,
    *,
    recipient: str,
    channel: OutOfBandChannel,
    audit: AuditLog,
) -> ProvenanceFinancialOutcome:
    """Evaluate ``result``; dispatch an OOB challenge iff provenance fails.

    A single F5 audit event is appended on every call (triggered or not) so
    the chain shows the F3+F4 evaluation ran — a silently-skipped check would
    be indistinguishable from a missed dispatch (§7: fail loudly).

    Raises :class:`ValueError` for an empty ``recipient`` *iff* the trigger
    would fire — a passing document never needs a recipient, but a failing
    one dispatched to nobody silently defeats the defence-in-depth check
    (ACM 1.2). The audit event is written before re-raising so the chain
    still reflects the evaluation attempt.
    """
    reason_code = _classify(result)
    receipt: ChallengeReceipt | None = None
    dispatch_error: str | None = None

    if reason_code is not None:
        try:
            challenge = build_provenance_challenge(
                recipient=recipient,
                reason_code=reason_code,
            )
        except ValueError as exc:
            dispatch_error = str(exc)
        else:
            receipt = channel.send(challenge)

    payload: dict[str, object] = {
        "triggered": reason_code is not None,
        "reason_code": reason_code,
        "has_valid_signature": result.has_valid_signature,
        "is_trusted_issuer": result.is_trusted_issuer,
        "issuer": result.issuer,
        "verifier_reason": result.reason,
        "channel_name": channel.name,
        "challenge_id": receipt.challenge_id if receipt is not None else None,
        "dispatched": receipt.dispatched if receipt is not None else False,
        "dispatch_error": dispatch_error,
    }
    audit.append(EVENT_NAME, payload)

    if dispatch_error is not None:
        raise ValueError(dispatch_error)

    return ProvenanceFinancialOutcome(
        triggered=reason_code is not None,
        reason_code=reason_code,
        receipt=receipt,
    )
