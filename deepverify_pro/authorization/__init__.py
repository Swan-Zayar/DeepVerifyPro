"""Authorization package — out-of-band financial trigger (F4).

Feature: F4 (Out-of-Band Financial Authorisation Trigger)
ACM: 1.2, 1.6, 2.5
Scope: in-product.md
"""

from deepverify_pro.authorization.channels import LocalFileChannel, RecordingChannel
from deepverify_pro.authorization.trigger import (
    ChallengeReceipt,
    FinancialTriggerResult,
    OutOfBandChallenge,
    OutOfBandChannel,
    TriggerReasonCode,
    build_challenge,
    build_provenance_challenge,
    evaluate_transcript,
)

__all__ = [
    "ChallengeReceipt",
    "FinancialTriggerResult",
    "LocalFileChannel",
    "OutOfBandChallenge",
    "OutOfBandChannel",
    "RecordingChannel",
    "TriggerReasonCode",
    "build_challenge",
    "build_provenance_challenge",
    "evaluate_transcript",
]
