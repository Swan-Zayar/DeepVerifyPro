"""F4 financial-trigger tests: keyword + amount matching, defence-in-depth, audit hygiene.

Feature: F4 (Out-of-Band Financial Authorisation Trigger)
ACM: 1.2 (fires independent of detectors — §4.3 DoD test), 1.6, 2.5
Scope: in-product.md
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deepverify_pro.audit.log import AuditLog
from deepverify_pro.authorization import (
    FinancialTriggerResult,
    LocalFileChannel,
    RecordingChannel,
    build_challenge,
    evaluate_transcript,
)
from deepverify_pro.tools.financial_trigger import EVENT_NAME, financial_trigger

THRESHOLD = 10_000.0


# ---------- evaluate_transcript ----------


def test_no_signal_does_not_trigger() -> None:
    result = evaluate_transcript("hello, just checking in on the project", threshold=THRESHOLD)
    assert result.triggered is False
    assert result.matched_categories == ()
    assert result.largest_amount is None
    assert result.amount_above_threshold is False


def test_wire_transfer_keyword_fires() -> None:
    result = evaluate_transcript("please initiate a wire transfer today", threshold=THRESHOLD)
    assert result.triggered is True
    assert "wire_transfer" in result.matched_categories


def test_account_number_keyword_fires() -> None:
    result = evaluate_transcript("the IBAN is GB29NWBK60161331926819", threshold=THRESHOLD)
    assert result.triggered is True
    assert "account_number" in result.matched_categories


def test_payment_approval_keyword_fires() -> None:
    result = evaluate_transcript("we need you to authorise the payment now", threshold=THRESHOLD)
    assert result.triggered is True
    assert "payment_approval" in result.matched_categories


def test_amount_above_threshold_fires_without_keywords() -> None:
    result = evaluate_transcript("send $25,000 over", threshold=THRESHOLD)
    assert result.triggered is True
    assert result.largest_amount == pytest.approx(25_000.0)
    assert result.amount_above_threshold is True


def test_amount_below_threshold_does_not_fire_alone() -> None:
    result = evaluate_transcript("a small $50 lunch reimbursement", threshold=THRESHOLD)
    assert result.matched_categories == ()
    assert result.amount_above_threshold is False
    assert result.triggered is False


def test_largest_amount_wins_when_multiple_present() -> None:
    result = evaluate_transcript("$500 today, USD 75,000 tomorrow", threshold=THRESHOLD)
    assert result.largest_amount == pytest.approx(75_000.0)


def test_suffix_million_is_expanded() -> None:
    result = evaluate_transcript("transfer $2 million urgently", threshold=THRESHOLD)
    assert result.largest_amount == pytest.approx(2_000_000.0)
    assert result.amount_above_threshold is True


def test_keyword_match_fires_even_when_amount_below_threshold() -> None:
    result = evaluate_transcript("tiny $5 wire transfer test", threshold=THRESHOLD)
    assert result.triggered is True
    assert "wire_transfer" in result.matched_categories
    assert result.amount_above_threshold is False


def test_negative_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate_transcript("anything", threshold=-1.0)


def test_build_challenge_refuses_non_triggered_result() -> None:
    result = FinancialTriggerResult(triggered=False, threshold=THRESHOLD)
    with pytest.raises(ValueError):
        build_challenge(result, recipient="alice@example.test")


# ---------- channels ----------


def test_recording_channel_records_dispatch() -> None:
    channel = RecordingChannel()
    result = evaluate_transcript("wire transfer of $50,000", threshold=THRESHOLD)
    challenge = build_challenge(result, recipient="bob@example.test")
    receipt = channel.send(challenge)
    assert receipt.dispatched is True
    assert receipt.channel_name == "recording"
    assert len(channel.sent) == 1
    assert channel.sent[0].challenge_id == challenge.challenge_id


def test_local_file_channel_appends_jsonl(tmp_path: Path) -> None:
    channel = LocalFileChannel(tmp_path / "challenges.jsonl")
    result = evaluate_transcript("authorise the payment for $20,000", threshold=THRESHOLD)
    challenge = build_challenge(result, recipient="cfo-device")
    receipt = channel.send(challenge)
    assert receipt.dispatched is True
    lines = channel.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    import json

    record = json.loads(lines[0])
    assert record["challenge_id"] == challenge.challenge_id
    assert record["recipient"] == "cfo-device"


# ---------- financial_trigger ADK tool ----------


def test_financial_trigger_dispatches_on_match(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    channel = RecordingChannel()
    outcome = financial_trigger(
        "wire transfer $80,000 to vendor",
        threshold=THRESHOLD,
        recipient="cfo-device",
        channel=channel,
        audit=audit,
    )
    assert outcome.result.triggered is True
    assert outcome.receipt is not None
    assert outcome.receipt.dispatched is True
    assert len(channel.sent) == 1


def test_financial_trigger_does_not_dispatch_on_no_signal(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    channel = RecordingChannel()
    outcome = financial_trigger(
        "hi, are we still on for the quarterly review?",
        threshold=THRESHOLD,
        recipient="cfo-device",
        channel=channel,
        audit=audit,
    )
    assert outcome.result.triggered is False
    assert outcome.receipt is None
    assert len(channel.sent) == 0


def test_financial_trigger_audit_contains_no_transcript(tmp_path: Path) -> None:
    """ACM 1.6: audit payload must NOT include the transcript text itself."""
    audit = AuditLog(tmp_path / "audit.jsonl")
    channel = RecordingChannel()
    transcript = "please wire transfer $40,000 to acme corp asap"
    financial_trigger(
        transcript,
        threshold=THRESHOLD,
        recipient="cfo-device",
        channel=channel,
        audit=audit,
    )
    records = audit.read_all()
    assert len(records) == 1
    rec = records[0]
    assert rec.event == EVENT_NAME
    # The transcript text and substrings (sender, amount-with-symbol) must
    # never appear in the audit payload.
    serialised = repr(rec.payload).lower()
    assert "acme corp" not in serialised
    assert "asap" not in serialised
    assert "please" not in serialised
    # Forbidden keys from the audit-log media guard would also reject these,
    # but the tool MUST not even try to embed them.
    assert "transcript" not in rec.payload
    assert audit.verify_chain() is True


def test_audit_event_emitted_even_when_no_trigger(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    financial_trigger(
        "just chatting",
        threshold=THRESHOLD,
        recipient="cfo-device",
        channel=RecordingChannel(),
        audit=audit,
    )
    records = audit.read_all()
    assert len(records) == 1
    assert records[0].payload["triggered"] is False
    assert records[0].payload["dispatched"] is False


# ---------- DEFENCE-IN-DEPTH (CODING_STANDARDS §4.3 / ACM 1.2) ----------


def test_f4_fires_with_all_detectors_absent(tmp_path: Path) -> None:
    """§4.3: the F4 path MUST fire on transcript alone — no detector wired.

    This test imports nothing from ``deepverify_pro.detection`` and passes no
    detector / score anywhere. If F4 ever grows a dependency on detector
    state, this test breaks first.
    """
    audit = AuditLog(tmp_path / "audit.jsonl")
    channel = RecordingChannel()
    outcome = financial_trigger(
        "approve the transfer of USD 100000 to the supplier",
        threshold=THRESHOLD,
        recipient="cfo-device",
        channel=channel,
        audit=audit,
    )
    assert outcome.result.triggered is True
    assert outcome.receipt is not None and outcome.receipt.dispatched is True
    assert len(channel.sent) == 1


def test_f4_signature_takes_no_detector() -> None:
    """Static guarantee: ``financial_trigger`` accepts no Detector / score arg."""
    import inspect

    sig = inspect.signature(financial_trigger)
    params = set(sig.parameters)
    assert "detector" not in params
    assert "score" not in params
    assert "synthetic_probability" not in params
    assert "indicator_state" not in params
