"""F3+F4 composition tests: dispatch on bad provenance, hold on trusted+valid.

Feature: F3 + F4
ACM: 1.2 (independent of detector scores — §4.3), 1.3 (valid ≠ trusted),
     1.6 (audit log carries metadata only), 2.5, 3.7
Scope: in-product.md  | composition (owner-approved widening of F4 trigger surface)

These tests pin the four outcomes of the composition:
    * valid + trusted → pass (no dispatch)
    * valid + untrusted issuer → dispatch (reason: untrusted_issuer)
    * invalid signature  → dispatch (reason: invalid_signature)
    * no manifest        → dispatch (reason: no_manifest)

Plus the defence-in-depth guarantees: the tool's signature takes no detector,
empty recipient on a fired trigger is refused, and the audit chain stays intact.

These tests construct :class:`ProvenanceResult` directly so they exercise the
F3+F4 wiring without depending on c2patool being installed.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from deepverify_pro.audit.log import AuditLog
from deepverify_pro.authorization import RecordingChannel
from deepverify_pro.provenance import ProvenanceResult
from deepverify_pro.tools.provenance_financial_trigger import (
    EVENT_NAME,
    provenance_financial_trigger,
)


def _valid_trusted() -> ProvenanceResult:
    return ProvenanceResult(
        has_valid_signature=True,
        issuer="Acme Org Signing",
        reason="valid C2PA manifest",
        is_trusted_issuer=True,
    )


def _valid_untrusted() -> ProvenanceResult:
    return ProvenanceResult(
        has_valid_signature=True,
        issuer="Some Attacker Self-Signed",
        reason="valid C2PA manifest (issuer not in deployment trust list)",
        is_trusted_issuer=False,
    )


def _invalid_signature() -> ProvenanceResult:
    return ProvenanceResult(
        has_valid_signature=False,
        issuer="Acme Org Signing",
        reason="manifest invalid: claimSignature.mismatch",
        is_trusted_issuer=False,
    )


def _no_manifest() -> ProvenanceResult:
    return ProvenanceResult(
        has_valid_signature=False,
        issuer=None,
        reason="no manifest",
        is_trusted_issuer=False,
    )


# ---------- passing case ----------


def test_trusted_and_valid_does_not_dispatch(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    channel = RecordingChannel()
    outcome = provenance_financial_trigger(
        _valid_trusted(),
        recipient="cfo-device",
        channel=channel,
        audit=audit,
    )
    assert outcome.triggered is False
    assert outcome.reason_code is None
    assert outcome.receipt is None
    assert len(channel.sent) == 0
    records = audit.read_all()
    assert len(records) == 1
    assert records[0].payload["triggered"] is False
    assert records[0].payload["dispatched"] is False


# ---------- the three fire cases ----------


def test_untrusted_issuer_dispatches(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    channel = RecordingChannel()
    outcome = provenance_financial_trigger(
        _valid_untrusted(),
        recipient="cfo-device",
        channel=channel,
        audit=audit,
    )
    assert outcome.triggered is True
    assert outcome.reason_code == "untrusted_issuer"
    assert outcome.receipt is not None and outcome.receipt.dispatched is True
    assert len(channel.sent) == 1
    assert channel.sent[0].reason_code == "untrusted_issuer"


def test_invalid_signature_dispatches(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    channel = RecordingChannel()
    outcome = provenance_financial_trigger(
        _invalid_signature(),
        recipient="cfo-device",
        channel=channel,
        audit=audit,
    )
    assert outcome.triggered is True
    assert outcome.reason_code == "invalid_signature"
    assert outcome.receipt is not None and outcome.receipt.dispatched is True
    assert len(channel.sent) == 1
    assert channel.sent[0].reason_code == "invalid_signature"


def test_no_manifest_dispatches(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    channel = RecordingChannel()
    outcome = provenance_financial_trigger(
        _no_manifest(),
        recipient="cfo-device",
        channel=channel,
        audit=audit,
    )
    assert outcome.triggered is True
    assert outcome.reason_code == "no_manifest"
    assert outcome.receipt is not None and outcome.receipt.dispatched is True
    assert len(channel.sent) == 1
    assert channel.sent[0].reason_code == "no_manifest"


# ---------- audit hygiene ----------


def test_audit_event_emitted_even_on_pass(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    provenance_financial_trigger(
        _valid_trusted(),
        recipient="cfo-device",
        channel=RecordingChannel(),
        audit=audit,
    )
    records = audit.read_all()
    assert len(records) == 1
    assert records[0].event == EVENT_NAME


def test_audit_event_records_trust_signal_on_failure(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    provenance_financial_trigger(
        _valid_untrusted(),
        recipient="cfo-device",
        channel=RecordingChannel(),
        audit=audit,
    )
    records = audit.read_all()
    payload = records[0].payload
    assert payload["has_valid_signature"] is True
    assert payload["is_trusted_issuer"] is False
    assert payload["reason_code"] == "untrusted_issuer"
    assert audit.verify_chain() is True


# ---------- recipient validation ----------


def test_empty_recipient_on_fired_trigger_raises_and_still_audits(tmp_path: Path) -> None:
    """ACM 1.2: an OOB challenge dispatched to no recipient defeats the check."""
    audit = AuditLog(tmp_path / "audit.jsonl")
    channel = RecordingChannel()
    with pytest.raises(ValueError, match="recipient"):
        provenance_financial_trigger(
            _no_manifest(),
            recipient="   ",
            channel=channel,
            audit=audit,
        )
    # Audit STILL records the attempt — silent skip would mask the failure.
    records = audit.read_all()
    assert len(records) == 1
    assert records[0].payload["dispatched"] is False
    assert records[0].payload["dispatch_error"] is not None
    assert len(channel.sent) == 0


def test_empty_recipient_on_passing_doc_is_fine(tmp_path: Path) -> None:
    """A doc that passes never needs a recipient — don't reject benign calls."""
    audit = AuditLog(tmp_path / "audit.jsonl")
    outcome = provenance_financial_trigger(
        _valid_trusted(),
        recipient="",
        channel=RecordingChannel(),
        audit=audit,
    )
    assert outcome.triggered is False


# ---------- DEFENCE IN DEPTH (CODING_STANDARDS §4.3 / ACM 1.2) ----------


def test_tool_signature_takes_no_detector_handle() -> None:
    """Static guarantee: the tool consults no detector / score (defence in depth).

    F4 must not be gated on F1/F2 output. This test breaks first if the tool
    ever grows a detector dependency.
    """
    sig = inspect.signature(provenance_financial_trigger)
    params = set(sig.parameters)
    assert "detector" not in params
    assert "score" not in params
    assert "synthetic_probability" not in params
    assert "indicator_state" not in params
    assert "audio" not in params
    assert "video" not in params
