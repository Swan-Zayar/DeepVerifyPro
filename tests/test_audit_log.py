"""Tests for the tamper-evident audit log (F5, ACM 3.1/3.7/1.6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from deepverify_pro.audit import AuditLog, AuditTampered, AuditViolation, SessionAuditLog
from deepverify_pro.tools.audit_log import audit_log


def test_append_and_verify_chain(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("detection", {"score": 0.12, "detector": "x"})
    log.append("provenance", {"valid": False})
    records = log.read_all()
    assert [r.seq for r in records] == [0, 1]
    assert records[1].prev_hash == records[0].hash
    assert log.verify_chain() is True


def test_tamper_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("detection", {"score": 0.10})
    log.append("detection", {"score": 0.20})

    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"score": 0.1', '"score": 0.99')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(AuditTampered):
        log.verify_chain()


def test_forbidden_media_key_rejected(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    with pytest.raises(AuditViolation):
        log.append("detection", {"mfcc": [1, 2, 3]})
    # nothing should have been written
    assert log.read_all() == []


# ---------- the audit_log ADK tool (CODING_STANDARDS §7: every tool tested) ----------


def test_audit_log_tool_appends_and_returns_record(tmp_path: Path) -> None:
    """The F5 ``audit_log`` tool appends one event and returns its record."""
    log = AuditLog(tmp_path / "audit.jsonl")
    record = audit_log("orchestrator.tick.start", {"tick_id": 1}, audit=log)
    assert record.seq == 0
    assert record.event == "orchestrator.tick.start"
    assert record.payload == {"tick_id": 1}
    assert log.read_all() == [record]
    assert log.verify_chain() is True


def test_audit_log_tool_enforces_forbidden_media_guard(tmp_path: Path) -> None:
    """The tool inherits the §4.4 / ACM 1.6 media-key guard from AuditLog."""
    log = AuditLog(tmp_path / "audit.jsonl")
    with pytest.raises(AuditViolation):
        audit_log("detection", {"landmarks": [1, 2, 3]}, audit=log)
    assert log.read_all() == []


# ---------- per-session slice (SessionAuditLog + AuditLog.read_session) ----------


def test_session_proxy_stamps_session_id_into_every_append(tmp_path: Path) -> None:
    """SessionAuditLog adds session_id to payloads; the underlying chain
    stays single (no fork) and verifies green."""
    log = AuditLog(tmp_path / "audit.jsonl")
    proxy_a = SessionAuditLog(log, "sess-A")
    proxy_b = SessionAuditLog(log, "sess-B")

    proxy_a.append("audio.detect", {"score": 0.1})
    proxy_b.append("audio.detect", {"score": 0.2})
    log.append("orchestrator.tick.end", {"tick_id": 1})  # global event, no session

    records = log.read_all()
    assert [r.payload.get("session_id") for r in records] == ["sess-A", "sess-B", None]
    assert log.verify_chain() is True


def test_session_proxy_refuses_payload_with_mismatched_session_id(tmp_path: Path) -> None:
    """A tool can't smuggle another session's id through the proxy."""
    log = AuditLog(tmp_path / "audit.jsonl")
    proxy = SessionAuditLog(log, "sess-A")
    with pytest.raises(AuditViolation):
        proxy.append("audio.detect", {"score": 0.1, "session_id": "sess-B"})
    # nothing should have been written
    assert log.read_all() == []


def test_session_proxy_rejects_empty_session_id(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    with pytest.raises(ValueError):
        SessionAuditLog(log, "")


def test_read_session_filters_to_the_session_slice(tmp_path: Path) -> None:
    """read_session returns only records stamped with the matching id; the
    returned records keep their original seq/prev_hash/hash so per-record
    hashes can still be re-checked."""
    log = AuditLog(tmp_path / "audit.jsonl")
    SessionAuditLog(log, "sess-A").append("audio.detect", {"score": 0.1})
    SessionAuditLog(log, "sess-B").append("audio.detect", {"score": 0.2})
    SessionAuditLog(log, "sess-A").append("video.detect", {"score": 0.3})
    log.append("orchestrator.tick.end", {"tick_id": 1})  # untagged

    slice_a = log.read_session("sess-A")
    assert [r.event for r in slice_a] == ["audio.detect", "video.detect"]
    # original seq preserved (0 and 2 in the global chain)
    assert [r.seq for r in slice_a] == [0, 2]
    assert log.read_session("sess-missing") == []
    assert log.read_session("") == []
