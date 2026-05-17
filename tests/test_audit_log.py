"""Tests for the tamper-evident audit log (F5, ACM 3.1/3.7/1.6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from deepverify_pro.audit import AuditLog, AuditTampered, AuditViolation


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
