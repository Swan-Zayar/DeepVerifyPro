"""Tamper-evident, append-only audit log (SHA-256 hash chain).

Feature: F5 (Audit Trail and Incident Reporting)
ACM: 3.1, 3.7, 1.6
Scope: in-product.md

Each record stores the previous record's hash; :meth:`AuditLog.verify_chain`
recomputes the chain and raises on any tampering (CODING_STANDARDS §4.4).
Records may never carry raw media or biometric vectors (ACM 1.6) — payload
keys that look like media are rejected at append time.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS: str = "0" * 64

# Payload keys that would risk carrying raw media / biometrics (ACM 1.6).
_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "data",
        "frame",
        "frames",
        "mfcc",
        "landmarks",
        "audio",
        "video",
        "pixels",
        "samples",
        "biometric",
        "embedding",
        "waveform",
    }
)


class AuditViolation(Exception):
    """Raised when a payload would violate the ACM 1.6 no-media-in-log rule."""


class AuditTampered(Exception):
    """Raised by :meth:`AuditLog.verify_chain` when the hash chain is broken."""


@dataclass(frozen=True)
class AuditRecord:
    """One immutable, hash-chained audit entry."""

    seq: int
    ts: str
    event: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str


def _canonical(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _compute_hash(prev_hash: str, payload: dict[str, Any], ts: str, seq: int, event: str) -> str:
    h = hashlib.sha256()
    h.update(prev_hash.encode("utf-8"))
    h.update(_canonical({"seq": seq, "ts": ts, "event": event, "payload": payload}))
    return h.hexdigest()


class AuditLog:
    """Append-only JSONL audit log with a verifiable SHA-256 chain."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # In-process append serialisation. The M5 orchestrator runs the
        # audio/video/provenance/financial pipelines concurrently and they
        # all append here; without this lock, two threads could read the
        # same prev_hash and write a forked chain (§4.4 / ACM 3.1, 3.7).
        self._append_lock: threading.Lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def _reject_forbidden(self, payload: dict[str, Any]) -> None:
        for key in payload:
            if key.lower() in _FORBIDDEN_KEYS:
                raise AuditViolation(
                    f"payload key {key!r} may carry raw media/biometrics — refused (ACM 1.6)"
                )

    def append(self, event: str, payload: dict[str, Any]) -> AuditRecord:
        """Append one event. Raises :class:`AuditViolation` for media-bearing payloads."""
        self._reject_forbidden(payload)
        with self._append_lock:
            records = self.read_all()
            seq = len(records)
            prev_hash = records[-1].hash if records else GENESIS
            ts = datetime.now(UTC).isoformat()
            digest = _compute_hash(prev_hash, payload, ts, seq, event)
            record = AuditRecord(
                seq=seq, ts=ts, event=event, payload=payload, prev_hash=prev_hash, hash=digest
            )
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dataclasses.asdict(record), sort_keys=True) + "\n")
        return record

    def read_all(self) -> list[AuditRecord]:
        if not self._path.exists():
            return []
        out: list[AuditRecord] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            raw: dict[str, Any] = json.loads(line)
            out.append(
                AuditRecord(
                    seq=int(raw["seq"]),
                    ts=str(raw["ts"]),
                    event=str(raw["event"]),
                    payload=dict(raw["payload"]),
                    prev_hash=str(raw["prev_hash"]),
                    hash=str(raw["hash"]),
                )
            )
        return out

    def verify_chain(self) -> bool:
        """Return ``True`` if intact; raise :class:`AuditTampered` otherwise."""
        prev = GENESIS
        for i, record in enumerate(self.read_all()):
            if record.seq != i or record.prev_hash != prev:
                raise AuditTampered(f"chain linkage broken at seq {i}")
            expected = _compute_hash(
                record.prev_hash, record.payload, record.ts, record.seq, record.event
            )
            if expected != record.hash:
                raise AuditTampered(f"hash mismatch at seq {i}")
            prev = record.hash
        return True
