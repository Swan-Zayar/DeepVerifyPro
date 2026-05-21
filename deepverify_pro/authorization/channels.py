"""Out-of-band channel implementations (prototype, on-prem only).

Feature: F4 (Out-of-Band Financial Authorisation Trigger)
ACM: 1.2, 1.6
Scope: in-product.md

Two concrete channels ship in the prototype, both on-prem (ACM 1.6 — no
third-party egress; SMS / push providers are out of scope per
CODING_STANDARDS §4.1 and need owner discussion before adding):

- :class:`LocalFileChannel` — appends a JSON challenge to a local file
  on the deploying organisation's filesystem (a registered-device handle
  the org's directory service would resolve in production).
- :class:`RecordingChannel` — in-memory test stub. Records every dispatch
  so tests can assert the F4 path fired.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from deepverify_pro.authorization.trigger import (
    ChallengeReceipt,
    OutOfBandChallenge,
    OutOfBandChannel,
)


class LocalFileChannel(OutOfBandChannel):
    """Append challenges as JSON lines to a local on-prem file.

    Pure local I/O — no network. The directory is created on first write
    so the prototype can be wired up without manual setup.
    """

    name: str = "local-file"

    def __init__(self, path: str | Path) -> None:
        self._path: Path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def send(self, challenge: OutOfBandChallenge) -> ChallengeReceipt:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            **dataclasses.asdict(challenge),
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return ChallengeReceipt(
            challenge_id=challenge.challenge_id,
            channel_name=self.name,
            dispatched=True,
            detail={"path": str(self._path)},
        )


class RecordingChannel(OutOfBandChannel):
    """In-memory channel — records every dispatch. Test / dry-run only."""

    name: str = "recording"

    def __init__(self) -> None:
        self._sent: list[OutOfBandChallenge] = []

    @property
    def sent(self) -> Sequence[OutOfBandChallenge]:
        return tuple(self._sent)

    def send(self, challenge: OutOfBandChallenge) -> ChallengeReceipt:
        self._sent.append(challenge)
        return ChallengeReceipt(
            challenge_id=challenge.challenge_id,
            channel_name=self.name,
            dispatched=True,
        )
