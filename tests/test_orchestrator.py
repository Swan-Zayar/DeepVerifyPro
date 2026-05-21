"""M5 orchestrator tests: tool surface, audit chain, F4 independence, parallel pipelines.

Feature: F1–F5 (CODING_STANDARDS §2 orchestration)
ACM: 1.2 (F4 fires under orchestration with detectors absent — §4.3 DoD test),
     1.3, 1.6, 2.5, 3.1, 3.7
Scope: in-product.md
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import numpy as np
import pytest

from deepverify_pro.agents import (
    ORCHESTRATOR_NAME,
    TICK_END_EVENT,
    TICK_START_EVENT,
    DeepVerifyOrchestrator,
)
from deepverify_pro.audit.log import AuditLog
from deepverify_pro.authorization import RecordingChannel
from deepverify_pro.detection.base import DetectionResult, Detector, Frame, Modality
from deepverify_pro.indicator import IndicatorState
from deepverify_pro.tools.audio_detect import EVENT_NAME as AUDIO_EVENT
from deepverify_pro.tools.financial_trigger import EVENT_NAME as FIN_EVENT
from deepverify_pro.tools.provenance_verify import EVENT_NAME as PROV_EVENT
from deepverify_pro.tools.video_detect import EVENT_NAME as VIDEO_EVENT

THRESHOLD = 10_000.0


class StubAudioDetector(Detector):
    """Deterministic stub — bypasses librosa to keep the orchestrator tests fast/pure."""

    name: str = "stub-audio"
    is_production: bool = False

    def score(self, frame: Frame) -> DetectionResult:
        if frame.modality is not Modality.AUDIO:
            raise ValueError("stub-audio scores AUDIO frames only")
        return self._result(0.35, source="stub")


class StubVideoDetector(Detector):
    """Deterministic stub — bypasses dlib so tests don't need the predictor weights."""

    name: str = "stub-video"
    is_production: bool = False

    def score(self, frame: Frame) -> DetectionResult:
        if frame.modality is not Modality.VIDEO:
            raise ValueError("stub-video scores VIDEO frames only")
        return self._result(0.45, source="stub")


class ExplodingAudioDetector(Detector):
    """Always raises — proves a crashed pipeline still closes the audit chain."""

    name: str = "exploding-audio"
    is_production: bool = False

    def score(self, frame: Frame) -> DetectionResult:
        raise RuntimeError("detector boom")


def _audio_frame() -> Frame:
    return Frame(
        modality=Modality.AUDIO,
        data=np.zeros(1600, dtype=np.float32),
        sample_rate=16_000,
        index=1,
    )


def _video_frame() -> Frame:
    return Frame(modality=Modality.VIDEO, data=np.zeros((16, 16, 3), dtype=np.uint8), index=1)


# ---------- ADK tool surface (CODING_STANDARDS §2) ----------


def test_orchestrator_exposes_all_six_tools(tmp_path: Path) -> None:
    """§2 requires exactly six deterministic tools on the orchestrator surface."""
    audit = AuditLog(tmp_path / "audit.jsonl")
    orchestrator = DeepVerifyOrchestrator(audit=audit)
    names = orchestrator.tool_names()
    assert set(names) == {
        "audio_detect",
        "video_detect",
        "provenance_verify",
        "sign_media",
        "financial_trigger",
        "audit_log",
    }
    assert len(orchestrator.tools) == 6


def test_orchestrator_name_is_stable(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    assert DeepVerifyOrchestrator(audit=audit).name == ORCHESTRATOR_NAME


def test_orchestrator_rejects_negative_threshold(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    with pytest.raises(ValueError):
        DeepVerifyOrchestrator(audit=audit, financial_threshold=-1.0)


def test_orchestrator_rejects_zero_workers(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    with pytest.raises(ValueError):
        DeepVerifyOrchestrator(audit=audit, max_workers=0)


# ---------- end-to-end tick wires the four runtime pipelines ----------


def test_tick_wires_audio_video_financial_in_one_chain(tmp_path: Path) -> None:
    """A tick with all four inputs writes a clean, verifiable hash chain (§4.4)."""
    audit = AuditLog(tmp_path / "audit.jsonl")
    channel = RecordingChannel()
    orchestrator = DeepVerifyOrchestrator(
        audit=audit,
        audio_detector=StubAudioDetector(),
        video_detector=StubVideoDetector(),
        channel=channel,
        financial_threshold=THRESHOLD,
    )

    tick = orchestrator.tick(
        audio_frame=_audio_frame(),
        video_frame=_video_frame(),
        transcript="please wire transfer $50,000 to vendor",
        recipient="cfo-device",
    )

    assert tick.tick_id == 1
    assert tick.audio is not None and tick.audio.indicator_state in {
        IndicatorState.GREEN,
        IndicatorState.AMBER,
        IndicatorState.RED,
    }
    assert tick.video is not None
    assert tick.financial is not None and tick.financial.result.triggered is True

    events = [r.event for r in audit.read_all()]
    assert events[0] == TICK_START_EVENT
    assert events[-1] == TICK_END_EVENT
    assert AUDIO_EVENT in events
    assert VIDEO_EVENT in events
    assert FIN_EVENT in events
    assert audit.verify_chain() is True


def test_audit_chain_verifies_after_tick(tmp_path: Path) -> None:
    """§8 DoD: audit chain must verify after a tick."""
    audit = AuditLog(tmp_path / "audit.jsonl")
    orchestrator = DeepVerifyOrchestrator(
        audit=audit,
        audio_detector=StubAudioDetector(),
        video_detector=StubVideoDetector(),
        channel=RecordingChannel(),
        financial_threshold=THRESHOLD,
    )
    orchestrator.tick(
        audio_frame=_audio_frame(),
        video_frame=_video_frame(),
        transcript="just checking in",
    )
    assert audit.verify_chain() is True


def test_two_ticks_increment_id_and_keep_chain(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    orchestrator = DeepVerifyOrchestrator(
        audit=audit,
        audio_detector=StubAudioDetector(),
    )
    a = orchestrator.tick(audio_frame=_audio_frame())
    b = orchestrator.tick(audio_frame=_audio_frame())
    assert (a.tick_id, b.tick_id) == (1, 2)
    assert audit.verify_chain() is True


def test_tick_with_no_inputs_still_audits_lifecycle(tmp_path: Path) -> None:
    """An empty tick still emits start+end (§7: fail loudly, never silently)."""
    audit = AuditLog(tmp_path / "audit.jsonl")
    orchestrator = DeepVerifyOrchestrator(audit=audit)
    tick = orchestrator.tick()
    assert tick.audio is None and tick.video is None
    assert tick.provenance is None and tick.financial is None
    events = [r.event for r in audit.read_all()]
    assert events == [TICK_START_EVENT, TICK_END_EVENT]
    assert audit.verify_chain() is True


def test_tick_rejects_empty_recipient_for_a_triggering_transcript(tmp_path: Path) -> None:
    """A fired F4 trigger must not dispatch a challenge to an empty recipient.

    §4.3 / ACM 1.2: a defence-in-depth challenge that reaches no one is worse
    than no challenge. ``build_challenge`` refuses it; the orchestrator records
    the failure in the F5 chain and re-raises rather than swallowing it.
    """
    audit = AuditLog(tmp_path / "audit.jsonl")
    orchestrator = DeepVerifyOrchestrator(
        audit=audit,
        channel=RecordingChannel(),
        financial_threshold=THRESHOLD,
    )
    with pytest.raises(ValueError, match="recipient"):
        orchestrator.tick(transcript="please wire transfer $50,000", recipient="")

    # The failure is audited, not swallowed — the chain opens and closes.
    records = audit.read_all()
    assert [r.event for r in records] == [TICK_START_EVENT, TICK_END_EVENT]
    assert "recipient" in records[-1].payload["error"]
    assert audit.verify_chain() is True


def test_tick_writes_end_event_even_when_a_pipeline_raises(tmp_path: Path) -> None:
    """A crashing pipeline must still leave a closed, orphan-free audit chain.

    §4.4 / ACM 3.1, 3.7: ``orchestrator.tick.end`` is written (with an
    ``error`` field) before the exception propagates, so the F5 chain never
    keeps a start record with no matching end.
    """
    audit = AuditLog(tmp_path / "audit.jsonl")
    orchestrator = DeepVerifyOrchestrator(audit=audit, audio_detector=ExplodingAudioDetector())

    with pytest.raises(RuntimeError, match="detector boom"):
        orchestrator.tick(audio_frame=_audio_frame())

    records = audit.read_all()
    events = [r.event for r in records]
    assert events[0] == TICK_START_EVENT
    assert events[-1] == TICK_END_EVENT  # end written despite the failure
    end_payload = records[-1].payload
    assert end_payload["error"] is not None
    assert "detector boom" in end_payload["error"]
    assert audit.verify_chain() is True


# ---------- §4.3 / ACM 1.2: F4 fires under orchestration without detectors ----------


def test_f4_fires_under_orchestration_with_no_detectors(tmp_path: Path) -> None:
    """§4.3 DoD: F4 must fire even when both detectors are absent.

    No audio_detector and no video_detector are wired. The orchestrator must
    still dispatch the F4 challenge purely on the transcript signal.
    """
    audit = AuditLog(tmp_path / "audit.jsonl")
    channel = RecordingChannel()
    orchestrator = DeepVerifyOrchestrator(
        audit=audit,
        audio_detector=None,
        video_detector=None,
        channel=channel,
        financial_threshold=THRESHOLD,
    )

    tick = orchestrator.tick(
        audio_frame=_audio_frame(),  # provided but ignored — no detector wired
        video_frame=_video_frame(),
        transcript="approve the transfer of USD 100000 to the supplier",
        recipient="cfo-device",
    )

    assert tick.audio is None and tick.video is None
    assert tick.financial is not None
    assert tick.financial.result.triggered is True
    assert tick.financial.receipt is not None
    assert tick.financial.receipt.dispatched is True
    assert len(channel.sent) == 1

    events = [r.event for r in audit.read_all()]
    assert AUDIO_EVENT not in events
    assert VIDEO_EVENT not in events
    assert FIN_EVENT in events
    assert audit.verify_chain() is True


def test_orchestrator_tick_signature_takes_no_detector_score() -> None:
    """Static guarantee: tick() accepts no detector-score parameter (§4.3)."""
    sig = inspect.signature(DeepVerifyOrchestrator.tick)
    params = set(sig.parameters)
    for forbidden in ("score", "synthetic_probability", "indicator_state"):
        assert forbidden not in params


# ---------- provenance pipeline is independent of the detectors ----------


@pytest.mark.skipif(shutil.which("c2patool") is None, reason="c2patool binary not installed")
def test_provenance_runs_independently_of_detectors(tmp_path: Path) -> None:
    """§3.4: the provenance pipeline runs independently of the detection engines."""
    # An empty path → c2patool reports no manifest. The point of this test is
    # that the provenance pipeline runs and audits even with no detectors wired.
    media = tmp_path / "no-such-input.png"
    media.write_bytes(b"not-a-real-image")

    audit = AuditLog(tmp_path / "audit.jsonl")
    orchestrator = DeepVerifyOrchestrator(audit=audit)
    tick = orchestrator.tick(provenance_path=media)

    assert tick.audio is None and tick.video is None
    assert tick.provenance is not None
    assert tick.provenance.has_valid_signature is False  # unsigned input
    events = [r.event for r in audit.read_all()]
    assert PROV_EVENT in events
    assert audit.verify_chain() is True


# ---------- ACM 1.6: orchestrator audit payloads carry no media ----------


def test_tick_audit_carries_no_media_keys(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    orchestrator = DeepVerifyOrchestrator(
        audit=audit,
        audio_detector=StubAudioDetector(),
        video_detector=StubVideoDetector(),
        channel=RecordingChannel(),
        financial_threshold=THRESHOLD,
    )
    orchestrator.tick(
        audio_frame=_audio_frame(),
        video_frame=_video_frame(),
        transcript="please wire transfer $50,000 to vendor",
        recipient="cfo-device",
    )
    for record in audit.read_all():
        for forbidden in ("data", "frame", "mfcc", "landmarks", "audio", "video", "waveform"):
            assert forbidden not in record.payload
