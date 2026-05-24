"""M6 API tests: F1–F5 over the localhost HTTP surface.

Feature: F1–F5 (HTTP surface)
ACM: 1.2 (F4 fires over HTTP with detectors absent — §4.3 DoD test),
     1.3, 1.6, 2.5, 3.1, 3.7
Scope: in-product.md
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from deepverify_pro.agents import TICK_START_EVENT, DeepVerifyOrchestrator
from deepverify_pro.api.app import create_app
from deepverify_pro.audit.log import AuditLog
from deepverify_pro.authorization import RecordingChannel
from deepverify_pro.config import Settings
from deepverify_pro.detection.audio import BaselineAudioDetector
from deepverify_pro.detection.base import DetectionResult, Detector, Frame, Modality
from deepverify_pro.detection.video.landmarks import NoFaceDetected
from deepverify_pro.tools.audio_detect import EVENT_NAME as AUDIO_EVENT
from scripts.gen_test_cert import mint_chain

requires_c2patool = pytest.mark.skipif(
    shutil.which("c2patool") is None,
    reason="c2patool binary not installed",
)


# ---------- detector stubs (bypass librosa/dlib for fast, pure API tests) ----------


class StubVideoDetector(Detector):
    """Deterministic video stub — proves the API wires the video pipeline."""

    name: str = "stub-video"
    is_production: bool = False

    def score(self, frame: Frame) -> DetectionResult:
        if frame.modality is not Modality.VIDEO:
            raise ValueError("stub-video scores VIDEO frames only")
        return self._result(0.45, source="stub")


class NoFaceVideoDetector(Detector):
    """Always raises NoFaceDetected — exercises the API's 422 error path."""

    name: str = "noface-video"
    is_production: bool = False

    def score(self, frame: Frame) -> DetectionResult:
        raise NoFaceDetected("no face detected in frame")


# ---------- media fixtures ----------


def _wav_bytes(seconds: float = 0.5, sample_rate: int = 16_000) -> bytes:
    """A short, deterministic tone encoded as WAV."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = (0.3 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, tone, sample_rate, format="WAV")
    return buffer.getvalue()


def _png_bytes() -> bytes:
    """A minimal, genuinely decodable PNG (8x8 black)."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return bytes(encoded.tobytes())


# ---------- app construction helpers ----------


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "audit_path": tmp_path / "audit.jsonl",
        "challenge_log_path": tmp_path / "challenges.jsonl",
        "financial_amount_threshold": 10_000.0,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _orchestrator(tmp_path: Path, **kwargs: object) -> DeepVerifyOrchestrator:
    audit = AuditLog(tmp_path / "audit.jsonl")
    kwargs.setdefault("channel", RecordingChannel())
    kwargs.setdefault("financial_threshold", 10_000.0)
    return DeepVerifyOrchestrator(audit=audit, **kwargs)  # type: ignore[arg-type]


def _client(tmp_path: Path, orchestrator: DeepVerifyOrchestrator) -> TestClient:
    return TestClient(create_app(settings=_settings(tmp_path), orchestrator=orchestrator))


# ---------- health ----------


def test_health_reports_the_six_tool_surface(tmp_path: Path) -> None:
    client = _client(tmp_path, _orchestrator(tmp_path))
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert len(body["tools"]) == 6


# ---------- F1 detect ----------


def test_detect_audio_returns_a_probabilistic_result(tmp_path: Path) -> None:
    """ACM 1.3: the response is a probability + colour state, never a verdict."""
    orchestrator = _orchestrator(tmp_path, audio_detector=BaselineAudioDetector())
    client = _client(tmp_path, orchestrator)
    resp = client.post("/detect", files={"audio": ("clip.wav", _wav_bytes(), "audio/wav")})
    assert resp.status_code == 200
    audio = resp.json()["audio"]
    assert audio is not None
    assert 0.0 <= audio["synthetic_probability"] <= 1.0
    assert audio["indicator_state"] in {"green", "amber", "red"}
    assert audio["is_production"] is False  # baseline — never presented as final
    assert orchestrator.audit.verify_chain() is True


def test_detect_undecodable_audio_returns_422(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, audio_detector=BaselineAudioDetector())
    client = _client(tmp_path, orchestrator)
    resp = client.post("/detect", files={"audio": ("clip.wav", b"not-audio", "audio/wav")})
    assert resp.status_code == 422


# ---------- F2 detect ----------


def test_detect_video_is_wired_through_the_orchestrator(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, video_detector=StubVideoDetector())
    client = _client(tmp_path, orchestrator)
    resp = client.post("/detect", files={"video": ("frame.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    video = resp.json()["video"]
    assert video is not None and video["detector_name"] == "stub-video"


def test_detect_video_with_no_face_returns_422_and_closes_chain(tmp_path: Path) -> None:
    """A faceless frame is a clean 422 — and the F5 chain still closes (§4.4)."""
    orchestrator = _orchestrator(tmp_path, video_detector=NoFaceVideoDetector())
    client = _client(tmp_path, orchestrator)
    resp = client.post("/detect", files={"video": ("frame.png", _png_bytes(), "image/png")})
    assert resp.status_code == 422
    assert orchestrator.audit.verify_chain() is True


# ---------- F4: fires over HTTP, independent of detectors (ACM 1.2 / §4.3) ----------


def test_f4_fires_over_http_with_no_detectors(tmp_path: Path) -> None:
    """§4.3 DoD: F4 must fire over the HTTP surface with both detectors absent."""
    channel = RecordingChannel()
    orchestrator = _orchestrator(tmp_path, channel=channel)
    client = _client(tmp_path, orchestrator)
    resp = client.post(
        "/detect",
        data={
            "transcript": "please wire transfer $50,000 to the new vendor",
            "recipient": "cfo-device",
        },
    )
    assert resp.status_code == 200
    financial = resp.json()["financial"]
    assert financial["triggered"] is True
    assert financial["dispatched"] is True
    assert len(channel.sent) == 1


def test_f4_triggering_transcript_without_recipient_is_422(tmp_path: Path) -> None:
    """A fired trigger with no recipient is refused — a challenge to no one
    silently defeats the defence-in-depth check (ACM 1.2)."""
    orchestrator = _orchestrator(tmp_path)
    client = _client(tmp_path, orchestrator)
    resp = client.post("/detect", data={"transcript": "please wire transfer $50,000"})
    assert resp.status_code == 422
    assert orchestrator.audit.verify_chain() is True


# ---------- F3 sign / verify ----------


@requires_c2patool
def test_verify_unsigned_file_reports_no_signature(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    client = _client(tmp_path, orchestrator)
    resp = client.post("/verify", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    assert resp.json()["has_valid_signature"] is False


@requires_c2patool
def test_sign_then_verify_roundtrip(tmp_path: Path) -> None:
    ca = tmp_path / "ca.crt"
    cert = tmp_path / "leaf.crt"
    key = tmp_path / "leaf.key"
    mint_chain(ca, cert, key, validity_days=30)
    settings = _settings(tmp_path, signing_cert_path=cert, signing_key_path=key)
    orchestrator = _orchestrator(tmp_path)
    client = TestClient(create_app(settings=settings, orchestrator=orchestrator))

    sign_resp = client.post("/sign", files={"file": ("photo.png", _png_bytes(), "image/png")})
    assert sign_resp.status_code == 200
    assert sign_resp.headers["x-dvp-issuer"] == "DeepVerify Pro Test Signer"
    signed = sign_resp.content
    assert signed

    verify_resp = client.post("/verify", files={"file": ("signed.png", signed, "image/png")})
    assert verify_resp.status_code == 200
    assert verify_resp.json()["has_valid_signature"] is True


# ---------- F5 audit ----------


def test_audit_endpoint_lists_chained_events(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, audio_detector=BaselineAudioDetector())
    client = _client(tmp_path, orchestrator)
    client.post("/detect", files={"audio": ("clip.wav", _wav_bytes(), "audio/wav")})
    body = client.get("/audit").json()
    assert body["count"] > 0
    events = [record["event"] for record in body["records"]]
    assert TICK_START_EVENT in events
    assert AUDIO_EVENT in events


def test_audit_verify_endpoint_reports_intact_chain(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, audio_detector=BaselineAudioDetector())
    client = _client(tmp_path, orchestrator)
    client.post("/detect", files={"audio": ("clip.wav", _wav_bytes(), "audio/wav")})
    body = client.get("/audit/verify").json()
    assert body["intact"] is True
    assert body["records_checked"] > 0


def test_audit_records_carry_no_media_keys(tmp_path: Path) -> None:
    """ACM 1.6: no audit payload may carry raw media or biometric vectors."""
    orchestrator = _orchestrator(tmp_path, audio_detector=BaselineAudioDetector())
    client = _client(tmp_path, orchestrator)
    client.post(
        "/detect",
        files={"audio": ("clip.wav", _wav_bytes(), "audio/wav")},
        data={"transcript": "wire transfer $50,000", "recipient": "cfo-device"},
    )
    records = client.get("/audit").json()["records"]
    forbidden = ("data", "frame", "mfcc", "landmarks", "audio", "video", "waveform")
    for record in records:
        for key in forbidden:
            assert key not in record["payload"]


# ---------- F5 per-session audit slice (download) ----------


def test_audit_session_endpoint_returns_only_matching_records(tmp_path: Path) -> None:
    """The slice contains the requested session's events; other sessions and
    untagged events stay out. Each record keeps its original seq so the
    global chain order is preserved in the download."""
    orchestrator = _orchestrator(tmp_path, audio_detector=BaselineAudioDetector())
    client = _client(tmp_path, orchestrator)
    client.post(
        "/detect",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"session_id": "sess-A"},
    )
    client.post(
        "/detect",
        files={"audio": ("b.wav", _wav_bytes(), "audio/wav")},
        data={"session_id": "sess-B"},
    )

    resp = client.get("/audit/session/sess-A")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    assert resp.headers["content-disposition"] == (
        'attachment; filename="audit-session-sess-A.jsonl"'
    )
    lines = [line for line in resp.text.splitlines() if line]
    records = [json.loads(line) for line in lines]
    assert records, "expected at least one record for sess-A"
    for record in records:
        assert record["payload"]["session_id"] == "sess-A"
    # the orchestrator's global chain is still one continuous chain
    assert orchestrator.audit.verify_chain() is True


def test_audit_session_endpoint_rejects_invalid_session_id(tmp_path: Path) -> None:
    """Path-component / Content-Disposition injection is refused at the boundary.

    Either the route handler's regex returns 400, or Starlette's path router
    refuses to match (404) when the decoded id contains a path separator —
    both are valid rejections at the boundary.
    """
    client = _client(tmp_path, _orchestrator(tmp_path))
    resp = client.get("/audit/session/has spaces")
    assert resp.status_code == 400
    resp = client.get("/audit/session/has%2Fslash")
    assert resp.status_code in {400, 404}


def test_detect_rejects_invalid_session_id_form_field(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, audio_detector=BaselineAudioDetector())
    client = _client(tmp_path, orchestrator)
    resp = client.post(
        "/detect",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"session_id": "bad id"},
    )
    assert resp.status_code == 400


# ---------- ACM 1.6: CORS is scoped to the local frontend origins ----------


def test_cors_allows_the_configured_local_origin(tmp_path: Path) -> None:
    client = _client(tmp_path, _orchestrator(tmp_path))
    resp = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"
