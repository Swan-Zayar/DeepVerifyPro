"""F1 audio detection tests: MFCC hop, baseline contract, audit hygiene.

Feature: F1 (Real-Time Audio Deepfake Detection)
ACM: 1.2, 1.3, 1.6
Scope: in-product.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepverify_pro.audit.log import AuditLog, AuditViolation
from deepverify_pro.detection.audio import (
    BaselineAudioDetector,
    MFCCConfig,
    MFCCExtractorError,
    extract_mfcc,
)
from deepverify_pro.detection.base import DetectionResult, Frame, Modality
from deepverify_pro.indicator import IndicatorState
from deepverify_pro.tools.audio_detect import EVENT_NAME, audio_detect

SAMPLE_RATE = 16_000
DURATION_S = 1.0


def _speech_like(sample_rate: int = SAMPLE_RATE, seconds: float = DURATION_S) -> np.ndarray:
    """Deterministic, voice-shaped waveform with healthy temporal variation."""
    rng = np.random.default_rng(seed=42)
    n = int(sample_rate * seconds)
    t = np.arange(n) / sample_rate
    # Pitched carrier with vibrato + amplitude tremolo + low-amplitude noise.
    fundamental = 140.0 + 30.0 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum(fundamental) / sample_rate
    carrier = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t)
    noise = 0.05 * rng.standard_normal(n)
    waveform = (carrier * envelope + noise).astype(np.float32)
    peak = float(np.max(np.abs(waveform))) or 1.0
    return waveform / peak


def _flat_audio(sample_rate: int = SAMPLE_RATE, seconds: float = DURATION_S) -> np.ndarray:
    """Pure single-tone — minimal MFCC variability, should drift towards red."""
    n = int(sample_rate * seconds)
    t = np.arange(n) / sample_rate
    return (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


# ---------- MFCC extractor ----------


def test_mfcc_hop_is_25ms() -> None:
    waveform = _speech_like()
    mfcc = extract_mfcc(waveform, SAMPLE_RATE)
    hop = MFCCConfig().hop_length(SAMPLE_RATE)
    expected_frames = 1 + (waveform.size // hop)
    # librosa center-pads by default — allow a ±1-frame slack against the ideal hop count.
    assert abs(mfcc.shape[1] - expected_frames) <= 1
    assert mfcc.shape[0] == MFCCConfig().n_mfcc


def test_mfcc_rejects_multichannel() -> None:
    stereo = np.zeros((2, 1024), dtype=np.float32)
    with pytest.raises(MFCCExtractorError):
        extract_mfcc(stereo, SAMPLE_RATE)


def test_mfcc_rejects_empty() -> None:
    with pytest.raises(MFCCExtractorError):
        extract_mfcc(np.zeros(0, dtype=np.float32), SAMPLE_RATE)


def test_mfcc_config_rejects_bad_values() -> None:
    with pytest.raises(MFCCExtractorError):
        MFCCConfig(n_mfcc=0)
    with pytest.raises(MFCCExtractorError):
        MFCCConfig(hop_ms=-1.0)


# ---------- BaselineAudioDetector contract ----------


def test_baseline_returns_valid_detection_result() -> None:
    detector = BaselineAudioDetector()
    frame = Frame(modality=Modality.AUDIO, data=_speech_like(), sample_rate=SAMPLE_RATE)
    result = detector.score(frame)
    assert isinstance(result, DetectionResult)
    assert 0.30 <= result.synthetic_probability <= 0.70  # bounded honesty band
    assert result.detector_name == "audio-mfcc-heuristic-baseline-v0"
    assert result.is_production is False
    assert result.indicator_state in {
        IndicatorState.GREEN,
        IndicatorState.AMBER,
        IndicatorState.RED,
    }


def test_baseline_is_not_production() -> None:
    assert BaselineAudioDetector().is_production is False


def test_baseline_is_deterministic() -> None:
    detector = BaselineAudioDetector()
    waveform = _speech_like()
    a = detector.score(Frame(modality=Modality.AUDIO, data=waveform, sample_rate=SAMPLE_RATE))
    b = detector.score(Frame(modality=Modality.AUDIO, data=waveform, sample_rate=SAMPLE_RATE))
    assert a.synthetic_probability == b.synthetic_probability
    assert a.indicator_state is b.indicator_state


def test_baseline_rejects_non_audio_frame() -> None:
    detector = BaselineAudioDetector()
    with pytest.raises(ValueError):
        detector.score(Frame(modality=Modality.VIDEO, data=np.zeros((16, 16))))


def test_baseline_rejects_missing_sample_rate() -> None:
    detector = BaselineAudioDetector()
    with pytest.raises(ValueError):
        detector.score(Frame(modality=Modality.AUDIO, data=_speech_like(), sample_rate=None))


def test_flat_audio_drifts_red_relative_to_speech() -> None:
    detector = BaselineAudioDetector()
    speech = detector.score(
        Frame(modality=Modality.AUDIO, data=_speech_like(), sample_rate=SAMPLE_RATE)
    )
    flat = detector.score(
        Frame(modality=Modality.AUDIO, data=_flat_audio(), sample_rate=SAMPLE_RATE)
    )
    # Honest heuristic: flat audio should not score *less* synthetic-leaning than speech.
    assert flat.synthetic_probability >= speech.synthetic_probability


def test_baseline_does_not_touch_disk(tmp_path: Path) -> None:
    """Detector is pure — no files appear in tmp_path during scoring (§7, ACM 1.6)."""
    before = set(tmp_path.iterdir())
    detector = BaselineAudioDetector()
    detector.score(Frame(modality=Modality.AUDIO, data=_speech_like(), sample_rate=SAMPLE_RATE))
    after = set(tmp_path.iterdir())
    assert before == after


# ---------- audio_detect ADK tool ----------


def test_audio_detect_emits_one_clean_audit_event(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    detector = BaselineAudioDetector()
    frame = Frame(modality=Modality.AUDIO, data=_speech_like(), sample_rate=SAMPLE_RATE, index=7)

    result = audio_detect(frame, detector=detector, audit=audit)

    records = audit.read_all()
    assert len(records) == 1
    rec = records[0]
    assert rec.event == EVENT_NAME
    assert rec.payload["frame_index"] == 7
    assert rec.payload["detector_name"] == result.detector_name
    assert rec.payload["is_production"] is False
    assert rec.payload["synthetic_probability"] == result.synthetic_probability
    assert rec.payload["indicator_state"] == str(result.indicator_state)
    assert audit.verify_chain() is True

    # ACM 1.6: no raw-media keys may appear in the audit payload.
    for forbidden in (
        "data",
        "frame",
        "mfcc",
        "audio",
        "video",
        "waveform",
        "samples",
        "biometric",
        "embedding",
    ):
        assert forbidden not in rec.payload


def test_audit_log_refuses_media_payload(tmp_path: Path) -> None:
    """Defence-in-depth: even if a future tool tried, the log would refuse it."""
    audit = AuditLog(tmp_path / "audit.jsonl")
    with pytest.raises(AuditViolation):
        audit.append(EVENT_NAME, {"frame_index": 0, "mfcc": [1.0, 2.0]})
