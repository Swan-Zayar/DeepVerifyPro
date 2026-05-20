"""F2 video detection tests: 68-landmark extractor, baseline contract, audit hygiene.

Feature: F2 (Live Video Face Authenticity Verification)
ACM: 1.3, 1.6
Scope: in-product.md

Real-dlib paths skip cleanly when the predictor weights file is not present
(it is ~99 MB and fetched on demand by ``scripts/fetch_landmarks.py``, never
committed). The heuristic math is exercised independently with synthetic
landmark arrays so the algorithm is covered regardless of the weights file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepverify_pro.audit.log import AuditLog, AuditViolation
from deepverify_pro.config import get_settings
from deepverify_pro.detection.base import DetectionResult, Frame, Modality
from deepverify_pro.detection.video import (
    LANDMARK_COUNT,
    BaselineVideoDetector,
    LandmarkExtractorError,
    LandmarksUnavailable,
    NoFaceDetected,
    extract_landmarks,
)
from deepverify_pro.detection.video import baseline as baseline_module
from deepverify_pro.indicator import IndicatorState
from deepverify_pro.tools.video_detect import EVENT_NAME, video_detect

_PREDICTOR_PATH = get_settings().dlib_landmarks_path
_NEEDS_PREDICTOR = pytest.mark.skipif(
    not _PREDICTOR_PATH.exists(),
    reason=f"68-landmark predictor not at {_PREDICTOR_PATH}; run scripts/fetch_landmarks.py",
)


# ---------- fixtures ----------


def _procedural_face(seed: int = 0) -> np.ndarray:
    """Deterministic procedural face image — dlib HOG reliably detects this layout."""
    rng = np.random.default_rng(seed)
    height, width = 200, 200
    img = np.full((height, width, 3), 200, dtype=np.uint8)
    ys, xs = np.mgrid[0:height, 0:width]
    cy, cx = height // 2, width // 2
    ellipse = ((xs - cx) / 60.0) ** 2 + ((ys - cy) / 80.0) ** 2
    img[ellipse > 1] = 100
    for eye_x in (cx - 22, cx + 22):
        img[cy - 18 : cy - 8, eye_x - 8 : eye_x + 8] = 30
    img[cy : cy + 18, cx - 4 : cx + 4] = 150
    img[cy + 28 : cy + 36, cx - 18 : cx + 18] = 80
    # mild noise so identical-seed runs are byte-identical but content is realistic.
    img = np.clip(img + rng.integers(-3, 4, img.shape, endpoint=False), 0, 255).astype(np.uint8)
    return img


def _symmetric_landmarks() -> np.ndarray:
    """Perfectly mirror-symmetric 68 landmarks → suspicious-low symmetry deviation."""
    landmarks = np.zeros((LANDMARK_COUNT, 2), dtype=np.float32)
    # Jaw 0..16 — symmetric arc across x=100.
    for i in range(17):
        offset = i - 8
        landmarks[i] = (100.0 + 10.0 * offset, 180.0 - 0.5 * offset * offset)
    # Eyebrows 17..26 — symmetric.
    for i in range(5):
        landmarks[17 + i] = (60.0 + 5.0 * i, 60.0)
        landmarks[26 - i] = (140.0 - 5.0 * i, 60.0)
    # Nose 27..30 (bridge), 31..35 (tip), with 27 → 30 vertical down the midline.
    for i in range(4):
        landmarks[27 + i] = (100.0, 70.0 + 10.0 * i)
    landmarks[31] = (88.0, 115.0)
    landmarks[32] = (94.0, 117.0)
    landmarks[33] = (100.0, 118.0)
    landmarks[34] = (106.0, 117.0)
    landmarks[35] = (112.0, 115.0)
    # Eyes 36..41 (left, indexed CW from outer corner) and 42..47 (right,
    # indexed CW from INNER corner per dlib's convention). With the offsets
    # below the right eye is the mirror image of the left eye across x=100,
    # placing the inter-ocular distance at 64px on a 160px face → ior 0.40.
    left_centre = np.array([68.0, 90.0], dtype=np.float32)
    right_centre = np.array([132.0, 90.0], dtype=np.float32)
    left_offsets = np.array([(-8, 0), (-4, -4), (4, -4), (8, 0), (4, 4), (-4, 4)], dtype=np.float32)
    right_offsets = np.array(
        [(-8, 0), (-4, -4), (4, -4), (8, 0), (4, 4), (-4, 4)], dtype=np.float32
    )
    landmarks[36:42] = left_centre + left_offsets
    landmarks[42:48] = right_centre + right_offsets
    # Mouth 48..67 — symmetric.
    landmarks[48] = (80.0, 150.0)
    landmarks[49] = (88.0, 145.0)
    landmarks[50] = (94.0, 142.0)
    landmarks[51] = (100.0, 142.0)
    landmarks[52] = (106.0, 142.0)
    landmarks[53] = (112.0, 145.0)
    landmarks[54] = (120.0, 150.0)
    landmarks[55] = (112.0, 158.0)
    landmarks[56] = (106.0, 162.0)
    landmarks[57] = (100.0, 162.0)
    landmarks[58] = (94.0, 162.0)
    landmarks[59] = (88.0, 158.0)
    landmarks[60] = (88.0, 150.0)
    landmarks[61] = (94.0, 148.0)
    landmarks[62] = (100.0, 148.0)
    landmarks[63] = (106.0, 148.0)
    landmarks[64] = (112.0, 150.0)
    landmarks[65] = (106.0, 155.0)
    landmarks[66] = (100.0, 155.0)
    landmarks[67] = (94.0, 155.0)
    return landmarks


def _asymmetric_landmarks() -> np.ndarray:
    """Add lateral noise to the right side → larger symmetry deviation."""
    landmarks = _symmetric_landmarks()
    rng = np.random.default_rng(seed=0)
    perturbation = rng.normal(0.0, 8.0, size=landmarks.shape).astype(np.float32)
    # Only perturb the right-hand half (x > midline) to break symmetry while
    # leaving the inter-ocular ratio measurable.
    right_side = landmarks[:, 0] > 100.0
    landmarks[right_side] += perturbation[right_side]
    return landmarks


# ---------- heuristic math ----------


def test_symmetry_component_low_for_symmetric() -> None:
    sym = _symmetric_landmarks()
    dev = baseline_module._symmetry_deviation(sym)
    component = baseline_module._symmetry_component(dev)
    assert dev < 1e-3  # essentially perfectly symmetric
    assert component > 0.9  # → strongly synthetic-leaning in the symmetry component


def test_symmetry_component_higher_for_asymmetric() -> None:
    sym_dev = baseline_module._symmetry_deviation(_symmetric_landmarks())
    asym_dev = baseline_module._symmetry_deviation(_asymmetric_landmarks())
    assert asym_dev > sym_dev


def test_inter_ocular_ratio_close_to_canonical() -> None:
    ior = baseline_module._inter_ocular_ratio(_symmetric_landmarks())
    assert abs(ior - baseline_module.HEALTHY_IOR) < 0.02


def test_combine_bounded_to_honesty_band() -> None:
    for sym in (0.0, 0.04, 0.10):
        for ior in (0.20, 0.40, 0.60):
            value = baseline_module._combine(sym, ior)
            assert baseline_module.LOW_PROB <= value <= baseline_module.HIGH_PROB


# ---------- baseline detector contract via monkeypatched extractor ----------


def _install_stub_extractor(monkeypatch: pytest.MonkeyPatch, landmarks: np.ndarray) -> None:
    """Swap ``extract_landmarks`` so the contract is testable without dlib weights."""
    monkeypatch.setattr(
        baseline_module,
        "extract_landmarks",
        lambda image, predictor_path=None: landmarks,
    )


def test_baseline_returns_valid_detection_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub_extractor(monkeypatch, _asymmetric_landmarks())
    detector = BaselineVideoDetector()
    frame = Frame(modality=Modality.VIDEO, data=_procedural_face())
    result = detector.score(frame)
    assert isinstance(result, DetectionResult)
    assert baseline_module.LOW_PROB <= result.synthetic_probability <= baseline_module.HIGH_PROB
    assert result.detector_name == "video-landmark-heuristic-baseline-v0"
    assert result.is_production is False
    assert result.indicator_state in {
        IndicatorState.GREEN,
        IndicatorState.AMBER,
        IndicatorState.RED,
    }


def test_baseline_is_not_production() -> None:
    assert BaselineVideoDetector().is_production is False


def test_baseline_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub_extractor(monkeypatch, _asymmetric_landmarks())
    detector = BaselineVideoDetector()
    frame = Frame(modality=Modality.VIDEO, data=_procedural_face())
    a = detector.score(frame)
    b = detector.score(frame)
    assert a.synthetic_probability == b.synthetic_probability
    assert a.indicator_state is b.indicator_state


def test_baseline_rejects_non_video_frame() -> None:
    detector = BaselineVideoDetector()
    with pytest.raises(ValueError):
        detector.score(Frame(modality=Modality.AUDIO, data=np.zeros(16), sample_rate=16_000))


def test_symmetric_drifts_more_synthetic_than_asymmetric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = BaselineVideoDetector()
    frame = Frame(modality=Modality.VIDEO, data=_procedural_face())

    _install_stub_extractor(monkeypatch, _symmetric_landmarks())
    symmetric = detector.score(frame)

    _install_stub_extractor(monkeypatch, _asymmetric_landmarks())
    asymmetric = detector.score(frame)

    # Honest heuristic: a perfectly-symmetric face should not score *less*
    # synthetic-leaning than a naturally-asymmetric one.
    assert symmetric.synthetic_probability >= asymmetric.synthetic_probability


def test_baseline_does_not_touch_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Detector is pure — no files appear in tmp_path during scoring (§7, ACM 1.6)."""
    _install_stub_extractor(monkeypatch, _asymmetric_landmarks())
    before = set(tmp_path.iterdir())
    detector = BaselineVideoDetector()
    detector.score(Frame(modality=Modality.VIDEO, data=_procedural_face()))
    after = set(tmp_path.iterdir())
    assert before == after


# ---------- real dlib landmark extraction (skip when weights absent) ----------


@_NEEDS_PREDICTOR
def test_extract_landmarks_returns_68_points_on_real_face() -> None:
    landmarks = extract_landmarks(_procedural_face())
    assert landmarks.shape == (LANDMARK_COUNT, 2)
    assert landmarks.dtype == np.float32
    # Every landmark must fall inside the frame bounds.
    assert np.all(landmarks[:, 0] >= 0) and np.all(landmarks[:, 0] < 200)
    assert np.all(landmarks[:, 1] >= 0) and np.all(landmarks[:, 1] < 200)


@_NEEDS_PREDICTOR
def test_extract_landmarks_raises_on_no_face() -> None:
    blank = np.full((100, 100, 3), 128, dtype=np.uint8)
    with pytest.raises(NoFaceDetected):
        extract_landmarks(blank)


@_NEEDS_PREDICTOR
def test_baseline_end_to_end_on_real_dlib_pipeline() -> None:
    detector = BaselineVideoDetector()
    frame = Frame(modality=Modality.VIDEO, data=_procedural_face(), index=3)
    result = detector.score(frame)
    assert baseline_module.LOW_PROB <= result.synthetic_probability <= baseline_module.HIGH_PROB
    assert result.detector_name == "video-landmark-heuristic-baseline-v0"


def test_extract_landmarks_rejects_bad_shape() -> None:
    with pytest.raises(LandmarkExtractorError):
        extract_landmarks(np.zeros((10,), dtype=np.uint8))


def test_extract_landmarks_rejects_empty() -> None:
    with pytest.raises(LandmarkExtractorError):
        extract_landmarks(np.zeros((0, 0, 3), dtype=np.uint8))


def test_extract_landmarks_raises_when_predictor_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_predictor.dat"
    with pytest.raises(LandmarksUnavailable):
        extract_landmarks(_procedural_face(), predictor_path=missing)


# ---------- video_detect ADK tool ----------


def test_video_detect_emits_one_clean_audit_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_stub_extractor(monkeypatch, _asymmetric_landmarks())
    audit = AuditLog(tmp_path / "audit.jsonl")
    detector = BaselineVideoDetector()
    frame = Frame(modality=Modality.VIDEO, data=_procedural_face(), index=11)

    result = video_detect(frame, detector=detector, audit=audit)

    records = audit.read_all()
    assert len(records) == 1
    rec = records[0]
    assert rec.event == EVENT_NAME
    assert rec.payload["frame_index"] == 11
    assert rec.payload["detector_name"] == result.detector_name
    assert rec.payload["is_production"] is False
    assert rec.payload["synthetic_probability"] == result.synthetic_probability
    assert rec.payload["indicator_state"] == str(result.indicator_state)
    assert audit.verify_chain() is True

    # ACM 1.6: no raw-media or landmark keys may appear in the audit payload.
    for forbidden in (
        "data",
        "frame",
        "frames",
        "video",
        "pixels",
        "landmarks",
        "biometric",
        "embedding",
    ):
        assert forbidden not in rec.payload


def test_audit_log_refuses_landmark_payload(tmp_path: Path) -> None:
    """Defence-in-depth: even if a future tool tried, the log would refuse it."""
    audit = AuditLog(tmp_path / "audit.jsonl")
    with pytest.raises(AuditViolation):
        audit.append(EVENT_NAME, {"frame_index": 0, "landmarks": [[1.0, 2.0]]})
