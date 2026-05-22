"""M8 EfficientNet-SBI detector tests — checkpoint loader and Detector contract.

Feature: F2 (Live Video Face Authenticity Verification)
ACM: 1.2, 1.3, 1.6, 2.5
Scope: in-product.md

The real SBI weights are research-only and never committed, so these tests
build a structurally-identical EfficientNet-B4 checkpoint with random weights
(SBI layout: state_dict under the "model" key, keys prefixed "net."). That
proves the loader, the architecture adapter, and the Detector contract; it does
NOT prove real-world accuracy — that is what `scripts/evaluate.py` measures on
a real test set, and why `is_production` stays False (M8 §10).

The suite skips cleanly when torch / efficientnet_pytorch / dlib are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepverify_pro.detection.base import DetectionResult, Frame, Modality
from deepverify_pro.detection.video import efficientnet_sbi as sbi
from deepverify_pro.detection.video.efficientnet_sbi import (
    EfficientNetSBIDetector,
    SBIWeightsUnavailable,
    verify_checkpoint,
)
from deepverify_pro.indicator import IndicatorState

torch = pytest.importorskip("torch")
efficientnet_module = pytest.importorskip("efficientnet_pytorch")


def _dlib_available() -> bool:
    try:
        import dlib  # noqa: F401
    except ImportError:
        return False
    return True


_NEEDS_DLIB = pytest.mark.skipif(not _dlib_available(), reason="needs the dlib package")


# ---------- fixtures ----------


@pytest.fixture(scope="module")
def sbi_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real EfficientNet-B4 checkpoint in SBI layout (random weights)."""
    net = efficientnet_module.EfficientNet.from_name("efficientnet-b4", num_classes=2)
    # SBI saves from a wrapper whose only child is `net` → keys prefixed "net.",
    # stored under the top-level "model" key.
    state_dict = {f"net.{key}": value for key, value in net.state_dict().items()}
    path = tmp_path_factory.mktemp("sbi_weights") / "sbi_efficientnet_b4.pth"
    torch.save({"model": state_dict}, path)
    return path


@pytest.fixture(autouse=True)
def _reset_model_cache() -> None:
    """Clear the module-level model cache so tests do not leak state."""
    sbi._MODEL = None
    sbi._MODEL_PATH = None


def _procedural_face() -> np.ndarray:
    """A procedural face image dlib's HOG detector reliably finds."""
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
    return img


# ---------- checkpoint loader ----------


def test_load_checkpoint_builds_model(sbi_checkpoint: Path) -> None:
    model = sbi._load_checkpoint(sbi_checkpoint)
    with torch.no_grad():
        output = model(torch.zeros(1, 3, sbi.INPUT_SIZE, sbi.INPUT_SIZE))
    assert tuple(output.shape) == (1, sbi.NUM_CLASSES)


def test_verify_checkpoint_accepts_valid(sbi_checkpoint: Path) -> None:
    verify_checkpoint(sbi_checkpoint)  # must not raise


def test_load_checkpoint_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SBIWeightsUnavailable):
        sbi._load_checkpoint(tmp_path / "absent.pth")


def test_load_checkpoint_rejects_no_model_key(tmp_path: Path) -> None:
    path = tmp_path / "bad.pth"
    torch.save({"not_model": {}}, path)
    with pytest.raises(SBIWeightsUnavailable):
        sbi._load_checkpoint(path)


def test_load_checkpoint_rejects_unprefixed_keys(tmp_path: Path) -> None:
    path = tmp_path / "unprefixed.pth"
    # A key without the "net." prefix → not an SBI-layout checkpoint.
    torch.save({"model": {"_conv_stem.weight": torch.zeros(2, 2)}}, path)
    with pytest.raises(SBIWeightsUnavailable):
        sbi._load_checkpoint(path)


def test_load_checkpoint_rejects_mismatched_architecture(tmp_path: Path) -> None:
    path = tmp_path / "mismatch.pth"
    # Correct prefix, but the tensors do not match EfficientNet-B4 → strict load fails.
    torch.save({"model": {"net.bogus.weight": torch.zeros(2, 2)}}, path)
    with pytest.raises(SBIWeightsUnavailable):
        sbi._load_checkpoint(path)


# ---------- Detector contract ----------


def test_detector_name_and_not_production() -> None:
    detector = EfficientNetSBIDetector()
    assert detector.name == "video-efficientnet-sbi-v0"
    assert detector.is_production is False


def test_detector_rejects_audio_frame() -> None:
    # Modality is checked before any weights load — no checkpoint needed.
    detector = EfficientNetSBIDetector()
    with pytest.raises(ValueError):
        detector.score(Frame(modality=Modality.AUDIO, data=np.zeros(16), sample_rate=16_000))


@_NEEDS_DLIB
def test_detector_scores_video_frame(sbi_checkpoint: Path) -> None:
    detector = EfficientNetSBIDetector(weights_path=sbi_checkpoint)
    result = detector.score(Frame(modality=Modality.VIDEO, data=_procedural_face(), index=4))
    assert isinstance(result, DetectionResult)
    assert 0.0 <= result.synthetic_probability <= 1.0
    assert result.detector_name == "video-efficientnet-sbi-v0"
    assert result.is_production is False
    assert result.indicator_state in {
        IndicatorState.GREEN,
        IndicatorState.AMBER,
        IndicatorState.RED,
    }


@_NEEDS_DLIB
def test_detector_is_deterministic(sbi_checkpoint: Path) -> None:
    detector = EfficientNetSBIDetector(weights_path=sbi_checkpoint)
    frame = Frame(modality=Modality.VIDEO, data=_procedural_face())
    first = detector.score(frame)
    second = detector.score(frame)
    assert first.synthetic_probability == second.synthetic_probability


@_NEEDS_DLIB
def test_detector_raises_no_face(sbi_checkpoint: Path) -> None:
    from deepverify_pro.detection.video import NoFaceDetected

    detector = EfficientNetSBIDetector(weights_path=sbi_checkpoint)
    blank = np.full((120, 120, 3), 128, dtype=np.uint8)
    with pytest.raises(NoFaceDetected):
        detector.score(Frame(modality=Modality.VIDEO, data=blank))


@_NEEDS_DLIB
def test_detector_missing_weights_raises(tmp_path: Path) -> None:
    detector = EfficientNetSBIDetector(weights_path=tmp_path / "no_weights.pth")
    with pytest.raises(SBIWeightsUnavailable):
        detector.score(Frame(modality=Modality.VIDEO, data=_procedural_face()))


@_NEEDS_DLIB
def test_extract_face_crop_shape(sbi_checkpoint: Path) -> None:
    detector = EfficientNetSBIDetector(weights_path=sbi_checkpoint)
    # Drive a real score so the cached face detector is built, then crop directly.
    detector.score(Frame(modality=Modality.VIDEO, data=_procedural_face()))
    crop = sbi._extract_face_crop(_procedural_face(), sbi._ensure_face_detector())
    assert crop.shape == (sbi.INPUT_SIZE, sbi.INPUT_SIZE, 3)
    assert crop.dtype == np.uint8


# ---------- harness integration ----------


def test_sbi_detector_registered_in_harness() -> None:
    from scripts.evaluate import _DETECTORS

    factory, modality = _DETECTORS["video-sbi"]
    assert modality is Modality.VIDEO
    assert isinstance(factory(), EfficientNetSBIDetector)
