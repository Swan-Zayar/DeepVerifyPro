"""F2 trained-model detector — EfficientNet-B4 / Self-Blended-Images adapter.

Feature: F2 (Live Video Face Authenticity Verification)
ACM: 1.2, 1.3, 1.6, 2.5
Scope: in-product.md

A :class:`Detector` subclass wrapping the EfficientNet-B4 network trained with
Self-Blended Images (Shiohara & Yamasaki, "Detecting Deepfakes with
Self-Blended Images", CVPR 2022). This is a **clean adapter** — only the
network architecture and the documented inference recipe are reused; the SBI
training framework is NOT vendored (M8 §8). The :class:`Detector` ABC is
unchanged.

RESEARCH-ONLY WEIGHTS. The SBI checkpoint and its FaceForensics++ training
data are licensed for non-commercial academic / research use only (M8 §7;
CODING_STANDARDS §3 M8 entry). DeepVerify Pro is designated a non-commercial
research prototype — this detector **must not be used in a commercial
deployment** while it carries SBI / FaceForensics++-derived weights. Full
disclosure lives in ``MODEL_CARD_efficientnet_sbi.md`` alongside this file.

``is_production`` is ``False`` and stays ``False`` until ``scripts/evaluate.py``
clears an owner-agreed bar on a real labelled test set (M8 §10 / ACM 2.5). The
score is a probability, never a guarantee (ACM 1.3).

Privacy (ACM 1.6): the architecture is built with ``from_name`` (no
pretrained-weight download); weights load from a local file only; inference
runs fully in-process. No audio, video, frame or biometric data leaves the
machine. ``Frame.data`` is expected in BGR channel order (OpenCV convention —
as produced by ``api/media.py`` and ``scripts/evaluate.py``); the crop fed to
the network is converted to RGB, which SBI trained on.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Final

import numpy as np

from deepverify_pro.config import get_settings
from deepverify_pro.detection.base import DetectionResult, Detector, Frame, Modality
from deepverify_pro.detection.video.landmarks import NoFaceDetected

DETECTOR_NAME: Final[str] = "video-efficientnet-sbi-v0"

EFFICIENTNET_VARIANT: Final[str] = "efficientnet-b4"
INPUT_SIZE: Final[int] = 380  # EfficientNet-B4 native input; SBI trained at 380.
NUM_CLASSES: Final[int] = 2
FAKE_CLASS_INDEX: Final[int] = 1  # softmax index 1 = "fake" (SBI inference recipe).
# SBI test-phase face-box margin: w/4, then halved → 1/8 of the box per side
# (mapooon/SelfBlendedImages, src/inference/preprocess.py).
FACE_MARGIN: Final[float] = 0.125
# The SBI checkpoint stores the state_dict under this key; its keys are
# prefixed "net." because the SBI wrapper module holds the network as self.net.
CHECKPOINT_STATE_DICT_KEY: Final[str] = "model"
_STATE_DICT_PREFIX: Final[str] = "net."


class SBIDetectorError(ValueError):
    """Base class for EfficientNet-SBI detector failures."""


class SBIDependencyMissing(SBIDetectorError):
    """Raised when torch / efficientnet_pytorch / dlib / cv2 are not installed."""


class SBIWeightsUnavailable(SBIDetectorError):
    """Raised when the SBI checkpoint is missing or not a compatible checkpoint."""


_LOCK = threading.Lock()
_MODEL: Any = None
_MODEL_PATH: Path | None = None
_FACE_DETECTOR: Any = None


def _load_torch() -> Any:
    try:
        import torch  # noqa: PLC0415 — lazy import keeps the non-ML paths light.
    except ImportError as exc:  # pragma: no cover — covered by the install path.
        raise SBIDependencyMissing(
            "PyTorch is not installed. Install the M8 model extra: "
            "`pip install -e '.[video,video-model]'` (CODING_STANDARDS §3 M8 entry)."
        ) from exc
    return torch


def _load_efficientnet() -> Any:
    try:
        from efficientnet_pytorch import EfficientNet  # noqa: PLC0415 — lazy import.
    except ImportError as exc:  # pragma: no cover — covered by the install path.
        raise SBIDependencyMissing(
            "efficientnet_pytorch is not installed. Install the M8 model extra: "
            "`pip install -e '.[video,video-model]'`."
        ) from exc
    return EfficientNet


def _load_dlib() -> Any:
    try:
        import dlib  # noqa: PLC0415 — lazy import.
    except ImportError as exc:  # pragma: no cover — covered by the install path.
        raise SBIDependencyMissing(
            "dlib is not installed. Install the video extra: `pip install -e '.[video]'`."
        ) from exc
    return dlib


def _load_cv2() -> Any:
    try:
        import cv2  # noqa: PLC0415 — lazy import.
    except ImportError as exc:  # pragma: no cover — covered by the install path.
        raise SBIDependencyMissing(
            "OpenCV is not installed. Install the video extra: `pip install -e '.[video]'`."
        ) from exc
    return cv2


def _build_network(state_dict: dict[str, Any]) -> Any:
    """Build EfficientNet-B4 and load the (de-prefixed) SBI state_dict into it.

    SBI builds the architecture via ``efficientnet_pytorch`` (NOT ``timm`` — its
    checkpoint keys match only that library). ``from_name`` builds the network
    with no pretrained-weight download (ACM 1.6 — no runtime network).
    """
    efficientnet = _load_efficientnet()
    model = efficientnet.from_name(EFFICIENTNET_VARIANT, num_classes=NUM_CLASSES)
    # SBI saves from a wrapper module whose only child is ``net``, so every key
    # is prefixed "net." — strip it to load straight into the EfficientNet.
    stripped: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not key.startswith(_STATE_DICT_PREFIX):
            raise SBIWeightsUnavailable(
                f"unexpected state_dict key {key!r} (every key must start with "
                f"{_STATE_DICT_PREFIX!r}) — this is not an SBI EfficientNet-B4 checkpoint"
            )
        stripped[key[len(_STATE_DICT_PREFIX) :]] = value
    try:
        model.load_state_dict(stripped, strict=True)
    except (RuntimeError, KeyError) as exc:
        raise SBIWeightsUnavailable(
            f"checkpoint does not match the EfficientNet-B4 architecture: {exc}"
        ) from exc
    model.eval()
    return model


def _load_checkpoint(weights_path: Path) -> Any:
    """Load and validate the SBI checkpoint from a local file (no network)."""
    if not weights_path.exists():
        raise SBIWeightsUnavailable(
            f"SBI weights not found at {weights_path}. Obtain the research-only "
            "checkpoint and install it via `python scripts/fetch_sbi_weights.py` "
            "(see deepverify_pro/detection/video/MODEL_CARD_efficientnet_sbi.md)."
        )
    torch = _load_torch()
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch.load surfaces several error types for bad files.
        raise SBIWeightsUnavailable(f"could not read checkpoint {weights_path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or CHECKPOINT_STATE_DICT_KEY not in checkpoint:
        raise SBIWeightsUnavailable(
            f"checkpoint {weights_path} has no {CHECKPOINT_STATE_DICT_KEY!r} key — "
            "not an SBI checkpoint"
        )
    state_dict = checkpoint[CHECKPOINT_STATE_DICT_KEY]
    if not isinstance(state_dict, dict):
        raise SBIWeightsUnavailable(
            f"checkpoint {weights_path}[{CHECKPOINT_STATE_DICT_KEY!r}] is not a state_dict"
        )
    return _build_network(state_dict)


def verify_checkpoint(weights_path: Path) -> None:
    """Confirm a file is a loadable SBI EfficientNet-B4 checkpoint.

    Raises :class:`SBIDetectorError` (or a subclass) if not. Used by
    ``scripts/fetch_sbi_weights.py`` to validate weights at install time —
    the strongest integrity check available without a pinned SHA-256.
    """
    _load_checkpoint(weights_path)


def _ensure_model(weights_path: Path) -> Any:
    """Load and cache the SBI model for ``weights_path`` (thread-safe)."""
    global _MODEL, _MODEL_PATH  # noqa: PLW0603 — module-level cache.
    with _LOCK:
        if _MODEL is not None and _MODEL_PATH == weights_path:
            return _MODEL
        _MODEL = _load_checkpoint(weights_path)
        _MODEL_PATH = weights_path
        return _MODEL


def _ensure_face_detector() -> Any:
    """Load and cache dlib's HOG frontal-face detector (ships in the wheel)."""
    global _FACE_DETECTOR  # noqa: PLW0603 — module-level cache.
    with _LOCK:
        if _FACE_DETECTOR is None:
            _FACE_DETECTOR = _load_dlib().get_frontal_face_detector()
        return _FACE_DETECTOR


def _extract_face_crop(frame_data: np.ndarray, face_detector: Any) -> np.ndarray:
    """Detect the largest face, expand by the SBI margin, return a 380×380 RGB crop.

    ``frame_data`` is an HxWx3 uint8 image in BGR order. dlib's HOG detector is
    colour-agnostic; the returned crop is converted to RGB for the network.
    Raises :class:`NoFaceDetected` when no face is found.
    """
    cv2 = _load_cv2()
    if frame_data.ndim != 3 or frame_data.shape[2] != 3:
        raise SBIDetectorError(f"video frame must be HxWx3; got shape {frame_data.shape}")
    image = np.ascontiguousarray(frame_data)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    faces = face_detector(image, 0)
    if len(faces) == 0:
        raise NoFaceDetected("no face detected in frame")
    face = max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))

    height, width = image.shape[:2]
    margin_w = (face.right() - face.left()) * FACE_MARGIN
    margin_h = (face.bottom() - face.top()) * FACE_MARGIN
    x0 = max(0, int(round(face.left() - margin_w)))
    y0 = max(0, int(round(face.top() - margin_h)))
    x1 = min(width, int(round(face.right() + margin_w)))
    y1 = min(height, int(round(face.bottom() + margin_h)))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:  # pragma: no cover — margins are clamped inside the frame.
        raise NoFaceDetected("face bounding box collapsed to an empty crop")

    resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


def _infer(model: Any, crop_rgb: np.ndarray) -> float:
    """Run the network on a 380×380 RGB uint8 crop → fake-class probability."""
    torch = _load_torch()
    tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
    with torch.no_grad():
        logits = model(tensor)
        probability = logits.softmax(dim=1)[0, FAKE_CLASS_INDEX].item()
    return float(probability)


class EfficientNetSBIDetector(Detector):
    """EfficientNet-B4 / Self-Blended-Images deepfake detector (research-only weights).

    A clean adapter over the SBI network architecture and inference recipe
    (M8 §8). Weights load lazily from a local file on the first ``score()``
    call and are cached process-wide. Pure at score time: no network, no disk
    writes (ACM 1.6). ``is_production`` is ``False`` until ``scripts/evaluate.py``
    clears an owner-agreed bar on a real test set (M8 §10).

    Pass ``weights_path`` to override the configured ``Settings.sbi_weights_path``.
    """

    name: str = DETECTOR_NAME
    is_production: bool = False

    def __init__(self, weights_path: Path | None = None) -> None:
        self._weights_path: Path | None = weights_path

    def score(self, frame: Frame) -> DetectionResult:
        if frame.modality is not Modality.VIDEO:
            raise ValueError(f"{self.name} only scores VIDEO frames; got {frame.modality}")

        weights_path = self._weights_path or get_settings().sbi_weights_path
        model = _ensure_model(weights_path)
        face_detector = _ensure_face_detector()

        crop = _extract_face_crop(frame.data, face_detector)
        synthetic_probability = _infer(model, crop)

        detail: dict[str, Any] = {
            "model": EFFICIENTNET_VARIANT,
            "input_size": INPUT_SIZE,
            "weights": "self-blended-images (research-only)",
            "frame_index": frame.index,
        }
        return self._result(synthetic_probability, **detail)
