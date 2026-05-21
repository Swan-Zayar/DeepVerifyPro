"""68-point facial landmark extraction (product.md §3.3).

Feature: F2 (Live Video Face Authenticity Verification)
ACM: 1.3, 1.6
Scope: in-product.md

product.md §3.3 specifies tracking **68 facial landmark points**. This module
is the deterministic feature pipeline behind the F2 baseline detector
(CODING_STANDARDS §3 stack: ``dlib``). It is **pure**: no network, no disk
writes, no media leaves the process (ACM 1.6 / §7). The predictor weights are
loaded once from a configured local path; the fetch script
(``scripts/fetch_landmarks.py``) is the only sanctioned ingress for those
weights.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Final

import numpy as np

from deepverify_pro.config import get_settings

LANDMARK_COUNT: Final[int] = 68


class LandmarkExtractorError(ValueError):
    """Base class for landmark-extraction failures."""


class LandmarksUnavailable(LandmarkExtractorError):
    """Raised when the predictor weights file cannot be loaded."""


class NoFaceDetected(LandmarkExtractorError):
    """Raised when the face detector finds no face in a frame."""


_LOCK = threading.Lock()
_DETECTOR: Any = None
_PREDICTOR: Any = None
_PREDICTOR_PATH: Path | None = None


def _load_dlib() -> Any:
    try:
        import dlib  # noqa: PLC0415 — lazy import keeps the audio-only path light.
    except ImportError as exc:  # pragma: no cover — covered by install path.
        raise LandmarksUnavailable(
            "dlib is not installed. Install the video extra: "
            "`pip install -e '.[video]'` (CODING_STANDARDS §3)."
        ) from exc
    return dlib


def _ensure_loaded(predictor_path: Path) -> tuple[Any, Any]:
    """Load and cache the dlib face detector + 68-point predictor."""
    global _DETECTOR, _PREDICTOR, _PREDICTOR_PATH  # noqa: PLW0603 — module cache.
    with _LOCK:
        if _DETECTOR is not None and _PREDICTOR is not None and _PREDICTOR_PATH == predictor_path:
            return _DETECTOR, _PREDICTOR
        if not predictor_path.exists():
            raise LandmarksUnavailable(
                f"68-point predictor not found at {predictor_path}. "
                "Run `python scripts/fetch_landmarks.py` to fetch it (F2 §3 deviation)."
            )
        dlib = _load_dlib()
        _DETECTOR = dlib.get_frontal_face_detector()
        _PREDICTOR = dlib.shape_predictor(str(predictor_path))
        _PREDICTOR_PATH = predictor_path
        return _DETECTOR, _PREDICTOR


def _validate_frame(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        arr = image
    elif image.ndim == 2:
        arr = image
    else:
        raise LandmarkExtractorError(
            "image must be HxWx3 (RGB/BGR) or HxW (grayscale) uint8; "
            f"got shape={image.shape}, ndim={image.ndim}"
        )
    if image.size == 0:
        raise LandmarkExtractorError("image is empty")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def extract_landmarks(
    image: np.ndarray,
    predictor_path: Path | None = None,
) -> np.ndarray:
    """Detect one face and return its 68 landmarks as a ``(68, 2)`` float32 array.

    ``image`` is an HxWx3 uint8 array (RGB or BGR — dlib's HOG detector is
    colour-channel agnostic on 8-bit input). Raises :class:`NoFaceDetected`
    when no face is found, and :class:`LandmarksUnavailable` when the
    predictor weights file is missing. Pure: no network, no disk writes.

    If multiple faces are present, the largest bounding box is selected
    (typical speaker-view assumption — documented in MODEL_CARD.md).
    """
    arr = _validate_frame(image)
    path = predictor_path or get_settings().dlib_landmarks_path
    detector, predictor = _ensure_loaded(path)

    faces = detector(arr, 0)
    if len(faces) == 0:
        raise NoFaceDetected("no face detected in frame")
    face = max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
    shape = predictor(arr, face)

    out = np.empty((LANDMARK_COUNT, 2), dtype=np.float32)
    for i in range(LANDMARK_COUNT):
        pt = shape.part(i)
        out[i, 0] = float(pt.x)
        out[i, 1] = float(pt.y)
    return out
