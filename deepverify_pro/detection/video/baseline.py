"""Prototype baseline video detector — documented heuristic over real 68 landmarks.

Feature: F2 (Live Video Face Authenticity Verification)
ACM: 1.3, 1.6
Scope: in-product.md

Per CODING_STANDARDS §5 a baseline detector performs real feature extraction
(dlib 68-point landmarks, §3) and may use "a lightweight or documented
heuristic" so long as it is honest about being a baseline. ``is_production``
MUST stay ``False`` (§4.2 / ACM 1.3 / 2.5). Full disclosure of method, limits,
and calibration placeholders lives in ``MODEL_CARD.md`` alongside this file.

The heuristic operates on two single-frame geometric signals:
  1. **Bilateral symmetry deviation** — mean Euclidean distance between
     mirror-pair landmarks reflected across the facial midline, normalised by
     face width. Real faces show natural asymmetry; over-symmetric output
     mildly raises the synthetic-probability estimate.
  2. **Inter-ocular ratio sanity** — distance between eye centres divided by
     face width. Healthy frontal faces sit near a canonical ratio; large
     deviations mildly raise the estimate.

Output is **deliberately bounded** to a narrow band around amber so the
prototype never overstates its confidence — exists to exercise the F2
pipeline end-to-end pending a trained model in a future round.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np

from deepverify_pro.detection.base import DetectionResult, Detector, Frame, Modality
from deepverify_pro.detection.video.landmarks import (
    LANDMARK_COUNT,
    LandmarkExtractorError,
    extract_landmarks,
)

DETECTOR_NAME: Final[str] = "video-landmark-heuristic-baseline-v0"

# Documented calibration placeholders — NOT empirically validated; see
# MODEL_CARD.md (§4.2). Output stays inside [LOW_PROB, HIGH_PROB] so the
# baseline never claims strong confidence.
HEALTHY_SYM_DEVIATION: Final[float] = 0.04  # ~4% of face width
HEALTHY_IOR: Final[float] = 0.40  # canonical frontal inter-ocular ratio
IOR_TOLERANCE: Final[float] = 0.10
LOW_PROB: Final[float] = 0.30
HIGH_PROB: Final[float] = 0.70

# dlib 68-landmark mirror pairs across the facial midline (zero-indexed).
# Source: the canonical Davis King layout — eight jaw pairs, five eyebrow
# pairs, two nose-tip pairs, six eye pairs, eight mouth pairs.
_MIRROR_PAIRS: Final[tuple[tuple[int, int], ...]] = (
    (0, 16),
    (1, 15),
    (2, 14),
    (3, 13),
    (4, 12),
    (5, 11),
    (6, 10),
    (7, 9),
    (17, 26),
    (18, 25),
    (19, 24),
    (20, 23),
    (21, 22),
    (31, 35),
    (32, 34),
    (36, 45),
    (37, 44),
    (38, 43),
    (39, 42),
    (40, 47),
    (41, 46),
    (48, 54),
    (49, 53),
    (50, 52),
    (60, 64),
    (61, 63),
    (55, 59),
    (56, 58),
    (65, 67),
)
_MIDLINE_TOP: Final[int] = 27  # bridge of nose
_MIDLINE_BOTTOM: Final[int] = 8  # chin tip
_JAW_LEFT: Final[int] = 0
_JAW_RIGHT: Final[int] = 16
_LEFT_EYE_SLICE: Final[slice] = slice(36, 42)
_RIGHT_EYE_SLICE: Final[slice] = slice(42, 48)


class BaselineVideoDetector(Detector):
    """68-landmark geometric heuristic. Pure: no network, no disk writes (§7, ACM 1.6).

    Pass ``predictor_path`` to override the configured weights location
    (defaults to ``Settings.dlib_landmarks_path``).
    """

    name: str = DETECTOR_NAME
    is_production: bool = False

    def __init__(self, predictor_path: Path | None = None) -> None:
        self._predictor_path: Path | None = predictor_path

    def score(self, frame: Frame) -> DetectionResult:
        if frame.modality is not Modality.VIDEO:
            raise ValueError(f"{self.name} only scores VIDEO frames; got {frame.modality}")

        landmarks = extract_landmarks(frame.data, self._predictor_path)
        if landmarks.shape != (LANDMARK_COUNT, 2):  # pragma: no cover — extractor guarantees shape
            raise LandmarkExtractorError(
                f"expected ({LANDMARK_COUNT}, 2) landmarks; got {landmarks.shape}"
            )

        sym_deviation = _symmetry_deviation(landmarks)
        ior = _inter_ocular_ratio(landmarks)
        synthetic_probability = _combine(sym_deviation, ior)

        detail: dict[str, Any] = {
            "n_landmarks": LANDMARK_COUNT,
            "symmetry_deviation": round(sym_deviation, 4),
            "inter_ocular_ratio": round(ior, 4),
            "calibration": "placeholder",
        }
        return self._result(synthetic_probability, **detail)


def _face_width(landmarks: np.ndarray) -> float:
    """Distance between outer jaw landmarks (0 and 16) — denominator floor at 1px."""
    width = float(np.linalg.norm(landmarks[_JAW_RIGHT] - landmarks[_JAW_LEFT]))
    return max(width, 1.0)


def _reflect(point: np.ndarray, axis_a: np.ndarray, axis_b: np.ndarray) -> np.ndarray:
    """Reflect ``point`` across the line through ``axis_a`` → ``axis_b``."""
    direction = axis_b - axis_a
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:  # pragma: no cover — degenerate midline impossible with real landmarks
        return point.copy()
    unit = direction / norm
    relative = point - axis_a
    projection = float(np.dot(relative, unit))
    foot = axis_a + projection * unit
    return 2.0 * foot - point


def _symmetry_deviation(landmarks: np.ndarray) -> float:
    """Mean reflected-pair distance normalised by face width (lower ≈ more symmetric)."""
    a = landmarks[_MIDLINE_TOP]
    b = landmarks[_MIDLINE_BOTTOM]
    distances = []
    for left_idx, right_idx in _MIRROR_PAIRS:
        left = landmarks[left_idx]
        right_reflected = _reflect(landmarks[right_idx], a, b)
        distances.append(float(np.linalg.norm(left - right_reflected)))
    return float(np.mean(distances)) / _face_width(landmarks)


def _inter_ocular_ratio(landmarks: np.ndarray) -> float:
    """Distance between eye centres divided by face width."""
    left_centre = landmarks[_LEFT_EYE_SLICE].mean(axis=0)
    right_centre = landmarks[_RIGHT_EYE_SLICE].mean(axis=0)
    inter = float(np.linalg.norm(left_centre - right_centre))
    return inter / _face_width(landmarks)


def _symmetry_component(value: float) -> float:
    """Map symmetry deviation → ``[0, 1]`` (small value → suspiciously symmetric)."""
    return float(max(0.0, 1.0 - min(1.0, max(0.0, value) / HEALTHY_SYM_DEVIATION)))


def _ior_component(value: float) -> float:
    """Map inter-ocular ratio → ``[0, 1]`` (far from canonical → suspicious)."""
    deviation = abs(value - HEALTHY_IOR)
    return float(min(1.0, deviation / IOR_TOLERANCE))


def _combine(sym_deviation: float, ior: float) -> float:
    """Average the two heuristic components, squash into ``[LOW_PROB, HIGH_PROB]``.

    The narrow band is intentional — see MODEL_CARD.md (§4.2): the baseline
    must not claim strong synthetic / genuine confidence.
    """
    s_sym = _symmetry_component(sym_deviation)
    s_ior = _ior_component(ior)
    blended = 0.5 * s_sym + 0.5 * s_ior
    return LOW_PROB + (HIGH_PROB - LOW_PROB) * blended
