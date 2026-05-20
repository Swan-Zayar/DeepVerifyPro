"""Prototype baseline audio detector — documented heuristic over real MFCCs.

Feature: F1 (Real-Time Audio Deepfake Detection)
ACM: 1.2, 1.3, 1.6
Scope: in-product.md

Per CODING_STANDARDS §5 a baseline detector performs real feature extraction
(librosa MFCC, §3) and may use "a lightweight or documented heuristic" so long
as it is honest about being a baseline. ``is_production`` MUST stay ``False``
(§4.2 / ACM 1.3 / 2.5). The full disclosure of method, limits, and calibration
placeholders lives in ``MODEL_CARD.md`` alongside this file (§4.2).

The heuristic: real speech tends to show healthy temporal MFCC variance and
non-trivial delta-MFCC magnitudes; flat or static spectra mildly raise the
synthetic-probability estimate. The output is **deliberately bounded to a
narrow band around amber** (uncertain) so the prototype never overstates its
confidence — it exists to exercise the F1 pipeline end-to-end pending model
training in a future round.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from deepverify_pro.detection.audio.mfcc import (
    MFCCConfig,
    MFCCExtractorError,
    extract_mfcc,
)
from deepverify_pro.detection.base import DetectionResult, Detector, Frame, Modality

DETECTOR_NAME: Final[str] = "audio-mfcc-heuristic-baseline-v0"

# Documented calibration placeholders. These are NOT empirically validated —
# see MODEL_CARD.md (§4.2). Output stays inside [LOW_PROB, HIGH_PROB] so the
# baseline never claims strong confidence.
HEALTHY_TEMPORAL_STD: Final[float] = 30.0
HEALTHY_DELTA_MAG: Final[float] = 8.0
LOW_PROB: Final[float] = 0.30
HIGH_PROB: Final[float] = 0.70


class BaselineAudioDetector(Detector):
    """MFCC-stats heuristic. Pure: no network, no disk writes (§7, ACM 1.6).

    Pass a :class:`MFCCConfig` override only if downstream code needs a hop
    other than the product.md-specified 25 ms.
    """

    name: str = DETECTOR_NAME
    is_production: bool = False

    def __init__(self, mfcc_config: MFCCConfig | None = None) -> None:
        self._mfcc_config: MFCCConfig = mfcc_config or MFCCConfig()

    def score(self, frame: Frame) -> DetectionResult:
        if frame.modality is not Modality.AUDIO:
            raise ValueError(f"{self.name} only scores AUDIO frames; got {frame.modality}")
        if frame.sample_rate is None or frame.sample_rate <= 0:
            raise ValueError(f"{self.name} requires a positive sample_rate on the Frame")

        mfcc = extract_mfcc(frame.data, frame.sample_rate, self._mfcc_config)
        if mfcc.shape[1] < 2:
            raise MFCCExtractorError(
                "audio segment too short for delta-MFCC; provide at least two hops of audio"
            )

        temporal_std = float(np.mean(np.std(mfcc, axis=1)))
        delta = np.diff(mfcc, axis=1)
        delta_mag = float(np.mean(np.abs(delta)))

        synthetic_probability = _combine(temporal_std, delta_mag)
        detail: dict[str, Any] = {
            "n_frames": int(mfcc.shape[1]),
            "mean_temporal_std": round(temporal_std, 4),
            "mean_delta_magnitude": round(delta_mag, 4),
            "calibration": "placeholder",
        }
        return self._result(synthetic_probability, **detail)


def _component(value: float, healthy: float) -> float:
    """Map a non-negative feature to ``[0, 1]`` (low value → suspicious / synthetic-leaning)."""
    if healthy <= 0:
        raise ValueError("healthy anchor must be positive")
    clipped = max(0.0, value)
    return float(max(0.0, 1.0 - min(1.0, clipped / healthy)))


def _combine(temporal_std: float, delta_mag: float) -> float:
    """Average the two heuristic components and squash into ``[LOW_PROB, HIGH_PROB]``.

    The narrow output band is intentional — see MODEL_CARD.md (§4.2): the
    baseline must not claim strong synthetic / genuine confidence.
    """
    s_var = _component(temporal_std, HEALTHY_TEMPORAL_STD)
    s_delta = _component(delta_mag, HEALTHY_DELTA_MAG)
    blended = 0.5 * s_var + 0.5 * s_delta
    return LOW_PROB + (HIGH_PROB - LOW_PROB) * blended
