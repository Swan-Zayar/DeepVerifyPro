"""The pluggable ``Detector`` contract (interface only — no models here).

Feature: F1 (Real-Time Audio Deepfake Detection), F2 (Live Video Face Authenticity Verification)
ACM: 1.2, 1.3, 1.6
Scope: in-product.md

The orchestrator depends only on this interface so detection models are
swappable without touching agents/tools (CODING_STANDARDS §5). Implementations
must be pure: no network, no disk writes (ACM 1.6).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from deepverify_pro.indicator.state import IndicatorState, classify


class Modality(StrEnum):
    """Media modality a :class:`Frame` carries."""

    AUDIO = "audio"
    VIDEO = "video"


@dataclass(frozen=True)
class Frame:
    """A unit of media handed to a :class:`Detector`.

    Raw media stays in-process and must never be logged or transmitted (ACM 1.6).
    """

    modality: Modality
    data: np.ndarray
    sample_rate: int | None = None  # required for AUDIO frames
    index: int = 0


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of scoring one :class:`Frame`.

    ``synthetic_probability`` is a probability, never a guarantee (ACM 1.3).
    """

    synthetic_probability: float
    indicator_state: IndicatorState
    detector_name: str
    is_production: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.synthetic_probability <= 1.0:
            raise ValueError("synthetic_probability must be in [0, 1]")


class Detector(ABC):
    """Base class for all detectors.

    ``is_production`` MUST stay ``False`` for prototype baselines (ACM 1.3 / 2.5).
    """

    name: str = "abstract-detector"
    is_production: bool = False

    @abstractmethod
    def score(self, frame: Frame) -> DetectionResult:
        """Return a :class:`DetectionResult`. Pure; no network; no disk writes."""
        raise NotImplementedError

    def _result(self, synthetic_probability: float, **detail: Any) -> DetectionResult:
        """Helper: clamp probability to ``[0, 1]`` and attach the colour state."""
        p = float(max(0.0, min(1.0, synthetic_probability)))
        return DetectionResult(
            synthetic_probability=p,
            indicator_state=classify(p),
            detector_name=self.name,
            is_production=self.is_production,
            detail=dict(detail),
        )
