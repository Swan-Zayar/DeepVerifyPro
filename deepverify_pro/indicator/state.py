"""Probabilistic colour-state model (green / amber / red).

Feature: F1 (Real-Time Audio Deepfake Detection — live colour indicator)
ACM: 1.3
Scope: in-product.md

The colour is a *probabilistic* indicator of synthetic likelihood, never an
absolute guarantee of authenticity (CODING_STANDARDS §4.2, ACM 1.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class IndicatorState(StrEnum):
    """Live indicator state. GREEN/AMBER/RED reflect *probability bands*, not certainty."""

    GREEN = "green"  # low synthetic probability
    AMBER = "amber"  # uncertain
    RED = "red"  # high synthetic probability


@dataclass(frozen=True)
class IndicatorThresholds:
    """Probability cut-points. ``>= amber_at`` → AMBER, ``>= red_at`` → RED."""

    amber_at: float = 0.40
    red_at: float = 0.70

    def __post_init__(self) -> None:
        if not 0.0 <= self.amber_at <= self.red_at <= 1.0:
            raise ValueError("require 0 <= amber_at <= red_at <= 1")


def classify(
    synthetic_probability: float,
    thresholds: IndicatorThresholds | None = None,
) -> IndicatorState:
    """Map a synthetic probability in ``[0, 1]`` to a probabilistic colour state."""
    if not 0.0 <= synthetic_probability <= 1.0:
        raise ValueError("synthetic_probability must be in [0, 1]")
    t = thresholds or IndicatorThresholds()
    if synthetic_probability >= t.red_at:
        return IndicatorState.RED
    if synthetic_probability >= t.amber_at:
        return IndicatorState.AMBER
    return IndicatorState.GREEN
