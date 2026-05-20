"""Colour-coded confidence indicator package.

Feature: F1 (Real-Time Audio Deepfake Detection — colour indicator)
ACM: 1.3
Scope: in-product.md
"""

from deepverify_pro.indicator.state import (
    IndicatorState,
    IndicatorThresholds,
    classify,
)

__all__ = ["IndicatorState", "IndicatorThresholds", "classify"]
