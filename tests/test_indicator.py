"""Tests for the probabilistic colour indicator (F1, ACM 1.3)."""

from __future__ import annotations

import pytest

from deepverify_pro.indicator import IndicatorState, IndicatorThresholds, classify


def test_classify_bands_default_thresholds() -> None:
    assert classify(0.00) is IndicatorState.GREEN
    assert classify(0.39) is IndicatorState.GREEN
    assert classify(0.40) is IndicatorState.AMBER
    assert classify(0.69) is IndicatorState.AMBER
    assert classify(0.70) is IndicatorState.RED
    assert classify(1.00) is IndicatorState.RED


def test_classify_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        classify(-0.01)
    with pytest.raises(ValueError):
        classify(1.01)


def test_thresholds_validate_ordering() -> None:
    with pytest.raises(ValueError):
        IndicatorThresholds(amber_at=0.8, red_at=0.5)


def test_custom_thresholds() -> None:
    t = IndicatorThresholds(amber_at=0.2, red_at=0.9)
    assert classify(0.5, t) is IndicatorState.AMBER
    assert classify(0.95, t) is IndicatorState.RED
