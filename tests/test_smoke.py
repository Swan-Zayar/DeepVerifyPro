"""M0 smoke tests: package imports, ABC contracts, Zoom stub, settings."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from deepverify_pro.adapters.base import MeetingAdapter
from deepverify_pro.adapters.zoom import ZoomAdapter
from deepverify_pro.config import get_settings
from deepverify_pro.detection.base import DetectionResult, Detector, Frame, Modality
from deepverify_pro.indicator import IndicatorState


def test_all_packages_import() -> None:
    for mod in (
        "deepverify_pro",
        "deepverify_pro.agents",
        "deepverify_pro.audit",
        "deepverify_pro.authorization",
        "deepverify_pro.cli",
        "deepverify_pro.config",
        "deepverify_pro.detection",
        "deepverify_pro.detection.audio",
        "deepverify_pro.detection.video",
        "deepverify_pro.indicator",
        "deepverify_pro.provenance",
        "deepverify_pro.tools",
    ):
        assert importlib.import_module(mod) is not None


def test_detector_is_abstract() -> None:
    with pytest.raises(TypeError):
        Detector()  # type: ignore[abstract]


def test_detector_subclass_result_helper() -> None:
    class Dummy(Detector):
        name = "dummy-baseline-v0"
        is_production = False

        def score(self, frame: Frame) -> DetectionResult:
            return self._result(1.5)  # clamps to 1.0

    frame = Frame(modality=Modality.AUDIO, data=np.zeros(4), sample_rate=16_000)
    result = Dummy().score(frame)
    assert result.synthetic_probability == 1.0
    assert result.indicator_state is IndicatorState.RED
    assert result.is_production is False


def test_zoom_adapter_is_documented_stub() -> None:
    assert issubclass(ZoomAdapter, MeetingAdapter)
    with pytest.raises(NotImplementedError):
        ZoomAdapter()


def test_detection_result_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        DetectionResult(
            synthetic_probability=2.0,
            indicator_state=IndicatorState.RED,
            detector_name="x",
            is_production=False,
        )


def test_settings_defaults() -> None:
    settings = get_settings()
    assert 0.0 <= settings.amber_at <= settings.red_at <= 1.0
    assert settings.financial_amount_threshold > 0
