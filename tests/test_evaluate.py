"""M8 evaluation-harness tests — metrics, manifest parsing, end-to-end scoring.

Feature: F1 (Real-Time Audio Deepfake Detection), F2 (Live Video Face Authenticity Verification)
ACM: 1.3, 2.5
Scope: in-product.md

The metric functions (ROC-AUC, EER, confusion matrix) are pure numpy and are
exercised against hand-computed known answers. The end-to-end ``evaluate`` path
is covered with a scripted stub detector so it needs no media files; the real
dlib path is additionally smoke-tested and skips cleanly when the predictor
weights are absent (the repo never commits the ~99 MB file).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from deepverify_pro.config import get_settings
from deepverify_pro.detection.base import DetectionResult, Detector, Frame, Modality
from scripts import evaluate as evaluate_mod
from scripts.evaluate import (
    ConfusionMatrix,
    Sample,
    SampleLoadError,
    confusion_at,
    equal_error_rate,
    evaluate,
    format_report,
    load_manifest,
    report_to_dict,
    roc_auc,
    score_sample,
)

# ---------- ROC-AUC ----------


def test_roc_auc_perfect_separation() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_perfect_inversion() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(0.0)


def test_roc_auc_all_ties_is_chance() -> None:
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(0.5)


def test_roc_auc_known_value() -> None:
    # fakes {0.35, 0.8} vs reals {0.1, 0.4}: 3 of 4 fake/real pairs correct.
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(0.75)


def test_roc_auc_undefined_for_single_class() -> None:
    scores = np.array([0.2, 0.6, 0.9])
    labels = np.array([0, 0, 0])
    assert math.isnan(roc_auc(scores, labels))


# ---------- Equal Error Rate ----------


def test_eer_zero_for_perfect_separation() -> None:
    scores = np.array([0.2, 0.3, 0.7, 0.8])
    labels = np.array([0, 0, 1, 1])
    eer, _ = equal_error_rate(scores, labels)
    assert eer == pytest.approx(0.0)


def test_eer_half_for_complete_overlap() -> None:
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 0, 1, 1])
    eer, _ = equal_error_rate(scores, labels)
    assert eer == pytest.approx(0.5)


def test_eer_bounded_for_partial_overlap() -> None:
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    labels = np.array([0, 0, 1, 1])
    eer, threshold = equal_error_rate(scores, labels)
    assert 0.0 <= eer <= 0.5
    assert math.isfinite(threshold)


def test_eer_undefined_for_single_class() -> None:
    eer, threshold = equal_error_rate(np.array([0.3, 0.7]), np.array([1, 1]))
    assert math.isnan(eer) and math.isnan(threshold)


# ---------- Confusion matrix ----------


def test_confusion_at_counts() -> None:
    scores = np.array([0.2, 0.5, 0.8, 0.9])
    labels = np.array([0, 1, 1, 0])
    matrix = confusion_at(scores, labels, threshold=0.6)
    assert (matrix.true_positive, matrix.false_negative) == (1, 1)
    assert (matrix.false_positive, matrix.true_negative) == (1, 1)
    assert matrix.total == 4


def test_confusion_derived_metrics() -> None:
    scores = np.array([0.2, 0.5, 0.8, 0.9])
    labels = np.array([0, 1, 1, 0])
    matrix = confusion_at(scores, labels, threshold=0.6)
    assert matrix.precision == pytest.approx(0.5)
    assert matrix.recall == pytest.approx(0.5)
    assert matrix.accuracy == pytest.approx(0.5)


def test_confusion_precision_undefined_when_nothing_flagged() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 0, 1, 1])
    matrix = confusion_at(scores, labels, threshold=0.99)
    assert matrix.true_positive == 0 and matrix.false_positive == 0
    assert math.isnan(matrix.precision)
    assert matrix.recall == pytest.approx(0.0)


# ---------- Manifest parsing ----------


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_manifest_parses_and_resolves_relative_paths(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "manifest.jsonl",
        '{"path": "real/a.mp4", "label": "real"}\n' '{"path": "fake/b.mp4", "label": "fake"}\n',
    )
    samples = load_manifest(manifest)
    assert [s.label for s in samples] == [0, 1]
    assert samples[0].path == tmp_path / "real" / "a.mp4"
    assert samples[1].path == tmp_path / "fake" / "b.mp4"


def test_load_manifest_keeps_absolute_paths(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere" / "clip.mp4"
    manifest = _write(
        tmp_path / "m.jsonl", json.dumps({"path": str(target), "label": "fake"}) + "\n"
    )
    assert load_manifest(manifest)[0].path == target


def test_load_manifest_accepts_label_synonyms(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "m.jsonl",
        '{"path": "a", "label": "genuine"}\n'
        '{"path": "b", "label": "synthetic"}\n'
        '{"path": "c", "label": "deepfake"}\n',
    )
    assert [s.label for s in load_manifest(manifest)] == [0, 1, 1]


def test_load_manifest_skips_blanks_and_comments(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "m.jsonl",
        "# a header comment\n" "\n" '{"path": "a", "label": "real"}\n' "   \n",
    )
    assert len(load_manifest(manifest)) == 1


def test_load_manifest_rejects_unknown_label(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "m.jsonl", '{"path": "a", "label": "maybe"}\n')
    with pytest.raises(SystemExit):
        load_manifest(manifest)


def test_load_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "m.jsonl", "{not json}\n")
    with pytest.raises(SystemExit):
        load_manifest(manifest)


def test_load_manifest_rejects_missing_fields(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "m.jsonl", '{"path": "a"}\n')
    with pytest.raises(SystemExit):
        load_manifest(manifest)


def test_load_manifest_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        load_manifest(tmp_path / "does_not_exist.jsonl")


def test_load_manifest_rejects_empty(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "m.jsonl", "# only comments\n")
    with pytest.raises(SystemExit):
        load_manifest(manifest)


# ---------- evaluate / score_sample (scripted stub detectors) ----------


class _ScriptedDetector(Detector):
    """Returns ``frame.data[0]`` as the synthetic probability — for tests only."""

    name = "scripted-test-detector"
    is_production = False

    def score(self, frame: Frame) -> DetectionResult:
        return self._result(float(np.asarray(frame.data).flat[0]))


class _AlwaysFailDetector(Detector):
    """Raises on every frame — exercises the unscorable-sample path."""

    name = "always-fail-test-detector"
    is_production = False

    def score(self, frame: Frame) -> DetectionResult:
        raise RuntimeError("boom")


class _FlakyDetector(Detector):
    """Fails on frame index 0, succeeds otherwise — exercises partial scoring."""

    name = "flaky-test-detector"
    is_production = False

    def score(self, frame: Frame) -> DetectionResult:
        if frame.index == 0:
            raise RuntimeError("no face in this frame")
        return self._result(0.6)


def _frame(value: float, index: int = 0) -> Frame:
    return Frame(modality=Modality.VIDEO, data=np.array([value], dtype=np.float64), index=index)


def test_evaluate_produces_expected_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    scores_by_name = {"a": 0.1, "b": 0.4, "c": 0.35, "d": 0.8}

    def fake_frames(sample: Sample, modality: Modality, frames_per_clip: int) -> list[Frame]:
        return [_frame(scores_by_name[sample.path.name])]

    monkeypatch.setattr(evaluate_mod, "frames_for_sample", fake_frames)
    samples = [
        Sample(Path("a"), 0),
        Sample(Path("b"), 0),
        Sample(Path("c"), 1),
        Sample(Path("d"), 1),
    ]
    report = evaluate(_ScriptedDetector(), samples, Modality.VIDEO, frames_per_clip=1)

    assert report.n_total == 4
    assert report.n_scored == 4 and report.n_errored == 0
    assert (report.n_real, report.n_fake) == (2, 2)
    assert report.roc_auc == pytest.approx(0.75)
    assert report.detector_name == "scripted-test-detector"
    assert report.is_production is False
    # Confusion is reported at the two indicator thresholds (0.40, 0.70).
    assert [m.threshold for m in report.confusion] == [pytest.approx(0.40), pytest.approx(0.70)]


def test_evaluate_counts_errored_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(sample: Sample, modality: Modality, frames_per_clip: int) -> list[Frame]:
        raise SampleLoadError("unreadable")

    monkeypatch.setattr(evaluate_mod, "frames_for_sample", boom)
    report = evaluate(
        _ScriptedDetector(), [Sample(Path("x"), 1)], Modality.VIDEO, frames_per_clip=1
    )
    assert report.n_scored == 0 and report.n_errored == 1
    assert math.isnan(report.roc_auc)


def test_score_sample_reports_missing_file() -> None:
    result = score_sample(
        _ScriptedDetector(), Sample(Path("/no/such/file.png"), 1), Modality.VIDEO, 1
    )
    assert result.score is None
    assert result.frames_scored == 0
    assert result.error is not None


def test_score_sample_all_frames_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluate_mod,
        "frames_for_sample",
        lambda sample, modality, fpc: [_frame(0.0, 0), _frame(0.0, 1)],
    )
    result = score_sample(_AlwaysFailDetector(), Sample(Path("x"), 1), Modality.VIDEO, 2)
    assert result.score is None
    assert result.error is not None and "boom" in result.error


def test_score_sample_partial_frame_failure_still_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluate_mod,
        "frames_for_sample",
        lambda sample, modality, fpc: [_frame(0.0, 0), _frame(0.0, 1), _frame(0.0, 2)],
    )
    result = score_sample(_FlakyDetector(), Sample(Path("x"), 1), Modality.VIDEO, 3)
    # Frame 0 fails; frames 1 and 2 score 0.6 each — the clip is not lost.
    assert result.score == pytest.approx(0.6)
    assert result.frames_scored == 2


def test_report_serialisation_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate_mod, "frames_for_sample", lambda s, m, f: [_frame(0.55)])
    report = evaluate(
        _ScriptedDetector(),
        [Sample(Path("a"), 0), Sample(Path("b"), 1)],
        Modality.VIDEO,
        frames_per_clip=1,
    )
    payload = report_to_dict(report)
    # JSON-serialisable, and carries no raw-media keys (ACM 1.6).
    text = json.dumps(payload)
    assert '"detector_name"' in text
    for forbidden in ("data", "pixels", "landmarks", "biometric", "frame_data"):
        assert forbidden not in payload
    assert isinstance(format_report(report), str)
    assert "measures; it does not assert" in format_report(report)


def test_confusion_matrix_is_immutable() -> None:
    matrix = ConfusionMatrix(0.5, 1, 1, 1, 1)
    with pytest.raises(AttributeError):
        matrix.threshold = 0.9  # type: ignore[misc]


# ---------- real dlib end-to-end (skips when the predictor is absent) ----------


def _dlib_available() -> bool:
    try:
        import dlib  # noqa: F401
    except ImportError:
        return False
    return True


def _cv2_available() -> bool:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


_PREDICTOR_PATH = get_settings().dlib_landmarks_path
_NEEDS_VIDEO_STACK = pytest.mark.skipif(
    not _PREDICTOR_PATH.exists() or not _dlib_available() or not _cv2_available(),
    reason="real end-to-end needs the dlib predictor weights, dlib, and OpenCV",
)


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


@_NEEDS_VIDEO_STACK
def test_evaluate_end_to_end_video_baseline(tmp_path: Path) -> None:
    import cv2

    from deepverify_pro.detection.video import BaselineVideoDetector

    for name in ("face_a.png", "face_b.png"):
        cv2.imwrite(str(tmp_path / name), _procedural_face())
    manifest = _write(
        tmp_path / "manifest.jsonl",
        '{"path": "face_a.png", "label": "real"}\n' '{"path": "face_b.png", "label": "real"}\n',
    )
    samples = load_manifest(manifest)
    report = evaluate(BaselineVideoDetector(), samples, Modality.VIDEO, frames_per_clip=4)

    assert report.n_scored == 2
    assert report.detector_name == "video-landmark-heuristic-baseline-v0"
    assert report.is_production is False
    # Both samples are the same class — ROC-AUC is genuinely undefined, not faked.
    assert math.isnan(report.roc_auc)
