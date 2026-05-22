"""Evaluation harness — measure a Detector against a labelled test set.

Feature: F1 (Real-Time Audio Deepfake Detection), F2 (Live Video Face Authenticity Verification)
ACM: 1.3, 2.5
Scope: in-product.md

M8 item E. Runs a :class:`Detector` over a labelled test set and reports
ROC-AUC, EER, and the confusion matrix at the indicator thresholds. It adds no
dependency — pure ``numpy`` + stdlib for the maths; ``cv2`` / ``soundfile`` are
lazy-imported only when a media file of that modality is actually loaded (the
same optional-extra pattern the detectors use).

This harness **measures; it never asserts**. Every number it prints comes from
the test set you give it (CODING_STANDARDS §4.2 — zero fabricated metrics). It
does **not** flip any detector's ``is_production`` flag — that stays a reviewed
human decision made after reading a measured number (M8 §10).

Privacy (ACM 1.6): reads local files only, scores in-process, no network. The
optional JSON report holds paths, labels, scalar scores and metrics only —
never raw media or biometric vectors.

Manifest format — one JSON object per line (JSONL); blank lines and lines
starting with ``#`` are ignored::

    {"path": "real/clip01.mp4", "label": "real"}
    {"path": "fake/clip02.mp4", "label": "fake"}

``path`` is resolved relative to the manifest file's own directory. ``label``
is ``real`` (genuine — expected low synthetic probability) or
``fake``/``synthetic`` (deepfake — expected high). Samples that cannot be
scored (no face, unreadable file) are counted and excluded from the metrics,
never silently dropped.

Frame sampling is deterministic (evenly spaced) — there is no randomness, so
runs are reproducible without a seed.

Usage::

    python scripts/evaluate.py --manifest path/to/manifest.jsonl \\
        --detector video-baseline [--frames-per-clip 16] [--json-out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from deepverify_pro.detection.audio import BaselineAudioDetector
from deepverify_pro.detection.base import Detector, Frame, Modality
from deepverify_pro.detection.video import BaselineVideoDetector, EfficientNetSBIDetector
from deepverify_pro.indicator.state import IndicatorThresholds

# Label vocabulary → integer class. 1 = fake (the positive class), 0 = real.
_LABEL_MAP: Final[dict[str, int]] = {
    "real": 0,
    "genuine": 0,
    "authentic": 0,
    "fake": 1,
    "synthetic": 1,
    "deepfake": 1,
}

_IMAGE_EXT: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
_VIDEO_EXT: Final[frozenset[str]] = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})
_AUDIO_EXT: Final[frozenset[str]] = frozenset({".wav", ".flac", ".ogg", ".mp3", ".m4a"})

DEFAULT_FRAMES_PER_CLIP: Final[int] = 16


# --------------------------------------------------------------------------
# Metrics — pure numpy, no dependency. Public so tests exercise them directly.
# --------------------------------------------------------------------------


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank ``values`` ascending, 1-based, assigning average ranks to ties."""
    n = values.size
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        # Average of the 1-based positions i+1 .. j+1 spanned by this tie group.
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve. ``labels``: 1 = fake (positive), 0 = real.

    Computed via the Mann-Whitney rank-sum identity — exact, with ties handled
    by average ranks. Returns ``nan`` when either class is empty (the metric is
    genuinely undefined; it is never faked — ACM 1.3).
    """
    score_arr = np.asarray(scores, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int64)
    n_pos = int(np.sum(label_arr == 1))
    n_neg = int(np.sum(label_arr == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(score_arr)
    rank_sum_pos = float(np.sum(ranks[label_arr == 1]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate and the threshold achieving it.

    A sample is predicted ``fake`` when ``score >= threshold``. ``FPR`` is the
    fraction of real samples flagged as fake; ``FNR`` is the fraction of fake
    samples missed. The EER is reported at the threshold minimising
    ``|FPR - FNR|``, as the mean of the two rates there. Returns
    ``(nan, nan)`` when either class is empty.
    """
    score_arr = np.asarray(scores, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int64)
    fake = score_arr[label_arr == 1]
    real = score_arr[label_arr == 0]
    if fake.size == 0 or real.size == 0:
        return float("nan"), float("nan")
    # Candidate thresholds: every observed score, plus one strictly above the
    # maximum so FPR can reach 0 (predict nothing as fake).
    candidates = np.unique(score_arr)
    candidates = np.append(candidates, np.nextafter(candidates[-1], np.inf))
    best_gap = np.inf
    best_eer = float("nan")
    best_threshold = float("nan")
    for threshold in candidates:
        fpr = float(np.mean(real >= threshold))
        fnr = float(np.mean(fake < threshold))
        gap = abs(fpr - fnr)
        if gap < best_gap:
            best_gap = gap
            best_eer = (fpr + fnr) / 2.0
            best_threshold = float(threshold)
    return best_eer, best_threshold


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Ratio, or ``nan`` when the denominator is zero (undefined, not faked)."""
    return numerator / denominator if denominator else float("nan")


@dataclass(frozen=True)
class ConfusionMatrix:
    """A 2x2 confusion matrix at one decision threshold (positive = fake)."""

    threshold: float
    true_positive: int  # fake, correctly flagged
    false_negative: int  # fake, missed
    false_positive: int  # real, wrongly flagged
    true_negative: int  # real, correctly cleared

    @property
    def total(self) -> int:
        return self.true_positive + self.false_negative + self.false_positive + self.true_negative

    @property
    def precision(self) -> float:
        """Of the samples flagged fake, the fraction that truly were."""
        return _safe_ratio(self.true_positive, self.true_positive + self.false_positive)

    @property
    def recall(self) -> float:
        """Of the truly-fake samples, the fraction caught."""
        return _safe_ratio(self.true_positive, self.true_positive + self.false_negative)

    @property
    def accuracy(self) -> float:
        return _safe_ratio(self.true_positive + self.true_negative, self.total)


def confusion_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> ConfusionMatrix:
    """Confusion matrix predicting ``fake`` when ``score >= threshold``."""
    score_arr = np.asarray(scores, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int64)
    predicted_fake = score_arr >= threshold
    is_fake = label_arr == 1
    return ConfusionMatrix(
        threshold=float(threshold),
        true_positive=int(np.sum(predicted_fake & is_fake)),
        false_negative=int(np.sum(~predicted_fake & is_fake)),
        false_positive=int(np.sum(predicted_fake & ~is_fake)),
        true_negative=int(np.sum(~predicted_fake & ~is_fake)),
    )


# --------------------------------------------------------------------------
# Test-set manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One labelled test-set entry: a media path and its ground-truth class."""

    path: Path
    label: int  # 1 = fake, 0 = real


class SampleLoadError(RuntimeError):
    """Raised when a sample's media file cannot be read or decoded."""


def load_manifest(manifest_path: Path) -> list[Sample]:
    """Parse a JSONL manifest into :class:`Sample` entries (paths resolved)."""
    if not manifest_path.exists():
        raise SystemExit(
            f"manifest not found: {manifest_path}\n"
            "Expected a JSONL file, one object per line: "
            '{"path": "...", "label": "real|fake"}'
        )
    base = manifest_path.resolve().parent
    samples: list[Sample] = []
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{manifest_path}:{lineno}: invalid JSON — {exc}") from exc
        if not isinstance(obj, dict) or "path" not in obj or "label" not in obj:
            raise SystemExit(
                f"{manifest_path}:{lineno}: each line needs a 'path' and a 'label' field"
            )
        label_key = str(obj["label"]).strip().lower()
        if label_key not in _LABEL_MAP:
            raise SystemExit(
                f"{manifest_path}:{lineno}: label {obj['label']!r} is not one of "
                f"{sorted(_LABEL_MAP)}"
            )
        path = Path(str(obj["path"]))
        if not path.is_absolute():
            path = base / path
        samples.append(Sample(path=path, label=_LABEL_MAP[label_key]))
    if not samples:
        raise SystemExit(f"manifest {manifest_path} contains no samples")
    return samples


# --------------------------------------------------------------------------
# Frame loading — lazy media imports keep the metrics path dependency-free.
# --------------------------------------------------------------------------


def _load_cv2() -> Any:
    try:
        import cv2  # noqa: PLC0415 — lazy: only video samples need OpenCV.
    except ImportError as exc:  # pragma: no cover — covered by install path.
        raise SystemExit(
            "OpenCV is not installed. Install the video extra: pip install -e '.[video]'"
        ) from exc
    return cv2


def _load_soundfile() -> Any:
    try:
        import soundfile  # noqa: PLC0415 — lazy: only audio samples need soundfile.
    except ImportError as exc:  # pragma: no cover — covered by install path.
        raise SystemExit(
            "soundfile is not installed. Install the audio extra: pip install -e '.[audio]'"
        ) from exc
    return soundfile


def _image_frame(path: Path) -> Frame:
    cv2 = _load_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SampleLoadError(f"could not read image: {path}")
    return Frame(modality=Modality.VIDEO, data=np.ascontiguousarray(image), index=0)


def _video_frames(path: Path, frames_per_clip: int) -> list[Frame]:
    cv2 = _load_cv2()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise SampleLoadError(f"could not open video: {path}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise SampleLoadError(f"video reports no frames: {path}")
        count = min(frames_per_clip, total)
        indices = np.linspace(0, total - 1, num=count, dtype=int)
        frames: list[Frame] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, image = capture.read()
            if ok and image is not None:
                frames.append(
                    Frame(
                        modality=Modality.VIDEO,
                        data=np.ascontiguousarray(image),
                        index=int(index),
                    )
                )
        if not frames:
            raise SampleLoadError(f"could not decode any frame from: {path}")
        return frames
    finally:
        capture.release()


def _audio_frame(path: Path) -> Frame:
    soundfile = _load_soundfile()
    try:
        data, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:  # soundfile surfaces several error types for bad files.
        raise SampleLoadError(f"could not read audio: {path} — {exc}") from exc
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 2:
        # Down-mix to mono by averaging channels — an explicit, documented
        # choice (the audio detector rejects multi-channel input upstream).
        array = array.mean(axis=1).astype(np.float32)
    return Frame(
        modality=Modality.AUDIO,
        data=np.ascontiguousarray(array),
        sample_rate=int(sample_rate),
        index=0,
    )


def frames_for_sample(sample: Sample, modality: Modality, frames_per_clip: int) -> list[Frame]:
    """Decode a sample into one or more :class:`Frame` objects for ``modality``."""
    if not sample.path.exists():
        raise SampleLoadError(f"file not found: {sample.path}")
    suffix = sample.path.suffix.lower()
    if modality is Modality.VIDEO:
        if suffix in _IMAGE_EXT:
            return [_image_frame(sample.path)]
        if suffix in _VIDEO_EXT:
            return _video_frames(sample.path, frames_per_clip)
        raise SampleLoadError(f"unsupported video sample extension {suffix!r}: {sample.path}")
    if suffix in _AUDIO_EXT:
        return [_audio_frame(sample.path)]
    raise SampleLoadError(f"unsupported audio sample extension {suffix!r}: {sample.path}")


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleResult:
    """The outcome of scoring one test-set sample."""

    path: str
    label: int
    score: float | None  # mean synthetic_probability; None when unscorable
    frames_scored: int
    error: str | None


@dataclass(frozen=True)
class EvalReport:
    """A full evaluation run — counts, metrics, and per-sample detail."""

    detector_name: str
    is_production: bool
    modality: str
    frames_per_clip: int
    n_total: int
    n_scored: int
    n_errored: int
    n_real: int
    n_fake: int
    roc_auc: float
    eer: float
    eer_threshold: float
    confusion: list[ConfusionMatrix]
    samples: list[SampleResult]


def score_sample(
    detector: Detector,
    sample: Sample,
    modality: Modality,
    frames_per_clip: int,
) -> SampleResult:
    """Score one sample: mean ``synthetic_probability`` over its frames.

    A per-frame failure (e.g. no face in one frame of a clip) is recorded and
    skipped, never silently swallowed — it surfaces in ``frames_scored`` and,
    if no frame scores at all, in ``error`` (CODING_STANDARDS §7).
    """
    try:
        frames = frames_for_sample(sample, modality, frames_per_clip)
    except SampleLoadError as exc:
        return SampleResult(str(sample.path), sample.label, None, 0, str(exc))

    probabilities: list[float] = []
    last_error: str | None = None
    for frame in frames:
        try:
            result = detector.score(frame)
        except Exception as exc:  # per-frame robustness — recorded, not swallowed.
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        probabilities.append(result.synthetic_probability)

    if not probabilities:
        return SampleResult(
            str(sample.path),
            sample.label,
            None,
            0,
            last_error or "no frame could be scored",
        )
    return SampleResult(
        str(sample.path),
        sample.label,
        float(np.mean(probabilities)),
        len(probabilities),
        None,
    )


def evaluate(
    detector: Detector,
    samples: list[Sample],
    modality: Modality,
    frames_per_clip: int,
) -> EvalReport:
    """Run ``detector`` over ``samples`` and assemble an :class:`EvalReport`."""
    results = [score_sample(detector, sample, modality, frames_per_clip) for sample in samples]
    scored = [r for r in results if r.score is not None]
    scores = np.array([r.score for r in scored], dtype=np.float64)
    labels = np.array([r.label for r in scored], dtype=np.int64)

    thresholds = IndicatorThresholds()
    eer, eer_threshold = equal_error_rate(scores, labels)
    confusion = [
        confusion_at(scores, labels, thresholds.amber_at),
        confusion_at(scores, labels, thresholds.red_at),
    ]
    return EvalReport(
        detector_name=detector.name,
        is_production=detector.is_production,
        modality=modality.value,
        frames_per_clip=frames_per_clip,
        n_total=len(samples),
        n_scored=len(scored),
        n_errored=len(samples) - len(scored),
        n_real=int(np.sum(labels == 0)),
        n_fake=int(np.sum(labels == 1)),
        roc_auc=roc_auc(scores, labels),
        eer=eer,
        eer_threshold=eer_threshold,
        confusion=confusion,
        samples=results,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _fmt(value: float) -> str:
    """Format a metric, printing 'undefined' for ``nan`` (never a fake number)."""
    return "undefined" if value != value else f"{value:.4f}"


def format_report(report: EvalReport) -> str:
    """Render an :class:`EvalReport` as a human-readable text block."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"DeepVerify Pro — evaluation harness  ·  detector: {report.detector_name}")
    lines.append("=" * 72)
    lines.append(f"  modality            : {report.modality}")
    lines.append(f"  is_production       : {report.is_production}")
    lines.append(f"  frames per clip     : {report.frames_per_clip}")
    lines.append(f"  samples (total)     : {report.n_total}")
    lines.append(
        f"  samples scored      : {report.n_scored}  (real={report.n_real}, "
        f"fake={report.n_fake})"
    )
    lines.append(f"  samples errored     : {report.n_errored}")
    lines.append("-" * 72)
    lines.append(f"  ROC-AUC             : {_fmt(report.roc_auc)}")
    lines.append(
        f"  EER                 : {_fmt(report.eer)}  "
        f"(at threshold {_fmt(report.eer_threshold)})"
    )
    lines.append("-" * 72)
    for matrix in report.confusion:
        lines.append(f"  confusion @ score >= {matrix.threshold:.2f}  (flag as synthetic):")
        lines.append(f"    true-positive (fake flagged) : {matrix.true_positive}")
        lines.append(f"    false-negative (fake missed) : {matrix.false_negative}")
        lines.append(f"    false-positive (real flagged): {matrix.false_positive}")
        lines.append(f"    true-negative (real cleared) : {matrix.true_negative}")
        lines.append(
            f"    precision={_fmt(matrix.precision)}  recall={_fmt(matrix.recall)}  "
            f"accuracy={_fmt(matrix.accuracy)}"
        )
    if report.n_real == 0 or report.n_fake == 0:
        lines.append("-" * 72)
        lines.append("  NOTE: ROC-AUC and EER need both classes present; with only one")
        lines.append("        class they are undefined and reported as such — not faked.")
    errored = [s for s in report.samples if s.score is None]
    if errored:
        lines.append("-" * 72)
        lines.append(f"  errored samples ({len(errored)}):")
        for sample in errored:
            lines.append(f"    [{_label_name(sample.label)}] {sample.path} — {sample.error}")
    lines.append("=" * 72)
    lines.append("  This harness measures; it does not assert. A detector's is_production")
    lines.append("  flag stays a reviewed human decision after reading these numbers")
    lines.append("  (M8 §10). Numbers reflect THIS test set only — not any benchmark.")
    lines.append("=" * 72)
    return "\n".join(lines)


def _label_name(label: int) -> str:
    return "fake" if label == 1 else "real"


def _confusion_to_dict(matrix: ConfusionMatrix) -> dict[str, Any]:
    return {
        "threshold": matrix.threshold,
        "true_positive": matrix.true_positive,
        "false_negative": matrix.false_negative,
        "false_positive": matrix.false_positive,
        "true_negative": matrix.true_negative,
        "precision": matrix.precision,
        "recall": matrix.recall,
        "accuracy": matrix.accuracy,
    }


def report_to_dict(report: EvalReport) -> dict[str, Any]:
    """Serialise a report to a JSON-friendly dict (scores + metrics only)."""
    return {
        "detector_name": report.detector_name,
        "is_production": report.is_production,
        "modality": report.modality,
        "frames_per_clip": report.frames_per_clip,
        "n_total": report.n_total,
        "n_scored": report.n_scored,
        "n_errored": report.n_errored,
        "n_real": report.n_real,
        "n_fake": report.n_fake,
        "roc_auc": report.roc_auc,
        "eer": report.eer,
        "eer_threshold": report.eer_threshold,
        "confusion": [_confusion_to_dict(m) for m in report.confusion],
        "samples": [
            {
                "path": s.path,
                "label": _label_name(s.label),
                "score": s.score,
                "frames_scored": s.frames_scored,
                "error": s.error,
            }
            for s in report.samples
        ],
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

_DETECTORS: Final[dict[str, tuple[Callable[[], Detector], Modality]]] = {
    "video-baseline": (BaselineVideoDetector, Modality.VIDEO),
    "video-sbi": (EfficientNetSBIDetector, Modality.VIDEO),
    "audio-baseline": (BaselineAudioDetector, Modality.AUDIO),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, required=True, help="path to the JSONL manifest")
    parser.add_argument(
        "--detector",
        choices=sorted(_DETECTORS),
        required=True,
        help="which registered Detector to evaluate",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=DEFAULT_FRAMES_PER_CLIP,
        help=f"frames sampled per video clip (default {DEFAULT_FRAMES_PER_CLIP})",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional path to write the full report as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.frames_per_clip < 1:
        raise SystemExit("--frames-per-clip must be >= 1")
    factory, modality = _DETECTORS[args.detector]
    detector = factory()
    samples = load_manifest(args.manifest)
    report = evaluate(detector, samples, modality, args.frames_per_clip)
    sys.stdout.write(format_report(report) + "\n")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report_to_dict(report), indent=2) + "\n", encoding="utf-8"
        )
        sys.stdout.write(f"\nwrote JSON report: {args.json_out}\n")


if __name__ == "__main__":
    main()
