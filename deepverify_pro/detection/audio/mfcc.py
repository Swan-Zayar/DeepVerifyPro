"""MFCC feature extraction at a 25 ms hop (product.md §3.3 / §3.4).

Feature: F1 (Real-Time Audio Deepfake Detection)
ACM: 1.2, 1.3, 1.6
Scope: in-product.md

product.md §3.3 / §3.4 specifies MFCC features extracted every 25 ms. This
module is the deterministic feature pipeline behind the F1 baseline detector
(CODING_STANDARDS §3 stack: ``librosa``). It is **pure**: no network, no disk
writes, no globals (ACM 1.6 / §7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import librosa
import numpy as np

DEFAULT_N_MFCC: Final[int] = 20
DEFAULT_HOP_MS: Final[float] = 25.0  # product.md §3.3 — "every 25 milliseconds"
DEFAULT_WINDOW_MS: Final[float] = 25.0


class MFCCExtractorError(ValueError):
    """Raised when the input audio cannot be turned into MFCCs."""


@dataclass(frozen=True)
class MFCCConfig:
    """Knobs for the MFCC pipeline.

    Defaults match product.md §3.3 ("every 25 milliseconds"). The constructor
    validates positivity so an upstream typo (negative hop, zero coefficients)
    fails loudly rather than silently producing bogus features (§7).
    """

    n_mfcc: int = DEFAULT_N_MFCC
    hop_ms: float = DEFAULT_HOP_MS
    window_ms: float = DEFAULT_WINDOW_MS

    def __post_init__(self) -> None:
        if self.n_mfcc <= 0:
            raise MFCCExtractorError("n_mfcc must be positive")
        if self.hop_ms <= 0 or self.window_ms <= 0:
            raise MFCCExtractorError("hop_ms and window_ms must be positive")

    def hop_length(self, sample_rate: int) -> int:
        """Hop length in samples, never less than 1."""
        return max(1, int(round(self.hop_ms * 1e-3 * sample_rate)))

    def n_fft(self, sample_rate: int) -> int:
        """FFT size: next power of two ≥ window length in samples (librosa-style)."""
        win_samples = max(1, int(round(self.window_ms * 1e-3 * sample_rate)))
        n = 1
        while n < win_samples:
            n <<= 1
        return n


def extract_mfcc(
    waveform: np.ndarray,
    sample_rate: int,
    config: MFCCConfig | None = None,
) -> np.ndarray:
    """Return MFCCs of shape ``(n_mfcc, n_frames)`` for a mono waveform.

    Mono float waveform in ``[-1, 1]``; multi-channel input is rejected so the
    caller makes the down-mixing choice explicitly (§7 — no silent surprises).
    Raises :class:`MFCCExtractorError` on empty / wrong-shape input.
    """
    cfg = config or MFCCConfig()
    if sample_rate <= 0:
        raise MFCCExtractorError("sample_rate must be positive")
    if waveform.ndim != 1:
        raise MFCCExtractorError(f"waveform must be 1-D mono float audio; got ndim={waveform.ndim}")
    if waveform.size == 0:
        raise MFCCExtractorError("waveform is empty")

    samples = np.ascontiguousarray(waveform, dtype=np.float32)
    mfcc = librosa.feature.mfcc(
        y=samples,
        sr=sample_rate,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft(sample_rate),
        hop_length=cfg.hop_length(sample_rate),
    )
    return np.asarray(mfcc, dtype=np.float32)
