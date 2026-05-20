"""Audio detection subpackage — MFCC pipeline + baseline F1 detector.

Feature: F1 (Real-Time Audio Deepfake Detection)
ACM: 1.2, 1.3, 1.6
Scope: in-product.md

The baseline detector is a documented heuristic, not a trained model
(CODING_STANDARDS §3 deviation; see ``MODEL_CARD.md``).
"""

from deepverify_pro.detection.audio.baseline import (
    DETECTOR_NAME,
    BaselineAudioDetector,
)
from deepverify_pro.detection.audio.mfcc import (
    DEFAULT_HOP_MS,
    DEFAULT_N_MFCC,
    MFCCConfig,
    MFCCExtractorError,
    extract_mfcc,
)

__all__ = [
    "BaselineAudioDetector",
    "DETECTOR_NAME",
    "DEFAULT_HOP_MS",
    "DEFAULT_N_MFCC",
    "MFCCConfig",
    "MFCCExtractorError",
    "extract_mfcc",
]
