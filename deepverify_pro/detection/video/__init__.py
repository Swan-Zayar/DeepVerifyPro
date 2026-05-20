"""Video detection subpackage — 68-landmark pipeline + baseline F2 detector.

Feature: F2 (Live Video Face Authenticity Verification)
ACM: 1.3, 1.6
Scope: in-product.md

The baseline detector is a documented heuristic, not a trained model
(CODING_STANDARDS §3 deviation; see ``MODEL_CARD.md``).
"""

from deepverify_pro.detection.video.baseline import (
    DETECTOR_NAME,
    BaselineVideoDetector,
)
from deepverify_pro.detection.video.landmarks import (
    LANDMARK_COUNT,
    LandmarkExtractorError,
    LandmarksUnavailable,
    NoFaceDetected,
    extract_landmarks,
)

__all__ = [
    "BaselineVideoDetector",
    "DETECTOR_NAME",
    "LANDMARK_COUNT",
    "LandmarkExtractorError",
    "LandmarksUnavailable",
    "NoFaceDetected",
    "extract_landmarks",
]
