"""Video detection subpackage — 68-landmark pipeline + F2 detectors.

Feature: F2 (Live Video Face Authenticity Verification)
ACM: 1.2, 1.3, 1.6, 2.5
Scope: in-product.md

Two F2 detectors share the ``Detector`` contract:

* ``BaselineVideoDetector`` — a documented 68-landmark geometric heuristic
  (CODING_STANDARDS §3 deviation; see ``MODEL_CARD.md``).
* ``EfficientNetSBIDetector`` — the EfficientNet-B4 / Self-Blended-Images
  trained model, adopted under the M8 non-commercial-research designation with
  research-only weights (see ``MODEL_CARD_efficientnet_sbi.md``).

Both keep ``is_production = False`` until the evaluation harness clears a bar.
"""

from deepverify_pro.detection.video.baseline import (
    DETECTOR_NAME,
    BaselineVideoDetector,
)
from deepverify_pro.detection.video.efficientnet_sbi import (
    EfficientNetSBIDetector,
    SBIDependencyMissing,
    SBIDetectorError,
    SBIWeightsUnavailable,
    verify_checkpoint,
)
from deepverify_pro.detection.video.landmarks import (
    LANDMARK_COUNT,
    LandmarkExtractorError,
    LandmarksUnavailable,
    NoFaceDetected,
    extract_landmarks,
)

__all__ = [
    "DETECTOR_NAME",
    "LANDMARK_COUNT",
    "BaselineVideoDetector",
    "EfficientNetSBIDetector",
    "LandmarkExtractorError",
    "LandmarksUnavailable",
    "NoFaceDetected",
    "SBIDependencyMissing",
    "SBIDetectorError",
    "SBIWeightsUnavailable",
    "extract_landmarks",
    "verify_checkpoint",
]
