"""Detection package — pluggable deepfake detectors.

Feature: F1 (Real-Time Audio Deepfake Detection), F2 (Live Video Face Authenticity Verification)
ACM: 1.2, 1.3, 1.6
Scope: in-product.md
"""

from deepverify_pro.detection.base import (
    DetectionResult,
    Detector,
    Frame,
    Modality,
)

__all__ = ["DetectionResult", "Detector", "Frame", "Modality"]
