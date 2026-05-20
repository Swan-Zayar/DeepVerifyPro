"""ADK tool — score one video frame with a :class:`Detector` and audit it.

Feature: F2 (Live Video Face Authenticity Verification)
ACM: 1.3, 1.6, 3.1, 3.7
Scope: in-product.md

Thin deterministic wrapper around a :class:`Detector` instance, exposed to
the ADK orchestrator (§2). Emits exactly one F5 audit event per frame
carrying score/state metadata only — never the image, landmarks, or any
biometric vector (ACM 1.6, enforced by the audit log's forbidden-key guard).
"""

from __future__ import annotations

from deepverify_pro.audit.log import AuditLog
from deepverify_pro.detection.base import DetectionResult, Detector, Frame

EVENT_NAME = "video.detect"


def video_detect(frame: Frame, *, detector: Detector, audit: AuditLog) -> DetectionResult:
    """Score ``frame`` with ``detector`` and append one audit event.

    Re-raises whatever the detector raises (no silent swallow — §7).
    """
    result = detector.score(frame)
    audit.append(
        EVENT_NAME,
        {
            "frame_index": frame.index,
            "detector_name": result.detector_name,
            "is_production": result.is_production,
            "synthetic_probability": result.synthetic_probability,
            "indicator_state": str(result.indicator_state),
        },
    )
    return result
