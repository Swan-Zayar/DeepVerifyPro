"""Zoom capture adapter — DOCUMENTED STUB ONLY.

Feature: F1, F2 (integration boundary)
ACM: 1.6
Scope: discuss-required

Live Zoom integration requires Zoom's Raw Data entitlement plus Marketplace /
security review (CODING_STANDARDS §9). It is deliberately NOT implemented in
the prototype; constructing this adapter raises ``NotImplementedError``.
"""

from __future__ import annotations

from collections.abc import Iterator

from deepverify_pro.adapters.base import MeetingAdapter
from deepverify_pro.detection.base import Frame

_REASON = (
    "ZoomAdapter is a documented stub. Live Zoom capture needs the Zoom Raw Data "
    "entitlement + Marketplace/security review (CODING_STANDARDS §9) — out of scope "
    "for the prototype."
)


class ZoomAdapter(MeetingAdapter):
    """Placeholder for a future, separately-approved Zoom integration."""

    name = "zoom-stub"

    def __init__(self) -> None:
        raise NotImplementedError(_REASON)

    def audio_frames(self) -> Iterator[Frame]:  # pragma: no cover - unreachable stub
        raise NotImplementedError(_REASON)

    def video_frames(self) -> Iterator[Frame]:  # pragma: no cover - unreachable stub
        raise NotImplementedError(_REASON)

    def close(self) -> None:  # pragma: no cover - unreachable stub
        raise NotImplementedError(_REASON)
