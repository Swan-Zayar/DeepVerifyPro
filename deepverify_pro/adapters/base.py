"""The ``MeetingAdapter`` capture contract.

Feature: F1 (Real-Time Audio Deepfake Detection), F2 (Live Video Face Authenticity Verification)
ACM: 1.6
Scope: in-product.md

An adapter yields :class:`Frame` objects from a capture source. All adapters
keep media in-process (ACM 1.6); the only sanctioned external boundary is a
future Zoom adapter under raw-data entitlement (CODING_STANDARDS §9).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from deepverify_pro.detection.base import Frame


class MeetingAdapter(ABC):
    """Base class for media capture sources feeding the detection pipelines."""

    name: str = "abstract-adapter"

    @abstractmethod
    def audio_frames(self) -> Iterator[Frame]:
        """Yield AUDIO :class:`Frame` objects until the source is exhausted."""
        raise NotImplementedError

    @abstractmethod
    def video_frames(self) -> Iterator[Frame]:
        """Yield VIDEO :class:`Frame` objects until the source is exhausted."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release any capture resources."""
        raise NotImplementedError
