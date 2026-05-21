"""Decode uploaded media bytes into in-process :class:`Frame` objects.

Feature: F1 (audio), F2 (video) — HTTP surface capture boundary
ACM: 1.6
Scope: in-product.md

Uploaded audio/video is decoded entirely in-process (ACM 1.6 — no third-party
egress) and handed to the existing detectors as :class:`Frame` objects. The
HTTP layer is the only place an upload becomes media; the orchestrator and
tools below it never see raw bytes.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import soundfile as sf

from deepverify_pro.detection.base import Frame, Modality


class MediaDecodeError(ValueError):
    """Raised when an upload cannot be decoded into a usable :class:`Frame`."""


def audio_frame_from_bytes(data: bytes, *, index: int = 0) -> Frame:
    """Decode an audio upload (WAV/FLAC/OGG) into a mono AUDIO :class:`Frame`.

    Multi-channel input is down-mixed to mono explicitly here so the choice is
    visible (CODING_STANDARDS §7 — no silent surprises).
    """
    if not data:
        raise MediaDecodeError("audio upload is empty")
    try:
        waveform, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except RuntimeError as exc:  # soundfile.LibsndfileError subclasses RuntimeError
        raise MediaDecodeError(f"could not decode audio upload: {exc}") from exc
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    array = np.ascontiguousarray(array, dtype=np.float32)
    if array.size == 0:
        raise MediaDecodeError("audio upload decoded to an empty waveform")
    return Frame(
        modality=Modality.AUDIO,
        data=array,
        sample_rate=int(sample_rate),
        index=index,
    )


def video_frame_from_bytes(data: bytes, *, index: int = 0) -> Frame:
    """Decode a video-frame upload (PNG/JPEG/...) into a VIDEO :class:`Frame`.

    The decoded array is HxWx3 uint8 (BGR); the F2 landmark extractor is
    colour-channel agnostic on 8-bit input, so no conversion is needed.
    """
    if not data:
        raise MediaDecodeError("video upload is empty")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise MediaDecodeError("video upload is not a decodable image")
    return Frame(modality=Modality.VIDEO, data=image, index=index)
