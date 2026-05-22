"""DeepVerify Pro HTTP surface — FastAPI backend over the F1–F5 orchestrator.

Feature: F1–F5 (HTTP surface — Part A sign, Part B detect/verify/audit)
ACM: 1.2, 1.3, 1.6, 2.5, 3.1, 3.7
Scope: in-product.md

Every endpoint is a thin translation onto an existing deterministic tool or
the ADK orchestrator (CODING_STANDARDS §2) — no detection, crypto, or audit
logic is re-implemented here.

ACM 1.6 (hard rule): the server binds ``127.0.0.1`` by default
(:class:`Settings.api_host`). Uploaded audio and video are decoded straight
from the request bytes in memory — they never touch disk. Provenance uploads
and the ``/sign`` / ``/verify`` inputs, which the external ``c2patool`` binary
must read from a path, are spooled to a private temp file that is always
deleted in a ``finally`` — media never reaches a third party. Audit payloads
carry metadata only (enforced by the F5 tools and :class:`AuditLog`).

ACM 1.3 / 2.5: responses use probabilistic field names and echo each
detector's ``is_production`` flag; a baseline is never surfaced as a verdict.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from deepverify_pro.agents import DeepVerifyOrchestrator
from deepverify_pro.agents.orchestrator import OrchestratorTick
from deepverify_pro.api.media import (
    MediaDecodeError,
    audio_frame_from_bytes,
    video_frame_from_bytes,
)
from deepverify_pro.api.schemas import (
    AuditRecordOut,
    AuditResponse,
    AuditVerifyResponse,
    DetectorResultOut,
    DetectResponse,
    FinancialOut,
    HealthResponse,
    ProvenanceOut,
)
from deepverify_pro.audit.log import AuditLog, AuditTampered
from deepverify_pro.authorization import LocalFileChannel
from deepverify_pro.config import Settings, get_settings
from deepverify_pro.detection.audio import BaselineAudioDetector
from deepverify_pro.detection.base import DetectionResult
from deepverify_pro.detection.video.baseline import BaselineVideoDetector
from deepverify_pro.provenance import ProvenanceSignerError, ProvenanceVerifierError
from deepverify_pro.provenance.verifier import C2PATOOL_BIN


def build_default_orchestrator(settings: Settings) -> DeepVerifyOrchestrator:
    """Construct the orchestrator the API serves from, wired to live tools.

    The video detector is wired only when the dlib 68-point predictor is
    present on disk; otherwise the orchestrator simply skips the video
    pipeline (it tolerates ``video_detector=None``). The audit log and the F4
    challenge channel are local files on the deploying machine (ACM 1.6).
    """
    audit = AuditLog(settings.audit_path)
    video_detector = (
        BaselineVideoDetector(settings.dlib_landmarks_path)
        if settings.dlib_landmarks_path.is_file()
        else None
    )
    return DeepVerifyOrchestrator(
        audit=audit,
        audio_detector=BaselineAudioDetector(),
        video_detector=video_detector,
        channel=LocalFileChannel(settings.challenge_log_path),
        financial_threshold=settings.financial_amount_threshold,
    )


def _spool(upload: UploadFile, suffix: str) -> Path:
    """Write an upload to a private temp file; the caller must delete it."""
    fd, name = tempfile.mkstemp(suffix=suffix)
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            shutil.copyfileobj(upload.file, handle)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _suffix(upload: UploadFile) -> str:
    """File extension for an upload — c2patool keys behaviour off it."""
    return Path(upload.filename or "upload").suffix or ".bin"


def _detector_out(result: DetectionResult | None) -> DetectorResultOut | None:
    if result is None:
        return None
    return DetectorResultOut(
        synthetic_probability=result.synthetic_probability,
        indicator_state=result.indicator_state,
        detector_name=result.detector_name,
        is_production=result.is_production,
        detail=dict(result.detail),
    )


def _detect_response(tick: OrchestratorTick) -> DetectResponse:
    provenance: ProvenanceOut | None = None
    if tick.provenance is not None:
        provenance = ProvenanceOut(
            has_valid_signature=tick.provenance.has_valid_signature,
            issuer=tick.provenance.issuer,
            reason=tick.provenance.reason,
        )
    financial: FinancialOut | None = None
    if tick.financial is not None:
        outcome = tick.financial
        receipt = outcome.receipt
        financial = FinancialOut(
            triggered=outcome.result.triggered,
            matched_categories=list(outcome.result.matched_categories),
            largest_amount=outcome.result.largest_amount,
            amount_above_threshold=outcome.result.amount_above_threshold,
            threshold=outcome.result.threshold,
            dispatched=receipt.dispatched if receipt is not None else False,
            challenge_id=receipt.challenge_id if receipt is not None else None,
        )
    return DetectResponse(
        tick_id=tick.tick_id,
        audio=_detector_out(tick.audio),
        video=_detector_out(tick.video),
        provenance=provenance,
        financial=financial,
    )


def create_app(
    *,
    settings: Settings | None = None,
    orchestrator: DeepVerifyOrchestrator | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Tests pass ``settings`` / ``orchestrator`` overrides; production calls it
    with no arguments and gets the default localhost wiring.
    """
    cfg = settings or get_settings()
    orch = orchestrator or build_default_orchestrator(cfg)

    app = FastAPI(
        title="DeepVerify Pro API",
        version="0.0.1",
        summary="Localhost HTTP surface over the F1–F5 deepfake-detection orchestrator.",
    )
    # ACM 1.6: only the locally-served frontend origins are allowed; the API
    # is never intended to answer a cross-origin caller on a public network.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.api_cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.state.settings = cfg
    app.state.orchestrator = orch

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness + capability probe."""
        return HealthResponse(
            status="ok",
            orchestrator=orch.name,
            tools=list(orch.tool_names()),
            c2patool_available=shutil.which(C2PATOOL_BIN) is not None,
        )

    @app.post("/detect", response_model=DetectResponse)
    def detect(
        audio: Annotated[UploadFile | None, File()] = None,
        video: Annotated[UploadFile | None, File()] = None,
        provenance: Annotated[UploadFile | None, File()] = None,
        transcript: Annotated[str | None, Form()] = None,
        recipient: Annotated[str, Form()] = "",
        frame_index: Annotated[int, Form()] = 0,
    ) -> DetectResponse:
        """Part B — run one orchestrator tick over the supplied media.

        Each pipeline is optional; whatever is provided is scored and the rest
        is skipped (the audit log records only what ran — §4.2). The F4
        financial trigger fires purely on the transcript and never consults a
        detector score (§4.3 / ACM 1.2).
        """
        cleanup: list[Path] = []
        try:
            audio_frame = (
                audio_frame_from_bytes(audio.file.read(), index=frame_index)
                if audio is not None
                else None
            )
            video_frame = (
                video_frame_from_bytes(video.file.read(), index=frame_index)
                if video is not None
                else None
            )
            provenance_path: Path | None = None
            if provenance is not None:
                provenance_path = _spool(provenance, _suffix(provenance))
                cleanup.append(provenance_path)
            try:
                tick = orch.tick(
                    audio_frame=audio_frame,
                    video_frame=video_frame,
                    provenance_path=provenance_path,
                    transcript=transcript,
                    recipient=recipient,
                )
            except ValueError as exc:
                # Detector / orchestrator domain errors (no face, audio too
                # short, F4 recipient missing). The orchestrator has already
                # closed the F5 chain in its ``finally``; surface a clean 422
                # rather than a 500 stack trace.
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return _detect_response(tick)
        except MediaDecodeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            for path in cleanup:
                path.unlink(missing_ok=True)

    @app.post("/sign")
    def sign(file: Annotated[UploadFile, File()]) -> FileResponse:
        """Part A — embed a C2PA manifest and return the signed asset.

        Signs with the server-configured prototype cert/key (the org's signing
        material, never client-supplied). The signed file streams back; its
        issuer is echoed in the ``X-DVP-Issuer`` header. The F5 audit event is
        written inside the ``sign_media`` tool.
        """
        suffix = _suffix(file)
        input_path = _spool(file, suffix)
        fd, out_name = tempfile.mkstemp(suffix=f".signed{suffix}")
        os.close(fd)
        output_path = Path(out_name)
        try:
            result = orch.sign(
                input_path,
                output_path,
                cert_path=cfg.signing_cert_path,
                key_path=cfg.signing_key_path,
            )
        except ProvenanceSignerError as exc:
            output_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            input_path.unlink(missing_ok=True)
        download_name = f"signed-{file.filename or 'media'}"
        return FileResponse(
            output_path,
            filename=download_name,
            media_type="application/octet-stream",
            headers={"X-DVP-Issuer": result.issuer},
            background=BackgroundTask(output_path.unlink, missing_ok=True),
        )

    @app.post("/verify", response_model=ProvenanceOut)
    def verify(file: Annotated[UploadFile, File()]) -> ProvenanceOut:
        """F3 — verify an uploaded asset's C2PA manifest."""
        input_path = _spool(file, _suffix(file))
        try:
            result = orch.verify(input_path)
        except ProvenanceVerifierError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            input_path.unlink(missing_ok=True)
        return ProvenanceOut(
            has_valid_signature=result.has_valid_signature,
            issuer=result.issuer,
            reason=result.reason,
        )

    @app.get("/audit", response_model=AuditResponse)
    def audit(limit: int | None = None) -> AuditResponse:
        """F5 — read the hash-chained audit log (optionally the last ``limit``)."""
        records = orch.audit.read_all()
        if limit is not None:
            records = records[-limit:] if limit > 0 else []
        return AuditResponse(
            count=len(records),
            records=[
                AuditRecordOut(
                    seq=record.seq,
                    ts=record.ts,
                    event=record.event,
                    payload=record.payload,
                    prev_hash=record.prev_hash,
                    hash=record.hash,
                )
                for record in records
            ],
        )

    @app.get("/audit/verify", response_model=AuditVerifyResponse)
    def audit_verify() -> AuditVerifyResponse:
        """F5 — recompute the hash chain and report whether it is intact."""
        checked = len(orch.audit.read_all())
        try:
            orch.audit.verify_chain()
        except AuditTampered as exc:
            return AuditVerifyResponse(
                intact=False,
                records_checked=checked,
                detail=str(exc),
            )
        return AuditVerifyResponse(
            intact=True,
            records_checked=checked,
            detail="hash chain intact",
        )

    return app
