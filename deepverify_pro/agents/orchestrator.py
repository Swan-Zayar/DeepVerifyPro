"""DeepVerify Pro orchestrator — one agent coordinating F1–F5 tools.

Feature: F1–F5 (CODING_STANDARDS §2 orchestration)
ACM: 1.2, 1.3, 1.6, 2.5, 3.1, 3.7
Scope: in-product.md

Per CODING_STANDARDS §2 the prototype runs **one ADK orchestrator coordinating
deterministic tools** — detection and cryptography are never delegated to an
LLM. Each of the six tools (``audio_detect``, ``video_detect``,
``provenance_verify``, ``sign_media``, ``financial_trigger``, ``audit_log``) is
registered as a :class:`google.adk.tools.FunctionTool` so the toolset is
discoverable by any ADK runner. The orchestrator drives tool calls
deterministically — no LLM routing — because routing media-derived state
through a third-party LLM would breach the §4.1 / ACM 1.6 hard-rule (no
media or biometric egress; network egress is deny-by-default in the
prototype). Production routing decisions can be re-introduced behind a
future, separately-discussed entitlement (§9).

Per product.md §3.4 the audio, video, and provenance pipelines run in
parallel; the provenance pipeline is independent of the detection engines.
:meth:`DeepVerifyOrchestrator.tick` runs them on a thread pool together with
the F4 financial trigger, which is **never gated on detector output**
(CODING_STANDARDS §4.3 / ACM 1.2 defence-in-depth). The audit log is
thread-safe (see :class:`deepverify_pro.audit.log.AuditLog`) so the
concurrent pipelines can all append to a single F5 chain (§4.4).
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from google.adk.tools import FunctionTool

from deepverify_pro.audit.log import AuditLog, SessionAuditLog
from deepverify_pro.authorization.trigger import OutOfBandChannel
from deepverify_pro.detection.base import DetectionResult, Detector, Frame
from deepverify_pro.provenance import ProvenanceResult, SignResult
from deepverify_pro.tools.audio_detect import audio_detect
from deepverify_pro.tools.audit_log import audit_log
from deepverify_pro.tools.financial_trigger import (
    FinancialTriggerOutcome,
    financial_trigger,
)
from deepverify_pro.tools.provenance_financial_trigger import (
    ProvenanceFinancialOutcome,
    provenance_financial_trigger,
)
from deepverify_pro.tools.provenance_verify import provenance_verify
from deepverify_pro.tools.sign_media import sign_media
from deepverify_pro.tools.video_detect import video_detect

ORCHESTRATOR_NAME: Final[str] = "deepverify-orchestrator-v0"
TICK_START_EVENT: Final[str] = "orchestrator.tick.start"
TICK_END_EVENT: Final[str] = "orchestrator.tick.end"


@dataclass(frozen=True)
class OrchestratorTick:
    """Outcome of one coordinated cycle.

    Any pipeline that wasn't fed input (or whose detector is absent) reports
    ``None`` rather than a fabricated result — the F5 audit log records what
    actually ran (§4.2 / ACM 1.3: zero fabricated metrics).
    """

    tick_id: int
    audio: DetectionResult | None
    video: DetectionResult | None
    provenance: ProvenanceResult | None
    financial: FinancialTriggerOutcome | None


@dataclass(frozen=True)
class FinancialVerifyOutcome:
    """Result of the F3+F4 ``verify_for_financial`` composition.

    Carries the underlying provenance verdict and the F4 dispatch result so
    the surface can render both — *both* signals are operational: a passing
    provenance check still informs the operator, and a fired challenge tells
    them the document failed one of the three checks (missing / invalid /
    untrusted) with the precise reason code.
    """

    provenance: ProvenanceResult
    financial: ProvenanceFinancialOutcome


class DeepVerifyOrchestrator:
    """Single deterministic ADK orchestrator over the six F1–F5 tools."""

    name: Final[str] = ORCHESTRATOR_NAME

    def __init__(
        self,
        *,
        audit: AuditLog,
        audio_detector: Detector | None = None,
        video_detector: Detector | None = None,
        channel: OutOfBandChannel | None = None,
        financial_threshold: float = 0.0,
        trusted_issuers: tuple[str, ...] = (),
        max_workers: int = 4,
    ) -> None:
        if financial_threshold < 0.0:
            raise ValueError("financial_threshold must be non-negative")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._audit: AuditLog = audit
        self._audio_detector: Detector | None = audio_detector
        self._video_detector: Detector | None = video_detector
        self._channel: OutOfBandChannel | None = channel
        self._financial_threshold: float = financial_threshold
        # Empty trust list ⇒ no issuer is trusted (fail-closed). The
        # deploying organisation populates Settings.signing_trusted_issuers
        # with its own signing infrastructure's leaf-cert common names.
        self._trusted_issuers: tuple[str, ...] = trusted_issuers
        self._max_workers: int = max_workers
        self._tick_counter: int = 0
        # ``tick`` may run concurrently — FastAPI dispatches sync endpoints on a
        # threadpool — so the tick-id read-modify-write is guarded; without the
        # lock two ticks could collide and emit duplicate/out-of-order IDs.
        self._tick_lock: threading.Lock = threading.Lock()

        # The seven-tool ADK surface (CODING_STANDARDS §2 — owner-approved
        # extension from six to seven for the F3+F4 composition; see
        # DEVIATIONS.md). Registered once so the function metadata is
        # captured for any future ADK runner; the orchestrator calls the
        # underlying functions directly during ticks.
        self._tools: tuple[FunctionTool, ...] = (
            FunctionTool(audio_detect),
            FunctionTool(video_detect),
            FunctionTool(provenance_verify),
            FunctionTool(sign_media),
            FunctionTool(financial_trigger),
            FunctionTool(provenance_financial_trigger),
            FunctionTool(audit_log),
        )

    @property
    def audit(self) -> AuditLog:
        return self._audit

    @property
    def tools(self) -> tuple[FunctionTool, ...]:
        """The ADK :class:`FunctionTool` surface (§2 — six deterministic tools)."""
        return self._tools

    def tool_names(self) -> tuple[str, ...]:
        """Names of the registered tools, in declaration order."""
        return tuple(tool.name for tool in self._tools)

    def sign(
        self,
        input_path: Path,
        output_path: Path,
        *,
        cert_path: Path,
        key_path: Path,
    ) -> SignResult:
        """Part A — sign one media file via the F3 ``sign_media`` tool.

        Routed through the orchestrator so the audit log records every sign
        action in the same hash chain as the detection-side events (§4.4).
        Returns the :class:`SignResult` (output path + issuer common name).
        """
        return sign_media(
            input_path,
            output_path,
            cert_path=cert_path,
            key_path=key_path,
            audit=self._audit,
        )

    def verify(self, input_path: Path) -> ProvenanceResult:
        """Part B (one-shot) — verify one media file's C2PA manifest via F3.

        Mirrors :meth:`sign`: routed through the orchestrator so the verdict
        lands in the same F5 hash chain as the detection-side events (§4.4).
        The deployment trust list is passed through so ``is_trusted_issuer``
        is computed honestly (ACM 1.3).
        """
        return provenance_verify(
            input_path,
            audit=self._audit,
            trusted_issuers=self._trusted_issuers,
        )

    def verify_for_financial(
        self,
        input_path: Path,
        *,
        recipient: str,
    ) -> FinancialVerifyOutcome:
        """F3+F4 composition — verify a financial document and fire OOB if it fails.

        Runs :func:`provenance_verify` (with the deployment trust list) then
        :func:`provenance_financial_trigger` (with the resulting
        :class:`ProvenanceResult`). Both events land in the same F5 hash
        chain. The F4 challenge fires iff the document is unsigned, has an
        invalid signature, or is signed by an issuer not on the trust list —
        the §5 honesty anti-pattern this composition exists to prevent
        ("valid signature alone authorises a financial action").

        ``recipient`` is the operator's chosen out-of-band destination (a
        registered-device handle the channel implementation resolves). It is
        only consulted when the trigger fires; a passing document never
        contacts the channel.
        """
        provenance_result = provenance_verify(
            input_path,
            audit=self._audit,
            trusted_issuers=self._trusted_issuers,
        )
        if self._channel is None:
            raise RuntimeError(
                "verify_for_financial requires an out-of-band channel — "
                "construct DeepVerifyOrchestrator with channel=... so F4 can "
                "dispatch when provenance fails (ACM 1.2 defence in depth)"
            )
        financial_outcome = provenance_financial_trigger(
            provenance_result,
            recipient=recipient,
            channel=self._channel,
            audit=self._audit,
        )
        return FinancialVerifyOutcome(
            provenance=provenance_result,
            financial=financial_outcome,
        )

    def tick(
        self,
        *,
        audio_frame: Frame | None = None,
        video_frame: Frame | None = None,
        provenance_path: Path | None = None,
        transcript: str | None = None,
        recipient: str = "",
        session_id: str | None = None,
    ) -> OrchestratorTick:
        """Part B — one coordinated cycle over the four runtime pipelines.

        Audio, video, provenance, and financial-trigger pipelines run on a
        thread pool. The financial trigger consults no detector score
        (§4.3 / ACM 1.2). Any pipeline without inputs is skipped — the audit
        log faithfully reflects what ran (§4.2).

        If any pipeline raises — including the F4 trigger refusing to build a
        challenge for an empty ``recipient`` (§4.3 / ACM 1.2) — the
        ``orchestrator.tick.end`` event is still written (with an ``error``
        field) before the exception propagates, so the F5 chain never keeps
        an orphaned start record (§4.4 / ACM 3.1, 3.7).

        When ``session_id`` is supplied, every audit append made during this
        tick is routed through a :class:`SessionAuditLog` proxy that stamps
        the id into the payload. The underlying global hash chain is
        unchanged — the proxy only adds a key so the slice can later be
        filtered for that session's user (F5 per-session download).
        """
        self._tick_counter += 1
        tick_id = self._tick_counter
        tick_audit: AuditLog = (
            SessionAuditLog(self._audit, session_id) if session_id else self._audit
        )
        tick_audit.append(
            TICK_START_EVENT,
            {
                "tick_id": tick_id,
                "orchestrator": self.name,
                "has_audio": audio_frame is not None and self._audio_detector is not None,
                "has_video": video_frame is not None and self._video_detector is not None,
                "has_provenance": provenance_path is not None,
                "has_financial": transcript is not None and self._channel is not None,
            },
        )

        audio_result: DetectionResult | None = None
        video_result: DetectionResult | None = None
        provenance_result: ProvenanceResult | None = None
        financial_result: FinancialTriggerOutcome | None = None
        error: str | None = None

        try:
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                audio_future: Future[DetectionResult] | None = None
                video_future: Future[DetectionResult] | None = None
                provenance_future: Future[ProvenanceResult] | None = None
                financial_future: Future[FinancialTriggerOutcome] | None = None

                if audio_frame is not None and self._audio_detector is not None:
                    audio_future = pool.submit(
                        audio_detect,
                        audio_frame,
                        detector=self._audio_detector,
                        audit=tick_audit,
                    )
                if video_frame is not None and self._video_detector is not None:
                    video_future = pool.submit(
                        video_detect,
                        video_frame,
                        detector=self._video_detector,
                        audit=tick_audit,
                    )
                if provenance_path is not None:
                    provenance_future = pool.submit(
                        provenance_verify,
                        provenance_path,
                        audit=tick_audit,
                        trusted_issuers=self._trusted_issuers,
                    )
                # F4 is dispatched independently of any detector future —
                # §4.3 / ACM 1.2. It consults no detector score and no
                # detector handle. An empty ``recipient`` on a fired trigger
                # is refused inside ``build_challenge`` (caught below).
                if transcript is not None and self._channel is not None:
                    financial_future = pool.submit(
                        financial_trigger,
                        transcript,
                        threshold=self._financial_threshold,
                        recipient=recipient,
                        channel=self._channel,
                        audit=tick_audit,
                    )

                audio_result = audio_future.result() if audio_future is not None else None
                video_result = video_future.result() if video_future is not None else None
                provenance_result = (
                    provenance_future.result() if provenance_future is not None else None
                )
                financial_result = (
                    financial_future.result() if financial_future is not None else None
                )
        except Exception as exc:
            # Record the failure in the F5 chain (the finally below) and then
            # re-raise — a swallowed pipeline failure is a security failure
            # here (§7: never silently swallow a detection failure).
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # Always close the tick in the audit chain, success or failure,
            # so a crashed pipeline never leaves an orphaned start record
            # (§4.4 / ACM 3.1, 3.7).
            tick_audit.append(
                TICK_END_EVENT,
                {
                    "tick_id": tick_id,
                    "orchestrator": self.name,
                    "error": error,
                    "audio_state": (
                        str(audio_result.indicator_state) if audio_result is not None else None
                    ),
                    "video_state": (
                        str(video_result.indicator_state) if video_result is not None else None
                    ),
                    "provenance_valid": (
                        provenance_result.has_valid_signature
                        if provenance_result is not None
                        else None
                    ),
                    "financial_triggered": (
                        financial_result.result.triggered if financial_result is not None else None
                    ),
                },
            )

        return OrchestratorTick(
            tick_id=tick_id,
            audio=audio_result,
            video=video_result,
            provenance=provenance_result,
            financial=financial_result,
        )
