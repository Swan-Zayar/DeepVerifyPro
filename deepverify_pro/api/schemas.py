"""Pydantic response models for the HTTP surface.

Feature: F1–F5 (HTTP surface)
ACM: 1.3, 2.5
Scope: in-product.md

Field names are deliberately probabilistic (``synthetic_probability``,
``indicator_state``) and every detector result echoes ``is_production`` — the
surface never presents a prototype baseline as an absolute verdict
(CODING_STANDARDS §4.2 / ACM 1.3, 2.5).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from deepverify_pro.indicator import IndicatorState


class HealthResponse(BaseModel):
    """Liveness + capability report for the running surface."""

    status: str
    orchestrator: str
    tools: list[str]
    c2patool_available: bool


class DetectorResultOut(BaseModel):
    """One detector's probabilistic outcome — a confidence score, never a
    guarantee (ACM 1.3). ``is_production`` is ``False`` for prototype
    baselines and is surfaced so the client cannot mistake one for a verdict.
    """

    synthetic_probability: float
    indicator_state: IndicatorState
    detector_name: str
    is_production: bool
    detail: dict[str, Any] = Field(default_factory=dict)


class ProvenanceOut(BaseModel):
    """F3 verdict.

    ``has_valid_signature`` is the cryptographic-integrity check only — it
    tells the client the bytes were signed by *someone* with a matching key.
    ``is_trusted_issuer`` is the separate allow-list check against the
    deployment's configured signing issuers. Both are surfaced because
    conflating them is the §5 anti-pattern this product exists to prevent
    (an attacker can produce a cryptographically valid manifest with a
    freshly self-signed cert). ``reason`` keeps any trust caveat visible
    in plain text (ACM 1.3).
    """

    has_valid_signature: bool
    is_trusted_issuer: bool
    issuer: str | None
    reason: str


class FinancialOut(BaseModel):
    """F4 outcome. ``triggered`` is decided purely on transcript signals and
    is independent of any detector score (CODING_STANDARDS §4.3 / ACM 1.2)."""

    triggered: bool
    matched_categories: list[str]
    largest_amount: float | None
    amount_above_threshold: bool
    threshold: float
    dispatched: bool
    challenge_id: str | None


class ProvenanceFinancialOut(BaseModel):
    """F3+F4 composition outcome — the OOB dispatch result for a financial doc.

    ``triggered`` is True iff the document failed one of the three checks
    (no manifest, invalid signature, untrusted issuer) and an out-of-band
    challenge was dispatched to ``recipient``. ``reason_code`` names which
    check failed; ``challenge_id`` is the channel receipt id when dispatched.
    Defence in depth (CODING_STANDARDS §4.3 / ACM 1.2): this trigger never
    consults a detector score.
    """

    triggered: bool
    reason_code: str | None
    dispatched: bool
    challenge_id: str | None


class VerifyFinancialResponse(BaseModel):
    """Response for ``/verify`` with ``financial_context=true``.

    Surfaces the F3 verdict (provenance) and the F4 composition outcome
    (financial) side-by-side so the client can render both honestly: a
    passing document still reports its trust signal, and a fired challenge
    tells the operator the document failed one of the three checks with the
    precise reason code (ACM 1.3 / 2.5 — slice ≠ verdict).
    """

    provenance: ProvenanceOut
    financial: ProvenanceFinancialOut


class DetectResponse(BaseModel):
    """Outcome of one orchestrator tick. Any pipeline without input is
    ``None`` rather than a fabricated result (ACM 1.3)."""

    tick_id: int
    audio: DetectorResultOut | None
    video: DetectorResultOut | None
    provenance: ProvenanceOut | None
    financial: FinancialOut | None


class AuditRecordOut(BaseModel):
    """One hash-chained F5 audit entry, surfaced read-only."""

    seq: int
    ts: str
    event: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str


class AuditResponse(BaseModel):
    """A page of audit records (``count`` is the number returned)."""

    count: int
    records: list[AuditRecordOut]


class AuditVerifyResponse(BaseModel):
    """F5 tamper check. ``intact=False`` is a valid, reportable finding — the
    audit feature exists to detect tampering, not to assume it never happens."""

    intact: bool
    records_checked: int
    detail: str
