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
    """F3 verdict. ``has_valid_signature`` is the cryptographic-integrity
    check only; ``reason`` keeps any trust caveat visible (ACM 1.3)."""

    has_valid_signature: bool
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
