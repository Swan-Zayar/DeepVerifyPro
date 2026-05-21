"""ADK tool — append one event to the F5 audit log.

Feature: F5 (Audit Trail and Incident Reporting)
ACM: 3.1, 3.7, 1.6
Scope: in-product.md

Thin deterministic wrapper around :meth:`AuditLog.append` so the F5 surface
appears in the ADK toolset alongside the other five tools per
CODING_STANDARDS §2. The audio/video/provenance/financial tools each emit
their own structured audit entries; this tool exists for orchestrator-level
lifecycle events (e.g. ``orchestrator.tick.start`` / ``orchestrator.tick.end``)
that no other tool covers.

The :class:`AuditLog` forbidden-key guard (§4.4 / ACM 1.6) still rejects any
payload that would carry raw media or biometric vectors.
"""

from __future__ import annotations

from typing import Any

from deepverify_pro.audit.log import AuditLog, AuditRecord


def audit_log(event: str, payload: dict[str, Any], *, audit: AuditLog) -> AuditRecord:
    """Append ``event`` with ``payload`` to ``audit`` and return the record."""
    return audit.append(event, payload)
