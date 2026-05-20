"""ADK tool — verify a media asset's C2PA manifest and audit the result.

Feature: F3 (Cryptographic Content Provenance Signing)
ACM: 2.5, 3.7, 3.1
Scope: in-product.md

Thin deterministic wrapper over :func:`deepverify_pro.provenance.verify`. The
audit event carries only the verdict, issuer, and a short reason — no raw
bytes, no hashes, no assertions (ACM 1.6).
"""

from __future__ import annotations

from pathlib import Path

from deepverify_pro.audit.log import AuditLog
from deepverify_pro.provenance import ProvenanceResult, verify

EVENT_NAME = "provenance.verify"


def provenance_verify(
    input_path: Path,
    *,
    audit: AuditLog,
) -> ProvenanceResult:
    """Verify ``input_path``'s C2PA manifest and record one audit event."""
    result = verify(input_path)
    audit.append(
        EVENT_NAME,
        {
            "input_name": input_path.name,
            "has_valid_signature": result.has_valid_signature,
            "issuer": result.issuer,
            "reason": result.reason,
        },
    )
    return result
