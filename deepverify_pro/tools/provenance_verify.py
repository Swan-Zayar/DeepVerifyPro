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
    trusted_issuers: tuple[str, ...] = (),
) -> ProvenanceResult:
    """Verify ``input_path``'s C2PA manifest and record one audit event.

    ``trusted_issuers`` is the deployment's allow-list of leaf-cert common
    names (typically wired from :class:`Settings.signing_trusted_issuers`).
    The result's ``is_trusted_issuer`` is logged in the audit payload so a
    later reader can tell ``valid-but-untrusted`` (an attacker self-signing)
    apart from ``valid-and-trusted`` (the deployment's own signing infra).
    """
    result = verify(input_path, trusted_issuers=trusted_issuers)
    audit.append(
        EVENT_NAME,
        {
            "input_name": input_path.name,
            "has_valid_signature": result.has_valid_signature,
            "is_trusted_issuer": result.is_trusted_issuer,
            "issuer": result.issuer,
            "reason": result.reason,
        },
    )
    return result
