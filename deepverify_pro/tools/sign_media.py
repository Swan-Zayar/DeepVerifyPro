"""ADK tool — sign one media asset with C2PA and audit the action.

Feature: F3 (Cryptographic Content Provenance Signing)
ACM: 2.5, 3.7, 3.1
Scope: in-product.md

Thin deterministic wrapper over :func:`deepverify_pro.provenance.sign`. Always
emits one F5 audit event with non-media metadata only — filename, issuer
common name. Raw bytes never enter the audit log (ACM 1.6).
"""

from __future__ import annotations

from pathlib import Path

from deepverify_pro.audit.log import AuditLog
from deepverify_pro.provenance import SignResult, sign

EVENT_NAME = "provenance.sign"


def sign_media(
    input_path: Path,
    output_path: Path,
    *,
    cert_path: Path,
    key_path: Path,
    audit: AuditLog,
) -> SignResult:
    """Sign ``input_path`` to ``output_path`` and record one audit event.

    Re-raises :class:`deepverify_pro.provenance.ProvenanceSignerError` on
    failure (no silent swallow — CODING_STANDARDS §7).
    """
    result = sign(
        input_path,
        output_path,
        cert_path=cert_path,
        key_path=key_path,
    )
    audit.append(
        EVENT_NAME,
        {
            "input_name": input_path.name,
            "output_name": result.output_path.name,
            "issuer": result.issuer,
        },
    )
    return result
