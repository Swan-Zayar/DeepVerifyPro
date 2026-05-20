"""Subprocess wrapper around c2patool — verify a C2PA manifest.

Feature: F3 (Cryptographic Content Provenance Signing)
ACM: 2.5, 3.7
Scope: in-product.md

Returns a frozen :class:`ProvenanceResult` carrying *only* a boolean
``has_valid_signature``, the issuer string from the cert, and a short reason.
No raw bytes, hashes, or assertions are surfaced — keeping the result safe
for the F5 audit log (ACM 1.6).

The cryptographic integrity check (``claimSignature``, ``assertion.dataHash``)
is what determines validity. A ``signingCredential.untrusted`` warning is
emitted by c2patool whenever the issuing CA is not in its bundled trust list
— the prototype intentionally uses a local test CA, so we treat that warning
as non-fatal but keep it in the reason string for transparency (ACM 1.3 /
2.5: never present an untrusted prototype cert as if it had real PKI trust).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

C2PATOOL_BIN: Final[str] = "c2patool"
_UNTRUSTED_CRED: Final[str] = "signingCredential.untrusted"


class ProvenanceVerifierError(RuntimeError):
    """Raised only when c2patool cannot be invoked at all (binary missing, etc.)."""


@dataclass(frozen=True)
class ProvenanceResult:
    """Outcome of a verify call.

    ``has_valid_signature`` is True only when c2patool reports
    ``validation_state == "Valid"`` (cryptographic integrity holds end-to-end).
    Missing manifests and any cryptographic mismatch produce ``False`` with a
    plain-English ``reason``; ``issuer`` is the leaf cert's common name when
    available.
    """

    has_valid_signature: bool
    issuer: str | None
    reason: str


def _failure_codes(report: dict[str, Any]) -> list[str]:
    results = report.get("validation_results")
    if not isinstance(results, dict):
        return []
    active = results.get("activeManifest")
    if not isinstance(active, dict):
        return []
    failures = active.get("failure")
    if not isinstance(failures, list):
        return []
    out: list[str] = []
    for entry in failures:
        if isinstance(entry, dict):
            code = entry.get("code")
            if isinstance(code, str):
                out.append(code)
    return out


def _issuer_common_name(report: dict[str, Any]) -> str | None:
    active_id = report.get("active_manifest")
    manifests = report.get("manifests")
    if not isinstance(active_id, str) or not isinstance(manifests, dict):
        return None
    active = manifests.get(active_id)
    if not isinstance(active, dict):
        return None
    info = active.get("signature_info")
    if not isinstance(info, dict):
        return None
    cn = info.get("common_name") or info.get("issuer")
    return str(cn) if isinstance(cn, str) else None


def _summarise_failure(codes: list[str]) -> str:
    real = [c for c in codes if c != _UNTRUSTED_CRED]
    if not real:
        return "signing credential untrusted"
    return "manifest invalid: " + ", ".join(real)


def verify(input_path: Path, *, timeout_s: float = 30.0) -> ProvenanceResult:
    """Run c2patool against ``input_path`` and translate the report.

    Behaviour:

    - **No manifest** (c2patool exits non-zero, e.g. "No claim found") →
      ``has_valid_signature=False``, ``reason="no manifest"``.
    - **Valid** (``validation_state == "Valid"``, any
      ``signingCredential.untrusted`` warning is tolerated) →
      ``has_valid_signature=True``.
    - **Invalid** (``validation_state == "Invalid"``, e.g.
      ``claimSignature.mismatch``, ``assertion.dataHash.mismatch``) →
      ``has_valid_signature=False`` with the failure code(s) in ``reason``.
    """
    if shutil.which(C2PATOOL_BIN) is None:
        raise ProvenanceVerifierError(
            "c2patool binary not found on PATH — install separately "
            "(e.g. `brew install c2patool`); see CODING_STANDARDS §3."
        )
    if not input_path.is_file():
        return ProvenanceResult(
            has_valid_signature=False,
            issuer=None,
            reason=f"input not found: {input_path}",
        )

    completed = subprocess.run(  # noqa: S603 — fixed argv, no shell.
        [C2PATOOL_BIN, str(input_path)],
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        reason = "no manifest"
        if stderr and "no claim" not in stderr.lower():
            reason = f"no manifest ({stderr.splitlines()[0]})"
        return ProvenanceResult(has_valid_signature=False, issuer=None, reason=reason)

    try:
        report: dict[str, Any] = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return ProvenanceResult(
            has_valid_signature=False,
            issuer=None,
            reason="could not parse c2patool report",
        )

    issuer = _issuer_common_name(report)
    state = report.get("validation_state")
    failure_codes = _failure_codes(report)

    if state == "Valid":
        non_trust_failures = [c for c in failure_codes if c != _UNTRUSTED_CRED]
        if non_trust_failures:
            return ProvenanceResult(
                has_valid_signature=False,
                issuer=issuer,
                reason=_summarise_failure(failure_codes),
            )
        reason = "valid C2PA manifest"
        if _UNTRUSTED_CRED in failure_codes:
            reason += " (issuer not in trust list)"
        return ProvenanceResult(has_valid_signature=True, issuer=issuer, reason=reason)

    return ProvenanceResult(
        has_valid_signature=False,
        issuer=issuer,
        reason=_summarise_failure(failure_codes) if failure_codes else "manifest invalid",
    )
