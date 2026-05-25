"""Subprocess wrapper around c2patool — verify a C2PA manifest.

Feature: F3 (Cryptographic Content Provenance Signing)
ACM: 2.5, 3.7
Scope: in-product.md

Returns a frozen :class:`ProvenanceResult` carrying ``has_valid_signature``
(cryptographic integrity), ``is_trusted_issuer`` (the leaf-cert common name
is on the deployment's allow-list), the issuer string, and a short reason.
No raw bytes, hashes, or assertions are surfaced — keeping the result safe
for the F5 audit log (ACM 1.6).

**``has_valid_signature`` vs ``is_trusted_issuer`` — keep these distinct.**
Cryptographic validity tells us the bytes were signed by *someone* with a
matching key; it does not tell us that someone is on our trust list. An
attacker can produce a cryptographically valid C2PA manifest with a freshly
self-signed cert. The honest answer is two booleans, both surfaced — callers
that gate authorisation on provenance (F3+F4 composition) must require both
(ACM 1.3 / 2.5 — conflating the two is the §5 anti-pattern this module exists
to prevent).

The cryptographic integrity check (``claimSignature``, ``assertion.dataHash``)
determines ``has_valid_signature``. A ``signingCredential.untrusted`` warning
is emitted by c2patool whenever the issuing CA is not in its bundled trust
list — the prototype intentionally uses a local test CA, so we treat that
warning as non-fatal for the crypto check but keep it in the reason string
for transparency (ACM 1.3). Trust is decided separately by checking the
leaf-cert common name against the deployment's allow-list.
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

    ``is_trusted_issuer`` is True iff ``issuer`` is non-empty AND appears in
    the ``trusted_issuers`` allow-list passed to :func:`verify`. This is a
    separate signal from cryptographic validity — see module docstring.
    Callers may have a valid signature from an untrusted issuer (an attacker
    self-signing), or a trusted issuer string with an invalid signature
    (tampering after signing). The honest answer surfaces both.
    """

    has_valid_signature: bool
    issuer: str | None
    reason: str
    is_trusted_issuer: bool = False


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


def _is_trusted(issuer: str | None, trusted_issuers: tuple[str, ...]) -> bool:
    """Trust check: leaf-cert CN must appear in the deployment's allow-list.

    Empty ``trusted_issuers`` ⇒ False for every issuer (fail-closed). The
    deploying organisation must configure ``Settings.signing_trusted_issuers``
    with its own signing infra's CNs; until they do, no signer is trusted —
    that is the honest answer rather than silently trusting everyone (ACM
    1.3 / §5 anti-pattern).
    """
    if not issuer:
        return False
    return issuer in trusted_issuers


def verify(
    input_path: Path,
    *,
    timeout_s: float = 30.0,
    trusted_issuers: tuple[str, ...] = (),
) -> ProvenanceResult:
    """Run c2patool against ``input_path`` and translate the report.

    Behaviour:

    - **No manifest** (c2patool exits non-zero, e.g. "No claim found") →
      ``has_valid_signature=False``, ``reason="no manifest"``,
      ``is_trusted_issuer=False``.
    - **Valid** (``validation_state == "Valid"``, any
      ``signingCredential.untrusted`` warning is tolerated for the crypto
      check) → ``has_valid_signature=True``. ``is_trusted_issuer`` is True
      iff the leaf-cert common name is in ``trusted_issuers``.
    - **Invalid** (``validation_state == "Invalid"``, e.g.
      ``claimSignature.mismatch``, ``assertion.dataHash.mismatch``) →
      ``has_valid_signature=False`` with the failure code(s) in ``reason``.
      ``is_trusted_issuer`` is False — trust on an invalid signature is
      meaningless (an attacker may have spoofed a trusted CN onto a forged
      manifest).
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
            is_trusted_issuer=False,
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
        return ProvenanceResult(
            has_valid_signature=False,
            issuer=None,
            reason=reason,
            is_trusted_issuer=False,
        )

    try:
        report: dict[str, Any] = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return ProvenanceResult(
            has_valid_signature=False,
            issuer=None,
            reason="could not parse c2patool report",
            is_trusted_issuer=False,
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
                is_trusted_issuer=False,
            )
        trusted = _is_trusted(issuer, trusted_issuers)
        reason = "valid C2PA manifest"
        if not trusted:
            # Honest tail on the reason string. Empty allow-list and a
            # known-bad issuer collapse to the same message — both mean the
            # deployment does not vouch for this signer.
            reason += " (issuer not in deployment trust list)"
        elif _UNTRUSTED_CRED in failure_codes:
            # Trusted by the deployment but c2patool's bundled CA chain
            # didn't recognise the CA — typical for an on-prem self-signed
            # CA. Surface both facts.
            reason += " (deployment-trusted issuer; c2patool CA bundle does not recognise the CA)"
        return ProvenanceResult(
            has_valid_signature=True,
            issuer=issuer,
            reason=reason,
            is_trusted_issuer=trusted,
        )

    return ProvenanceResult(
        has_valid_signature=False,
        issuer=issuer,
        reason=_summarise_failure(failure_codes) if failure_codes else "manifest invalid",
        is_trusted_issuer=False,
    )
