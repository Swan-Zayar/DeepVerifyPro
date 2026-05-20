"""Subprocess wrapper around the c2patool binary — embeds a C2PA manifest.

Feature: F3 (Cryptographic Content Provenance Signing)
ACM: 2.5, 3.7
Scope: in-product.md

Why c2patool rather than the ``c2pa-python`` binding: the binding cannot
produce a verifiable claim signature for an offline self-signed signer (see
CODING_STANDARDS §3 — owner-approved deviation). c2patool is the CAI reference
implementation and is invoked locally only — no network, no TSA (ACM 1.6).

The signing key path is passed to c2patool via a manifest JSON file the
process reads; the key never enters the Python process. Stdout/stderr from
c2patool are returned to the caller but never logged at module import time
(the audit-event emission happens in the F5 tool wrapper, not here).

Future-seam note (§0.8 / CODING_STANDARDS §3): swapping the local cert for
Cloud KMS / a remote signer means replacing the manifest JSON's
``private_key`` / ``sign_cert`` fields with ``--signer-path`` invoking a
local signer binary. This module is the single seam where that swap lands —
do not introduce KMS abstractions until that work is actually scheduled.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import NameOID

C2PATOOL_BIN: Final[str] = "c2patool"
DEFAULT_CLAIM_GENERATOR: Final[str] = "deepverify_pro/0.0.1"


class ProvenanceSignerError(RuntimeError):
    """Raised when the c2patool subprocess cannot produce a signed asset."""


@dataclass(frozen=True)
class SignResult:
    """Outcome of a successful sign call.

    ``issuer`` is the leaf cert's common name (the human-meaningful signer
    identity for the operator UI). It comes from the cert, not from any
    user-supplied input.
    """

    output_path: Path
    issuer: str


def _read_leaf_common_name(cert_path: Path) -> str:
    """Read the common name from the first cert in a PEM bundle."""
    pem = cert_path.read_bytes()
    cert = x509.load_pem_x509_certificate(pem)
    cns = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    if not cns:
        raise ProvenanceSignerError(f"leaf cert {cert_path} has no Common Name")
    cn = cns[0].value
    if isinstance(cn, bytes):
        cn = cn.decode("utf-8", errors="replace")
    return str(cn)


def _ensure_pkcs8(key_path: Path) -> None:
    """Sanity check: the signing key parses and is an EC private key."""
    key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    # The presence of `private_numbers` on EC keys is the cheap structural check.
    if not hasattr(key, "private_numbers"):
        raise ProvenanceSignerError(f"signing key {key_path} is not an EC private key")


def _build_manifest(
    cert_path: Path,
    key_path: Path,
    claim_generator: str,
) -> dict[str, object]:
    return {
        "claim_generator": claim_generator,
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {"actions": [{"action": "c2pa.created"}]},
            }
        ],
        "alg": "es256",
        "private_key": str(key_path.resolve()),
        "sign_cert": str(cert_path.resolve()),
    }


def sign(
    input_path: Path,
    output_path: Path,
    *,
    cert_path: Path,
    key_path: Path,
    claim_generator: str = DEFAULT_CLAIM_GENERATOR,
    timeout_s: float = 30.0,
) -> SignResult:
    """Embed a C2PA manifest into ``input_path`` and write to ``output_path``.

    Raises :class:`ProvenanceSignerError` if c2patool is missing, the cert/key
    pair is malformed, or c2patool exits non-zero.
    """
    if shutil.which(C2PATOOL_BIN) is None:
        raise ProvenanceSignerError(
            "c2patool binary not found on PATH — install separately "
            "(e.g. `brew install c2patool`); see CODING_STANDARDS §3."
        )
    if not input_path.is_file():
        raise ProvenanceSignerError(f"input asset not found: {input_path}")
    if not cert_path.is_file():
        raise ProvenanceSignerError(f"signing cert not found: {cert_path}")
    if not key_path.is_file():
        raise ProvenanceSignerError(f"signing key not found: {key_path}")

    _ensure_pkcs8(key_path)
    issuer = _read_leaf_common_name(cert_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.parent / f".{output_path.name}.manifest.json"
    manifest_path.write_text(
        json.dumps(_build_manifest(cert_path, key_path, claim_generator)),
        encoding="utf-8",
    )

    try:
        completed = subprocess.run(  # noqa: S603 — fixed argv, no shell.
            [
                C2PATOOL_BIN,
                str(input_path),
                "-m",
                str(manifest_path),
                "-o",
                str(output_path),
                "-f",
            ],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    finally:
        manifest_path.unlink(missing_ok=True)

    if completed.returncode != 0 or not output_path.is_file():
        raise ProvenanceSignerError(
            f"c2patool sign failed (exit {completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )

    return SignResult(output_path=output_path, issuer=issuer)
