"""Provenance package — C2PA sign + verify via c2patool.

Feature: F3 (Cryptographic Content Provenance Signing)
ACM: 2.5, 3.7
Scope: in-product.md
"""

from deepverify_pro.provenance.signer import (
    ProvenanceSignerError,
    SignResult,
    sign,
)
from deepverify_pro.provenance.verifier import (
    ProvenanceResult,
    ProvenanceVerifierError,
    verify,
)

__all__ = [
    "ProvenanceResult",
    "ProvenanceSignerError",
    "ProvenanceVerifierError",
    "SignResult",
    "sign",
    "verify",
]
