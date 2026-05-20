"""Mint a CA + leaf ES256 test cert chain for c2patool prototype signing.

Feature: F3 (Cryptographic Content Provenance Signing)
ACM: 2.5, 3.7
Scope: in-product.md

Produces three PEM files under ``keys/`` (gitignored, test-only —
CODING_STANDARDS §4.1):

- ``keys/test_ca.crt``: self-signed P-256 root CA (the trust anchor).
- ``keys/test_signing.crt``: leaf signer cert chain (leaf + CA concatenated),
  signed by the CA, with EKU ``emailProtection`` (the OID c2patool / C2PA
  require on end-entity signing certs). c2patool 0.26 refuses to embed a
  manifest with a self-signed end-entity cert, so a real chain is required.
- ``keys/test_signing.key``: leaf private key (P-256 ECDSA, PKCS#8, unencrypted
  — prototype only).

The CA cert is the file to register with c2patool as a trust anchor
(``C2PATOOL_TRUST_ANCHORS`` / ``--trust_anchors``).

This material does not authenticate any real identity and must not be used
outside the local prototype.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

EMAIL_PROTECTION_OID = ExtendedKeyUsageOID.EMAIL_PROTECTION


def _name(common_name: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "AU"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "DeepVerify Pro (test only)"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Prototype"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )


def _mint_ca(validity_days: int) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    key = ec.generate_private_key(ec.SECP256R1())
    name = _name("DeepVerify Pro Test Root CA")
    now = dt.datetime.now(tz=dt.UTC)
    ski = x509.SubjectKeyIdentifier.from_public_key(key.public_key())
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=validity_days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(ski, critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(ski),
            critical=False,
        )
        .sign(private_key=key, algorithm=hashes.SHA256())
    )
    return cert, key


def _mint_leaf(
    ca_cert: x509.Certificate,
    ca_key: ec.EllipticCurvePrivateKey,
    validity_days: int,
) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    key = ec.generate_private_key(ec.SECP256R1())
    subject = _name("DeepVerify Pro Test Signer")
    now = dt.datetime.now(tz=dt.UTC)
    ca_ski_ext = ca_cert.extensions.get_extension_for_class(x509.SubjectKeyIdentifier)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=validity_days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([EMAIL_PROTECTION_OID]), critical=True)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(ca_ski_ext.value),
            critical=False,
        )
        .sign(private_key=ca_key, algorithm=hashes.SHA256())
    )
    return cert, key


def mint_chain(
    ca_cert_path: Path,
    signing_cert_path: Path,
    signing_key_path: Path,
    *,
    validity_days: int = 365,
) -> None:
    """Mint and write a CA + leaf PEM chain plus the leaf private key."""
    ca_cert, ca_key = _mint_ca(validity_days=validity_days * 2)
    leaf_cert, leaf_key = _mint_leaf(ca_cert, ca_key, validity_days=validity_days)

    ca_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
    leaf_pem = leaf_cert.public_bytes(serialization.Encoding.PEM)

    for p in (ca_cert_path, signing_cert_path, signing_key_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    ca_cert_path.write_bytes(ca_pem)
    # c2patool's `sign_cert` expects leaf-first, anchor-last PEM bundle.
    signing_cert_path.write_bytes(leaf_pem + ca_pem)
    signing_key_path.write_bytes(
        leaf_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    try:
        signing_key_path.chmod(0o600)
    except OSError:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca-cert", type=Path, default=Path("keys/test_ca.crt"))
    parser.add_argument("--cert", type=Path, default=Path("keys/test_signing.crt"))
    parser.add_argument("--key", type=Path, default=Path("keys/test_signing.key"))
    parser.add_argument("--days", type=int, default=365)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mint_chain(args.ca_cert, args.cert, args.key, validity_days=args.days)


if __name__ == "__main__":
    main()
