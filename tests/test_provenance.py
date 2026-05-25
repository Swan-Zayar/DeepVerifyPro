"""F3 provenance tests: sign→verify roundtrip, tamper detection, unsigned reject.

Feature: F3 (Cryptographic Content Provenance Signing)
ACM: 2.5, 3.7
Scope: in-product.md
"""

from __future__ import annotations

import shutil
import struct
import wave
import zlib
from pathlib import Path

import pytest

from deepverify_pro.agents import DeepVerifyOrchestrator
from deepverify_pro.audit.log import AuditLog
from deepverify_pro.provenance import SignResult, sign, verify
from deepverify_pro.tools.provenance_verify import provenance_verify
from deepverify_pro.tools.sign_media import sign_media
from scripts.gen_test_cert import mint_chain

pytestmark = pytest.mark.skipif(
    shutil.which("c2patool") is None,
    reason="c2patool binary not installed",
)


def _write_tiny_png(path: Path) -> None:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + typ
            + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    idat = zlib.compress(b"\x00" + b"\x00\x00\x00\x00")
    path.write_bytes(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def _write_tiny_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 800)  # 0.1 s of silence


@pytest.fixture
def tiny_png(tmp_path: Path) -> Path:
    p = tmp_path / "in.png"
    _write_tiny_png(p)
    return p


@pytest.fixture
def tiny_wav(tmp_path: Path) -> Path:
    p = tmp_path / "in.wav"
    _write_tiny_wav(p)
    return p


@pytest.fixture
def signing_chain(tmp_path: Path) -> tuple[Path, Path]:
    ca = tmp_path / "ca.crt"
    cert = tmp_path / "leaf.crt"
    key = tmp_path / "leaf.key"
    mint_chain(ca, cert, key, validity_days=30)
    return cert, key


def test_sign_verify_roundtrip_png(
    tiny_png: Path,
    signing_chain: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    cert, key = signing_chain
    out = tmp_path / "signed.png"

    sign_result = sign(tiny_png, out, cert_path=cert, key_path=key)
    assert isinstance(sign_result, SignResult)
    assert sign_result.output_path.is_file()
    assert sign_result.issuer == "DeepVerify Pro Test Signer"

    result = verify(out)
    assert result.has_valid_signature is True
    assert result.issuer == "DeepVerify Pro Test Signer"
    assert "valid C2PA manifest" in result.reason


def test_tamper_after_sign_is_detected_png(
    tiny_png: Path,
    signing_chain: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    cert, key = signing_chain
    out = tmp_path / "signed.png"
    sign(tiny_png, out, cert_path=cert, key_path=key)

    # Flip one byte in the manifest region (near end-of-file). c2patool will
    # report either claimSignature.mismatch or assertion.dataHash.mismatch —
    # any cryptographic-integrity failure is acceptable here.
    blob = bytearray(out.read_bytes())
    blob[-100] ^= 0x55
    out.write_bytes(bytes(blob))

    result = verify(out)
    assert result.has_valid_signature is False
    lowered = result.reason.lower()
    assert "mismatch" in lowered or "invalid" in lowered


def test_verify_unsigned_returns_no_manifest(tiny_png: Path) -> None:
    result = verify(tiny_png)
    assert result.has_valid_signature is False
    assert result.issuer is None
    assert "no manifest" in result.reason.lower()


def test_sign_verify_roundtrip_wav(
    tiny_wav: Path,
    signing_chain: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    cert, key = signing_chain
    out = tmp_path / "signed.wav"

    sign(tiny_wav, out, cert_path=cert, key_path=key)
    result = verify(out)
    assert result.has_valid_signature is True
    assert result.issuer == "DeepVerify Pro Test Signer"


def test_tools_emit_clean_audit_events(
    tiny_png: Path,
    signing_chain: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    cert, key = signing_chain
    out = tmp_path / "signed.png"
    audit = AuditLog(tmp_path / "audit.jsonl")

    sign_media(tiny_png, out, cert_path=cert, key_path=key, audit=audit)
    provenance_verify(out, audit=audit)

    records = audit.read_all()
    assert [r.event for r in records] == ["provenance.sign", "provenance.verify"]
    assert audit.verify_chain() is True

    # No record may carry raw-media keys (re-asserts ACM 1.6 via the audit guard).
    for record in records:
        for forbidden in ("data", "frame", "mfcc", "audio", "video"):
            assert forbidden not in record.payload


def test_orchestrator_sign_routes_f3_through_shared_audit_chain(
    tiny_png: Path,
    signing_chain: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """M5: ``orchestrator.sign()`` routes F3 signing into the shared F5 chain.

    Verifies the orchestrator docstring claim — Part A signing is recorded in
    the same hash chain as the detection-side events (CODING_STANDARDS §4.4).
    """
    cert, key = signing_chain
    out = tmp_path / "signed.png"
    audit = AuditLog(tmp_path / "audit.jsonl")
    orchestrator = DeepVerifyOrchestrator(audit=audit)
    assert orchestrator.audit is audit

    orchestrator.sign(tiny_png, out, cert_path=cert, key_path=key)

    assert verify(out).has_valid_signature is True
    assert [r.event for r in audit.read_all()] == ["provenance.sign"]
    assert audit.verify_chain() is True


# ---------- F3 trust list (deployment allow-list) ----------
#
# These tests pin the ACM 1.3 / 2.5 honesty fix: a cryptographically valid
# signature does NOT imply the signer is on the deployment's trust list. The
# two booleans must remain distinct, and callers gating financial decisions
# on provenance must consult both.

_TEST_ISSUER = "DeepVerify Pro Test Signer"


def test_empty_trust_list_fails_closed(
    tiny_png: Path,
    signing_chain: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """No allow-list configured ⇒ no issuer is trusted, even a valid one."""
    cert, key = signing_chain
    out = tmp_path / "signed.png"
    sign(tiny_png, out, cert_path=cert, key_path=key)

    result = verify(out, trusted_issuers=())
    assert result.has_valid_signature is True
    assert result.is_trusted_issuer is False
    assert "not in deployment trust list" in result.reason


def test_trust_list_with_matching_issuer_marks_trusted(
    tiny_png: Path,
    signing_chain: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    cert, key = signing_chain
    out = tmp_path / "signed.png"
    sign(tiny_png, out, cert_path=cert, key_path=key)

    result = verify(out, trusted_issuers=(_TEST_ISSUER,))
    assert result.has_valid_signature is True
    assert result.is_trusted_issuer is True


def test_trust_list_with_unknown_issuer_remains_untrusted(
    tiny_png: Path,
    signing_chain: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    cert, key = signing_chain
    out = tmp_path / "signed.png"
    sign(tiny_png, out, cert_path=cert, key_path=key)

    result = verify(out, trusted_issuers=("Some Other Org",))
    assert result.has_valid_signature is True
    assert result.is_trusted_issuer is False


def test_no_manifest_is_never_trusted(tiny_png: Path) -> None:
    result = verify(tiny_png, trusted_issuers=(_TEST_ISSUER,))
    assert result.has_valid_signature is False
    assert result.is_trusted_issuer is False


def test_tampered_signature_is_never_trusted_even_with_matching_cn(
    tiny_png: Path,
    signing_chain: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Tampered manifest ⇒ ``is_trusted_issuer=False`` even if the CN matches.

    Trust on an invalid signature is meaningless — an attacker may have spoofed
    the CN onto a forged manifest. Defence-in-depth (ACM 1.2): trust requires
    BOTH the allow-list match AND cryptographic validity.
    """
    cert, key = signing_chain
    out = tmp_path / "signed.png"
    sign(tiny_png, out, cert_path=cert, key_path=key)

    blob = bytearray(out.read_bytes())
    blob[-100] ^= 0x55
    out.write_bytes(bytes(blob))

    result = verify(out, trusted_issuers=(_TEST_ISSUER,))
    assert result.has_valid_signature is False
    assert result.is_trusted_issuer is False


def test_provenance_verify_tool_logs_trust_signal(
    tiny_png: Path,
    signing_chain: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """The audit payload must carry ``is_trusted_issuer`` so the F5 chain
    distinguishes valid-but-untrusted from valid-and-trusted (ACM 3.1 / 3.7)."""
    cert, key = signing_chain
    out = tmp_path / "signed.png"
    sign(tiny_png, out, cert_path=cert, key_path=key)
    audit = AuditLog(tmp_path / "audit.jsonl")

    provenance_verify(out, audit=audit, trusted_issuers=(_TEST_ISSUER,))
    records = audit.read_all()
    assert len(records) == 1
    payload = records[0].payload
    assert payload["has_valid_signature"] is True
    assert payload["is_trusted_issuer"] is True
    assert audit.verify_chain() is True
