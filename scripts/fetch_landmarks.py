"""Fetch the dlib 68-point facial-landmark predictor (opt-in, explicit user action).

Feature: F2 (Live Video Face Authenticity Verification)
ACM: 1.3, 1.6
Scope: in-product.md

The runtime detector never touches the network — this script is the **only**
sanctioned ingress for the predictor weights, parallel to ``gen_test_cert.py``
for F3 keys. Run it once by hand to populate ``models/`` (gitignored). The
detector then loads the file from ``DVP_DLIB_LANDMARKS_PATH`` (pydantic-settings
``Settings.dlib_landmarks_path``) or the default ``models/`` location.

Source: https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
(the canonical Davis King distribution, CC0).

Verified by SHA-256 on the decompressed ``.dat``; mismatches abort. Pass
``--sha256-skip`` to override only with a known reason.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Final

CANONICAL_URL: Final[str] = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
# SHA-256 of the **decompressed** shape_predictor_68_face_landmarks.dat file,
# pinned to the canonical Davis King distribution. Recomputed on every run; a
# mismatch aborts (CODING_STANDARDS §7 — fail loudly).
EXPECTED_SHA256: Final[str] = "fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f"
DEFAULT_DEST: Final[Path] = Path("models/shape_predictor_68_face_landmarks.dat")


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:  # noqa: S310 — fixed, audited URL
        return bytes(response.read())


def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def fetch(dest: Path, *, url: str, expected_sha256: str | None) -> None:
    """Download, decompress, verify, and write the predictor file to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    compressed = _download(url)
    decompressed = bz2.decompress(compressed)
    digest = _sha256(decompressed)
    if expected_sha256 is not None and digest != expected_sha256:
        raise SystemExit(
            "SHA-256 mismatch on decompressed predictor:\n"
            f"  expected: {expected_sha256}\n"
            f"  computed: {digest}\n"
            "Refusing to write file. Re-verify the upstream source or pass "
            "--sha256-skip with a known reason."
        )
    dest.write_bytes(decompressed)
    sys.stdout.write(f"wrote {dest} ({len(decompressed):,} bytes, sha256={digest})\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--url", type=str, default=CANONICAL_URL)
    parser.add_argument(
        "--sha256-skip",
        action="store_true",
        help="Skip SHA-256 verification (use only with a known reason).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing destination file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dest.exists() and not args.force:
        raise SystemExit(f"{args.dest} already exists — pass --force to overwrite.")
    fetch(
        args.dest,
        url=args.url,
        expected_sha256=None if args.sha256_skip else EXPECTED_SHA256,
    )


if __name__ == "__main__":
    main()
