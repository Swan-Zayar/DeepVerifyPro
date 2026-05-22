"""Fetch / verify the EfficientNet-B4 Self-Blended-Images checkpoint (opt-in).

Feature: F2 (Live Video Face Authenticity Verification)
ACM: 1.3, 1.6
Scope: in-product.md

RESEARCH-ONLY WEIGHTS. The SBI checkpoint and its FaceForensics++ training data
are licensed for non-commercial academic / research use only (M8 §7). DeepVerify
Pro is a non-commercial research prototype — do not install these weights for a
commercial deployment.

The runtime detector never touches the network — this script is the only
sanctioned ingress for the weights, parallel to ``fetch_landmarks.py``. The SBI
checkpoint is distributed via Google Drive from the SBI repository
(https://github.com/mapooon/SelfBlendedImages); obtain the ``FFraw.tar`` (or
``FFc23.tar``) checkpoint from there under its research licence.

Because the upstream file is served from Google Drive there is no stable direct
URL and no SHA-256 this project can pin ahead of time — pinning a hash the
repository has never computed would be a fabricated value (CODING_STANDARDS
§4.2). Instead this script:

  * accepts the checkpoint via ``--from-file`` (recommended) or ``--url``,
  * computes and PRINTS its SHA-256 so you can pin it yourself,
  * enforces ``--expected-sha256`` when you supply one,
  * structurally VERIFIES the file really is a loadable SBI EfficientNet-B4
    checkpoint (the strongest integrity check without a pinned hash),
  * installs it to ``models/`` (gitignored — never committed, ACM 1.6).

Usage::

    python scripts/fetch_sbi_weights.py --from-file ~/Downloads/FFraw.tar
    python scripts/fetch_sbi_weights.py --from-file FFraw.tar \\
        --expected-sha256 <hash pinned on a previous verified run>
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Final

DEFAULT_DEST: Final[Path] = Path("models/sbi_efficientnet_b4.pth")
DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 120
# Not pinned: the upstream checkpoint is distributed via Google Drive and this
# repository has never computed its hash. Pin it yourself after a first verified
# run (the script prints the computed SHA-256). Never commit a hash the project
# has not actually verified (CODING_STANDARDS §4.2 — zero fabricated values).
EXPECTED_SHA256: Final[str | None] = None


def _sha256(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def _download(url: str) -> bytes:
    try:
        with urllib.request.urlopen(  # noqa: S310 — caller-supplied, audited URL.
            url, timeout=DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            return bytes(response.read())
    except OSError as exc:
        raise SystemExit(
            f"Failed to download {url} (timeout {DOWNLOAD_TIMEOUT_SECONDS}s): {exc}\n"
            "Google Drive large-file links often need a manual download — fetch the "
            "checkpoint by hand and pass it with --from-file instead."
        ) from exc


def _verify_is_sbi_checkpoint(path: Path) -> None:
    """Confirm the file loads as an SBI EfficientNet-B4 checkpoint."""
    try:
        from deepverify_pro.detection.video.efficientnet_sbi import (  # noqa: PLC0415
            SBIDetectorError,
            verify_checkpoint,
        )
    except ImportError as exc:
        raise SystemExit(
            "Cannot import the SBI detector — install the model extra first: "
            "pip install -e '.[video,video-model]'"
        ) from exc
    try:
        verify_checkpoint(path)
    except SBIDetectorError as exc:
        raise SystemExit(
            f"Verification failed — this is not a usable SBI checkpoint:\n  {exc}"
        ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--from-file", type=Path, default=None, help="a pre-downloaded checkpoint")
    parser.add_argument("--url", type=str, default=None, help="a direct download URL")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="install destination")
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        help="enforce this SHA-256 on the checkpoint file",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing destination")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.from_file is None and args.url is None:
        raise SystemExit("provide --from-file PATH or --url URL")
    if args.from_file is not None and args.url is not None:
        raise SystemExit("provide only one of --from-file / --url")
    if args.dest.exists() and not args.force:
        raise SystemExit(f"{args.dest} already exists — pass --force to overwrite.")

    if args.from_file is not None:
        source: Path = args.from_file
        if not source.exists():
            raise SystemExit(f"file not found: {source}")
        data = source.read_bytes()
    else:
        sys.stdout.write(f"downloading {args.url} ...\n")
        data = _download(args.url)

    digest = _sha256(data)
    sys.stdout.write(f"SHA-256: {digest}\n")
    expected = args.expected_sha256 or EXPECTED_SHA256
    if expected is not None and digest != expected:
        raise SystemExit(
            f"SHA-256 mismatch:\n  expected: {expected}\n  computed: {digest}\n"
            "Refusing to install."
        )
    if expected is None:
        sys.stdout.write(
            "NOTE: no SHA-256 was pinned. Record the hash above and pass it as "
            "--expected-sha256 on future runs to pin it.\n"
        )

    # Write to a temp file beside the destination, verify it really is an SBI
    # checkpoint, then atomically move it into place.
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=args.dest.parent) as handle:
        handle.write(data)
        temp_path = Path(handle.name)
    try:
        sys.stdout.write("verifying the checkpoint loads as EfficientNet-B4 / SBI ...\n")
        _verify_is_sbi_checkpoint(temp_path)
        shutil.move(str(temp_path), str(args.dest))
    finally:
        if temp_path.exists():
            temp_path.unlink()

    sys.stdout.write(f"installed {args.dest} ({len(data):,} bytes)\n")
    sys.stdout.write(
        "Reminder: these are research-only weights (M8 §7). Do not use them in a "
        "commercial deployment.\n"
    )


if __name__ == "__main__":
    main()
