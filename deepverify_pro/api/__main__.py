"""Launch the DeepVerify Pro API on localhost.

Feature: F1–F5 (HTTP surface launcher)
ACM: 1.6
Scope: in-product.md

Runs uvicorn bound to :class:`Settings.api_host` — ``127.0.0.1`` by default.
ACM 1.6: the prototype must not expose media-bearing endpoints on a public
interface; ``DVP_API_HOST`` should only be overridden for a trusted on-prem
deployment.

Run with ``python -m deepverify_pro.api``.
"""

from __future__ import annotations

import uvicorn

from deepverify_pro.api.app import create_app
from deepverify_pro.config import get_settings


def main() -> None:
    """Start the API server on the configured localhost host/port."""
    settings = get_settings()
    uvicorn.run(create_app(), host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
