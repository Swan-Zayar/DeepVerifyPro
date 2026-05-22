"""HTTP surface package — FastAPI backend over the F1–F5 orchestrator.

Feature: F1–F5 (HTTP surface — Part A sign, Part B detect/verify/audit)
ACM: 1.2, 1.3, 1.6, 2.5, 3.1, 3.7
Scope: in-product.md

The web surface is a thin adapter onto the existing deterministic tools and
the ADK orchestrator (CODING_STANDARDS §2 lists ``api/`` as a surface;
FastAPI was deferred in earlier rounds and is reintroduced here via owner
discussion). It binds localhost only — uploaded media is decoded in-process
and never leaves the machine (ACM 1.6).

Run it with ``python -m deepverify_pro.api``.
"""

from deepverify_pro.api.app import build_default_orchestrator, create_app

__all__ = ["build_default_orchestrator", "create_app"]
