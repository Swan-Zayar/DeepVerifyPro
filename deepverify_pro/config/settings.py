"""Typed, env-driven runtime configuration.

Feature: F1–F5 (shared configuration)
ACM: 1.6
Scope: in-product.md

Settings are loaded from ``DVP_*`` env vars / a local ``.env`` (gitignored).
No secrets are hard-coded (CODING_STANDARDS §7); key material lives only under
``keys_dir`` which is gitignored (ACM 1.6).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the prototype."""

    model_config = SettingsConfigDict(env_prefix="DVP_", env_file=".env", extra="ignore")

    keys_dir: Path = Field(default=Path("keys"))
    audit_path: Path = Field(default=Path("var/audit.jsonl"))
    amber_at: float = Field(default=0.40, ge=0.0, le=1.0)
    red_at: float = Field(default=0.70, ge=0.0, le=1.0)
    financial_amount_threshold: float = Field(default=10_000.0, ge=0.0)
    # F2 — path to the dlib 68-point predictor (fetch via
    # ``scripts/fetch_landmarks.py``; weights are never committed, ACM 1.6).
    dlib_landmarks_path: Path = Field(default=Path("models/shape_predictor_68_face_landmarks.dat"))
    # F4 — out-of-band challenge log written by the on-prem file channel
    # (``var/`` is gitignored; challenges never leave the machine, ACM 1.6).
    challenge_log_path: Path = Field(default=Path("var/challenges.jsonl"))
    # F3 — prototype signing materials. ``keys/`` is gitignored and key
    # material is never committed (§4.1 / ACM 1.6); override via env if needed.
    signing_cert_path: Path = Field(default=Path("keys/test_signing.crt"))
    signing_key_path: Path = Field(default=Path("keys/test_signing.key"))
    # M6 HTTP surface. ACM 1.6: the API binds localhost by default — media
    # endpoints must not be exposed on a public interface in the prototype.
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_cors_origins: list[str] = Field(default=["http://localhost:5173", "http://127.0.0.1:5173"])


def get_settings() -> Settings:
    """Construct :class:`Settings` from the environment."""
    return Settings()
