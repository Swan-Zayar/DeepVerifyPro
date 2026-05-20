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


def get_settings() -> Settings:
    """Construct :class:`Settings` from the environment."""
    return Settings()
