"""
MythosForge API — Settings loaded from environment variables.

All configuration is centralised here.  Override via env vars prefixed with
MYTHOSFORGE_ or via a .env file in the project root.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — one source of truth for all configuration."""

    # ── Application ───────────────────────────────────
    app_name: str = "MythosForge API"
    env: str = "dev"
    debug: bool = False

    # ── Server ────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1

    # ── CORS ──────────────────────────────────────────
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    cors_allow_credentials: bool = True
    cors_methods: list[str] = ["GET", "POST", "DELETE", "OPTIONS"]
    cors_headers: list[str] = ["Authorization", "Content-Type", "X-Request-ID"]

    # ── Auth ──────────────────────────────────────────
    auth_enabled: bool = False
    api_keys: list[str] = []

    # ── Logging ───────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

    # ── Rate limiting ─────────────────────────────────
    rate_limit_per_minute: int = 60

    # ── Inference ─────────────────────────────────────
    inference_max_tokens: int = 512
    inference_max_loops: int = 32

    # ── Observability ─────────────────────────────────
    metrics_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MYTHOSFORGE_",
        extra="ignore",
    )


# Singleton — import this wherever settings are needed
settings = Settings()
