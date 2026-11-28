"""Central, typed configuration (environment variables prefixed with ``DOCAI_``).

Every setting is read and validated here, once. The rest of the code receives
values through function arguments or the composition root
(:mod:`docai.container`); nothing else reads ``os.environ``.

API keys for the optional real providers keep their conventional names
(``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``) and are held
as :class:`pydantic.SecretStr`, so they never appear in ``repr`` or logs.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ProviderName = Literal["simulated", "anthropic", "gemini", "bedrock", "auto"]
GuardrailPreset = Literal["off", "dv-only", "text", "full"]

MIN_TOKEN_LENGTH = 24


def find_project_root() -> Path:
    """Folder that holds ``pyproject.toml`` and ``knowledge/`` (cwd first, then this file)."""
    env = os.environ.get("DOCAI_HOME")
    if env:
        return Path(env).resolve()
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").exists() and (candidate / "knowledge").is_dir():
                return candidate
    return Path.cwd()


class Settings(BaseSettings):
    """All runtime configuration. Construct with keyword arguments in tests."""

    model_config = SettingsConfigDict(
        env_prefix="DOCAI_", extra="ignore", frozen=True, validate_default=True
    )

    # ------------------------------------------------------------- locations
    home: Path = Field(default_factory=find_project_root)
    data_dir: Path = Field(default_factory=lambda d: d["home"] / "data")
    samples_dir: Path = Field(default_factory=lambda d: d["home"] / "samples")
    knowledge_dir: Path = Field(default_factory=lambda d: d["home"] / "knowledge")
    web_dir: Path = Field(default_factory=lambda d: d["home"] / "web")
    reports_dir: Path = Field(default_factory=lambda d: d["home"] / "reports")
    model_path: Path = Field(
        default_factory=lambda d: d["home"] / "models" / "doctype_classifier.pt"
    )
    db_path: Path = Field(
        default_factory=lambda d: d["home"] / "var" / "docai.db",
        description="SQLite file for the review queue and the run history",
    )
    mcp_root: Path = Field(
        default_factory=lambda d: d["samples_dir"],
        description="the only folder MCP tools may read (defaults to the samples folder)",
    )

    # ------------------------------------------------------------- pipeline
    provider: ProviderName = "simulated"
    guardrails: GuardrailPreset = "full"
    xml_first: bool = Field(True, description="use the fiscal XML instead of the LLM when present")
    llm_timeout_s: float = Field(60.0, gt=0, le=600)
    llm_max_output_tokens: int = Field(2048, ge=256, le=32_768)

    # ------------------------------------------------------------- input limits
    max_upload_bytes: int = Field(10 * 1024 * 1024, ge=1024, le=50 * 1024 * 1024)
    max_xml_bytes: int = Field(1024 * 1024, ge=1024, le=10 * 1024 * 1024)
    max_pdf_pages: int = Field(20, ge=1, le=500)
    max_text_chars: int = Field(200_000, ge=1_000, le=5_000_000)
    pdf_timeout_s: float = Field(15.0, gt=0, le=300)
    pdf_isolation: bool = Field(True, description="parse uploads in a killable child process")

    # ------------------------------------------------------------- API security
    api_token: SecretStr | None = Field(
        None, description=f"bearer token for mutating endpoints (>= {MIN_TOKEN_LENGTH} chars)"
    )
    review_public_read: bool = Field(
        False, description="GET /review/* without a token (off: the queue holds document data)"
    )
    review_max_pending: int = Field(1000, ge=1, le=1_000_000)
    rate_limit_per_minute: int = Field(30, ge=0, le=100_000, description="0 disables")
    rate_limit_burst: int = Field(10, ge=1, le=10_000)
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    docs_enabled: bool = True

    # ------------------------------------------------------------- logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    # ------------------------------------------------------------- real providers
    anthropic_api_key: SecretStr | None = Field(
        None, validation_alias=AliasChoices("DOCAI_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
    )
    gemini_api_key: SecretStr | None = Field(
        None,
        validation_alias=AliasChoices("DOCAI_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    anthropic_model: str | None = Field(
        None, description="Anthropic model id, required for that provider"
    )
    gemini_model: str = "gemini-2.5-flash"
    bedrock_model_id: str | None = None
    aws_region: str | None = Field(
        None, validation_alias=AliasChoices("DOCAI_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION")
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("cors_origins")
    @classmethod
    def _check_origins(cls, value: list[str]) -> list[str]:
        for origin in value:
            if origin == "*" or not origin.startswith(("https://", "http://")):
                raise ValueError(f"CORS origins must be explicit http(s) origins, got {origin!r}")
        return value

    @field_validator("anthropic_api_key", "gemini_api_key")
    @classmethod
    def _empty_secret_is_none(cls, value: SecretStr | None) -> SecretStr | None:
        return value if value is not None and value.get_secret_value().strip() else None

    @field_validator("anthropic_model", "bedrock_model_id")
    @classmethod
    def _empty_model_is_none(cls, value: str | None) -> str | None:
        return value.strip() if value is not None and value.strip() else None

    @field_validator("api_token")
    @classmethod
    def _check_token(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None or not value.get_secret_value():
            return None
        if len(value.get_secret_value()) < MIN_TOKEN_LENGTH:
            raise ValueError(f"DOCAI_API_TOKEN must have at least {MIN_TOKEN_LENGTH} characters")
        return value

    @field_validator(
        "home",
        "data_dir",
        "samples_dir",
        "knowledge_dir",
        "web_dir",
        "reports_dir",
        "model_path",
        "db_path",
        "mcp_root",
    )
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @property
    def mutations_enabled(self) -> bool:
        """Mutating endpoints are disabled unless a token is configured (secure by default)."""
        return self.api_token is not None

    def secret_values(self) -> list[str]:
        """Every configured secret, for log redaction."""
        secrets = (self.api_token, self.anthropic_api_key, self.gemini_api_key)
        return [s.get_secret_value() for s in secrets if s is not None]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read from the environment once."""
    return Settings()
