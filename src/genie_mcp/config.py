from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_DATABRICKS_HOST = "https://adb-415889795140801.1.azuredatabricks.net"
DEFAULT_GENIE_SPACE_ID = "01f0a8c81e88142fadad408f820867c3"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class ConfigError(RuntimeError):
    """Raised when required runtime configuration is missing."""


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {value!r}.") from exc


@dataclass(frozen=True)
class GenieConfig:
    databricks_host: str
    genie_space_id: str
    databricks_token: str | None = None
    databricks_auth_type: str | None = None
    databricks_config_profile: str | None = None
    databricks_cli_path: str | None = None
    poll_timeout_seconds: float = 600
    poll_initial_interval_seconds: float = 1
    poll_max_interval_seconds: float = 10

    @classmethod
    def from_env(cls) -> "GenieConfig":
        _load_env()

        host = os.getenv("DATABRICKS_HOST", DEFAULT_DATABRICKS_HOST).strip().rstrip("/")
        space_id = os.getenv("GENIE_SPACE_ID", DEFAULT_GENIE_SPACE_ID).strip()
        token = _empty_to_none(os.getenv("DATABRICKS_TOKEN", "").strip())
        auth_type = _empty_to_none(os.getenv("DATABRICKS_AUTH_TYPE", "").strip())
        config_profile = _empty_to_none(os.getenv("DATABRICKS_CONFIG_PROFILE", "").strip())
        cli_path = _empty_to_none(os.getenv("DATABRICKS_CLI_PATH", "").strip())

        missing = []
        if not host:
            missing.append("DATABRICKS_HOST")
        if not space_id:
            missing.append("GENIE_SPACE_ID")
        if missing:
            joined = ", ".join(missing)
            raise ConfigError(f"Missing required environment variable(s): {joined}.")

        return cls(
            databricks_host=host,
            genie_space_id=space_id,
            databricks_token=token,
            databricks_auth_type=auth_type,
            databricks_config_profile=config_profile,
            databricks_cli_path=cli_path,
            poll_timeout_seconds=_env_float("GENIE_POLL_TIMEOUT_SECONDS", 600),
            poll_initial_interval_seconds=_env_float("GENIE_POLL_INITIAL_INTERVAL_SECONDS", 1),
            poll_max_interval_seconds=_env_float("GENIE_POLL_MAX_INTERVAL_SECONDS", 10),
        )

    @classmethod
    def health_from_env(cls) -> dict[str, object]:
        _load_env()
        host = os.getenv("DATABRICKS_HOST", DEFAULT_DATABRICKS_HOST).strip().rstrip("/")
        space_id = os.getenv("GENIE_SPACE_ID", DEFAULT_GENIE_SPACE_ID).strip()
        token = os.getenv("DATABRICKS_TOKEN", "").strip()
        auth_type = os.getenv("DATABRICKS_AUTH_TYPE", "").strip()
        config_profile = os.getenv("DATABRICKS_CONFIG_PROFILE", "").strip()
        cli_path = os.getenv("DATABRICKS_CLI_PATH", "").strip()
        client_id = os.getenv("DATABRICKS_CLIENT_ID", "").strip()
        client_secret = os.getenv("DATABRICKS_CLIENT_SECRET", "").strip()
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        openai_model = os.getenv("OPENAI_MODEL", "gpt-5.5").strip()
        return {
            "databricks_host": host,
            "genie_space_id": space_id,
            "databricks_token_configured": bool(token),
            "databricks_auth_type": auth_type or None,
            "databricks_config_profile": config_profile or None,
            "databricks_cli_path": cli_path or None,
            "databricks_client_id_configured": bool(client_id),
            "databricks_client_secret_configured": bool(client_secret),
            "databricks_auth_hint": _auth_hint(
                token=token,
                auth_type=auth_type,
                config_profile=config_profile,
                client_id=client_id,
                client_secret=client_secret,
            ),
            "openai_api_key_configured": bool(openai_api_key),
            "openai_model": openai_model,
            "poll_timeout_seconds": _env_float("GENIE_POLL_TIMEOUT_SECONDS", 600),
            "poll_initial_interval_seconds": _env_float("GENIE_POLL_INITIAL_INTERVAL_SECONDS", 1),
            "poll_max_interval_seconds": _env_float("GENIE_POLL_MAX_INTERVAL_SECONDS", 10),
        }


def _empty_to_none(value: str) -> str | None:
    return value or None


def _load_env() -> None:
    load_dotenv(ENV_FILE)


def _auth_hint(
    *,
    token: str,
    auth_type: str,
    config_profile: str,
    client_id: str,
    client_secret: str,
) -> str:
    if token:
        return "personal-access-token"
    if auth_type:
        return auth_type
    if client_id and client_secret:
        return "oauth-m2m"
    if config_profile:
        return "config-profile"
    return "databricks-sdk-default-chain"
