from __future__ import annotations

import json
import subprocess
import time
from typing import Any

import httpx

from .config import GenieConfig, PROJECT_ROOT


class DatabricksAuthError(RuntimeError):
    """Raised when a Databricks bearer token cannot be resolved."""


_TOKEN_CACHE: dict[tuple[str, str, str], tuple[str, float]] = {}


def resolve_bearer_token(config: GenieConfig) -> str:
    if config.databricks_token:
        return config.databricks_token

    if config.databricks_auth_type == "databricks-cli":
        return get_cli_access_token(
            cli_path=config.databricks_cli_path,
            profile=config.databricks_config_profile,
        )

    if config.databricks_client_id and config.databricks_client_secret:
        return get_oauth_m2m_token(config)

    raise DatabricksAuthError(
        "Databricks auth needs DATABRICKS_TOKEN, DATABRICKS_AUTH_TYPE=databricks-cli, "
        "or DATABRICKS_CLIENT_ID plus DATABRICKS_CLIENT_SECRET."
    )


def get_cli_access_token(cli_path: str | None, profile: str | None) -> str:
    resolved_cli_path = _resolve_cli_path(cli_path)
    command = [
        resolved_cli_path,
        "auth",
        "token",
        "-o",
        "json",
        "--timeout",
        "10m",
    ]
    if profile:
        command.extend(["--profile", profile])

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=660,
        )
    except FileNotFoundError as exc:
        raise DatabricksAuthError(
            "Databricks CLI was not found. Set DATABRICKS_CLI_PATH or install the Databricks CLI."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()[:2000]
        raise DatabricksAuthError(f"Databricks CLI could not provide an OAuth token: {stderr}") from exc
    except subprocess.TimeoutExpired as exc:
        raise DatabricksAuthError("Databricks CLI token lookup timed out.") from exc

    try:
        payload = json.loads(completed.stdout)
    except ValueError as exc:
        raise DatabricksAuthError("Databricks CLI returned a non-JSON token response.") from exc

    access_token = payload.get("access_token")
    if not access_token:
        raise DatabricksAuthError("Databricks CLI token response did not include access_token.")
    return str(access_token)


def get_oauth_m2m_token(config: GenieConfig) -> str:
    client_id = config.databricks_client_id or ""
    client_secret = config.databricks_client_secret or ""
    scope = config.databricks_oauth_scope or "all-apis"
    cache_key = (config.databricks_host, client_id, scope)
    cached = _TOKEN_CACHE.get(cache_key)
    if cached and cached[1] > time.time() + 60:
        return cached[0]

    token_endpoint = _discover_token_endpoint(config.databricks_host)
    response = httpx.post(
        token_endpoint,
        data={
            "grant_type": "client_credentials",
            "scope": scope,
        },
        auth=(client_id, client_secret),
        timeout=httpx.Timeout(30.0, connect=10.0),
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise DatabricksAuthError(
            f"Databricks OAuth token request failed with HTTP {response.status_code}: {response.text[:1000]}"
        ) from exc

    payload = response.json()
    access_token = str(payload.get("access_token") or "")
    if not access_token:
        raise DatabricksAuthError("Databricks OAuth token response did not include access_token.")

    expires_in = _safe_int(payload.get("expires_in"), default=3600)
    _TOKEN_CACHE[cache_key] = (access_token, time.time() + expires_in)
    return access_token


def _discover_token_endpoint(host: str) -> str:
    host = host.rstrip("/")
    discovery_urls = []

    try:
        metadata = httpx.get(
            f"{host}/.well-known/databricks-config",
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        if metadata.status_code < 400:
            oidc_endpoint = (metadata.json().get("oidc_endpoint") or "").rstrip("/")
            if oidc_endpoint:
                discovery_urls.append(f"{oidc_endpoint}/.well-known/oauth-authorization-server")
    except Exception:
        pass

    discovery_urls.append(f"{host}/oidc/.well-known/oauth-authorization-server")

    for url in discovery_urls:
        try:
            response = httpx.get(url, timeout=httpx.Timeout(10.0, connect=5.0))
            if response.status_code >= 400:
                continue
            token_endpoint = response.json().get("token_endpoint")
            if token_endpoint:
                return str(token_endpoint)
        except Exception:
            continue

    return f"{host}/oidc/v1/token"


def _resolve_cli_path(cli_path: str | None) -> str:
    if not cli_path:
        bundled = PROJECT_ROOT / ".tools" / "databricks.exe"
        return str(bundled) if bundled.exists() else "databricks"

    path = cli_path.replace("/", "\\")
    candidate = PROJECT_ROOT / path
    if not _path_like(path) and candidate.exists():
        return str(candidate)
    return path


def _path_like(path: str) -> bool:
    return ":" in path or path.startswith("\\")


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
