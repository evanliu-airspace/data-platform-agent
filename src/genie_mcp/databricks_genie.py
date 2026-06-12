from __future__ import annotations

import asyncio
import json
import subprocess
import time
from typing import Any
from urllib.parse import quote

import httpx
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError

from .config import GenieConfig, PROJECT_ROOT


TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}


class GenieAPIError(RuntimeError):
    """Raised when Databricks Genie returns an API error."""


class GenieTimeoutError(RuntimeError):
    """Raised when Genie does not reach a terminal message status in time."""


class DatabricksGenieClient:
    def __init__(self, config: GenieConfig):
        self.config = config
        self._sdk_client: WorkspaceClient | None = None
        self._http_client: httpx.AsyncClient | None = None

        bearer_token = self._resolve_bearer_token()
        if bearer_token:
            self._http_client = httpx.AsyncClient(
                base_url=config.databricks_host,
                headers={
                    "Authorization": f"Bearer {bearer_token}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(60.0, connect=20.0),
            )
        else:
            self._sdk_client = WorkspaceClient(
                host=config.databricks_host,
                auth_type=config.databricks_auth_type,
                profile=config.databricks_config_profile,
                product="databricks-genie-mcp-agent",
                product_version="0.1.0",
            )

    async def __aenter__(self) -> "DatabricksGenieClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
        return None

    async def get_space(self, include_serialized_space: bool = False) -> dict[str, Any]:
        space_id = self._space_id()
        params = {"include_serialized_space": str(include_serialized_space).lower()}
        return await self._request("GET", f"/api/2.0/genie/spaces/{space_id}", params=params)

    async def start_conversation(self, content: str) -> dict[str, Any]:
        space_id = self._space_id()
        return await self._request(
            "POST",
            f"/api/2.0/genie/spaces/{space_id}/start-conversation",
            json={"content": content},
        )

    async def create_message(self, conversation_id: str, content: str) -> dict[str, Any]:
        space_id = self._space_id()
        conversation_id = _quote_id(conversation_id)
        return await self._request(
            "POST",
            f"/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages",
            json={"content": content},
        )

    async def get_message(self, conversation_id: str, message_id: str) -> dict[str, Any]:
        space_id = self._space_id()
        conversation_id = _quote_id(conversation_id)
        message_id = _quote_id(message_id)
        return await self._request(
            "GET",
            f"/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}",
        )

    async def wait_for_message(
        self,
        conversation_id: str,
        message_id: str,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        timeout_seconds = timeout_seconds or self.config.poll_timeout_seconds
        deadline = time.monotonic() + timeout_seconds
        delay = self.config.poll_initial_interval_seconds

        while True:
            message = await self.get_message(conversation_id, message_id)
            status = str(message.get("status", "")).upper()
            if status in TERMINAL_STATUSES:
                return message

            if time.monotonic() >= deadline:
                raise GenieTimeoutError(
                    f"Genie message {message_id} did not complete within {timeout_seconds:g} seconds. "
                    f"Last status: {status or 'UNKNOWN'}."
                )

            await asyncio.sleep(delay)
            delay = min(delay * 1.5, self.config.poll_max_interval_seconds)

    async def get_query_result(
        self,
        conversation_id: str,
        message_id: str,
        attachment_id: str,
    ) -> dict[str, Any]:
        space_id = self._space_id()
        conversation_id = _quote_id(conversation_id)
        message_id = _quote_id(message_id)
        attachment_id = _quote_id(attachment_id)
        return await self._request(
            "GET",
            "/api/2.0/genie/spaces/"
            f"{space_id}/conversations/{conversation_id}/messages/{message_id}"
            f"/query-result/{attachment_id}",
        )

    async def ask(
        self,
        question: str,
        conversation_id: str | None = None,
        wait: bool = True,
        include_query_results: bool = True,
        timeout_seconds: float | None = None,
        max_result_rows: int = 50,
    ) -> dict[str, Any]:
        if conversation_id:
            initial = await self.create_message(conversation_id, question)
            message = _message_from_response(initial)
            resolved_conversation_id = conversation_id
        else:
            initial = await self.start_conversation(question)
            message = _message_from_response(initial)
            conversation = initial.get("conversation") or {}
            resolved_conversation_id = str(
                conversation.get("id") or message.get("conversation_id") or ""
            )

        message_id = str(message.get("id") or "")
        if not resolved_conversation_id or not message_id:
            raise GenieAPIError("Genie response did not include conversation_id and message_id.")

        if wait:
            message = await self.wait_for_message(
                resolved_conversation_id,
                message_id,
                timeout_seconds=timeout_seconds,
            )

        compiled = compile_message(message, max_result_rows=max_result_rows)
        compiled["conversation_id"] = resolved_conversation_id
        compiled["message_id"] = message_id

        if include_query_results and str(message.get("status", "")).upper() == "COMPLETED":
            query_results: dict[str, Any] = {}
            for attachment_id in compiled["query_attachment_ids"]:
                raw_result = await self.get_query_result(
                    resolved_conversation_id,
                    message_id,
                    attachment_id,
                )
                query_results[attachment_id] = compact_query_result(raw_result, max_result_rows)
            compiled["query_results"] = query_results

        return compiled

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._http_client:
            return await self._http_request(method, path, params=params, json_body=json)

        if not self._sdk_client:
            raise GenieAPIError("Databricks client was not initialized.")

        try:
            result = await asyncio.to_thread(
                self._sdk_client.api_client.do,
                method,
                path,
                query=params,
                body=json,
            )
        except DatabricksError as exc:
            raise GenieAPIError(
                f"Databricks Genie API failed for {method} {path}: {exc}"
            ) from exc
        except Exception as exc:
            raise GenieAPIError(
                f"Databricks Genie API request could not be completed for {method} {path}: {exc}"
            ) from exc

        if result is None:
            return {}
        if isinstance(result, dict):
            return result
        return {"result": result}

    def _space_id(self) -> str:
        return _quote_id(self.config.genie_space_id)

    async def _http_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not self._http_client:
            raise GenieAPIError("HTTP client was not initialized.")

        response = await self._http_client.request(method, path, params=params, json=json_body)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = response.text[:2000]
            raise GenieAPIError(
                f"Databricks Genie API returned HTTP {response.status_code} for {method} {path}: {body}"
            ) from exc

        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise GenieAPIError(
                f"Databricks Genie API returned non-JSON response: {response.text[:2000]}"
            ) from exc

    def _resolve_bearer_token(self) -> str | None:
        if self.config.databricks_token:
            return self.config.databricks_token

        if self.config.databricks_auth_type != "databricks-cli":
            return None

        return _get_cli_access_token(
            cli_path=self.config.databricks_cli_path,
            profile=self.config.databricks_config_profile,
        )


def _get_cli_access_token(cli_path: str | None, profile: str | None) -> str:
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
        raise GenieAPIError(
            "Databricks CLI was not found. Set DATABRICKS_CLI_PATH or install the Databricks CLI."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()[:2000]
        raise GenieAPIError(f"Databricks CLI could not provide an OAuth token: {stderr}") from exc
    except subprocess.TimeoutExpired as exc:
        raise GenieAPIError("Databricks CLI token lookup timed out.") from exc

    try:
        payload = json.loads(completed.stdout)
    except ValueError as exc:
        raise GenieAPIError("Databricks CLI returned a non-JSON token response.") from exc

    access_token = payload.get("access_token")
    if not access_token:
        raise GenieAPIError("Databricks CLI token response did not include access_token.")
    return str(access_token)


def _resolve_cli_path(cli_path: str | None) -> str:
    if not cli_path:
        bundled = PROJECT_ROOT / ".tools" / "databricks.exe"
        return str(bundled) if bundled.exists() else "databricks"

    path = cli_path.replace("/", "\\")
    candidate = PROJECT_ROOT / path
    if not PathLike(path) and candidate.exists():
        return str(candidate)
    return path


def PathLike(path: str) -> bool:
    return ":" in path or path.startswith("\\")


def compile_message(message: dict[str, Any], max_result_rows: int = 50) -> dict[str, Any]:
    attachments = _as_list(message.get("attachments"))
    answer_text = []
    sql = []
    query_attachment_ids = []

    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue

        text = _extract_text(attachment)
        if text:
            answer_text.append(text)

        statement = _extract_sql(attachment)
        if statement:
            sql.append(statement)

        attachment_id = _extract_attachment_id(attachment)
        if attachment_id and _looks_like_query_attachment(attachment):
            query_attachment_ids.append(attachment_id)

    return {
        "status": message.get("status"),
        "error": message.get("error"),
        "answer_text": answer_text,
        "sql": sql,
        "query_attachment_ids": query_attachment_ids,
        "attachments": attachments,
    }


def compact_query_result(raw_result: dict[str, Any], max_rows: int = 50) -> dict[str, Any]:
    rows = _extract_rows(raw_result)
    truncated = len(rows) > max_rows
    return {
        "columns": _extract_columns(raw_result),
        "rows": rows[:max_rows],
        "row_count": len(rows),
        "truncated": truncated,
        "raw_result": raw_result if not rows else None,
    }


def _message_from_response(response: dict[str, Any]) -> dict[str, Any]:
    message = response.get("message") or response
    if not isinstance(message, dict):
        raise GenieAPIError("Genie response message is not an object.")
    return message


def _quote_id(value: str) -> str:
    return quote(str(value), safe="")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _extract_attachment_id(attachment: dict[str, Any]) -> str | None:
    for key in ("attachment_id", "id"):
        value = attachment.get(key)
        if value:
            return str(value)
    return None


def _extract_text(attachment: dict[str, Any]) -> str | None:
    value = attachment.get("text")
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "value"):
            if value.get(key):
                return str(value[key])
    return None


def _extract_sql(attachment: dict[str, Any]) -> str | None:
    query = attachment.get("query")
    if query is None:
        return None
    if isinstance(query, str):
        return query
    if isinstance(query, list):
        return "".join(str(part) for part in query)
    if isinstance(query, dict):
        for key in ("query", "sql", "statement", "content"):
            value = query.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return "".join(str(part) for part in value)
    return None


def _looks_like_query_attachment(attachment: dict[str, Any]) -> bool:
    return attachment.get("query") is not None


def _extract_columns(raw_result: dict[str, Any]) -> list[str]:
    schema_columns = _find_first(
        raw_result,
        [
            ("statement_response", "manifest", "schema", "columns"),
            ("manifest", "schema", "columns"),
            ("schema", "columns"),
            ("columns",),
        ],
    )
    if not isinstance(schema_columns, list):
        return []

    columns = []
    for column in schema_columns:
        if isinstance(column, dict):
            columns.append(str(column.get("name") or column.get("column_name") or column.get("label") or ""))
        else:
            columns.append(str(column))
    return [column for column in columns if column]


def _extract_rows(raw_result: dict[str, Any]) -> list[Any]:
    data = _find_first(
        raw_result,
        [
            ("statement_response", "result", "data_array"),
            ("result", "data_array"),
            ("data_array",),
            ("rows",),
        ],
    )
    if not isinstance(data, list):
        return []

    columns = _extract_columns(raw_result)
    if not columns:
        return data

    normalized_rows = []
    for row in data:
        if isinstance(row, list):
            normalized_rows.append(dict(zip(columns, row, strict=False)))
        else:
            normalized_rows.append(row)
    return normalized_rows


def _find_first(payload: dict[str, Any], paths: list[tuple[str, ...]]) -> Any:
    for path in paths:
        current: Any = payload
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return current
    return None
