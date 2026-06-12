from __future__ import annotations

import time
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import Disposition, ExecuteStatementRequestOnWaitTimeout, Format

from .config import GenieConfig
from .databricks_genie import _get_cli_access_token
from .sql_guard import validate_read_only_sql


READ_ONLY_TOOLS = {"execute_sql_read_only", "poll_sql_result"}
TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELED", "CANCELLED", "CLOSED"}


class DBSQLMCPClient:
    def __init__(self, config: GenieConfig):
        self.config = config
        self.server_url = f"{config.databricks_host}/api/2.0/sql/statements"
        token = _resolve_bearer_token(config)
        if token:
            self._client = WorkspaceClient(
                host=config.databricks_host,
                token=token,
                auth_type="pat",
            )
        else:
            self._client = WorkspaceClient(
                host=config.databricks_host,
                auth_type=config.databricks_auth_type,
                profile=config.databricks_config_profile,
                product="databricks-genie-mcp-agent",
                product_version="0.1.0",
            )

    def list_read_only_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "execute_sql_read_only",
                "description": "Execute one read-only Databricks SQL statement through the SQL Statements API.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A single SELECT, WITH, SHOW, DESCRIBE, DESC, or EXPLAIN statement.",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "poll_sql_result",
                "description": "Poll a Databricks SQL statement by statement_id.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "statement_id": {
                            "type": "string",
                            "description": "Statement ID returned by execute_sql_read_only.",
                        }
                    },
                    "required": ["statement_id"],
                },
            },
        ]

    def execute_read_only_sql(
        self,
        query: str,
        *,
        poll: bool = True,
        timeout_seconds: float = 300,
        poll_interval_seconds: float = 2,
    ) -> dict[str, Any]:
        safe_query = validate_read_only_sql(query)
        result = self._execute_statement(safe_query)
        if not poll:
            return result

        statement_id = result.get("statement_id")
        state = _state(result)
        if not statement_id or state in TERMINAL_STATES:
            return result

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            time.sleep(poll_interval_seconds)
            result = self.call_tool("poll_sql_result", {"statement_id": statement_id})
            if _state(result) in TERMINAL_STATES:
                return result

        result["poll_timeout"] = True
        return result

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in READ_ONLY_TOOLS:
            raise ValueError(f"Tool {tool_name!r} is not allowed. Allowed tools: {sorted(READ_ONLY_TOOLS)}.")

        if tool_name == "execute_sql_read_only":
            return self.execute_read_only_sql(str(arguments.get("query") or ""))
        return self._poll_statement(str(arguments.get("statement_id") or ""))

    def _execute_statement(self, query: str) -> dict[str, Any]:
        warehouse_id = self.config.databricks_warehouse_id
        if not warehouse_id:
            raise ValueError(
                "DBSQL agent needs DATABRICKS_WAREHOUSE_ID. "
                "Set it to the SQL warehouse ID that should run read-only queries."
            )

        response = self._client.statement_execution.execute_statement(
            statement=query,
            warehouse_id=warehouse_id,
            disposition=Disposition.INLINE,
            format=Format.JSON_ARRAY,
            on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
            row_limit=100,
            wait_timeout="10s",
        )
        return compact_sql_result(_as_dict(response))

    def _poll_statement(self, statement_id: str) -> dict[str, Any]:
        if not statement_id:
            raise ValueError("poll_sql_result needs statement_id.")
        response = self._client.statement_execution.get_statement(statement_id)
        return compact_sql_result(_as_dict(response))


def compact_sql_result(payload: dict[str, Any], max_rows: int = 100) -> dict[str, Any]:
    compacted: dict[str, Any] = {
        "statement_id": payload.get("statement_id"),
        "status": payload.get("status"),
        "message": payload.get("message"),
        "is_error": payload.get("is_error", False),
    }

    manifest = payload.get("manifest") if isinstance(payload.get("manifest"), dict) else {}
    compacted["total_row_count"] = manifest.get("total_row_count")
    compacted["truncated"] = manifest.get("truncated")

    columns = _extract_columns(manifest)
    rows = _extract_rows(payload, columns)
    if columns:
        compacted["columns"] = columns
    if rows:
        compacted["rows"] = rows[:max_rows]
        compacted["returned_row_count"] = len(rows[:max_rows])
        compacted["rows_truncated_locally"] = len(rows) > max_rows

    if not rows and payload.get("result"):
        compacted["result"] = payload.get("result")

    return {key: value for key, value in compacted.items() if value is not None}


def _resolve_bearer_token(config: GenieConfig) -> str | None:
    if config.databricks_token:
        return config.databricks_token
    if config.databricks_auth_type == "databricks-cli":
        return _get_cli_access_token(
            cli_path=config.databricks_cli_path,
            profile=config.databricks_config_profile,
        )
    return None


def _extract_columns(manifest: dict[str, Any]) -> list[str]:
    schema = manifest.get("schema") if isinstance(manifest.get("schema"), dict) else {}
    raw_columns = schema.get("columns") if isinstance(schema.get("columns"), list) else []
    return [str(column.get("name")) for column in raw_columns if isinstance(column, dict) and column.get("name")]


def _extract_rows(payload: dict[str, Any], columns: list[str]) -> list[dict[str, Any] | list[Any]]:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    typed_rows = result.get("data_typed_array")
    if isinstance(typed_rows, list):
        return _rows_from_value_rows(typed_rows, columns)

    data_array = result.get("data_array")
    if isinstance(data_array, list):
        return _rows_from_value_rows(data_array, columns)

    return []


def _rows_from_value_rows(raw_rows: list[Any], columns: list[str]) -> list[dict[str, Any] | list[Any]]:
    rows = []
    for raw_row in raw_rows:
        values = raw_row.get("values") if isinstance(raw_row, dict) else raw_row
        if isinstance(values, list):
            row_values = [_typed_value(value) for value in values]
            rows.append(dict(zip(columns, row_values, strict=False)) if columns else row_values)
        else:
            rows.append(raw_row)
    return rows


def _typed_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in (
        "str",
        "string_value",
        "long",
        "long_value",
        "int",
        "int_value",
        "double",
        "double_value",
        "float",
        "float_value",
        "boolean",
        "boolean_value",
        "bytes",
        "bytes_value",
        "decimal",
        "decimal_value",
    ):
        if key in value:
            return value[key]
    if value.get("null") is True:
        return None
    if len(value) == 1:
        return next(iter(value.values()))
    return value


def _state(payload: dict[str, Any]) -> str:
    status = payload.get("status")
    if isinstance(status, dict):
        return str(status.get("state") or "").upper()
    return ""


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return {"result": value}
