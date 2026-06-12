from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from .config import GenieConfig
from .databricks_genie import DatabricksGenieClient, compile_message, compact_query_result


mcp = FastMCP("databricks-genie")


@mcp.tool()
def health() -> dict[str, Any]:
    """Return non-secret MCP server configuration and readiness hints."""
    return GenieConfig.health_from_env()


@mcp.tool()
async def get_genie_space(include_serialized_space: bool = False) -> dict[str, Any]:
    """Fetch the configured Databricks Genie Space metadata."""
    config = GenieConfig.from_env()
    async with DatabricksGenieClient(config) as client:
        return await client.get_space(include_serialized_space=include_serialized_space)


@mcp.tool()
async def ask_genie(
    question: str,
    conversation_id: str | None = None,
    wait: bool = True,
    include_query_results: bool = True,
    timeout_seconds: float | None = None,
    max_result_rows: int = 50,
) -> dict[str, Any]:
    """Ask Databricks Genie a data question and return answer text, SQL, and query rows."""
    config = GenieConfig.from_env()
    async with DatabricksGenieClient(config) as client:
        return await client.ask(
            question=question,
            conversation_id=conversation_id,
            wait=wait,
            include_query_results=include_query_results,
            timeout_seconds=timeout_seconds,
            max_result_rows=max_result_rows,
        )


@mcp.tool()
async def get_genie_message(
    conversation_id: str,
    message_id: str,
    max_result_rows: int = 50,
) -> dict[str, Any]:
    """Retrieve and normalize a Genie message by conversation and message ID."""
    config = GenieConfig.from_env()
    async with DatabricksGenieClient(config) as client:
        message = await client.get_message(conversation_id, message_id)
    compiled = compile_message(message, max_result_rows=max_result_rows)
    compiled["conversation_id"] = conversation_id
    compiled["message_id"] = message_id
    return compiled


@mcp.tool()
async def get_genie_query_result(
    conversation_id: str,
    message_id: str,
    attachment_id: str,
    max_result_rows: int = 100,
) -> dict[str, Any]:
    """Fetch and compact query results for a Genie message attachment."""
    config = GenieConfig.from_env()
    async with DatabricksGenieClient(config) as client:
        raw_result = await client.get_query_result(conversation_id, message_id, attachment_id)
    return compact_query_result(raw_result, max_result_rows)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

