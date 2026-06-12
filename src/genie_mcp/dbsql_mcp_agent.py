from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from typing import Any

from dotenv import load_dotenv

from .config import ENV_FILE, GenieConfig
from .databricks_llm import DEFAULT_LLM_ENDPOINT, DatabricksLLMClient
from .dbsql_mcp_client import DBSQLMCPClient
from .sql_guard import SQLSafetyError


SYSTEM_PROMPT = """
You are a read-only Databricks SQL agent.

You can ask for tool execution by returning JSON only:
{
  "action": "call_tool",
  "tool_name": "execute_sql_read_only",
  "arguments": {"query": "read-only DBSQL query"},
  "reason": "brief reason"
}

Or finish by returning JSON only:
{
  "action": "final",
  "answer": "final answer in Traditional Chinese"
}

Rules:
- You may only use execute_sql_read_only and poll_sql_result.
- Never request execute_sql.
- Never write data or metadata.
- Only produce SELECT, WITH, SHOW, DESCRIBE, DESC, or EXPLAIN statements.
- Do not use INSERT, UPDATE, DELETE, MERGE, CREATE, DROP, ALTER, TRUNCATE, GRANT, REVOKE, OPTIMIZE, VACUUM, REFRESH, USE, or SET.
- Use fully qualified Unity Catalog table names whenever possible.
- If you do not know available catalogs, schemas, tables, or columns, discover them with SHOW and DESCRIBE.
- Prefer concise SQL with LIMIT for discovery queries.
- After tool results are provided, answer from the observed rows only.
- If the question cannot be answered with read-only SQL, say so.
""".strip()


async def run_once(args: argparse.Namespace) -> str:
    output, _ = await run_question(args, " ".join(args.question), history=[])
    return output


async def run_question(
    args: argparse.Namespace,
    question: str,
    *,
    history: list[dict[str, str]],
) -> tuple[str, dict[str, str]]:
    load_dotenv(ENV_FILE)
    config = GenieConfig.from_env()
    endpoint = args.endpoint or os.getenv("DATABRICKS_LLM_ENDPOINT") or DEFAULT_LLM_ENDPOINT
    dbsql = await asyncio.to_thread(DBSQLMCPClient, config)
    tool_specs = await asyncio.to_thread(dbsql.list_read_only_tools)
    messages = _initial_messages(question, tool_specs, history)
    tool_log: list[dict[str, Any]] = []

    async with DatabricksLLMClient(config, endpoint=endpoint) as llm:
        for step in range(1, args.max_steps + 1):
            content = await llm.chat(messages, temperature=0, max_tokens=1200)
            action = _parse_action(content)

            if action.get("action") == "final":
                answer = str(action.get("answer") or "").strip()
                if args.show_trace:
                    answer = _format_trace(endpoint, tool_log) + "\n\n" + answer
                return answer, {
                    "user_question": question,
                    "answer_excerpt": _compact(answer),
                }

            if action.get("action") != "call_tool":
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Return valid JSON with action final or call_tool."})
                continue

            tool_name = str(action.get("tool_name") or "")
            arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
            try:
                tool_result = await asyncio.to_thread(_call_allowed_tool, dbsql, tool_name, arguments)
            except (SQLSafetyError, ValueError) as exc:
                tool_result = {"is_error": True, "error": str(exc)}

            tool_log.append(
                {
                    "step": step,
                    "tool_name": tool_name,
                    "arguments": _redact_large(arguments),
                    "result": tool_result,
                }
            )
            messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
            messages.append(
                {
                    "role": "user",
                    "content": "Tool result:\n"
                    + json.dumps(tool_result, ensure_ascii=False, indent=2)
                    + "\nContinue. Return JSON only.",
                }
            )

    answer = "Reached the maximum number of tool-call steps before a final answer was produced."
    if args.show_trace:
        answer = _format_trace(endpoint, tool_log) + "\n\n" + answer
    return answer, {"user_question": question, "answer_excerpt": _compact(answer)}


async def interactive(args: argparse.Namespace) -> None:
    print("Databricks DBSQL read-only agent. Type 'exit' or Ctrl+C to quit.")
    history: list[dict[str, str]] = []
    while True:
        question = input("\nquestion> ").strip()
        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            continue
        output, history_entry = await run_question(args, question, history=history)
        print(output)
        history.append(history_entry)


def _initial_messages(
    question: str,
    tool_specs: list[dict[str, Any]],
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    context = {
        "allowed_tools": tool_specs,
        "recent_conversation": history[-6:],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Context:\n" + json.dumps(context, ensure_ascii=False, indent=2)},
        {"role": "user", "content": question},
    ]


def _call_allowed_tool(dbsql: DBSQLMCPClient, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "execute_sql_read_only":
        return dbsql.execute_read_only_sql(str(arguments.get("query") or ""))
    if tool_name == "poll_sql_result":
        return dbsql.call_tool("poll_sql_result", {"statement_id": str(arguments.get("statement_id") or "")})
    raise ValueError(f"Tool {tool_name!r} is not allowed.")


def _parse_action(content: str) -> dict[str, Any]:
    text = content.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        return {"action": "final", "answer": content}
    return parsed if isinstance(parsed, dict) else {"action": "final", "answer": content}


def _format_trace(endpoint: str, tool_log: list[dict[str, Any]]) -> str:
    trace = {
        "llm_endpoint": endpoint,
        "sql_backend": "Databricks SQL Statements API",
        "allowed_tools": sorted(["execute_sql_read_only", "poll_sql_result"]),
        "tool_calls": tool_log,
    }
    return json.dumps(trace, ensure_ascii=False, indent=2)


def _redact_large(value: Any, limit: int = 4000) -> Any:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= limit:
        return value
    return text[: limit - 3] + "..."


def _compact(answer: str, limit: int = 700) -> str:
    compacted = " ".join(answer.split())
    return compacted if len(compacted) <= limit else compacted[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Databricks DBSQL agent."
    )
    parser.add_argument("question", nargs="*", help="Question to ask. Omit to enter interactive mode.")
    parser.add_argument("--endpoint", help="Databricks Model Serving endpoint for the language model.")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum LLM/tool iterations.")
    parser.add_argument("--show-trace", action="store_true", help="Print LLM endpoint and MCP tool call trace.")
    args = parser.parse_args()

    try:
        if args.question:
            print(asyncio.run(run_once(args)))
        else:
            asyncio.run(interactive(args))
    except KeyboardInterrupt:
        print()
    except Exception as exc:
        raise SystemExit(f"dbsql-mcp-agent failed: {exc}") from exc


if __name__ == "__main__":
    main()
