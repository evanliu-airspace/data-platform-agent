from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from .config import GenieConfig
from .databricks_genie import DatabricksGenieClient


async def get_space() -> dict[str, Any]:
    config = GenieConfig.from_env()
    async with DatabricksGenieClient(config) as client:
        return await client.get_space(include_serialized_space=False)


async def ask_once(
    question: str,
    *,
    conversation_id: str | None = None,
    include_query_results: bool = True,
    max_result_rows: int = 50,
) -> dict[str, Any]:
    config = GenieConfig.from_env()
    async with DatabricksGenieClient(config) as client:
        return await client.ask(
            question=question,
            conversation_id=conversation_id,
            include_query_results=include_query_results,
            max_result_rows=max_result_rows,
        )


async def interactive(args: argparse.Namespace) -> None:
    print("Databricks Genie direct mode. Type 'exit' or Ctrl+C to quit.")
    conversation_id = args.conversation_id
    while True:
        question = input("\nquestion> ").strip()
        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            continue

        result = await ask_once(
            question,
            conversation_id=conversation_id,
            include_query_results=not args.no_results,
            max_result_rows=args.max_result_rows,
        )
        conversation_id = result.get("conversation_id") or conversation_id
        print(format_answer(result, show_sql=args.show_sql, show_ids=args.show_ids))


def format_space(space: dict[str, Any]) -> str:
    visible = {
        key: space.get(key)
        for key in ("space_id", "title", "description", "warehouse_id", "parent_path")
        if space.get(key) is not None
    }
    return json.dumps(visible, ensure_ascii=False, indent=2)


def format_answer(result: dict[str, Any], *, show_sql: bool, show_ids: bool) -> str:
    parts: list[str] = []

    status = result.get("status")
    if status and status != "COMPLETED":
        parts.append(f"status: {status}")

    error = result.get("error")
    if error:
        parts.append(f"error: {json.dumps(error, ensure_ascii=False)}")

    answer_text = result.get("answer_text") or []
    for text in answer_text:
        parts.append(str(text))

    if show_sql:
        sql_statements = result.get("sql") or []
        for index, statement in enumerate(sql_statements, start=1):
            parts.append(f"sql[{index}]:\n{statement}")

    query_results = result.get("query_results") or {}
    for attachment_id, query_result in query_results.items():
        rows = query_result.get("rows") or []
        row_count = query_result.get("row_count")
        truncated = query_result.get("truncated")
        header = f"query_result[{attachment_id}]"
        if row_count is not None:
            header += f" row_count={row_count}"
        if truncated:
            header += " truncated=true"
        parts.append(header)
        parts.append(json.dumps(rows, ensure_ascii=False, indent=2))

    if show_ids:
        ids = {
            "conversation_id": result.get("conversation_id"),
            "message_id": result.get("message_id"),
        }
        parts.append(json.dumps(ids, ensure_ascii=False, indent=2))

    if not parts:
        return json.dumps(result, ensure_ascii=False, indent=2)
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask Databricks Genie directly without calling the OpenAI API."
    )
    parser.add_argument("question", nargs="*", help="Question to ask. Omit to enter interactive mode.")
    parser.add_argument("--conversation-id", help="Continue an existing Genie conversation.")
    parser.add_argument("--max-result-rows", type=int, default=50, help="Maximum rows to print per query result.")
    parser.add_argument("--no-results", action="store_true", help="Do not fetch query result rows.")
    parser.add_argument("--show-sql", action="store_true", help="Print generated SQL when Genie returns it.")
    parser.add_argument("--show-ids", action="store_true", help="Print conversation_id and message_id.")
    parser.add_argument("--space", action="store_true", help="Print configured Genie Space metadata and exit.")
    args = parser.parse_args()

    try:
        if args.space:
            print(format_space(asyncio.run(get_space())))
            return

        if args.question:
            result = asyncio.run(
                ask_once(
                    " ".join(args.question),
                    conversation_id=args.conversation_id,
                    include_query_results=not args.no_results,
                    max_result_rows=args.max_result_rows,
                )
            )
            print(format_answer(result, show_sql=args.show_sql, show_ids=args.show_ids))
            return

        asyncio.run(interactive(args))
    except KeyboardInterrupt:
        print()
    except Exception as exc:
        raise SystemExit(f"genie-direct failed: {exc}") from exc


if __name__ == "__main__":
    main()

