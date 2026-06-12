from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from dotenv import load_dotenv

from .config import ENV_FILE, GenieConfig
from .databricks_genie import DatabricksGenieClient
from .databricks_llm import DEFAULT_LLM_ENDPOINT, DatabricksLLMClient
from .direct_client import format_answer


async def run_once(args: argparse.Namespace) -> str:
    output, _, _ = await run_question(
        args,
        question=" ".join(args.question),
        conversation_id=args.conversation_id,
        history=[],
    )
    return output


async def run_question(
    args: argparse.Namespace,
    *,
    question: str,
    conversation_id: str | None,
    history: list[dict[str, str]],
) -> tuple[str, str | None, dict[str, str]]:
    load_dotenv(ENV_FILE)
    config = GenieConfig.from_env()
    endpoint = args.endpoint or os.getenv("DATABRICKS_LLM_ENDPOINT") or DEFAULT_LLM_ENDPOINT

    async with DatabricksLLMClient(config, endpoint=endpoint) as llm:
        rewrite = await llm.rewrite_for_genie(question, history=history)

    if rewrite["needs_clarification"] and not args.no_clarify:
        output = f"需要釐清：{rewrite['clarifying_question']}"
        return output, conversation_id, {
            "user_question": question,
            "genie_question": "",
            "answer_excerpt": output,
        }

    genie_question = rewrite["genie_question"]
    async with DatabricksGenieClient(config) as genie:
        result = await genie.ask(
            question=genie_question,
            conversation_id=conversation_id,
            include_query_results=not args.no_results,
            max_result_rows=args.max_result_rows,
        )

    answer = format_answer(result, show_sql=args.show_sql, show_ids=args.show_ids)
    next_conversation_id = result.get("conversation_id") or conversation_id
    history_entry = {
        "user_question": question,
        "genie_question": genie_question,
        "answer_excerpt": _compact_answer(answer),
    }

    if not args.show_rewrite:
        return answer, next_conversation_id, history_entry

    rewrite_section = _format_rewrite_section(endpoint, genie_question, rewrite)
    return "\n".join(rewrite_section) + "\n\n" + answer, next_conversation_id, history_entry


async def interactive(args: argparse.Namespace) -> None:
    print("Databricks Genie conversational mode. Type 'exit' or Ctrl+C to quit.")
    conversation_id = args.conversation_id
    history: list[dict[str, str]] = []

    while True:
        question = input("\nquestion> ").strip()
        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            continue

        output, conversation_id, history_entry = await run_question(
            args,
            question=question,
            conversation_id=conversation_id,
            history=history,
        )
        print(output)
        history.append(history_entry)


def _format_rewrite_section(
    endpoint: str,
    genie_question: str,
    rewrite: dict[str, Any],
) -> list[str]:
    rewrite_section = [
        f"llm_endpoint: {endpoint}",
        f"genie_question: {genie_question}",
    ]
    assumptions = rewrite.get("assumptions") or []
    if assumptions:
        rewrite_section.append("assumptions: " + "; ".join(str(item) for item in assumptions))
    return rewrite_section


def _compact_answer(answer: str, limit: int = 700) -> str:
    compacted = " ".join(answer.split())
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use a Databricks-hosted LLM to rewrite a question, then ask Databricks Genie."
    )
    parser.add_argument("question", nargs="*", help="Question to ask. Omit to enter interactive mode.")
    parser.add_argument("--endpoint", help="Databricks Model Serving endpoint for the language model.")
    parser.add_argument("--conversation-id", help="Continue an existing Genie conversation.")
    parser.add_argument("--max-result-rows", type=int, default=50, help="Maximum rows to print per query result.")
    parser.add_argument("--no-results", action="store_true", help="Do not fetch query result rows.")
    parser.add_argument("--no-clarify", action="store_true", help="Send the rewritten question even if LLM asks to clarify.")
    parser.add_argument("--show-rewrite", action="store_true", help="Print the rewritten Genie question.")
    parser.add_argument("--show-sql", action="store_true", help="Print generated SQL when Genie returns it.")
    parser.add_argument("--show-ids", action="store_true", help="Print conversation_id and message_id.")
    args = parser.parse_args()

    try:
        if args.question:
            print(asyncio.run(run_once(args)))
        else:
            asyncio.run(interactive(args))
    except KeyboardInterrupt:
        print()
    except Exception as exc:
        raise SystemExit(f"genie-dbx-agent failed: {exc}") from exc


if __name__ == "__main__":
    main()
