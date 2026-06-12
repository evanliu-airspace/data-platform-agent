from __future__ import annotations

import argparse
import asyncio

from . import dbsql_mcp_agent
from . import dbx_agent_client


AGENTS = {
    "genie": "Databricks Genie agent. Best for business questions covered by the Genie Space.",
    "dbsql": "Read-only DBSQL agent. Best for catalog/schema/table discovery and direct SQL-style analysis.",
}


async def run_once(args: argparse.Namespace) -> str:
    question = " ".join(args.question)
    if args.agent == "genie":
        output, _, _ = await dbx_agent_client.run_question(
            args,
            question=question,
            conversation_id=args.conversation_id,
            history=[],
        )
        return output

    output, _ = await dbsql_mcp_agent.run_question(args, question, history=[])
    return output


async def interactive(args: argparse.Namespace) -> None:
    current_agent = args.agent
    genie_conversation_id = args.conversation_id
    genie_history: list[dict[str, str]] = []
    dbsql_history: list[dict[str, str]] = []

    print("Databricks unified agent. Type /help for commands, /exit to quit.")
    print(f"Current agent: {current_agent} - {AGENTS[current_agent]}")

    while True:
        try:
            question = input(f"\n{current_agent}> ").strip()
        except EOFError:
            return
        if not question:
            continue

        command = _parse_command(question)
        if command:
            next_agent = _handle_command(command, current_agent)
            if next_agent:
                current_agent = next_agent
            continue

        if current_agent == "genie":
            output, genie_conversation_id, history_entry = await dbx_agent_client.run_question(
                args,
                question=question,
                conversation_id=genie_conversation_id,
                history=genie_history,
            )
            print(output)
            genie_history.append(history_entry)
            continue

        output, history_entry = await dbsql_mcp_agent.run_question(
            args,
            question,
            history=dbsql_history,
        )
        print(output)
        dbsql_history.append(history_entry)


def _parse_command(text: str) -> list[str] | None:
    text = text.strip()
    if not text:
        return None

    if text.startswith(("/", "／")):
        text = text[1:].strip()
    else:
        command = text.split(maxsplit=1)[0].lower()
        if command not in {"agent", "agents", "current", "help", "exit", "quit", "genie", "dbsql", "switch"}:
            return None

    parts = text.split()
    if not parts:
        return None

    command = parts[0].lower()
    if command in AGENTS and len(parts) == 1:
        return ["agent", command]
    if command == "switch" and len(parts) >= 2:
        return ["agent", parts[1].lower()]
    return [command, *parts[1:]]


def _handle_command(parts: list[str], current_agent: str) -> str | None:
    if not parts:
        return None

    command = parts[0].lower()
    if command in {"exit", "quit"}:
        raise KeyboardInterrupt

    if command == "help":
        print("Commands:")
        print("  /agent genie    switch to Genie agent")
        print("  /agent dbsql    switch to read-only DBSQL agent")
        print("  /genie          shortcut for /agent genie")
        print("  /dbsql          shortcut for /agent dbsql")
        print("  agent dbsql     slash is optional")
        print("  /agents         list available agents")
        print("  /current        show current agent")
        print("  /exit           quit")
        return None

    if command == "agents":
        for name, description in AGENTS.items():
            print(f"{name}: {description}")
        return None

    if command == "current":
        print(f"Current agent: {current_agent} - {AGENTS[current_agent]}")
        return None

    if command == "agent":
        target = parts[1].lower() if len(parts) >= 2 else ""
        if target not in AGENTS:
            print("Usage: /agent genie|dbsql")
            return None
        next_agent = target
        print(f"Switched to {next_agent} - {AGENTS[next_agent]}")
        return next_agent

    print(f"Unknown command: /{command}. Type /help for commands.")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified interface for choosing between Databricks agents."
    )
    parser.add_argument("question", nargs="*", help="Question to ask. Omit to enter interactive mode.")
    parser.add_argument("--agent", choices=sorted(AGENTS), default="genie", help="Agent to use.")
    parser.add_argument("--endpoint", help="Databricks Model Serving endpoint for the language model.")

    parser.add_argument("--conversation-id", help="Genie only: continue an existing Genie conversation.")
    parser.add_argument("--max-result-rows", type=int, default=50, help="Genie only: max rows per query result.")
    parser.add_argument("--no-results", action="store_true", help="Genie only: do not fetch query result rows.")
    parser.add_argument("--no-clarify", action="store_true", help="Genie only: skip clarification stop.")
    parser.add_argument("--show-rewrite", action="store_true", help="Genie only: print rewritten Genie question.")
    parser.add_argument("--show-sql", action="store_true", help="Genie only: print Genie SQL.")
    parser.add_argument("--show-ids", action="store_true", help="Genie only: print Genie conversation/message IDs.")

    parser.add_argument("--max-steps", type=int, default=8, help="DBSQL only: max LLM/tool iterations.")
    parser.add_argument("--show-trace", action="store_true", help="DBSQL only: print MCP tool-call trace.")
    args = parser.parse_args()

    try:
        if args.question:
            print(asyncio.run(run_once(args)))
        else:
            asyncio.run(interactive(args))
    except KeyboardInterrupt:
        print()
    except Exception as exc:
        raise SystemExit(f"databricks-agent failed: {exc}") from exc


if __name__ == "__main__":
    main()
