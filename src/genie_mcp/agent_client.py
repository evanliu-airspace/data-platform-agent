from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
DEFAULT_MODEL = "gpt-5.5"


AGENT_INSTRUCTIONS = """
You are a precise data analyst agent for a Databricks Genie Space.

Your job:
- Understand the user's business question and rewrite it mentally into a clear Genie-ready question.
- Ask a short clarification before using tools if the metric, entity, filter, or time range is too ambiguous.
- Use the Databricks Genie MCP tools for all data-backed answers.
- Treat Genie output as the only source of truth for numbers, SQL, and result rows.
- Never invent values, columns, SQL, or trends that were not returned by Genie.
- When Genie returns SQL and rows, summarize the answer in the user's language.
- Mention important filters, time ranges, and whether rows were truncated.
- If Genie fails or times out, explain that clearly and suggest a narrower follow-up question.
""".strip()


async def run_once(question: str) -> str:
    load_dotenv(ROOT / ".env")

    try:
        from agents import Agent, Runner
        from agents.mcp import MCPServerStdio
    except ImportError as exc:
        raise RuntimeError(
            "Missing OpenAI Agents SDK. Install with: python -m pip install -e \".[client]\""
        ) from exc

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = _prepend_path(str(SRC), child_env.get("PYTHONPATH", ""))

    async with MCPServerStdio(
        name="databricks_genie",
        client_session_timeout_seconds=60,
        params={
            "command": sys.executable,
            "args": ["-m", "genie_mcp.server"],
            "cwd": str(ROOT),
            "env": child_env,
        },
    ) as server:
        agent = Agent(
            name="Databricks Genie Analyst",
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            instructions=AGENT_INSTRUCTIONS,
            mcp_servers=[server],
            mcp_config={
                "include_server_in_tool_names": True,
                "convert_schemas_to_strict": True,
            },
        )
        result = await Runner.run(agent, question)
        return str(result.final_output)


async def interactive() -> None:
    print("Databricks Genie agent. Type 'exit' or Ctrl+C to quit.")
    while True:
        question = input("\nquestion> ").strip()
        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            continue
        print(await run_once(question))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a Databricks Genie Space through an OpenAI MCP agent.")
    parser.add_argument("question", nargs="*", help="Question to ask. Omit to enter interactive mode.")
    args = parser.parse_args()

    try:
        if args.question:
            print(asyncio.run(run_once(" ".join(args.question))))
        else:
            asyncio.run(interactive())
    except KeyboardInterrupt:
        print()
    except Exception as exc:
        raise SystemExit(f"genie-agent failed: {exc}") from exc


def _prepend_path(path: str, existing: str) -> str:
    if not existing:
        return path
    return f"{path}{os.pathsep}{existing}"


if __name__ == "__main__":
    main()
