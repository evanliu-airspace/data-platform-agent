from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from urllib.parse import quote

import httpx
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError

from .config import GenieConfig
from .databricks_genie import GenieAPIError, _get_cli_access_token


DEFAULT_LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"


class DatabricksLLMClient:
    def __init__(self, config: GenieConfig, endpoint: str | None = None):
        self.config = config
        self.endpoint = endpoint or DEFAULT_LLM_ENDPOINT
        self._client: httpx.AsyncClient | None = None
        self._sdk_client: WorkspaceClient | None = None

        bearer_token = self._resolve_bearer_token()
        if bearer_token:
            self._client = httpx.AsyncClient(
                base_url=config.databricks_host,
                headers={
                    "Authorization": f"Bearer {bearer_token}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(120.0, connect=20.0),
            )
        else:
            self._sdk_client = WorkspaceClient(
                host=config.databricks_host,
                auth_type=config.databricks_auth_type,
                profile=config.databricks_config_profile,
                product="databricks-genie-mcp-agent",
                product_version="0.1.0",
            )

    async def __aenter__(self) -> "DatabricksLLMClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> str:
        endpoint_name = quote(self.endpoint, safe="")
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self._client:
            response = await self._client.post(
                f"/serving-endpoints/{endpoint_name}/invocations",
                json=payload,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                body = response.text[:2000]
                raise GenieAPIError(
                    f"Databricks LLM endpoint returned HTTP {response.status_code}: {body}"
                ) from exc

            return _extract_chat_content(response.json())

        if not self._sdk_client:
            raise GenieAPIError("Databricks LLM client was not initialized.")

        try:
            result = await asyncio.to_thread(
                self._sdk_client.api_client.do,
                "POST",
                f"/serving-endpoints/{endpoint_name}/invocations",
                body=payload,
            )
        except DatabricksError as exc:
            raise GenieAPIError(f"Databricks LLM endpoint request failed: {exc}") from exc
        except Exception as exc:
            raise GenieAPIError(f"Databricks LLM endpoint request could not be completed: {exc}") from exc

        return _extract_chat_content(result if isinstance(result, dict) else {"result": result})

    async def rewrite_for_genie(
        self,
        user_question: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        messages = [{"role": "system", "content": REWRITE_SYSTEM_PROMPT}]
        if history:
            messages.append({"role": "user", "content": _format_history_context(history)})
        messages.append({"role": "user", "content": user_question})

        content = await self.chat(
            messages,
            temperature=0,
            max_tokens=700,
        )
        return _parse_rewrite_response(content, user_question)

    def _resolve_bearer_token(self) -> str | None:
        if self.config.databricks_token:
            return self.config.databricks_token

        if self.config.databricks_auth_type == "databricks-cli":
            return _get_cli_access_token(
                cli_path=self.config.databricks_cli_path,
                profile=self.config.databricks_config_profile,
            )

        return None


REWRITE_SYSTEM_PROMPT = """
You rewrite business data questions for a Databricks Genie Space.

Return only JSON with this exact schema:
{
  "needs_clarification": boolean,
  "clarifying_question": string,
  "genie_question": string,
  "assumptions": [string]
}

Rules:
- If the user question is too ambiguous to query safely, set needs_clarification=true and write one concise clarifying question.
- Otherwise set needs_clarification=false and rewrite the user question into a clear Databricks Genie prompt.
- Write clarifying_question, genie_question, and assumptions in Traditional Chinese by default, unless the user explicitly asks for another language.
- Preserve table names, column names, metric names, and product names exactly when they are mentioned by the user.
- Do not invent table names, column names, numbers, or business definitions.
- If the user did not specify a time range, do not invent one; mention that as an assumption only if needed.
- If the current question is a follow-up, use the supplied conversation context to resolve references like "that store", "second place", "same period", or "compare with last week".
- Keep genie_question concise and directly answerable by Genie.
""".strip()


def _format_history_context(history: list[dict[str, str]]) -> str:
    recent = history[-6:]
    lines = ["Conversation context for resolving follow-up questions:"]
    for index, item in enumerate(recent, start=1):
        user_question = item.get("user_question", "")
        genie_question = item.get("genie_question", "")
        answer_excerpt = item.get("answer_excerpt", "")
        lines.append(f"{index}. user_question: {user_question}")
        if genie_question:
            lines.append(f"   genie_question: {genie_question}")
        if answer_excerpt:
            lines.append(f"   answer_excerpt: {answer_excerpt}")
    return "\n".join(lines)


def _extract_chat_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict)
                )

    predictions = payload.get("predictions")
    if isinstance(predictions, list) and predictions:
        first = predictions[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return json.dumps(first, ensure_ascii=False)

    if "result" in payload:
        return str(payload["result"])

    return json.dumps(payload, ensure_ascii=False)


def _parse_rewrite_response(content: str, fallback_question: str) -> dict[str, Any]:
    text = content.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    try:
        parsed = json.loads(text)
    except ValueError:
        return {
            "needs_clarification": False,
            "clarifying_question": "",
            "genie_question": fallback_question,
            "assumptions": [
                "Databricks LLM 回覆不是有效 JSON，因此改用原始問題。"
            ],
            "raw_llm_response": content,
        }

    return {
        "needs_clarification": bool(parsed.get("needs_clarification")),
        "clarifying_question": str(parsed.get("clarifying_question") or ""),
        "genie_question": str(parsed.get("genie_question") or fallback_question),
        "assumptions": parsed.get("assumptions") if isinstance(parsed.get("assumptions"), list) else [],
        "raw_llm_response": content,
    }
