from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


DEFAULT_DATABRICKS_HOST = "https://adb-415889795140801.1.azuredatabricks.net"
DEFAULT_GENIE_SPACE_ID = "01f0a8c81e88142fadad408f820867c3"
DEFAULT_LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
TERMINAL_GENIE_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}
TERMINAL_SQL_STATES = {"SUCCEEDED", "FAILED", "CANCELED", "CANCELLED", "CLOSED"}

PROJECT_ROOT = Path(__file__).resolve().parent
SESSIONS: dict[str, dict[str, Any]] = {}
TOKEN_CACHE: dict[tuple[str, str, str], tuple[str, float]] = {}


AGENTS = {
    "genie": "Databricks Genie agent. Best for business questions covered by the Genie Space.",
    "dbsql": "Read-only DBSQL agent. Best for catalog/schema/table discovery and direct SQL-style analysis.",
}


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
- If the current question is a follow-up, use the supplied conversation context to resolve references.
- Keep genie_question concise and directly answerable by Genie.
""".strip()


DBSQL_SYSTEM_PROMPT = """
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
- Never write data or metadata.
- Only produce SELECT, WITH, SHOW, DESCRIBE, DESC, or EXPLAIN statements.
- Do not use INSERT, UPDATE, DELETE, MERGE, CREATE, DROP, ALTER, TRUNCATE, GRANT, REVOKE, OPTIMIZE, VACUUM, REFRESH, USE, or SET.
- Use fully qualified Unity Catalog table names whenever possible.
- If you do not know available catalogs, schemas, tables, or columns, discover them with SHOW and DESCRIBE.
- Prefer concise SQL with LIMIT for discovery queries.
- After tool results are provided, answer from the observed rows only.
- If the question cannot be answered with read-only SQL, say so.
""".strip()


READ_ONLY_START_RE = re.compile(r"^\s*(select|with|show|describe|desc|explain)\b", re.IGNORECASE)
BLOCKED_SQL_RE = re.compile(
    r"\b("
    r"alter|analyze|attach|cache|clone|copy\s+into|create|delete|drop|grant|insert|"
    r"merge|msck|optimize|put|recover|refresh|replace|restore|revoke|set|truncate|"
    r"uncache|update|use|vacuum"
    r")\b",
    re.IGNORECASE,
)


class AppError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def main() -> None:
    load_env_file()
    port = int(os.getenv("DATABRICKS_APP_PORT") or os.getenv("PORT") or "8000")
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Databricks Genie MCP Agent API listening on 0.0.0.0:{port}", flush=True)
    server.serve_forever()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path in {"/", ""}:
            self.send_json(
                200,
                {
                    "name": "Databricks Genie MCP Agent API",
                    "health": "/health",
                    "agents": "/agents",
                    "query": "POST /query",
                },
            )
            return

        if self.path == "/health":
            self.send_json(200, {"ok": True, "agents": AGENTS, "config": health_config()})
            return

        if self.path == "/agents":
            self.send_json(200, AGENTS)
            return

        self.send_json(404, {"detail": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/query":
            self.send_json(404, {"detail": "Not found"})
            return

        try:
            payload = self.read_json()
            response = handle_query(payload)
            self.send_json(200, response)
        except AppError as exc:
            self.send_json(exc.status_code, {"detail": str(exc)})
        except Exception as exc:
            self.send_json(500, {"detail": f"agent execution failed: {exc}"})

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise AppError(f"Invalid JSON request body: {exc}", 400) from exc
        if not isinstance(payload, dict):
            raise AppError("JSON request body must be an object.", 400)
        return payload

    def send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def handle_query(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question") or "").strip()
    if not question:
        raise AppError("question is required.", 400)

    agent = str(payload.get("agent") or "genie").strip().lower()
    if agent not in AGENTS:
        raise AppError("agent must be genie or dbsql.", 400)

    session_id = str(payload.get("session_id") or uuid.uuid4())
    state = SESSIONS.setdefault(
        session_id,
        {"genie_conversation_id": None, "genie_history": [], "dbsql_history": []},
    )

    if agent == "genie":
        answer, conversation_id, history_entry = query_genie(payload, question, state)
    else:
        answer, conversation_id, history_entry = query_dbsql(payload, question, state)

    return {
        "agent": agent,
        "answer": answer,
        "session_id": session_id,
        "conversation_id": conversation_id,
        "history_entry": history_entry,
    }


def query_genie(
    payload: dict[str, Any],
    question: str,
    state: dict[str, Any],
) -> tuple[str, str | None, dict[str, str]]:
    if bool(payload.get("new_conversation")):
        state["genie_conversation_id"] = None
        state["genie_history"] = []

    cfg = config()
    token = resolve_bearer_token(cfg)
    conversation_id = payload.get("conversation_id") or state.get("genie_conversation_id")
    history = payload.get("history") if isinstance(payload.get("history"), list) else state["genie_history"]
    rewrite = rewrite_for_genie(
        cfg,
        token,
        question,
        history=history,
        endpoint=str(payload.get("endpoint") or cfg["llm_endpoint"]),
    )

    if rewrite.get("needs_clarification") and not bool(payload.get("no_clarify")):
        answer = f"需要釐清：{rewrite.get('clarifying_question') or ''}"
        history_entry = {"user_question": question, "genie_question": "", "answer_excerpt": compact(answer)}
        state["genie_history"].append(history_entry)
        return answer, conversation_id, history_entry

    genie_question = str(rewrite.get("genie_question") or question)
    result = ask_genie_api(
        cfg,
        token,
        genie_question,
        conversation_id=str(conversation_id) if conversation_id else None,
        include_query_results=bool(payload.get("include_query_results", True)),
        max_result_rows=int(payload.get("max_result_rows") or 50),
    )

    answer = format_genie_answer(
        result,
        show_sql=bool(payload.get("show_sql")),
        show_ids=bool(payload.get("show_ids")),
    )
    next_conversation_id = result.get("conversation_id") or conversation_id
    history_entry = {
        "user_question": question,
        "genie_question": genie_question,
        "answer_excerpt": compact(answer),
    }
    state["genie_conversation_id"] = next_conversation_id
    state["genie_history"].append(history_entry)

    if bool(payload.get("show_rewrite")):
        prefix = [
            f"llm_endpoint: {payload.get('endpoint') or cfg['llm_endpoint']}",
            f"genie_question: {genie_question}",
        ]
        assumptions = rewrite.get("assumptions") or []
        if assumptions:
            prefix.append("assumptions: " + "; ".join(str(item) for item in assumptions))
        answer = "\n".join(prefix) + "\n\n" + answer

    return answer, str(next_conversation_id) if next_conversation_id else None, history_entry


def query_dbsql(
    payload: dict[str, Any],
    question: str,
    state: dict[str, Any],
) -> tuple[str, None, dict[str, str]]:
    if bool(payload.get("new_conversation")):
        state["dbsql_history"] = []

    cfg = config()
    token = resolve_bearer_token(cfg)
    history = payload.get("history") if isinstance(payload.get("history"), list) else state["dbsql_history"]
    max_steps = int(payload.get("max_steps") or 8)
    endpoint = str(payload.get("endpoint") or cfg["llm_endpoint"])

    messages = initial_dbsql_messages(question, history)
    tool_log: list[dict[str, Any]] = []
    answer = "已達工具呼叫步數上限，但尚未產生最終回答。"

    for step in range(1, max_steps + 1):
        content = llm_chat(cfg, token, endpoint, messages, temperature=0, max_tokens=1200)
        action = parse_action(content)

        if action.get("action") == "final":
            answer = str(action.get("answer") or "").strip()
            break

        if action.get("action") != "call_tool":
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Return valid JSON with action final or call_tool."})
            continue

        tool_name = str(action.get("tool_name") or "")
        arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
        try:
            tool_result = call_dbsql_tool(cfg, token, tool_name, arguments)
        except Exception as exc:
            tool_result = {"is_error": True, "error": str(exc)}

        tool_log.append(
            {
                "step": step,
                "tool_name": tool_name,
                "arguments": redact_large(arguments),
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

    if bool(payload.get("show_trace")):
        trace = {
            "llm_endpoint": endpoint,
            "sql_backend": "Databricks SQL Statements API",
            "allowed_tools": ["execute_sql_read_only", "poll_sql_result"],
            "tool_calls": tool_log,
        }
        answer = json.dumps(trace, ensure_ascii=False, indent=2) + "\n\n" + answer

    history_entry = {"user_question": question, "answer_excerpt": compact(answer)}
    state["dbsql_history"].append(history_entry)
    return answer, None, history_entry


def ask_genie_api(
    cfg: dict[str, str],
    token: str,
    question: str,
    *,
    conversation_id: str | None,
    include_query_results: bool,
    max_result_rows: int,
) -> dict[str, Any]:
    space_id = quote(cfg["genie_space_id"], safe="")
    if conversation_id:
        initial = databricks_json(
            cfg,
            token,
            "POST",
            f"/api/2.0/genie/spaces/{space_id}/conversations/{quote(conversation_id, safe='')}/messages",
            {"content": question},
        )
        message = message_from_response(initial)
        resolved_conversation_id = conversation_id
    else:
        initial = databricks_json(
            cfg,
            token,
            "POST",
            f"/api/2.0/genie/spaces/{space_id}/start-conversation",
            {"content": question},
        )
        message = message_from_response(initial)
        conversation = initial.get("conversation") or {}
        resolved_conversation_id = str(conversation.get("id") or message.get("conversation_id") or "")

    message_id = str(message.get("id") or "")
    if not resolved_conversation_id or not message_id:
        raise AppError("Genie response did not include conversation_id and message_id.", 502)

    message = wait_for_genie_message(cfg, token, resolved_conversation_id, message_id)
    compiled = compile_genie_message(message)
    compiled["conversation_id"] = resolved_conversation_id
    compiled["message_id"] = message_id

    if include_query_results and str(message.get("status", "")).upper() == "COMPLETED":
        query_results: dict[str, Any] = {}
        for attachment_id in compiled["query_attachment_ids"]:
            raw = databricks_json(
                cfg,
                token,
                "GET",
                "/api/2.0/genie/spaces/"
                f"{space_id}/conversations/{quote(resolved_conversation_id, safe='')}/messages/"
                f"{quote(message_id, safe='')}/query-result/{quote(attachment_id, safe='')}",
            )
            query_results[attachment_id] = compact_query_result(raw, max_result_rows)
        compiled["query_results"] = query_results

    return compiled


def wait_for_genie_message(
    cfg: dict[str, str],
    token: str,
    conversation_id: str,
    message_id: str,
) -> dict[str, Any]:
    timeout_seconds = float(os.getenv("GENIE_POLL_TIMEOUT_SECONDS") or "600")
    delay = float(os.getenv("GENIE_POLL_INITIAL_INTERVAL_SECONDS") or "1")
    max_delay = float(os.getenv("GENIE_POLL_MAX_INTERVAL_SECONDS") or "10")
    deadline = time.monotonic() + timeout_seconds
    space_id = quote(cfg["genie_space_id"], safe="")
    path = (
        f"/api/2.0/genie/spaces/{space_id}/conversations/{quote(conversation_id, safe='')}"
        f"/messages/{quote(message_id, safe='')}"
    )
    while True:
        message = databricks_json(cfg, token, "GET", path)
        status = str(message.get("status", "")).upper()
        if status in TERMINAL_GENIE_STATUSES:
            return message
        if time.monotonic() >= deadline:
            raise AppError(f"Genie message {message_id} did not complete. Last status: {status}", 504)
        time.sleep(delay)
        delay = min(delay * 1.5, max_delay)


def rewrite_for_genie(
    cfg: dict[str, str],
    token: str,
    question: str,
    *,
    history: list[dict[str, str]],
    endpoint: str,
) -> dict[str, Any]:
    messages = [{"role": "system", "content": REWRITE_SYSTEM_PROMPT}]
    if history:
        messages.append({"role": "user", "content": format_history(history)})
    messages.append({"role": "user", "content": question})
    content = llm_chat(cfg, token, endpoint, messages, temperature=0, max_tokens=700)
    return parse_rewrite(content, question)


def llm_chat(
    cfg: dict[str, str],
    token: str,
    endpoint: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    endpoint_name = quote(endpoint, safe="")
    payload = databricks_json(
        cfg,
        token,
        "POST",
        f"/serving-endpoints/{endpoint_name}/invocations",
        {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    return extract_chat_content(payload)


def call_dbsql_tool(
    cfg: dict[str, str],
    token: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if tool_name == "execute_sql_read_only":
        return execute_read_only_sql(cfg, token, str(arguments.get("query") or ""))
    if tool_name == "poll_sql_result":
        return poll_sql_result(cfg, token, str(arguments.get("statement_id") or ""))
    raise ValueError(f"Tool {tool_name!r} is not allowed.")


def execute_read_only_sql(cfg: dict[str, str], token: str, query: str) -> dict[str, Any]:
    warehouse_id = cfg.get("warehouse_id")
    if not warehouse_id:
        raise ValueError("DBSQL agent needs DATABRICKS_WAREHOUSE_ID.")
    safe_query = validate_read_only_sql(query)
    result = compact_sql_result(
        databricks_json(
            cfg,
            token,
            "POST",
            "/api/2.0/sql/statements",
            {
                "statement": safe_query,
                "warehouse_id": warehouse_id,
                "disposition": "INLINE",
                "format": "JSON_ARRAY",
                "on_wait_timeout": "CONTINUE",
                "row_limit": 100,
                "wait_timeout": "10s",
            },
        )
    )

    statement_id = result.get("statement_id")
    if not statement_id or state_from_sql_result(result) in TERMINAL_SQL_STATES:
        return result

    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        time.sleep(2)
        result = poll_sql_result(cfg, token, str(statement_id))
        if state_from_sql_result(result) in TERMINAL_SQL_STATES:
            return result
    result["poll_timeout"] = True
    return result


def poll_sql_result(cfg: dict[str, str], token: str, statement_id: str) -> dict[str, Any]:
    if not statement_id:
        raise ValueError("poll_sql_result needs statement_id.")
    return compact_sql_result(databricks_json(cfg, token, "GET", f"/api/2.0/sql/statements/{statement_id}"))


def databricks_json(
    cfg: dict[str, str],
    token: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: int = 60,
) -> dict[str, Any]:
    url = cfg["host"].rstrip("/") + path
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    return http_json(method, url, body=body, headers=headers, timeout=timeout)


def http_json(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise AppError(f"HTTP {exc.code} for {url}: {detail}", 502) from exc
    except URLError as exc:
        raise AppError(f"Request failed for {url}: {exc}", 502) from exc

    if not raw:
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise AppError(f"Non-JSON response from {url}: {raw[:1000]!r}", 502) from exc
    return payload if isinstance(payload, dict) else {"result": payload}


def resolve_bearer_token(cfg: dict[str, str]) -> str:
    if cfg.get("token"):
        return cfg["token"]

    if cfg.get("auth_type") == "databricks-cli":
        return cli_access_token(cfg.get("cli_path") or "", cfg.get("profile") or "")

    client_id = cfg.get("client_id")
    client_secret = cfg.get("client_secret")
    if client_id and client_secret:
        return oauth_m2m_token(cfg)

    raise AppError(
        "Databricks auth needs DATABRICKS_TOKEN, databricks-cli auth, or DATABRICKS_CLIENT_ID plus DATABRICKS_CLIENT_SECRET.",
        500,
    )


def oauth_m2m_token(cfg: dict[str, str]) -> str:
    scope = cfg.get("oauth_scope") or "all-apis"
    cache_key = (cfg["host"], cfg["client_id"], scope)
    cached = TOKEN_CACHE.get(cache_key)
    if cached and cached[1] > time.time() + 60:
        return cached[0]

    token_endpoint = discover_token_endpoint(cfg["host"])
    credentials = f"{cfg['client_id']}:{cfg['client_secret']}".encode("utf-8")
    body = urlencode({"grant_type": "client_credentials", "scope": scope}).encode("utf-8")
    request = Request(
        token_endpoint,
        data=body,
        headers={
            "Authorization": "Basic " + base64.b64encode(credentials).decode("ascii"),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise AppError(f"Databricks OAuth token request failed with HTTP {exc.code}: {detail}", 502) from exc
    except Exception as exc:
        raise AppError(f"Databricks OAuth token request failed: {exc}", 502) from exc

    access_token = str(payload.get("access_token") or "")
    if not access_token:
        raise AppError("Databricks OAuth token response did not include access_token.", 502)

    expires_in = safe_int(payload.get("expires_in"), 3600)
    TOKEN_CACHE[cache_key] = (access_token, time.time() + expires_in)
    return access_token


def discover_token_endpoint(host: str) -> str:
    host = host.rstrip("/")
    discovery_urls: list[str] = []
    try:
        metadata = http_json("GET", f"{host}/.well-known/databricks-config", timeout=10)
        oidc_endpoint = str(metadata.get("oidc_endpoint") or "").rstrip("/")
        if oidc_endpoint:
            discovery_urls.append(f"{oidc_endpoint}/.well-known/oauth-authorization-server")
    except Exception:
        pass

    discovery_urls.append(f"{host}/oidc/.well-known/oauth-authorization-server")
    for url in discovery_urls:
        try:
            payload = http_json("GET", url, timeout=10)
            token_endpoint = payload.get("token_endpoint")
            if token_endpoint:
                return str(token_endpoint)
        except Exception:
            continue
    return f"{host}/oidc/v1/token"


def cli_access_token(cli_path: str, profile: str) -> str:
    path = cli_path or str(PROJECT_ROOT / ".tools" / "databricks.exe")
    command = [path if Path(path).exists() else "databricks", "auth", "token", "-o", "json", "--timeout", "10m"]
    if profile:
        command.extend(["--profile", profile])
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True, timeout=660)
    except Exception as exc:
        raise AppError(f"Databricks CLI could not provide an OAuth token: {exc}", 500) from exc
    payload = json.loads(completed.stdout)
    access_token = payload.get("access_token")
    if not access_token:
        raise AppError("Databricks CLI token response did not include access_token.", 500)
    return str(access_token)


def validate_read_only_sql(query: str) -> str:
    normalized = strip_sql_comments(query or "").strip()
    statements = [part.strip() for part in normalized.split(";") if part.strip()]
    if len(statements) != 1:
        raise ValueError("Only one read-only SQL statement is allowed.")

    statement = statements[0].strip()
    if not READ_ONLY_START_RE.match(statement):
        raise ValueError("Only SELECT, WITH, SHOW, DESCRIBE, DESC, and EXPLAIN queries are allowed.")
    blocked = BLOCKED_SQL_RE.search(statement)
    if blocked:
        raise ValueError(f"Blocked non-read-only SQL keyword: {blocked.group(1).upper()}.")
    return statement


def strip_sql_comments(query: str) -> str:
    without_block = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)
    return re.sub(r"--.*?$", "", without_block, flags=re.MULTILINE)


def initial_dbsql_messages(question: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
    context = {
        "allowed_tools": [
            {
                "name": "execute_sql_read_only",
                "description": "Execute one read-only Databricks SQL statement.",
            },
            {
                "name": "poll_sql_result",
                "description": "Poll a Databricks SQL statement by statement_id.",
            },
        ],
        "recent_conversation": history[-6:],
    }
    return [
        {"role": "system", "content": DBSQL_SYSTEM_PROMPT},
        {"role": "user", "content": "Context:\n" + json.dumps(context, ensure_ascii=False, indent=2)},
        {"role": "user", "content": question},
    ]


def parse_action(content: str) -> dict[str, Any]:
    text = content.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        return {"action": "final", "answer": content}
    return parsed if isinstance(parsed, dict) else {"action": "final", "answer": content}


def parse_rewrite(content: str, fallback_question: str) -> dict[str, Any]:
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
            "assumptions": ["Databricks LLM response was not valid JSON, so the original question was used."],
        }
    return {
        "needs_clarification": bool(parsed.get("needs_clarification")),
        "clarifying_question": str(parsed.get("clarifying_question") or ""),
        "genie_question": str(parsed.get("genie_question") or fallback_question),
        "assumptions": parsed.get("assumptions") if isinstance(parsed.get("assumptions"), list) else [],
    }


def extract_chat_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))

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


def compile_genie_message(message: dict[str, Any]) -> dict[str, Any]:
    attachments = as_list(message.get("attachments"))
    answer_text = []
    sql = []
    query_attachment_ids = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        text = extract_attachment_text(attachment)
        if text:
            answer_text.append(text)
        statement = extract_attachment_sql(attachment)
        if statement:
            sql.append(statement)
        attachment_id = str(attachment.get("attachment_id") or attachment.get("id") or "")
        if attachment_id and attachment.get("query") is not None:
            query_attachment_ids.append(attachment_id)
    return {
        "status": message.get("status"),
        "error": message.get("error"),
        "answer_text": answer_text,
        "sql": sql,
        "query_attachment_ids": query_attachment_ids,
        "attachments": attachments,
    }


def format_genie_answer(result: dict[str, Any], *, show_sql: bool, show_ids: bool) -> str:
    parts: list[str] = []
    status = result.get("status")
    if status and status != "COMPLETED":
        parts.append(f"status: {status}")
    if result.get("error"):
        parts.append("error: " + json.dumps(result["error"], ensure_ascii=False))
    for text in result.get("answer_text") or []:
        parts.append(str(text))
    if show_sql:
        for index, statement in enumerate(result.get("sql") or [], start=1):
            parts.append(f"sql[{index}]:\n{statement}")
    for attachment_id, query_result in (result.get("query_results") or {}).items():
        header = f"query_result[{attachment_id}]"
        if query_result.get("row_count") is not None:
            header += f" row_count={query_result['row_count']}"
        if query_result.get("truncated"):
            header += " truncated=true"
        parts.append(header)
        parts.append(json.dumps(query_result.get("rows") or [], ensure_ascii=False, indent=2))
    if show_ids:
        parts.append(
            json.dumps(
                {"conversation_id": result.get("conversation_id"), "message_id": result.get("message_id")},
                ensure_ascii=False,
                indent=2,
            )
        )
    return "\n\n".join(parts) if parts else json.dumps(result, ensure_ascii=False, indent=2)


def compact_query_result(raw_result: dict[str, Any], max_rows: int) -> dict[str, Any]:
    rows = extract_rows(raw_result)
    return {
        "columns": extract_columns(raw_result),
        "rows": rows[:max_rows],
        "row_count": len(rows),
        "truncated": len(rows) > max_rows,
        "raw_result": raw_result if not rows else None,
    }


def compact_sql_result(payload: dict[str, Any], max_rows: int = 100) -> dict[str, Any]:
    manifest = payload.get("manifest") if isinstance(payload.get("manifest"), dict) else {}
    columns = extract_manifest_columns(manifest)
    rows = extract_statement_rows(payload, columns)
    compacted: dict[str, Any] = {
        "statement_id": payload.get("statement_id"),
        "status": payload.get("status"),
        "message": payload.get("message"),
        "is_error": payload.get("is_error", False),
        "total_row_count": manifest.get("total_row_count"),
        "truncated": manifest.get("truncated"),
    }
    if columns:
        compacted["columns"] = columns
    if rows:
        compacted["rows"] = rows[:max_rows]
        compacted["returned_row_count"] = len(rows[:max_rows])
        compacted["rows_truncated_locally"] = len(rows) > max_rows
    if not rows and payload.get("result"):
        compacted["result"] = payload.get("result")
    return {key: value for key, value in compacted.items() if value is not None}


def extract_rows(raw_result: dict[str, Any]) -> list[Any]:
    data = find_first(raw_result, [
        ("statement_response", "result", "data_array"),
        ("result", "data_array"),
        ("data_array",),
        ("rows",),
    ])
    if not isinstance(data, list):
        return []
    columns = extract_columns(raw_result)
    if not columns:
        return data
    return [dict(zip(columns, row, strict=False)) if isinstance(row, list) else row for row in data]


def extract_columns(raw_result: dict[str, Any]) -> list[str]:
    schema_columns = find_first(raw_result, [
        ("statement_response", "manifest", "schema", "columns"),
        ("manifest", "schema", "columns"),
        ("schema", "columns"),
        ("columns",),
    ])
    if not isinstance(schema_columns, list):
        return []
    columns = []
    for column in schema_columns:
        if isinstance(column, dict):
            columns.append(str(column.get("name") or column.get("column_name") or column.get("label") or ""))
        else:
            columns.append(str(column))
    return [column for column in columns if column]


def extract_manifest_columns(manifest: dict[str, Any]) -> list[str]:
    schema = manifest.get("schema") if isinstance(manifest.get("schema"), dict) else {}
    raw_columns = schema.get("columns") if isinstance(schema.get("columns"), list) else []
    return [str(column.get("name")) for column in raw_columns if isinstance(column, dict) and column.get("name")]


def extract_statement_rows(payload: dict[str, Any], columns: list[str]) -> list[dict[str, Any] | list[Any]]:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    data_array = result.get("data_array")
    if isinstance(data_array, list):
        return [dict(zip(columns, row, strict=False)) if columns and isinstance(row, list) else row for row in data_array]
    typed_rows = result.get("data_typed_array")
    if isinstance(typed_rows, list):
        rows = []
        for raw_row in typed_rows:
            values = raw_row.get("values") if isinstance(raw_row, dict) else raw_row
            if isinstance(values, list):
                row_values = [typed_value(value) for value in values]
                rows.append(dict(zip(columns, row_values, strict=False)) if columns else row_values)
            else:
                rows.append(raw_row)
        return rows
    return []


def typed_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in (
        "str", "string_value", "long", "long_value", "int", "int_value",
        "double", "double_value", "float", "float_value", "boolean", "boolean_value",
        "bytes", "bytes_value", "decimal", "decimal_value",
    ):
        if key in value:
            return value[key]
    if value.get("null") is True:
        return None
    return next(iter(value.values())) if len(value) == 1 else value


def state_from_sql_result(payload: dict[str, Any]) -> str:
    status = payload.get("status")
    if isinstance(status, dict):
        return str(status.get("state") or "").upper()
    return ""


def message_from_response(response: dict[str, Any]) -> dict[str, Any]:
    message = response.get("message") or response
    if not isinstance(message, dict):
        raise AppError("Genie response message is not an object.", 502)
    return message


def extract_attachment_text(attachment: dict[str, Any]) -> str | None:
    value = attachment.get("text")
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "value"):
            if value.get(key):
                return str(value[key])
    return None


def extract_attachment_sql(attachment: dict[str, Any]) -> str | None:
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


def format_history(history: list[dict[str, str]]) -> str:
    lines = ["Conversation context for resolving follow-up questions:"]
    for index, item in enumerate(history[-6:], start=1):
        lines.append(f"{index}. user_question: {item.get('user_question', '')}")
        if item.get("genie_question"):
            lines.append(f"   genie_question: {item.get('genie_question')}")
        if item.get("answer_excerpt"):
            lines.append(f"   answer_excerpt: {item.get('answer_excerpt')}")
    return "\n".join(lines)


def health_config() -> dict[str, Any]:
    cfg = config()
    return {
        "databricks_host": cfg["host"],
        "genie_space_id": cfg["genie_space_id"],
        "databricks_warehouse_id": cfg.get("warehouse_id") or None,
        "databricks_token_configured": bool(cfg.get("token")),
        "databricks_auth_type": cfg.get("auth_type") or None,
        "databricks_client_id_configured": bool(cfg.get("client_id")),
        "databricks_client_secret_configured": bool(cfg.get("client_secret")),
        "databricks_oauth_scope": cfg.get("oauth_scope") or "all-apis",
        "llm_endpoint": cfg["llm_endpoint"],
    }


def config() -> dict[str, str]:
    return {
        "host": (os.getenv("DATABRICKS_HOST") or DEFAULT_DATABRICKS_HOST).strip().rstrip("/"),
        "genie_space_id": (os.getenv("GENIE_SPACE_ID") or DEFAULT_GENIE_SPACE_ID).strip(),
        "warehouse_id": (os.getenv("DATABRICKS_WAREHOUSE_ID") or "").strip(),
        "token": (os.getenv("DATABRICKS_TOKEN") or "").strip(),
        "auth_type": (os.getenv("DATABRICKS_AUTH_TYPE") or "").strip(),
        "profile": (os.getenv("DATABRICKS_CONFIG_PROFILE") or "").strip(),
        "cli_path": (os.getenv("DATABRICKS_CLI_PATH") or "").strip(),
        "client_id": (os.getenv("DATABRICKS_CLIENT_ID") or "").strip(),
        "client_secret": (os.getenv("DATABRICKS_CLIENT_SECRET") or "").strip(),
        "oauth_scope": (os.getenv("DATABRICKS_OAUTH_SCOPE") or "all-apis").strip(),
        "llm_endpoint": (os.getenv("DATABRICKS_LLM_ENDPOINT") or DEFAULT_LLM_ENDPOINT).strip(),
    }


def load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def find_first(payload: dict[str, Any], paths: list[tuple[str, ...]]) -> Any:
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


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def compact(answer: str, limit: int = 700) -> str:
    value = " ".join(answer.split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def redact_large(value: Any, limit: int = 4000) -> Any:
    text = json.dumps(value, ensure_ascii=False)
    return value if len(text) <= limit else text[: limit - 3] + "..."


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
