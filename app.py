from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from genie_mcp import dbsql_mcp_agent, dbx_agent_client  # noqa: E402
from genie_mcp.config import ConfigError, GenieConfig  # noqa: E402
from genie_mcp.databricks_genie import GenieAPIError, GenieTimeoutError  # noqa: E402
from genie_mcp.unified_agent import AGENTS  # noqa: E402


AgentName = Literal["genie", "dbsql"]


@dataclass
class SessionState:
    genie_conversation_id: str | None = None
    genie_history: list[dict[str, str]] = field(default_factory=list)
    dbsql_history: list[dict[str, str]] = field(default_factory=list)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    agent: AgentName = "genie"
    session_id: str | None = None
    conversation_id: str | None = None
    new_conversation: bool = False
    history: list[dict[str, str]] | None = None
    endpoint: str | None = None
    include_query_results: bool = True
    max_result_rows: int = Field(default=50, ge=1, le=500)
    no_clarify: bool = False
    show_rewrite: bool = False
    show_sql: bool = False
    show_ids: bool = False
    max_steps: int = Field(default=8, ge=1, le=50)
    show_trace: bool = False


class QueryResponse(BaseModel):
    agent: AgentName
    answer: str
    session_id: str
    conversation_id: str | None = None
    history_entry: dict[str, str] | None = None


class HealthResponse(BaseModel):
    ok: bool
    agents: dict[str, str]
    config: dict[str, object]


app = FastAPI(
    title="Databricks Genie MCP Agent API",
    version="0.1.0",
    description="HTTP API for Databricks Genie and read-only DBSQL MCP agents.",
)

_sessions: dict[str, SessionState] = {}


@app.get("/")
async def root() -> dict[str, object]:
    return {
        "name": "Databricks Genie MCP Agent API",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query",
        "agents": sorted(AGENTS),
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        config = GenieConfig.health_from_env()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"無法讀取設定：{exc}") from exc
    return HealthResponse(ok=True, agents=AGENTS, config=config)


@app.get("/agents")
async def agents() -> dict[str, str]:
    return AGENTS


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    session_id = request.session_id or str(uuid.uuid4())
    state = _sessions.setdefault(session_id, SessionState())

    try:
        if request.agent == "genie":
            return await _query_genie(request, session_id, state)
        return await _query_dbsql(request, session_id, state)
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=f"設定錯誤：{exc}") from exc
    except GenieTimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"Databricks Genie 查詢逾時：{exc}") from exc
    except GenieAPIError as exc:
        raise HTTPException(status_code=502, detail=f"Databricks API 呼叫失敗：{exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"agent 執行失敗：{exc}") from exc


async def _query_genie(
    request: QueryRequest,
    session_id: str,
    state: SessionState,
) -> QueryResponse:
    if request.new_conversation:
        state.genie_conversation_id = None
        state.genie_history.clear()

    conversation_id = request.conversation_id or state.genie_conversation_id
    history = _request_history_or_session(request, state.genie_history)
    args = _args_from_request(request, agent="genie", conversation_id=conversation_id)

    answer, next_conversation_id, history_entry = await dbx_agent_client.run_question(
        args,
        question=request.question.strip(),
        conversation_id=conversation_id,
        history=history,
    )
    state.genie_conversation_id = next_conversation_id
    state.genie_history.append(history_entry)

    return QueryResponse(
        agent="genie",
        answer=answer,
        session_id=session_id,
        conversation_id=next_conversation_id,
        history_entry=history_entry,
    )


async def _query_dbsql(
    request: QueryRequest,
    session_id: str,
    state: SessionState,
) -> QueryResponse:
    if request.new_conversation:
        state.dbsql_history.clear()

    history = _request_history_or_session(request, state.dbsql_history)
    args = _args_from_request(request, agent="dbsql", conversation_id=None)

    answer, history_entry = await dbsql_mcp_agent.run_question(
        args,
        request.question.strip(),
        history=history,
    )
    state.dbsql_history.append(history_entry)

    return QueryResponse(
        agent="dbsql",
        answer=answer,
        session_id=session_id,
        conversation_id=None,
        history_entry=history_entry,
    )


def _request_history_or_session(
    request: QueryRequest,
    session_history: list[dict[str, str]],
) -> list[dict[str, str]]:
    if request.history is not None:
        return [dict(item) for item in request.history]
    return list(session_history)


def _args_from_request(
    request: QueryRequest,
    *,
    agent: AgentName,
    conversation_id: str | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        agent=agent,
        endpoint=request.endpoint,
        conversation_id=conversation_id,
        max_result_rows=request.max_result_rows,
        no_results=not request.include_query_results,
        no_clarify=request.no_clarify,
        show_rewrite=request.show_rewrite,
        show_sql=request.show_sql,
        show_ids=request.show_ids,
        max_steps=request.max_steps,
        show_trace=request.show_trace,
    )
