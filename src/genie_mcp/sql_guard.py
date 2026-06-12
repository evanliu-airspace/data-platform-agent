from __future__ import annotations

import re

import sqlparse


READ_ONLY_START_RE = re.compile(r"^\s*(select|with|show|describe|desc|explain)\b", re.IGNORECASE)
BLOCKED_SQL_RE = re.compile(
    r"\b("
    r"alter|analyze|attach|cache|clone|copy\s+into|create|delete|drop|grant|insert|"
    r"merge|msck|optimize|put|recover|refresh|replace|restore|revoke|set|truncate|"
    r"uncache|update|use|vacuum"
    r")\b",
    re.IGNORECASE,
)


class SQLSafetyError(ValueError):
    """Raised when SQL is not safe for read-only execution."""


def validate_read_only_sql(query: str) -> str:
    normalized = _normalize_sql(query)
    statements = [statement.strip() for statement in sqlparse.split(normalized) if statement.strip()]
    if len(statements) != 1:
        raise SQLSafetyError("Only one read-only SQL statement is allowed.")

    statement = statements[0].rstrip(";").strip()
    if not statement:
        raise SQLSafetyError("SQL query is empty.")

    if not READ_ONLY_START_RE.match(statement):
        raise SQLSafetyError("Only SELECT, WITH, SHOW, DESCRIBE, DESC, and EXPLAIN queries are allowed.")

    blocked = BLOCKED_SQL_RE.search(statement)
    if blocked:
        raise SQLSafetyError(f"Blocked non-read-only SQL keyword: {blocked.group(1).upper()}.")

    return statement


def _normalize_sql(query: str) -> str:
    return sqlparse.format(query or "", strip_comments=True).strip()

