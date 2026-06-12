# Databricks Genie MCP Agent

This project exposes a Databricks Genie Space as MCP tools and includes a Python OpenAI Agents SDK client. The LLM handles user-language understanding and prompt shaping; Databricks Genie remains the source of truth for SQL generation, query execution, and tabular results.

## Architecture

```text
User question
  -> OpenAI agent client (default model: gpt-5.5)
  -> local MCP server over stdio
  -> Databricks Genie Spaces API
  -> generated SQL + query results
  -> agent summarizes only the returned Genie evidence
```

## Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[client]"
```

Create your local environment file if it does not already exist:

```powershell
Copy-Item .env.example .env
```

## Authentication

Set the shared Databricks and OpenAI values in `.env`:

```env
DATABRICKS_HOST=https://adb-415889795140801.1.azuredatabricks.net
GENIE_SPACE_ID=01f0a8c81e88142fadad408f820867c3
OPENAI_API_KEY=<your OpenAI API key>
OPENAI_MODEL=gpt-5.5
```

### Option A: Databricks Personal Access Token

Use this only if your workspace allows PAT creation:

```env
DATABRICKS_TOKEN=<your Databricks PAT>
```

### Option B: Databricks OAuth User Login

Use this when PAT creation is disabled but browser login is allowed.

Install the Databricks CLI, then run:

```powershell
databricks auth login --host https://adb-415889795140801.1.azuredatabricks.net
```

Then set:

```env
DATABRICKS_TOKEN=
DATABRICKS_AUTH_TYPE=databricks-cli
DATABRICKS_CONFIG_PROFILE=
```

If the CLI created a named profile and auto-discovery does not find it, set:

```env
DATABRICKS_CONFIG_PROFILE=<profile-name>
```

### Option C: Databricks OAuth M2M Service Principal

Use this for production or unattended execution:

```env
DATABRICKS_TOKEN=
DATABRICKS_AUTH_TYPE=oauth-m2m
DATABRICKS_CLIENT_ID=<service-principal-client-id>
DATABRICKS_CLIENT_SECRET=<service-principal-oauth-secret>
```

The service principal must be assigned to the workspace and must have access to the Genie Space, SQL warehouse, catalogs, schemas, and tables that Genie uses.

Check non-secret configuration:

```powershell
.\scripts\check-config.ps1
```

## Run The MCP Server

The MCP server uses stdio transport by default:

```powershell
genie-mcp
```

Most MCP clients should launch it as a subprocess:

```text
python -m genie_mcp.server
```

## Run The Recommended Agent Client

This mode uses the OpenAI API for language understanding and Databricks Genie for data:

Ask one question:

```powershell
genie-agent "What was revenue by product line last month?"
```

Or start an interactive shell:

```powershell
genie-agent
```

## Run Without OpenAI API

Direct mode skips the OpenAI API completely. It sends the question straight to Databricks Genie:

```powershell
genie-direct "What was revenue by product line last month?"
```

Verify the configured Genie Space:

```powershell
genie-direct --space
```

Print generated SQL and message IDs:

```powershell
genie-direct --show-sql --show-ids "What was revenue by product line last month?"
```

In this mode, Databricks Genie performs the natural-language-to-SQL work. There is no extra LLM layer to clarify ambiguous questions or polish the answer.

## Run With A Databricks-Hosted LLM

This mode does not call OpenAI. It uses a Databricks Model Serving or Foundation Model endpoint to understand and rewrite the question, then sends the rewritten question to Genie.

Set the model endpoint in `.env`:

```env
DATABRICKS_LLM_ENDPOINT=databricks-meta-llama-3-3-70b-instruct
```

Run:

```powershell
genie-dbx-agent --show-rewrite --show-sql "What was revenue by product line last month?"
```

Start a continuing conversation by omitting the question:

```powershell
genie-dbx-agent --show-rewrite
```

The interactive session keeps the same Genie conversation ID and gives the Databricks-hosted LLM recent context so follow-up questions can refer to the prior turn.

If that endpoint is not enabled in your workspace, choose an available endpoint from Databricks Serving UI, or list endpoints:

```powershell
.\.tools\databricks.exe serving-endpoints list --profile genie-mcp
```

## MCP Tools

The server exposes:

- `ask_genie`: start or continue a Genie conversation, wait for completion, and return answer text, generated SQL, and compact query rows.
- `get_genie_message`: poll or inspect a Genie message.
- `get_genie_query_result`: fetch query results for a specific Genie attachment.
- `get_genie_space`: verify the configured Genie Space.
- `health`: show non-secret runtime configuration.

## Run A Read-Only DBSQL MCP Agent

This mode connects to the Databricks managed DBSQL MCP server:

```text
https://adb-415889795140801.1.azuredatabricks.net/api/2.0/mcp/sql
```

It only exposes `execute_sql_read_only` and `poll_sql_result` to the agent, and it also validates SQL locally before execution.

Ask one question:

```powershell
dbsql-mcp-agent "請列出我可以存取的 catalogs"
```

Show the LLM endpoint and MCP tool-call trace:

```powershell
dbsql-mcp-agent --show-trace "請列出 dev_002_kant_v3 裡有哪些 schemas"
```

Start a continuing conversation:

```powershell
dbsql-mcp-agent --show-trace
```

## Run The Unified Agent Interface

Use one CLI and choose which agent answers:

```powershell
.\scripts\databricks-agent.ps1 --agent genie "請問這週銷量最好的門市是哪間"
.\scripts\databricks-agent.ps1 --agent dbsql --show-trace "請列出我可以存取的 catalogs"
```

Start interactive mode and switch agents as you go:

```powershell
.\scripts\databricks-agent.ps1
```

Interactive commands:

```text
/agent genie
/agent dbsql
/agents
/current
/exit
```

## Client Behavior

The included OpenAI client instructs the model to:

- clarify ambiguous metrics, filters, entities, or time ranges before querying;
- rewrite user wording into a concise Genie-ready question;
- use Genie for data-backed answers;
- avoid inventing numbers, columns, SQL, or conclusions not present in Genie output;
- summarize the result in the user's language and mention relevant filters and time ranges.
