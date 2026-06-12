$ErrorActionPreference = "Stop"

.\.venv\Scripts\python.exe -c "from genie_mcp.config import GenieConfig; print(GenieConfig.health_from_env())"

