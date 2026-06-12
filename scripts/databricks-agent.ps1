$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$srcPath = Join-Path $projectRoot "src"
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python virtual environment was not found at $pythonPath"
}

$env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $srcPath
} else {
    "$srcPath$([IO.Path]::PathSeparator)$env:PYTHONPATH"
}

& $pythonPath -m genie_mcp.unified_agent @args

