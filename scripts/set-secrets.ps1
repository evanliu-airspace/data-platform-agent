param(
    [string]$EnvPath = ".env"
)

$ErrorActionPreference = "Stop"

function ConvertTo-PlainText {
    param([Security.SecureString]$SecureValue)

    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureValue)
    try {
        [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

function Get-ExistingValue {
    param(
        [string[]]$Lines,
        [string]$Name,
        [string]$DefaultValue = ""
    )

    foreach ($line in $Lines) {
        if ($line -match "^\s*$([Regex]::Escape($Name))=(.*)$") {
            return $Matches[1]
        }
    }

    return $DefaultValue
}

$existingLines = @()
if (Test-Path -LiteralPath $EnvPath) {
    $existingLines = Get-Content -LiteralPath $EnvPath
}

$databricksHost = Get-ExistingValue $existingLines "DATABRICKS_HOST" "https://adb-415889795140801.1.azuredatabricks.net"
$genieSpaceId = Get-ExistingValue $existingLines "GENIE_SPACE_ID" "01f0a8c81e88142fadad408f820867c3"
$openaiModel = Get-ExistingValue $existingLines "OPENAI_MODEL" "gpt-5.5"
$databricksAuthType = Get-ExistingValue $existingLines "DATABRICKS_AUTH_TYPE" ""
$databricksConfigProfile = Get-ExistingValue $existingLines "DATABRICKS_CONFIG_PROFILE" ""
$databricksCliPath = Get-ExistingValue $existingLines "DATABRICKS_CLI_PATH" ".tools/databricks.exe"
$databricksClientId = Get-ExistingValue $existingLines "DATABRICKS_CLIENT_ID" ""
$databricksClientSecret = Get-ExistingValue $existingLines "DATABRICKS_CLIENT_SECRET" ""

$databricksToken = ConvertTo-PlainText (Read-Host "DATABRICKS_TOKEN (optional; leave blank for OAuth)" -AsSecureString)
$openaiApiKey = ConvertTo-PlainText (Read-Host "OPENAI_API_KEY" -AsSecureString)

if ([string]::IsNullOrWhiteSpace($openaiApiKey)) {
    throw "OPENAI_API_KEY cannot be empty."
}

$content = @(
    "DATABRICKS_HOST=$databricksHost",
    "GENIE_SPACE_ID=$genieSpaceId",
    "",
    "# Option A: Personal access token (legacy)",
    "DATABRICKS_TOKEN=$databricksToken",
    "",
    "# Option B: Databricks OAuth / unified authentication",
    "DATABRICKS_AUTH_TYPE=$databricksAuthType",
    "DATABRICKS_CONFIG_PROFILE=$databricksConfigProfile",
    "DATABRICKS_CLI_PATH=$databricksCliPath",
    "",
    "# Option C: Databricks OAuth M2M service principal",
    "DATABRICKS_CLIENT_ID=$databricksClientId",
    "DATABRICKS_CLIENT_SECRET=$databricksClientSecret",
    "",
    "OPENAI_API_KEY=$openaiApiKey",
    "OPENAI_MODEL=$openaiModel",
    "",
    "GENIE_POLL_TIMEOUT_SECONDS=600",
    "GENIE_POLL_INITIAL_INTERVAL_SECONDS=1",
    "GENIE_POLL_MAX_INTERVAL_SECONDS=10"
)

Set-Content -LiteralPath $EnvPath -Value $content -Encoding UTF8
Write-Host "Updated $EnvPath. Secret values were not printed."
