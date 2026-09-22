param(
  [string]$CaseId = "TCGA-38-4626",
  [string]$Message = "",
  [string[]]$Image = @(),
  [string[]]$ImageUrl = @()
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  $Python = "python"
}

if (-not ($env:MEDCLAW_LOCAL_API_KEY -or $env:VLLM_API_KEY)) {
  Write-Warning "Set MEDCLAW_LOCAL_API_KEY to the vLLM --api-key value."
}

$ArgsList = @("examples/run_qwen_agent.py", "--case-id", $CaseId)
if ($Message) {
  $ArgsList += @("--message", $Message)
}
foreach ($item in $Image) {
  $ArgsList += @("--image", $item)
}
foreach ($item in $ImageUrl) {
  $ArgsList += @("--image-url", $item)
}

& $Python @ArgsList
