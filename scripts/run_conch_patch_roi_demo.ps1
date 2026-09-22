param(
  [string]$Prompt = ""
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

$ArgsList = @("examples/run_conch_patch_roi_demo.py")
if ($Prompt) {
  $ArgsList += @("--prompt", $Prompt)
}

& $Python @ArgsList
