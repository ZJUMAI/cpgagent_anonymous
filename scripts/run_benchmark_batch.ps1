param(
  [string]$CasesRoot = "/data4/share/cpgtrajbench/LUNG",
  [string]$RunId = "run_001",
  [string]$RunsRoot = "runs",
  [ValidateSet("local_openai", "qwen", "openai", "gemini", "deepseek", "claude")]
  [string]$Agent = "local_openai",
  [ValidateSet("local_openai", "qwen", "openai", "gemini", "deepseek", "claude")]
  [string]$JudgeProvider = "local_openai",
  [int]$Limit = 0,
  [switch]$Force,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  $Python = "python"
}

if (($Agent -eq "local_openai" -or $JudgeProvider -eq "local_openai") -and
    -not ($env:MEDCLAW_LOCAL_API_KEY -or $env:VLLM_API_KEY)) {
  Write-Warning "Set MEDCLAW_LOCAL_API_KEY to the vLLM --api-key value."
}
if (($Agent -eq "qwen" -or $JudgeProvider -eq "qwen") -and
    -not $env:DASHSCOPE_API_KEY) {
  Write-Warning "DASHSCOPE_API_KEY is required for the selected qwen provider."
}
$env:MEDCLAW_CASES_ROOT = $CasesRoot
if (-not $env:MEDCLAW_GUIDELINE_RERANK_PROVIDER -and
    ($Agent -eq "local_openai" -or $Agent -eq "qwen")) {
  $env:MEDCLAW_GUIDELINE_RERANK_PROVIDER = $Agent
}
Write-Host "Using MEDCLAW_CASES_ROOT=$env:MEDCLAW_CASES_ROOT"

$ArgsList = @(
  "-m", "medclaw_benchmark.cli", "batch-run",
  "--cases-root", $CasesRoot,
  "--run-id", $RunId,
  "--runs-root", $RunsRoot,
  "--agent", $Agent,
  "--judge-provider", $JudgeProvider,
  "--limit", "$Limit"
)

if ($Force) {
  $ArgsList += "--force"
}
if ($DryRun) {
  $ArgsList += "--dry-run"
}

& $Python @ArgsList
