param(
  [string]$CaseDir = "examples/cases/TCGA-38-4626",
  [string]$RunId = "run_001",
  [string]$RunsRoot = "runs",
  [ValidateSet("local_openai", "qwen", "openai", "gemini", "deepseek", "claude")]
  [string]$Agent = "local_openai",
  [ValidateSet("local_openai", "qwen", "openai", "gemini", "deepseek", "claude")]
  [string]$JudgeProvider = "local_openai"
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

$CaseId = Split-Path $CaseDir -Leaf
$RubricPath = Join-Path $CaseDir "evaluation/${CaseId}_rubric.json"
if (-not (Test-Path $RubricPath)) {
  $RubricPath = Join-Path $CaseDir "evaluation/${CaseId}_T1_rubric.json"
}
if (-not $env:MEDCLAW_GUIDELINE_RERANK_PROVIDER -and
    ($Agent -eq "local_openai" -or $Agent -eq "qwen")) {
  $env:MEDCLAW_GUIDELINE_RERANK_PROVIDER = $Agent
}

& $Python -m medclaw_benchmark.cli run-and-judge `
  --case-dir $CaseDir `
  --rubric-path $RubricPath `
  --agent $Agent `
  --judge-provider $JudgeProvider `
  --judge llm `
  --run-id $RunId `
  --runs-root $RunsRoot
