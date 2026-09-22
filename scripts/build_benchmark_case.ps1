param(
  [string]$CaseDir = "examples/cases/TCGA-38-4626",
  [string]$ReportPath = "",
  [string]$RubricPath = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  $Python = "python"
}

$ArgsList = @(
  "-m", "medclaw_benchmark.cli", "build-case",
  "--case-dir", $CaseDir
)
if ($ReportPath) {
  $ArgsList += @("--report-path", $ReportPath)
}
if ($RubricPath) {
  $ArgsList += @("--rubric-path", $RubricPath)
}

& $Python @ArgsList
