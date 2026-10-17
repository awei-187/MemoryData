$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path 'results/logs/omnimemeval' | Out-Null
Start-Transcript -Path 'results/logs/omnimemeval/smoke.log' -Append
try {
    $env:PYTHONIOENCODING = 'utf-8'
    $ErrorActionPreference = 'Continue'
    & 'D:\conda\envs\memorydata\python.exe' 'scripts/smoke_omnimemeval.py' 2>&1 | Tee-Object -FilePath 'results/logs/omnimemeval/smoke-output.log'
    $smokeExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($smokeExit -ne 0) { 'blocked_or_failed' | Set-Content -LiteralPath 'results/logs/omnimemeval/smoke.status' }
    else { 'success' | Set-Content -LiteralPath 'results/logs/omnimemeval/smoke.status' }
} finally { Stop-Transcript }
