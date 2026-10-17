$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path 'results/logs/omnimemeval','results/test-tmp' | Out-Null
$env:PYTHONIOENCODING = 'utf-8'
$env:TEMP = (Resolve-Path -LiteralPath 'results/test-tmp').Path
$env:TMP = $env:TEMP
Start-Transcript -Path 'results/logs/omnimemeval/verify.log' -Append
try {
    $ErrorActionPreference = 'Continue'
    & 'D:\conda\envs\memorydata\python.exe' -m unittest discover -s tests -p 'test_omnimemeval*.py' -v 2>&1 | Tee-Object -FilePath 'results/logs/omnimemeval/tests.log'
    $verificationExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($verificationExit -eq 0) { 'success' | Set-Content -LiteralPath 'results/logs/omnimemeval/verify.status' }
    else { 'failed' | Set-Content -LiteralPath 'results/logs/omnimemeval/verify.status' }
} finally { Stop-Transcript }
