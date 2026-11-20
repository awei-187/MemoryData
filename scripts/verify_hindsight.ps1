$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path 'results/logs/hindsight','results/test-tmp' | Out-Null
$env:PYTHONIOENCODING = 'utf-8'
$env:TEMP = (Resolve-Path -LiteralPath 'results/test-tmp').Path
$env:TMP = $env:TEMP
'running' | Set-Content -LiteralPath 'results/logs/hindsight/verify.status'
try {
    $ErrorActionPreference = 'Continue'
    & 'D:\conda\envs\memorydata\python.exe' -m unittest discover -s tests -p 'test_hindsight.py' -v 2>&1 | Tee-Object -FilePath 'results/logs/hindsight/tests.log'
    $verificationExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($verificationExit -ne 0) { throw ('Hindsight verification failed with exit code ' + $verificationExit) }
    'success' | Set-Content -LiteralPath 'results/logs/hindsight/verify.status'
} catch {
    'failed' | Set-Content -LiteralPath 'results/logs/hindsight/verify.status'
    Write-Host $_
}
