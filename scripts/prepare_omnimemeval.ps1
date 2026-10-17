$ErrorActionPreference = 'Stop'
$memoryDataRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $memoryDataRoot
New-Item -ItemType Directory -Force -Path 'results/logs/omnimemeval' | Out-Null
Start-Transcript -Path 'results/logs/omnimemeval/prepare.log' -Append
$failedBenchmarks = @()
try {
    $env:PYTHONIOENCODING = 'utf-8'
    Set-Location -LiteralPath (Join-Path (Split-Path -Parent $memoryDataRoot) 'OmniMemEval')
    foreach ($benchmark in @('locomo','longmemeval','personamem_v2','halumem','beam')) {
        $scriptName = if ($benchmark -eq 'personamem_v2') { 'prepare_personamem.py' } else { 'prepare_' + $benchmark + '.py' }
        try {
            & 'D:\conda\envs\memorydata\python.exe' ('data/' + $benchmark + '/' + $scriptName)
            if ($LASTEXITCODE -ne 0) { throw ($benchmark + ' data preparation failed') }
        } catch {
            $failedBenchmarks += $benchmark
            Write-Host $_
        }
    }
    if ($failedBenchmarks.Count) { ('failed: ' + ($failedBenchmarks -join ', ')) | Set-Content -LiteralPath (Join-Path $memoryDataRoot 'results/logs/omnimemeval/prepare.status') }
    else { 'success' | Set-Content -LiteralPath (Join-Path $memoryDataRoot 'results/logs/omnimemeval/prepare.status') }
} finally { Stop-Transcript }
