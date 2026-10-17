$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path 'results/logs/omnimemeval' | Out-Null
Start-Transcript -Path 'results/logs/omnimemeval/setup.log' -Append
try {
    if (-not (Test-Path -LiteralPath 'D:\conda\envs\memorydata\python.exe')) {
        & 'C:\Users\wjp\anaconda3\Scripts\conda.exe' create --prefix 'D:\conda\envs\memorydata' python=3.11 pip -y
        if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed' }
    }
    $ErrorActionPreference = 'Continue'
    & 'D:\conda\envs\memorydata\python.exe' -m pip install -r requirements-omnimemeval.txt 2>&1 | Tee-Object -FilePath 'results/logs/omnimemeval/pip.log'
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    'success' | Set-Content -LiteralPath 'results/logs/omnimemeval/setup.status'
} catch {
    $_ | Out-String | Set-Content -LiteralPath 'results/logs/omnimemeval/setup.status'
    throw
} finally { Stop-Transcript }
