$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE) { throw 'Install Python 3.12 x64, then run setup.ps1 again.' }
}
& './.venv/Scripts/python.exe' -m pip install --timeout 120 -r requirements-lock.txt
if ($LASTEXITCODE) { throw 'Python dependency installation failed.' }
& npm.cmd ci
if ($LASTEXITCODE) { throw 'Frontend dependency installation failed.' }
& npm.cmd run build
if ($LASTEXITCODE) { throw 'Frontend build failed.' }
Write-Host 'Base app installed. Run the engine install script for photo reconstruction.'
