$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$toolsDir = Join-Path $PSScriptRoot 'tools'
New-Item -ItemType Directory -Path $toolsDir -Force | Out-Null
$archive = Join-Path $toolsDir 'OpenMVS_Windows_x64.zip'
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest -Uri 'https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_Windows_x64.zip' -OutFile $archive -TimeoutSec 600
}
Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $toolsDir 'openmvs') -Force
$program = Get-ChildItem -LiteralPath (Join-Path $toolsDir 'openmvs') -Recurse -Filter 'DensifyPointCloud.exe' | Select-Object -First 1
if (-not $program) { throw 'OpenMVS binary not found in the official archive.' }
& $program.FullName --help
Write-Host 'OpenMVS CPU engine installed. Refresh the browser to update engine status.'
