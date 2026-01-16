$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $root '.venv'
$pythonExe = Join-Path $venvPath 'Scripts\python.exe'

if (-Not (Test-Path $pythonExe)) {
  python -m venv $venvPath
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install pandas openpyxl

& $pythonExe (Join-Path $PSScriptRoot 'run_pipeline.py')
