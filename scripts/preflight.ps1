$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
& python -X utf8 (Join-Path $PSScriptRoot 'study.py') preflight
exit $LASTEXITCODE
