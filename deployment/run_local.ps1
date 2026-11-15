# SGP-5: Windows local deployment launcher

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"

# Check Python environment.
if (-not (Test-Path $python)) {
    Write-Error "Python environment not found. Create .venv first."
    exit 1
}

# Require an explicitly selected checkpoint.
if (-not $env:CHECKPOINT_PATH) {
    Write-Error "CHECKPOINT_PATH is required."
    exit 1
}

# Resolve relative checkpoint paths.
$checkpoint = $env:CHECKPOINT_PATH

if (-not [System.IO.Path]::IsPathRooted($checkpoint)) {
    $checkpoint = Join-Path $root $checkpoint
}

if (-not (Test-Path -PathType Leaf $checkpoint)) {
    Write-Error "Checkpoint not found: $checkpoint"
    exit 1
}

$env:CHECKPOINT_PATH = $checkpoint
$env:GRADIO_SERVER_PORT = "8080"

Write-Host "Starting local deployment..."
Write-Host "Model: $checkpoint"
Write-Host "URL: http://localhost:8080"

& $python -m app.app

exit $LASTEXITCODE