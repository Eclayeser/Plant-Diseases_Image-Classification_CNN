# SGP-5: Windows local deployment setup

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

# Require an explicitly selected checkpoint.
if (-not $env:CHECKPOINT_PATH) {
    Write-Error "CHECKPOINT_PATH is required. Train and select a model first."
    exit 1
}

$python = Join-Path $root ".venv\Scripts\python.exe"

# Create virtual environment if necessary.
if (-not (Test-Path $python)) {
    Write-Host "[SGP-5] Creating Python environment..."

    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv .venv
    } else {
        & python -m venv .venv
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create Python environment."
    }
}

# Install project dependencies.
Write-Host "[SGP-5] Installing dependencies..."
& $python -m pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

# Launch the existing Gradio application.
Write-Host "[SGP-5] Starting Gradio at http://localhost:8080"
& (Join-Path $PSScriptRoot "run_local.ps1")

exit $LASTEXITCODE