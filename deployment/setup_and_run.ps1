# SGP-5: Windows local deployment setup

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

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

# Install the app's runtime dependencies (the same as the cloud deployment).
Write-Host "[SGP-5] Installing dependencies..."
& $python -m pip install -r requirements-deploy.txt

if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

# Download the model and launch the Gradio application.
Write-Host "[SGP-5] Starting Gradio..."
& (Join-Path $PSScriptRoot "run_local.ps1")

exit $LASTEXITCODE
