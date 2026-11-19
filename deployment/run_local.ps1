# SGP-5: Windows local deployment launcher

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"

# Check Python environment.
if (-not (Test-Path $python)) {
    Write-Error "Python environment not found. Run deployment\setup_and_run.ps1 first."
    exit 1
}

# Download the model pinned in deployment/model.lock.json (quick when cached).
& $python -m scripts.fetch_model_from_clearml

if ($LASTEXITCODE -ne 0) {
    Write-Error "Model download failed. Check your ClearML credentials (clearml.conf)."
    exit 1
}

Write-Host "Starting local deployment..."
Write-Host "URL: http://localhost:8080"

& $python -m app.app

exit $LASTEXITCODE