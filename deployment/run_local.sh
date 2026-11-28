#!/usr/bin/env bash
set -euo pipefail

# SGP-5: Local deployment launcher
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"

# Verify Python environment.
if [[ ! -x "$PYTHON" ]]; then
    echo "Error: Python environment not found."
    echo "Run deployment/setup_and_run.sh first."
    exit 1
fi

# Download the model pinned in deployment/model.lock.json (quick when cached).
if ! "$PYTHON" -m scripts.fetch_model_from_clearml; then
    echo "Error: model download failed. Check your ClearML credentials (clearml.conf)."
    exit 1
fi

echo "Starting local deployment..."

exec "$PYTHON" -m app.app
