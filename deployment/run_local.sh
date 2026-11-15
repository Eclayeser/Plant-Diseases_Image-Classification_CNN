#!/usr/bin/env bash
set -euo pipefail

# SGP-5: Local deployment launcher
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"

# Verify Python environment.
if [[ ! -x "$PYTHON" ]]; then
    echo "Error: Python environment not found."
    echo "Create .venv and install requirements.txt first."
    exit 1
fi

# Require an explicitly selected checkpoint.
if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
    echo "Error: CHECKPOINT_PATH is required."
    exit 1
fi

# Resolve relative checkpoint paths.
if [[ "$CHECKPOINT_PATH" != /* ]]; then
    CHECKPOINT_PATH="$ROOT/$CHECKPOINT_PATH"
fi

# Verify checkpoint exists.
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

export CHECKPOINT_PATH
export GRADIO_SERVER_PORT=8080

echo "Starting local deployment..."
echo "Model: $CHECKPOINT_PATH"
echo "URL: http://localhost:8080"

exec "$PYTHON" -m app.app