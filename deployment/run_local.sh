#!/usr/bin/env bash
set -euo pipefail

# SGP-5: Local deployment launcher
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"

# Check that the local Python environment exists.
if [[ ! -x "$PYTHON" ]]; then
    echo "Error: Python environment not found."
    echo "Create .venv and install requirements.txt first."
    exit 1
fi

# Use an explicitly provided checkpoint, or select the newest.
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
    if [[ "$CHECKPOINT_PATH" != /* ]]; then
        CHECKPOINT_PATH="$ROOT/$CHECKPOINT_PATH"
    fi
else
    shopt -s nullglob
    checkpoints=("$ROOT"/outputs/models/*.pt)

    if [[ ${#checkpoints[@]} -eq 0 ]]; then
        echo "Error: No trained checkpoint found in outputs/models/"
        echo "Train a model using train.py first."
        exit 1
    fi

    CHECKPOINT_PATH="$(ls -t "${checkpoints[@]}" | head -n 1)"
fi

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
