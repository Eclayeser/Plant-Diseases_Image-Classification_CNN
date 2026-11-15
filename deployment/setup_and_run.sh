#!/usr/bin/env bash
set -euo pipefail

# SGP-5: One-command local deployment setup
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 1. Require a deliberately selected trained model.
if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
    echo "Error: CHECKPOINT_PATH is required."
    echo "Train a model using train.py first."
    exit 1
fi

# 2. Create a virtual environment if necessary.
if [[ ! -x ".venv/bin/python" ]]; then
    echo "[SGP-5] Creating Python environment..."
    python3 -m venv .venv
fi

PYTHON="$ROOT/.venv/bin/python"

# 3. Install the team's official dependencies.
echo "[SGP-5] Installing dependencies..."
"$PYTHON" -m pip install -r requirements.txt

# 4. Open the browser when the local server is ready (macOS).
if [[ "$(uname -s)" == "Darwin" ]]; then
    (
        for ((i=0; i<120; i++)); do
            if curl -fsS --max-time 2 \
                -o /dev/null http://127.0.0.1:8080/; then
                open "http://localhost:8080"
                exit 0
            fi
            sleep 1
        done
    ) &
fi

# 5. Launch the existing local Gradio application.
echo "[SGP-5] Starting Gradio at http://localhost:8080"
exec bash deployment/run_local.sh
