#!/usr/bin/env bash
set -euo pipefail

# SGP-5: One-command local deployment setup
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 1. Create a virtual environment if necessary.
if [[ ! -x ".venv/bin/python" ]]; then
    echo "[SGP-5] Creating Python environment..."
    python3 -m venv .venv
fi

PYTHON="$ROOT/.venv/bin/python"

# 2. Install the app's runtime dependencies (the same as the cloud deployment).
echo "[SGP-5] Installing dependencies..."
"$PYTHON" -m pip install -r requirements-deploy.txt

# 3. Open the browser when the local server is ready (macOS).
if [[ "$(uname -s)" == "Darwin" ]]; then
    (
        url="http://localhost:${PORT:-8080}"
        for ((i=0; i<120; i++)); do
            if curl -fsS --max-time 2 \
                -o /dev/null "${url}/"; then
                open "$url"
                exit 0
            fi
            sleep 1
        done
    ) &
fi

# 4. Download the model and launch the Gradio application.
echo "[SGP-5] Starting Gradio..."
exec bash deployment/run_local.sh
