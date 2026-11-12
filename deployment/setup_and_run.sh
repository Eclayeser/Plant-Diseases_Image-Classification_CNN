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

# 2. Install the team's official dependencies.
echo "[SGP-5] Installing dependencies..."
"$PYTHON" -m pip install -r requirements.txt

# 3. Train a baseline model if no checkpoint is available.
if [[ -z "${CHECKPOINT_PATH:-}" ]] && \
   ! compgen -G "outputs/models/*.pt" > /dev/null; then

    echo "[SGP-5] No checkpoint found. Training baseline CNN..."

    CLEARML_OFFLINE_MODE=1 "$PYTHON" train.py \
        --config configs/prototype_small_data_frac.yaml
else
    echo "[SGP-5] Existing checkpoint available."
fi

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
