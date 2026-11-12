# SGP-5 — Local Deployment

This deployment runs the Gradio interface locally at http://localhost:8080.

## Quick Start

From the repository root, run:

```bash
bash deployment/setup_and_run.sh
```

The script automatically:

1. Creates a Python virtual environment (`.venv`) if needed.
2. Installs dependencies from `requirements.txt`.
3. Trains a baseline CNN using `train.py` and ClearML offline mode if no checkpoint exists.
4. Loads the latest trained checkpoint and launches Gradio on port `8080`.
5. Opens the interface automatically on macOS.

Press `Ctrl + C` in the terminal to stop the server.

## Subsequent Runs

If the environment and trained checkpoint already exist, you can launch the interface directly:

```bash
bash deployment/run_local.sh
```

This skips dependency installation and training.