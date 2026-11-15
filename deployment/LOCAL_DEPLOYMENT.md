# SGP-5 — Local Deployment

Runs Gradio interface with a trained CNN at http://localhost:8080.

## Requirements

- Python 3
- A trained `.pt` checkpoint produced by `train.py`
- Project dependencies from `requirements.txt`

The deployment requires an explicitly selected checkpoint.
It does not automatically train or select a model.

## macOS / Linux

From the repository root:

```bash
export CHECKPOINT_PATH="outputs/models/your-model.pt"
bash deployment/setup_and_run.sh
```

The script creates `.venv` if necessary.

For subsequent runs:

```bash
bash deployment/run_local.sh
```

## Windows (PowerShell)

From the repository root:

```powershell
$env:CHECKPOINT_PATH = "outputs/models/your-model.pt"
.\deployment\setup_and_run.ps1
```

The script creates a Windows virtual environment if necessary.

For subsequent runs:

```powershell
.\deployment\run_local.ps1
```

Open http://localhost:8080 in your browser.

## Notes

- Replace `your-model.pt` with an existing checkpoint filename.
- `CHECKPOINT_PATH` must be set before launching the application.
- Stop the server with Ctrl+C.
- If port 8080 is already occupied, stop the other application first.