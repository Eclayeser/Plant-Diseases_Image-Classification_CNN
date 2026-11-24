# SGP-5 — Local Deployment

Runs the Gradio interface with the deployed ONNX model at http://localhost:8080 by default.

## Requirements

- Python 3
- On macOS: Apple Silicon and macOS 14 or later (the pinned `onnxruntime` has no wheels for older Macs)
- ClearML credentials in `clearml.conf` (see [Running ClearML](../README.md#running-clearml))

The model is not stored in the repository. The scripts download the model
pinned in `deployment/model.lock.json` from ClearML, which is the same model
as the cloud deployment.

## macOS / Linux

From the repository root:

```bash
bash deployment/setup_and_run.sh
```

The script creates `.venv` if necessary and installs `requirements-deploy.txt`,
the app's runtime packages (no torch).

For subsequent runs:

```bash
bash deployment/run_local.sh
```

## Windows (PowerShell)

From the repository root:

```powershell
.\deployment\setup_and_run.ps1
```

The script creates a Windows virtual environment if necessary.

For subsequent runs:

```powershell
.\deployment\run_local.ps1
```

Open the local URL printed by Gradio in your browser.

## Trying a model that is not deployed yet

`run_local` always downloads the pinned model again, which would replace your
own export. To try a new export, run the app directly instead:

```bash
python -m scripts.export_onnx --task-id <training task ID>
python -m app.app
```

The export needs the full `requirements.txt`, because it uses torch.

## Notes

- Stop the server with Ctrl+C.
- To use another port, set `$env:PORT = "9090"` in PowerShell or `export PORT=9090` in Bash before running either launcher.
