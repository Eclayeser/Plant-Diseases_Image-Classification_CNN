# Architecture

Explains the project structure and how its components connect.

---

## 1. Overview

<!-- 2–4 sentences: leaf image in → disease label out, and the main parts. -->

### High-level diagram

![Architecture diagram](architecture_diagram.png)

---

## 2. Repository layout

| Path | Purpose |
|------|---------|
| `leafapp/` | Serves as a core library for the shared code used by the runnables (training, evaluation, and other scripts) and the user-facing app. |
| `configs/` | Holds configuration files for model training. Includes settings for processing data, defining model type and its hyperparameters |
| `train.py` | Entry point for training. Trains and evaluates (limited, against training and validation splits only) a model from a config file, logs the run to ClearML and saves checkpoints (both locally and to ClearML) |
| `evaluate.py` | Entry point for evaluation. Evaluates a checkpoint (local or from a ClearML task) on the validation or test split and produces metrics, a confusion matrix, training curves and a report |
| `scripts/` | Supporting runnable scripts.|
| `app/` | Defines the Gradio UI that serves predictions from the deployed ONNX model |
| `deployment/` | Holds `model.lock.json`, which sets the ClearML ID of the model to deploy, and the scripts for running the app locally. The downloaded model goes into `deployment/model/`, which is git-ignored |
| `render.yaml` | Render service configuration for the cloud deployment |
| `requirements.txt` / `requirements-deploy.txt` | Full dependencies for development and training / reduced dependencies for running the app (no torch) |
| `pyproject.toml` | Test coverage settings |
| `tests/` | Contains codebase-wide pytests |
| `.github/workflows/` | CI workflows: `tests.yml` runs the test suite, and `deploy-check.yml` checks the deployment install has no torch, fetches the pinned model and checks the app can predict with it |
| `documents/` | Project documentation |
| `outputs/` | Generated locally by training and evaluation (checkpoints, reports, diagrams). Git-ignored |

---

## 3. Components

| Component | Files | Description |
|-----------|-------|-------------|
| Config | `leafapp/config.py`, `configs/` | |
| Data pipeline | `leafapp/data.py`, `leafapp/labels.py` | |
| Models | `leafapp/models/`, `leafapp/model_constructor.py` | |
| Training | `train.py`, `leafapp/optimizer_constructor.py`, `leafapp/early_stopping.py`, `leafapp/checkpoints.py` | |
| Evaluation | `evaluate.py`, `leafapp/eval_functions.py` | |
| Experiment tracking | `leafapp/tracking.py` | |
| Inference | `leafapp/predict.py` | |
| Web app | `app/` | |
| Deployment | `render.yaml`, `deployment/`, `scripts/`, `.github/workflows/` | |

---

## 4. Data flow

**Training:** <!-- config → data → model → train loop → tracking → checkpoint -->

**Evaluation:**

**Prediction:** <!-- upload → preprocessing → model → label + confidence → UI -->

---
