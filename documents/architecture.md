# Architecture

Explains the project structure and how its components connect.

> **Last updated:** <!-- YYYY-MM-DD --> · **Owner(s):** <!-- names -->

---

## 1. Overview

<!-- 2–4 sentences: what the system does end to end (leaf image in → disease label out),
     who uses it, and the main parts (training pipeline, model, web app, deployment). -->

### High-level diagram

![Architecture diagram](architecture_diagram.png)

---

## 2. Repository layout

| Path | Purpose |
|------|---------|
| `leafapp/` | Serves as a core library for the shared code used by training, evaluation, the scripts and the app: config loading, data loading and splitting, labels, model and optimiser construction, early stopping, checkpoints, metrics, ClearML tracking and prediction |
| `leafapp/models/` | Defines the model architectures: `BasicCNN` (`cnn.py`), the regularised CNN with batch norm, dropout and pooling options (`regularised_cnn.py`) and ResNet18 (`resnet.py`) |
| `configs/` | Holds configuration files for model training. Includes settings for processing data, defining model type and its hyperparameters |
| `train.py` | Entry point for training. Trains and validates a model from a config file, logs the run to ClearML and saves checkpoints |
| `evaluate.py` | Entry point for evaluation. Evaluates a checkpoint (local or from a ClearML task) on the validation or test split and produces metrics, a confusion matrix, training curves and a report |
| `compute_norm_stats.py` | One-off script that computes the per-channel mean and std of the PlantVillage training split, used for the `plant_village` normalisation option |
| `scripts/` | Supporting runnable scripts: exporting a checkpoint to ONNX and uploading it to ClearML (`export_onnx.py`), downloading the pinned deployment model (`fetch_model_from_clearml.py`) and saving the gallery images (`save_example_images.py`) |
| `app/` | Defines the Gradio UI that serves predictions from the deployed ONNX model, plus the example image gallery (`app/gallery/`) |
| `deployment/` | Holds `model.lock.json`, which sets the ClearML ID of the model to deploy, and the scripts for running the app locally. The downloaded model goes into `deployment/model/`, which is git-ignored |
| `render.yaml` | Render service configuration for the cloud deployment |
| `requirements.txt` / `requirements-deploy.txt` | Full dependencies for development and training / reduced dependencies for running the app (no torch) |
| `pyproject.toml` | Test coverage settings (measures `leafapp/` and `app/`) |
| `tests/` | Contains pytests for `leafapp/`, `app/`, the `train.py` and `evaluate.py` entry points and the ONNX export script |
| `.github/workflows/` | CI workflows: `tests.yml` runs the test suite, and `deploy-check.yml` checks the deployment install has no torch, fetches the pinned model and checks the app can predict with it |
| `notebooks/` | Exploratory work |
| `documents/` | Project documentation: experiment write-ups (`experiments/`), meeting summaries (`meetings/`), the AI use statement and this file |
| `outputs/` | Generated locally by training and evaluation (checkpoints, reports, diagrams). Git-ignored |

---

<!--
Shared Python code belongs in `leafapp/`, the interface in `app/`, and configuration files in `configs/`.
The `train.py` and `evaluate.py` files are entry points for training and evaluating the model.
The `notebooks/`, `tests/`, and `documents/` folders hold exploratory work, automated checks, and project documentation.
-->

## 3. Components

<!-- For each component: responsibility, key functions/classes, inputs → outputs,
     and which other components it depends on. Delete any that don't apply. -->

### 3.1 Configuration — `leafapp/config.py`, `configs/*.yaml`

- **Responsibility:**
- **Key fields / schema:**
- **How a config is selected and loaded:**

### 3.2 Data pipeline — `leafapp/data.py`, `leafapp/labels.py`

- **Dataset(s) and source:**
- **Splits and seeding:**
- **Transforms / augmentation / normalisation:**
- **Loader settings (batch size, `num_workers`):**
- **Label mapping:**

### 3.3 Models — `leafapp/models/`, `leafapp/model_constructor.py`

| Model | File | Notes |
|-------|------|-------|
| CNN | `cnn.py` | |
| Regularised CNN | `regularised_cnn.py` | |
| ResNet-18 | `resnet.py` | |

- **How the constructor picks a model from config:**
- **Final / deployed architecture:**

### 3.4 Training — `train.py`, `leafapp/optimizer_constructor.py`, `leafapp/early_stopping.py`, `leafapp/checkpoints.py`

- **Training loop outline:**
- **Optimisers supported:**
- **Early stopping criteria:**
- **Checkpoint format and location:**

### 3.5 Evaluation — `evaluate.py`, `leafapp/eval_functions.py`

- **Metrics:**
- **Outputs (reports, confusion matrices, curves):**

### 3.6 Experiment tracking — `leafapp/tracking.py`

- **Tool (ClearML) and what is logged:**
- **How models are registered / promoted:**

### 3.7 Inference — `leafapp/predict.py`

- **Input preprocessing:**
- **Runtime (PyTorch / ONNX):**
- **Output format:**

### 3.8 Web app — `app/app.py`

- **Framework:**
- **User flow:**
- **Example gallery:**

### 3.9 Deployment — `render.yaml`, `deployment/`, `scripts/`

- **Hosting (Render) and build steps:**
- **How the model is fetched (ClearML → `model.lock.json`):**
- **ONNX export:**
- **Running locally:** see [`deployment/LOCAL_DEPLOYMENT.md`](../deployment/LOCAL_DEPLOYMENT.md)

---

## 4. Data & control flow

### 4.1 Training run

<!-- Step-by-step: config → data → model → train loop → tracking → checkpoint. -->

1.
2.
3.

### 4.2 Prediction request

<!-- Step-by-step: user upload → preprocessing → model → label + confidence → UI. -->

1.
2.
3.

---

## 5. Testing & CI

- **Test strategy (unit / integration, fakes in `leafapp/fakes.py`):**
- **`tests.yml`:**
- **`deploy-check.yml`:**

---

## 6. Key design decisions

<!-- Short records of significant choices. Link to experiment write-ups in documents/experiments/. -->

| Decision | Alternatives considered | Rationale | Reference |
|----------|------------------------|-----------|-----------|
| | | | |

<!--
The prototype and final product will share the same codebase.
Prototype settings will be kept in `configs/prototype.yaml`, and ClearML tags will identify prototype runs.
-->

---

## 7. Dependencies

| Dependency | Used for | File |
|------------|----------|------|
| | | `requirements.txt` |
| | | `requirements-deploy.txt` |

---
