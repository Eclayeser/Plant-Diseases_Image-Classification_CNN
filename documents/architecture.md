# Architecture

Explains the project structure and how its components connect.

> **Last updated:** <!-- YYYY-MM-DD --> · **Owner(s):** <!-- names -->

---

## 1. Overview

<!-- 2–4 sentences: what the system does end to end (leaf image in → disease label out),
     who uses it, and the main parts (training pipeline, model, web app, deployment). -->

### High-level diagram

```text
<!-- Replace with a diagram (ASCII, Mermaid, or an image link). Suggested flow:

  Dataset ──► data.py ──► train.py ──► checkpoint ──► ClearML Models
                              │                            │
                              ▼                            ▼
                         evaluate.py              fetch_model_from_clearml.py
                                                           │
                                                     export_onnx.py
                                                           │
                                                      app/app.py  ──► Render
-->
```

---

## 2. Repository layout

| Path | Purpose |
|------|---------|
| `leafapp/` | <!-- core library: data, models, training helpers, inference --> |
| `leafapp/models/` | <!-- model definitions --> |
| `configs/` | <!-- YAML experiment configs --> |
| `train.py` | <!-- --> |
| `evaluate.py` | <!-- --> |
| `compute_norm_stats.py` | <!-- --> |
| `scripts/` | <!-- export / fetch / gallery helper scripts --> |
| `app/` | <!-- web app + example gallery --> |
| `deployment/` | <!-- local run scripts, model lock file --> |
| `tests/` | <!-- --> |
| `.github/workflows/` | <!-- CI --> |
| `documents/` | <!-- experiments, meetings, this file --> |

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

## 8. Limitations & future work

-
-

---

## 9. Glossary

| Term | Meaning |
|------|---------|
| | |
