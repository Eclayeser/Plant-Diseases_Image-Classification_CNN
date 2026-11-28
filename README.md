# Plant Disease Classification

A web app that identifies plant diseases from a photo of a leaf. It shows the five most likely of 39 classes (26 diseases, 12 healthy leaf types and photos without a leaf, across 14 crops), using a CNN trained from scratch in PyTorch and a ResNet18 transfer-learning model, served with Gradio.

## Team

**Team Pumpkin Jr**: 7 members (5CCSAMLF Small Group Project, King's College London)

| Name | GitHub |
|---|---|
| Mark Tarnavskyi | [Eclayeser](https://github.com/Eclayeser) |
| Pranay Singh | [Amiri05](https://github.com/Amiri05) |
| Illia Putintsev | [illiaputintsev](https://github.com/illiaputintsev) |
| Javier Fimia | [JaviF96](https://github.com/JaviF96) |
| Reon Carroll-Ito | [ReoMaxx](https://github.com/ReoMaxx) |
| Gul Nil Kayalioglu | [nilkayalioglu](https://github.com/nilkayalioglu) |
| Mario Westphalen Cavicchioli | [MarioCavicchioli](https://github.com/MarioCavicchioli) |

## Links

- **Repository:** https://github.com/Eclayeser/Plant-Diseases_Image-Classification_CNN
- **Live app:** https://plant-disease-classifier-4r08.onrender.com (free tier: the first visit after 15 idle minutes takes about a minute)

### Final models

| Model | ClearML model ID | Training task | Write-up |
|---|---|---|---|
| Wider+Deeper CNN (seed 45) | `e10c56f879504f13837d47965be973f3` | `f7139668aca54192892ad23dbe6c5b87` | [SGP-81](documents/experiments/SGP-81/sgp81_final_production_runs.md) |
| Wider CNN (seed 44) | `3fd958c7abcf45af8a420609d8862d94` | `6c0a69e2d95f4a839db41b10a401ac7b` | [SGP-81](documents/experiments/SGP-81/sgp81_final_production_runs.md) |
| ResNet18, transfer learning (seed 44) | `b42092f277924c0a83afebe689dfdb5f` | `23edd3eb3f8e436aacbeeef0bbc1775c` | [SGP-69](documents/experiments/SGP-69/sgp69_resnet18_full_data.md) |

The model used by the app is pinned in [`deployment/model.lock.json`](deployment/model.lock.json).

## Documentation

| Document | What it contains |
|---|---|
| [`documents/architecture.md`](documents/architecture.md) | Architecture diagram, repository layout and how the components connect |
| [`documents/testing.md`](documents/testing.md) | Test strategy: what is tested at which level and why |
| [`documents/experiments/`](documents/experiments/) | One write-up per experiment, named by Linear issue, with results and ClearML links |
| [`documents/meetings/`](documents/meetings/) | Summaries of the team meetings |
| [`deployment/LOCAL_DEPLOYMENT.md`](deployment/LOCAL_DEPLOYMENT.md) | Detailed guide to running the app locally |
| [`app/about_models.md`](app/about_models.md) | Intended use and limitations, shown in the app's "Read more" panel |
| [`AI_USE.md`](AI_USE.md) | AI use declared per file and change |

## Codebase

| Path | Purpose |
|---|---|
| `train.py`, `evaluate.py` | Entry points for training a model from a config and evaluating a checkpoint |
| `leafapp/` | Shared library: data pipeline, models, training helpers, evaluation, ClearML tracking and ONNX inference |
| `configs/` | YAML configs for every training run (data, model and hyperparameters) |
| `app/` | Gradio web app (single image and batch prediction) |
| `scripts/` | ONNX export, model download from ClearML, example images |
| `deployment/`, `render.yaml` | Model pinned for deployment, local run scripts and the Render configuration |
| `tests/`, `.github/workflows/` | Pytest suite and CI (`tests.yml`, `deploy-check.yml`) |

See [`documents/architecture.md`](documents/architecture.md) for how these parts connect.

## Tools

### Technology stack

| Tool | What we use it for | Why |
|---|---|---|
| **PyTorch, torchvision** | Building and training the CNNs and ResNet18; image transforms and augmentation | Standard deep-learning library; pretrained ResNet18 weights available in torchvision |
| **Hugging Face Datasets** | Loading PlantVillage ([`DScomp380/plant_village`](https://huggingface.co/datasets/DScomp380/plant_village)) | Downloads and caches the dataset, so it never has to be stored in the repository |
| **ClearML** | Experiment tracking: metrics, plots, checkpoints and the published ONNX models | Every run is recorded with its config and tags, so results can be compared and reproduced; shared workspace for the whole team |
| **scikit-learn, matplotlib, seaborn** | Evaluation metrics, confusion matrices and plots | Well-tested metric implementations and standard plotting |
| **ONNX, ONNX Runtime** | Exporting trained models and running them in the app | The app runs without torch, which keeps the deployment small enough for Render's free tier |
| **Gradio** | The web interface | Builds an interactive ML demo in Python with little code |
| **Render** | Hosting the live app | Free tier, deploys automatically from `main` |
| **GitHub Actions** | CI: runs the tests and the deployment check on every push | Catches broken code before it is merged |
| **pytest, pytest-cov** | Unit, integration and end-to-end tests with coverage | See [`documents/testing.md`](documents/testing.md) |

### Collaboration tools

| Tool | What we use it for | Why | How to view |
|---|---|---|---|
| **GitHub** | Code, pull requests and reviews; every PR needs two approvals (CODEOWNERS) | Keeps `main` and `dev` stable and every change reviewed | This repository |
| **Linear** | Issue tracking, development cycles and the project kanban; synced with GitHub issues | Easy to use, free tier, integrates with GitHub and Slack | Invite link only (no public read-only access); ask a team member |
| **Slack** | Asynchronous team communication in dedicated channels | Free, integrates with Linear, keeps discussions organised | N/A |
| **Discord** | Online meetings and screen sharing | Free and familiar to the whole team | N/A |
| **CodeRabbit** | AI code review on every PR | Catches bugs early and summarises PRs before human review | Comments on each PR |
| **LLMs (Claude, Gemini, ChatGPT)** | Code suggestions, reviews and brainstorming | Speeds up development; all use is declared in [`AI_USE.md`](AI_USE.md) | N/A |

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```bash
# Mac / Linux
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Training the model

### Run training
```bash
python train.py --responsible "<your name>"                                                                                  # uses configs/prototype.yaml, the default run type and a checkpoint interval of 5 epochs
python train.py --config configs/prototype_small_data_frac.yaml --responsible "<your name>" --run-type comparison --issue-id SGP-30  # or any other config file and run type; --issue-id is optional
python train.py --responsible "<your name>" --save-interval 1                                                                 # save a snapshot every epoch
```
- Settings (data fraction, image size, batch size, model channels, learning rate, epochs, seed) are read from the chosen config file.
- `normalize` under `data:` sets how image colours are normalised: `false` (default), `imagenet` (ImageNet's mean and std, needed for ResNet18; older configs write this as `true`), or `plant_village` (our training split's own mean and std, computed by `compute_norm_stats.py`).
- `patience` and `min_delta` under `train:` turn on early stopping: training stops once the validation loss fails to beat its best by more than `min_delta` (default `0.0`) for `patience` epochs in a row, so `epochs` becomes the maximum. Leave `patience` out to train all epochs. The stop epoch is printed and logged to ClearML as `stopped_epoch`, and the best checkpoint is still the epoch with the lowest validation loss. Example: `configs/cnn_wider_normalize_plant_village_early_stopping.yaml`.
- Each epoch prints train and validation loss, accuracy and f1 score.
- Each epoch's validation confusion matrix is reported to the ClearML task's Plots tab, with actual labels on rows and predicted labels on columns.
- At the end, the model is saved to `outputs/models/<model type>_<timestamp>_<id>.pt`, e.g. `basiccnn_20261105-184800_a3f9c1.pt`.
- Set `augment: false` under `data:` to turn off training augmentation (random flips, rotations and colour changes). It is on by default.
- `optimizer` under `train:` chooses the optimiser: `AdamW` (default) or `Adam`. See `documents/experiments/SGP-27/` for why AdamW is the default.
- Same-seed GPU runs are reproducible in our tests: `train.py` sets `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`, and two runs with the same config and `seed` gave identical weights and metrics (checked on `BasicCNN` and `ResNet18`, RTX 4070, PyTorch 2.14.1 with CUDA 13.2). Results can still differ across GPUs, PyTorch/CUDA versions, or between CPU and GPU. This can slow GPU training slightly.

### Understanding --save-interval

`--save-interval` is optional (default: 5). It configures the frequency of saving checkpoints during training.
`--save-interval 1` saves one after every epoch. The best-validation and final checkpoints are still saved as before.

### Understanding --run-type

`--run-type` is optional (default: `debugging`). It describes the purpose of the run and is recorded as the `@run-type:<type>` tag on the ClearML task.

| Run type | When to use it |
|---|---|
| `debugging` | Making the code or pipeline work. The metrics don't matter, only that nothing crashes. |
| `exploring` | Trying something new (a model, loss or augmentation) and looking at the metrics for a rough idea of whether it works. |
| `comparison` | Measuring performance in a controlled way: tuning hyperparameters, or comparing models with the same seed, data fraction and epochs. |
| `production` | The final run with the chosen model and settings. |

### Load a trained model
```python
from leafapp.checkpoints import load_checkpoint

model, class_names, history_metrics, config = load_checkpoint("outputs/models/<model type>_<timestamp>_<id>.pt")
```
The checkpoint stores the weights, metrics recorded during the training, the full config and the class names, so the model is rebuilt with the same settings and the same data split (`config["seed"]`, `config["data"]["fraction"]`) can be recreated for evaluation. The model is returned in eval mode, on the CPU.

## Evaluate a trained model

### From a local checkpoint
```bash
python evaluate.py --checkpoint outputs/models/basiccnn_20261105-184800_a3f9c1.pt --responsible "<your name>"
python evaluate.py --checkpoint outputs/models/basiccnn_20261105-184800_a3f9c1.pt --responsible "<your name>" --split test --run-type comparison --issue-id SGP-33
```
Replace the filename with your own checkpoint from `outputs/models/`. `--responsible`, `--run-type` and `--issue-id` work the same as for training (see [Understanding --run-type](#understanding---run-type)).

### From a ClearML training task
```bash
python evaluate.py --task-id <training task ID> --responsible "<your name>"
python evaluate.py --task-id <training task ID> --artifact final_checkpoint --responsible "<your name>"
```
- The checkpoint is downloaded from the training task's artifacts. Copy the task ID from the task's info panel in the ClearML WebUI.
- `--artifact` chooses which checkpoint to download: `best_checkpoint` (default, lowest validation loss) or `final_checkpoint`. It only works together with `--task-id`.
- Give either `--checkpoint` or `--task-id`, not both.
- Needs ClearML online, so it won't work in offline mode.

### What it produces
- Each evaluation is its own ClearML task, tagged `@task-type:evaluate.py` and `@split:<split>`. It logs the final metrics (Scalars → Summary), the confusion matrix and training curve (Debug Samples), and the report (Artifacts).
- The report and diagrams are also saved locally to `outputs/eval_reports/` and `outputs/diagrams/`.
- `--split` is `validation` by default, so the test set isn't used by accident while you're still choosing between models.

## Running ClearML

Every training and evaluation run is recorded as a task in ClearML: metrics for each epoch, the best and final checkpoints, plots and reports. The team shares one account; tasks are in Pumpkin Jr's workspace, in the `Plant-Disease-Image-Classification` project, which is created automatically on the first run.

### Setting up

`clearml` is installed with `requirements.txt`. Run:

```bash
clearml-init
```

and paste the API credentials from the ClearML WebUI (Profile Settings > Workspace > Create new credentials). If you have run it before, edit `access_key` and `secret_key` in `~/clearml.conf` (Windows: `C:\Users\<username>\clearml.conf`) instead.

### Tags

Each task is tagged with `@run-type:<type>`, `@model:<model type>` and `@responsible:<name>`, plus `@issue:<id>` when `--issue-id` is given. Training tasks also get `@config-name:<config name>`, and evaluation tasks get `@task-type:evaluate.py` and `@split:<split>`. Filter the task list by these tags, e.g. `@run-type:production` for the final runs.


## Running the tests

```bash
python -m pytest tests/
python -m pytest tests/ --cov --cov-report=term-missing   # also shows line coverage of leafapp/ and app/
```

See [`documents/testing.md`](documents/testing.md) for what is tested at each level and why.

## Deployment

There are two ways to run the app. Both serve `app/app.py` with the same ONNX
model, which is pinned in `deployment/model.lock.json` and downloaded from
ClearML. Both need ClearML credentials (see [Running ClearML](#running-clearml)).

### Option 1: Local (SGP-5)

Runs the app on your machine at http://localhost:8080.

```powershell
# Windows (PowerShell)
.\deployment\setup_and_run.ps1   # first run: creates .venv and installs requirements-deploy.txt
.\deployment\run_local.ps1       # later runs
```

```bash
# Mac / Linux
bash deployment/setup_and_run.sh
bash deployment/run_local.sh
```

Each run downloads the pinned model, then starts the app. For more detail, see
the [Local Deployment Guide](deployment/LOCAL_DEPLOYMENT.md).

### Option 2: Cloud on Render (SGP-49)

Live at: https://plant-disease-classifier-4r08.onrender.com

- A merge to `main` redeploys the app once the `tests` and `deploy-check`
  workflows pass, but only if it changes files outside the build filter in
  `render.yaml`. Merges that only change Markdown files, `documents/`,
  `configs/` or `tests/` don't redeploy.
- Render installs `requirements-deploy.txt` (no torch), downloads the pinned
  model and runs `python -m app.app`. The configuration is in `render.yaml`.
- If a build fails, the previous version stays live.
- On the free tier, the app sleeps after 15 minutes without visitors, and the
  first visit after that takes about a minute.

### Changing the deployed model

1. Export a training task's `best_checkpoint` and try it locally:
   ```bash
   python -m scripts.export_onnx --task-id <training task ID>
   python -m app.app
   ```
2. Upload it to the ClearML Models tab.
   ```bash
   python -m scripts.export_onnx --task-id <training task ID> --upload --responsible "<your name>"
   ```
3. Put the printed model ID in `deployment/model.lock.json`, then open a PR and
   merge it to `main`.

To roll back, set the previous model ID (from the git history of
`model.lock.json`) and merge to `main`.

### Example images (SGP-64)

The app shows a gallery of example leaf images from `app/gallery/`, so users
can click one to get a prediction without uploading their own. The script
saves one image per class from the PlantVillage test split (with the same
20% subset and seed as our experiments). To recreate them, or after changing
the classes in the script:

```bash
python -m scripts.save_example_images
```

Any `.jpg` added to `app/gallery/` is shown in the gallery automatically.

