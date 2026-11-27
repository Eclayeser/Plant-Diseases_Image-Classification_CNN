# Plant-Diseases_Image-Classification_CNN

Our aim is to classify plant diseases from leaf images using a PyTorch CNN and a Gradio interface.
The repository currently contains a project skeleton, with purpose comments to guide future implementation.

Shared Python code belongs in `leafapp/`, the interface in `app/`, and configuration files in `configs/`.
The `train.py` and `evaluate.py` files are entry points for training and evaluating the model.
The `notebooks/`, `tests/`, and `documents/` folders hold exploratory work, automated checks, and project documentation.

The prototype and final product will share the same codebase.
Prototype settings will be kept in `configs/prototype.yaml`, and ClearML tags will identify prototype runs.

## Tools

Despite the **Technology Stack** used directly for product development, our team uses a range of different software tools to accelerate the speed of our work and improve the code quality.

| Tool | Use | Why this tool | How to View |
|---|---|---|---|
| **Linear** | Issue tracking, development cycles management, main project's kanban | Easy to learn and intuitive to use; commonly used across businesses; some team members have previous experience using this tool; free tier available; supports integration with GitHub | Join workspace via a link (cannot be shared publicly since no read-only access available; request one of the team members for access) |
| **Slack** | Asynchronous team communication (via different dedicated channels) | Free to use; supports integration with Linear; commonly used across businesses; some team memebrs have previous experience using this tool; highly customisable which allows for clear information exchange | N/A |
| **Discord** | Real-time communication, online meetings | Free to use; all team memebrs have previous experience using this tool; easy and quick to use (including comfortable screen share) | N/A |
| **Coderabbit AI**| AI-driven code reviews for PRs; integrated as a required check before PR can be merged | Catches bugs early and allows developers to debug quicker without waiting for others to review; provides useful PR summaries; free to use and easy to integrate; common practice across most companies | N/A |
| **LLMs: Claude, Gemini, Chat-GPT**| Code generation/review, technical brainstorming | Useful assistants across multiple tech stacks; speeds up overall development (crucial for time-constrained project); widely used across most developers | N/A |


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

The ClearML WebUI records experiments in the form of tasks.
Each task:
- Logs training and validation metrics at every epoch
- Saves the lowest-validation-loss checkpoint alongside the final checkpoint
- Uploads both as ClearML artifacts

### Setting up

A ClearML server is assumed to be accessible and `clearml` should already be installed in the virtual environment from `requirements.txt` during initial setup.

#### ClearML initialisation
```
clearml-init
```
Paste the API credentials generated in the ClearML WebUI: Profile Settings > Workspace > Create new credentials

If the initialisation command above has already been run previously, edit the ClearML configuration file that already exists.

##### To edit the ClearML configuration file:
1. Open the configuration file:
    - Linux - `~/clearml.conf`
    - Mac - `$HOME/clearml.conf`
    - Windows - `\User\<username>\clearml.conf`

2. Modify the `access_key` and `secret_key` fields respectively with the generated API credentials.
3. Save configuration file.

### For collaborators

A shared ClearML account will be used with identical API credentials.

Tasks run inside Pumpkin Jr's workspace within the project `Plant-Disease-Image-Classification`

### Interface

After running the training a new task should appear to the left inside the `Plant-Disease-Image-Classification` project (a new project with this name should be created automatically).

The default configuration will run 5 iterations, saves a best and final checkpoint, and exits after successfully completing a task.

Each task is tagged automatically with `@run-type:<type>`, `@model:<model type>` and `@responsible:<name>`, plus `@issue:<id>` when `--issue-id` is given. Training tasks also get `@config-name:<config file name without extension>`, and evaluation tasks get `@task-type:evaluate.py` and `@split:<split>`. Use these tags to filter the "Tasks List", e.g. filter by `@run-type:production` to see only the official runs.

Every logged task in "Tasks List" has the following tabs: "Exection", "Configuration", "Artifacts", "Info", "Console", "Scalars", "Plots" and "Debug samples".

## Running the tests

```bash
python -m pytest tests/
```

To also see which lines of `leafapp` and `app` are covered by the tests:

```bash
python -m pytest tests/ --cov --cov-report=term-missing
```
The coverage settings are in `pyproject.toml`.

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

Live at: *(URL added once the Render service is created)*

- A merge to `main` redeploys the app once the `tests` and `deploy-check`
  workflows pass, but only if it changes files outside the build filter in
  `render.yaml`. Merges that only change Markdown files, `documents/`,
  `notebooks/`, `configs/` or `tests/` don't redeploy.
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

### Example images (SGP-64, SGP-72)

The app shows a gallery of example leaf images from `app/gallery/`, so users
can click one to get a prediction without uploading their own. The script
saves one image per class from the PlantVillage test split (with the same
20% subset and seed as our experiments). To recreate them, or after changing
the classes in the script:

```bash
python -m scripts.save_example_images
```

Any `.jpg` added to `app/gallery/` is shown in the gallery automatically.

The same script also saves 3 train-split images per class to
`app/class_examples/<class name>/`. After a prediction, the app shows the ones
for the top class, or none when no leaf is found.

