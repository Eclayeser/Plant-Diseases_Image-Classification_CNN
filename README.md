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
python train.py --responsible "<your name>"                                                                                  # uses configs/prototype.yaml and the default run type
python train.py --config configs/prototype_small_data_frac.yaml --responsible "<your name>" --run-type comparison --issue-id SGP-30  # or any other config file and run type; --issue-id is optional
```
- Settings (data fraction, image size, batch size, model channels, learning rate, epochs, seed) are read from the chosen config file.
- Each epoch prints train and validation loss, accuracy and f1 score.
- At the end, the model is saved to `outputs/models/<model type>_<timestamp>_<id>.pt`, e.g. `basiccnn_20261105-184800_a3f9c1.pt`.

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

```bash
python evaluate.py --checkpoint outputs/models/basiccnn_20261105-184800_a3f9c1.pt
```

Replace the filename with your own checkpoint from `outputs/models/`. This evaluates on the validation set by default. To evaluate on the test set instead:

```bash
python evaluate.py --checkpoint outputs/models/basiccnn_20261105-184800_a3f9c1.pt --split test
```

- The evaluation report is saved to `outputs/eval_reports/` and the diagrams to `outputs/diagrams/`.

The default is validation so the test set isn't used by accident while you're still choosing between models.

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

Each task is tagged automatically with `@run-type:<type>`, `@model:<model type>` and `@responsible:<name>` and `@config-name:<config file name without extension>`, plus `@issue:<id>` when `--issue-id` is given. Use these tags to filter the "Tasks List", e.g. filter by `@run-type:production` to see only the official runs.

Every logged task in "Tasks List" has the following tabs: "Exection", "Configuration", "Artifacts", "Info", "Console", "Scalars", "Plots" and "Debug samples".


## Running the UI

The Gradio app in `app/app.py` loads a trained checkpoint and shows the top 5 predicted diseases for an uploaded leaf image.

First train a model (see **Training the model**), then start the app with the path to the checkpoint:

```powershell
# Windows (PowerShell)
$env:CHECKPOINT_PATH="outputs/models/<model type>_<timestamp>_<id>.pt"
python -m app.app
```

```bash
# Mac / Linux
CHECKPOINT_PATH=outputs/models/<model type>_<timestamp>_<id>.pt python -m app.app
```
- Replace `<model type>_<timestamp>_<id>.pt` with the name of your checkpoint in `outputs/models/`, e.g. `basiccnn_20261105-184800_a3f9c1.pt`.
- Open the local link shown in the terminal (usually http://127.0.0.1:7860) and upload a leaf image.
- The image is preprocessed with the same settings as during evaluation (image size and normalisation are read from the checkpoint config).
- If `CHECKPOINT_PATH` is not set, the app still starts but shows an error when an image is uploaded.
- Uploads are limited to 10 MB.

## Running the tests

```bash
python -m pytest tests/
```