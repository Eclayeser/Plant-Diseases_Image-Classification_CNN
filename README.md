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
python train.py                                                     # uses configs/prototype.yaml
python train.py --config configs/prototype_small_data_frac.yaml     # or any other config file
```
- Settings (data fraction, image size, batch size, model channels, learning rate, epochs, seed) are read from the chosen config file.
- Each epoch prints train and validation loss, accuracy and f1 score.
- ClearML records the full config and per-epoch train/validation metrics, and stores both the final and lowest-validation-loss checkpoints as task artifacts. Tasks share the `Plant-Disease-Image-Classification` project and are tagged with their config name for comparison.
- Configure ClearML credentials and a server with `clearml-init` before training; see the [ClearML setup guide](https://clear.ml/docs/latest/docs/clearml_sdk/clearml_sdk_setup/).
- At the end, the model is saved to `outputs/models/<model type>_<timestamp>_<id>.pt`, e.g. `basiccnn_20261105-184800_a3f9c1.pt`.

### Load a trained model
```python
from leafapp.checkpoints import load_checkpoint

model, class_names, history_metrics, config = load_checkpoint("outputs/models/<model type>_<timestamp>_<id>.pt")
```
The checkpoint stores the weights, metrics recorded during the training, the full config and the class names, so the model is rebuilt with the same settings and the same data split (`config["seed"]`, `config["data"]["fraction"]`) can be recreated for evaluation. The model is returned in eval mode, on the CPU.

### Evaluate a trained model
```bash
python evaluate.py {RELATIVE_PATH_TO_CHECKPOINT} {DATA_SET_TYPE}
```
- `{RELATIVE_PATH_TO_CHECKPOINT}` is path to the model you want to evaluate, e.g. `outputs\models\example_cnn_20261104-215131_f55919.pt`
- `{DATA_SET_TYPE}` determines which data set you want to use for evaluation. Only accepts either `"validation"` or `"test"`
- At the end of the run, the evaluation report is produced and saved to `outputs/eval_reports/<data_type>_<timestamp>_report.txt`, and any diagrams/graphs are saved to `outputs/diagrams/<data_type>_<timestamp>_<diagram_type>.png`.

*Important Note:* Remember! Do not use test data set for any sort of model tunning. The test data set exists solely for final evaluation and designed to prevent biased model adjustments.

## Running the tests

```bash
python -m pytest tests/
```