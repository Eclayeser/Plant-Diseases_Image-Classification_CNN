# Plant-Diseases_Image-Classification_CNN

Our aim is to classify plant diseases from leaf images using a PyTorch CNN and a Gradio interface.
The repository currently contains a project skeleton, with purpose comments to guide future implementation.

Shared Python code belongs in `leafapp/`, the interface in `app/`, and configuration files in `configs/`.
The `train.py` and `evaluate.py` files are entry points for training and evaluating the model.
The `notebooks/`, `tests/`, and `documents/` folders hold exploratory work, automated checks, and project documentation.

The prototype and final product will share the same codebase.
Prototype settings will be kept in `configs/prototype.yaml`, and ClearML tags will identify prototype runs.
A `v0.1-prototype` Git tag will mark the version delivered as the prototype.

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

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   |   Mac/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```
On Windows with an NVIDIA GPU, the default `torch` from pip is CPU-only. Install the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) using the same versions as `requirements.txt`. Linux installs CUDA automatically.

## Training the model

### Run training
```bash
python train.py
```
- Settings (data fraction, image size, batch size, model channels, learning rate, epochs, seed) are read from `configs/prototype.yaml`.
- Each epoch prints train and validation loss and accuracy.
- At the end, the model is saved to `outputs/basic_cnn_<timestamp>.pt`. The `outputs/` folder is ignored by git.

### Load a trained model
```python
from leafapp.checkpoints import load_checkpoint

model, class_names, config = load_checkpoint("outputs/basic_cnn_<timestamp>.pt")
```
The checkpoint stores the weights, the full config and the class names, so the model is rebuilt with the same settings and the same data split (`config["seed"]`, `config["data"]["fraction"]`) can be recreated for evaluation. The model is returned in eval mode, on the CPU.

## Running the tests

```bash
python -m pytest tests/
```