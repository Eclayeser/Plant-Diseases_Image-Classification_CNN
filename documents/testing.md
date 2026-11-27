# Test Strategy

We use different levels of testing to check that individual parts of our project work correctly and that they work together as expected. Automated tests are written using pytest and stored in the `tests/` directory.

## Unit Testing

Unit tests check individual functions and components.

| What is tested | Why |
|---|---|
| Data splitting, augmentation and normalisation | To make sure images are prepared correctly and data splits do not overlap. |
| Model architectures and output shapes | To check that the models produce the expected outputs. |
| Model and optimiser configuration | To make sure the correct settings are used. |
| Early stopping | To check that training stops when there is no improvement. |
| Evaluation metrics and error analysis | To make sure metrics, confusion matrices and misclassified images are reported correctly. |
| Invalid settings and label formatting | To reject invalid model and optimiser settings and display readable class names. |
| Normalisation statistics | To make sure the mean and standard deviation are calculated only from the training images, without augmentation or normalisation. |
| Command-line arguments | To make sure training and evaluation accept valid arguments and reject invalid ones. |
| Softmax and top-k predictions | To make sure prediction probabilities are calculated and sorted correctly. |

## Integration Testing

Integration tests check how different parts of the project work together.

| What is tested | Why |
|---|---|
| Training loop and weight updates | To make sure model weights are updated correctly during training. |
| Saving and loading checkpoints | To make sure trained models can be restored correctly. |
| Image preprocessing and prediction | To check that images are processed correctly before prediction. |
| Gradio interface and example gallery | To make sure images can be passed to the prediction function and results are displayed. |
| Examples of the top prediction | To make sure the app shows 3 images of the top class, or none when no leaf is found. |
| ClearML tracking | To check metric logging, checkpoint handling and task management. |
| ONNX export preparation | To make sure the model uses the correct normalisation when exported to ONNX. |
| App warnings and errors | To make sure users see a clear message when no leaf is found or a file cannot be read. |
| Training failure handling | To make sure available checkpoints are preserved and the ClearML task is marked as failed when training fails. |

The pytest tests do not need the full dataset, a GPU or a connection to ClearML. They mainly use small test images, fake models and fake ClearML objects. We use pytest's `monkeypatch` to replace slow or external parts and `tmp_path` to create temporary files. This keeps the tests fast.

## End-to-End Testing

`tests/test_end_to_end.py` tests the whole pipeline by training a small model for one epoch, evaluating it, exporting it to ONNX and predicting one image using `LeafClassifier`.

It checks that each step works with the output of the previous step and that the top-5 probabilities are between 0 and 1. Fake data and mocked ClearML objects are used to keep the test fast and avoid external connections.

## Continuous Integration

Two GitHub Actions workflows run on every push:

| Workflow | What it does | Why |
|---|---|---|
| `tests.yml` | Runs all pytest tests with coverage on Python 3.12 (CPU only). | To catch broken code before it is merged. |
| `deploy-check.yml` | Installs only the deployment requirements, checks that torch is not installed, downloads the real model from ClearML and makes one prediction with the app. | To check that the app works with the real model and deployment dependencies. |

## System Testing

The complete application should also be tested manually by uploading leaf images, selecting example images and checking the prediction results, the examples of the top prediction and error messages.

This helps us check the user experience, which is not fully covered by automated tests.

## Model Evaluation

Model performance is evaluated separately using accuracy, precision, recall, F1-score and confusion matrices.

The validation set is used to compare models, while the test set is used for final evaluation.

## Running the Tests

Run all pytest tests:

`python -m pytest tests/`

Run tests with code coverage:

`python -m pytest tests/ --cov --cov-report=term-missing`

As of 25 November 2026, all 160 pytest tests passed, with 92% code coverage for `leafapp/` and `app/`. `train.py`, `evaluate.py` and some scripts in `scripts/` have tests but are not included in the coverage measurement.

New features and bug fixes should include relevant tests to prevent existing functionality from breaking.