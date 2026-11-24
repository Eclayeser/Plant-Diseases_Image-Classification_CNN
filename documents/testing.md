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
| Evaluation metrics | To make sure model performance is calculated correctly. |
| Input validation and label formatting | To handle invalid inputs and display readable class names. |

## Integration Testing

Integration tests check how different parts of the project work together.

| What is tested | Why |
|---|---|
| Training loop and weight updates | To make sure model weights are updated correctly during training. |
| Saving and loading checkpoints | To make sure trained models can be restored correctly. |
| Image preprocessing and prediction | To check that images are processed correctly before prediction. |
| Gradio interface and example gallery | To make sure images can be passed to the prediction function and results are displayed. |
| ClearML tracking | To check metric logging and checkpoint handling. |
| ONNX export preparation | To make sure the model uses the correct normalisation when exported to ONNX. |

The pytest tests do not need the full dataset, a GPU or a connection to ClearML. They mainly use small test images, fake models and fake ClearML objects. We use pytest's monkeypatch to replace slow or external parts and tmp_path to create temporary files. This keeps the tests fast.

## Continuous Integration

Two GitHub Actions workflows run on every push:

| Workflow | What it does | Why |
|---|---|---|
| `tests.yml` | Runs all pytest tests with coverage on Python 3.12 (CPU only). | To catch broken code before it is merged. |
| `deploy-check.yml` | Installs only the deployment requirements, checks that torch is not installed, downloads the real model from ClearML and makes one prediction with the app. | To check that the app works with the real model and deployment dependencies. |

## System Testing

The complete application should also be tested manually by uploading leaf images, selecting example images and checking the prediction results and error messages.

This helps us check the user experience, which is not fully covered by automated tests.

## Model Evaluation

Model performance is evaluated separately using accuracy, precision, recall, F1-score and confusion matrices.

The validation set is used to compare models, while the test set is used for final evaluation.

## Running the Tests

Run all pytest tests:

`python -m pytest tests/`

Run tests with code coverage:

`python -m pytest tests/ --cov --cov-report=term-missing`

As of 24 November 2026, all 159 pytest tests passed, with 83% code coverage for the measured modules.

New features and bug fixes should include relevant tests to prevent existing functionality from breaking.