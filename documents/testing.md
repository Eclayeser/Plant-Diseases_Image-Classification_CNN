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

