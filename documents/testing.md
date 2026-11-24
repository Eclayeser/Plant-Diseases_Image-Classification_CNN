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

