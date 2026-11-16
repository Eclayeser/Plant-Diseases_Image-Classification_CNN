# SGP-44: CNN regularisation comparison

## Setup

This experiment compares the wider CNN baseline with two variants that add either `Dropout2d` (`p=0.2`) or `BatchNorm2d` to each convolutional block. All three runs use channels `[32, 64, 128]`, the same split of a 20% PlantVillage subset, seed 42, 128 × 128 images, batch size 32, PlantVillage normalisation and training augmentation. Each run uses AdamW (learning rate 0.001, weight decay 0.01) for five epochs.

## Results (validation set, epoch 5)

| Variant | Config | Accuracy | Macro F1 | Loss | ClearML run |
| --- | --- | ---: | ---: | ---: | --- |
| Wider baseline | `cnn_wider_normalize_plant_village.yaml` | **84.25%** | **0.794** | **0.543** | [Baseline](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/7e13345fe54843bfa1f129f6fc7bbc5c/output/log) |
| Wider + dropout | `cnn_wider_dropout.yaml` | 81.78% | 0.752 | 0.647 | [Dropout](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/04b32ab0dab04c8e95472b74e90691a4/output/log) |
| Wider + batch normalisation | `cnn_wider_batchnorm.yaml` | 82.14% | 0.763 | 0.623 | [Batch normalisation](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/e0fb872e7c194d168cead84490300eb3/output/log) |

Each ClearML run has a `best_checkpoint` and a `final_checkpoint`. The baseline's lowest validation loss was 0.515 at epoch 4. Both regularised variants reached their lowest validation loss at epoch 5.

## Recommendation and limitations

Keep the wider baseline for the tested setup. It had the highest validation accuracy and macro F1 and the lowest validation loss at epoch 5. Both variants were still improving in validation loss at epoch 5, so a longer run could change the ranking. This comparison used one seed and a subset of the data, and all reported metrics come from the validation split.
