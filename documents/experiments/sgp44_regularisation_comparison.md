# SGP-44: CNN regularisation comparison

## Initial five-epoch setup

This experiment compares the wider CNN baseline with two variants that add either `Dropout2d` (`p=0.2`) or `BatchNorm2d` to each convolutional block. All three runs use channels `[32, 64, 128]`, the same split of a 20% PlantVillage subset, seed 42, 128 × 128 images, batch size 32, PlantVillage normalisation and training augmentation. Each run uses AdamW (learning rate 0.001, weight decay 0.01) for five epochs.

## Initial results (validation set, epoch 5)

| Variant | Config | Accuracy | Macro F1 | Loss | ClearML run |
| --- | --- | ---: | ---: | ---: | --- |
| Wider baseline | `cnn_wider_normalize_plant_village.yaml` | **84.25%** | **0.794** | **0.543** | [Baseline](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/7e13345fe54843bfa1f129f6fc7bbc5c/output/log) |
| Wider + dropout | `cnn_wider_dropout.yaml` | 81.78% | 0.752 | 0.647 | [Dropout](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/04b32ab0dab04c8e95472b74e90691a4/output/log) |
| Wider + batch normalisation | `cnn_wider_batchnorm.yaml` | 82.14% | 0.763 | 0.623 | [Batch normalisation](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/e0fb872e7c194d168cead84490300eb3/output/log) |

Each ClearML run has a `best_checkpoint` and a `final_checkpoint`. The baseline's lowest validation loss was 0.515 at epoch 4. Both regularised variants reached their lowest validation loss at epoch 5.

## Early-stopping comparison

Following review feedback, all three variants were trained with seeds 42, 43 and 44, a maximum of 30 epochs and early-stopping patience of 3. Any decrease in validation loss counts as improvement (min_delta 0.0). All other model, preprocessing and optimiser settings match the initial experiment, including the fixed PlantVillage normalisation statistics.

Changing the seed changes the sampled subset, data split and random training behaviour. Within each seed, the three variants use the same subset and split. This gives nine runs in total.

The source configs are in configs/ and default to seed 42.

- Baseline uses `cnn_wider_normalize_plant_village_early_stopping.yaml`.
- Dropout uses `cnn_wider_dropout_early_stopping.yaml`.
- Batch normalisation uses `cnn_wider_batchnorm_early_stopping.yaml`.

For each seed, a copy of each config was made under outputs/sgp44_early_stopping/configs/, setting seed to 42, 43 or 44 and checkpoint_dir to outputs/sgp44_early_stopping/models. Each was run with `python train.py --config <config-path> --responsible "Javier Fimia" --run-type comparison --issue-id SGP-44`. The full run settings and checkpoints are recorded in ClearML.

## Results at the best validation-loss epoch

Each row uses the epoch with the lowest validation loss for that run. Accuracy and macro F1 come from that same epoch. Every run stopped after three epochs without a new lowest validation loss.

| Variant | Seed | Best epoch | Epochs trained | Val accuracy | Macro F1 | Val loss | ClearML run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 42 | 11 | 14 | 89.72% | 0.8692 | 0.3841 | [Run](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/2382ecb462fd4dffa5541cd42b639c06/output/log) |
| Baseline | 43 | 9 | 12 | 88.51% | 0.8649 | 0.3823 | [Run](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/db6d96d969ba4488b858498d03110728/output/log) |
| Baseline | 44 | 6 | 9 | 87.25% | 0.8385 | 0.4514 | [Run](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/845e3278293045c29851dd16970ceb6a/output/log) |
| Dropout | 42 | 14 | 17 | 88.27% | 0.8511 | 0.3872 | [Run](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/255b25d0bff640ceab8b8006f0aaa519/output/log) |
| Dropout | 43 | 14 | 17 | 89.96% | 0.8672 | 0.3467 | [Run](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/221f61f17629429196d18b7f5cd458d7/output/log) |
| Dropout | 44 | 17 | 20 | 89.96% | 0.8653 | 0.3536 | [Run](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/9503e09314114afdb1cf8b11780474d2/output/log) |
| Batch normalisation | 42 | 7 | 10 | 85.93% | 0.8108 | 0.4804 | [Run](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/f17dfe5ec2834dabbf5fc3832d605e1c/output/log) |
| Batch normalisation | 43 | 11 | 14 | 89.78% | 0.8524 | 0.3559 | [Run](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/4cc46b0634d34a7daa649c58980ade3e/output/log) |
| Batch normalisation | 44 | 13 | 16 | 90.38% | 0.8532 | 0.3633 | [Run](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/40b59a581de6488fb42afcd5242b6949/output/log) |

## Mean results across three seeds

Values are mean +/- sample standard deviation across the three runs, calculated before rounding. Accuracy is shown as a percentage, with its standard deviation in percentage points.

| Variant | Val accuracy (%) | Macro F1 | Val loss |
| --- | ---: | ---: | ---: |
| Baseline | 88.49 +/- 1.23 | 0.8575 +/- 0.0166 | 0.4059 +/- 0.0394 |
| Dropout | 89.40 +/- 0.97 | 0.8612 +/- 0.0088 | 0.3625 +/- 0.0217 |
| Batch normalisation | 88.70 +/- 2.41 | 0.8388 +/- 0.0242 | 0.3999 +/- 0.0698 |

## Observations and recommendation

Dropout had the highest mean validation accuracy and macro F1 and the lowest mean validation loss. Its accuracy was 0.90 percentage points above the baseline on average. It improved all three metrics for seeds 43 and 44, while the baseline led on seed 42. Batch normalisation had slightly higher mean accuracy than the baseline but lower mean macro F1.

The longer runs changed the initial ranking. Dropout reached its lowest validation loss at epochs 14, 14 and 17, compared with epochs 11, 9 and 6 for the baseline. Dropout trained for 18 epochs on average, compared with 11.7 for the baseline, so its improvement required more training epochs. The linked ClearML runs include the complete training and validation curves.

Based on these runs, use dropout (p=0.2) as the preferred regularisation setting for this wider CNN setup. This replaces the initial recommendation to keep the baseline after five epochs. The average advantage is modest and varies by seed.

## Limitations

- Three seeds and a 20% subset give limited evidence for how the models will perform on the full dataset.
- Variation across seeds includes changes in the subset and split as well as model initialisation and training randomness.
- The validation set selected each checkpoint and informed the recommendation. Test-set performance has yet to be established.
- Only dropout p=0.2 and the current batch-normalisation setup were compared.
