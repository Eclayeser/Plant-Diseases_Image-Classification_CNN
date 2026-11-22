# SGP-52: Dropout on the Full Dataset

## Setup
Compared the wider CNN with the same model but data augmentation turned off.
Shared settings:
- `data`: `image_size: 128`, full data (`fraction: 1.0`), `batch_size: 32`, `normalize: plant_village`, `num_workers: 6`
- `model`: `type: RegularisedCNN`, `channels: [32, 64, 128]`, `regularisation: dropout`, `dropout_p: 0.2`
- `train`: AdamW (`lr: 0.001`, `weight_decay 0.01`), early stopping (`patience: 3`, max 30 epochs `epochs: 30`)

Configs: `configs/cnn_wider_dropout_plant_village_full_dataset_early_stopping_seed{42,43,44}_augmentation_on_num_workers6` and `configs/cnn_wider_dropout_plant_village_full_dataset_early_stopping_seed{42,43,44}_augmentation_off_num_workers6`

## Results (best validation-loss epoch)
| Seed | Augmentation | Best epoch | Stopped at | Val accuracy | Val F1 | Val loss | ClearML |
|---|---:|---:|---:|---:|---:|---:|---|
| 42 | On | 16 | 19 | 0.955 | 0.939 | 0.143 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/experiments/e93357e6d70042c0b5bd007226a80ec5/output/execution) |
| 42 | Off | 7 | 10 | 0.935 | 0.909 | 0.243 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/experiments/c4bce04646b74d799901680243843528/output/execution) |
| 43 | On | 19 | 22 | 0.955 | 0.939 | 0.160 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/experiments/ced781cf184044738a053ca63830ffce/output/execution) |
| 43 | Off | 4 | 7 | 0.920 | 0.895 | 0.278 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/experiments/88a484ea838c4816bc6ff7d1ed2e3fd4/output/execution) |
| 44 | On | 12 | 15 | 0.956 | 0.943 | 0.142 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/experiments/0f51fb36092f4241b70510dc6d6bc4bd/output/execution) |
| 44 | Off | 7 | 10 | 0.939 | 0.917 | 0.218 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/experiments/e585ec03551c45279371ff89cb84e116/output/execution) |

Difference per seed (augmentation off - augmentation on)
| Seed | Val accuracy | Val F1 | Val loss |
|---|---:|---:|---:|
| 42 | -2.0 pts | -0.030 | +0.100 |
| 43 | -3.5 pts | -0.044 | +0.118 |
| 44 | -1.7 pts | -0.026 | +0.076 |
| Mean | -2.4 pts (SD 1.0 pts) | -0.033 (SD 0.009) | +0.098 (SD 0.021) |

## Observations
- Augmentation-on runs outperformed the corresponding augmentation-off runs on all three seeds: validation accuracy was higher by 1.7–3.5 percentage points, macro F1 by 0.026–0.044, and validation loss was lower by 0.076–0.118.
- Averaged across the three seeds, augmentation on improved validation accuracy by 2.4 percentage points and macro F1 by 0.033, while reducing validation loss by 0.098.
- The augmentation-off runs reached their best validation-loss epoch sooner (epochs 4–7) than the augmentation-on runs (epochs 12–19). These results suggest augmentation helped in this experiment, though the comparison covers only three seeds.

## Recommendation
- A final evaluation on the test set would provide a stronger check, but these results are encouraging enough to continue with data augmentation.
- Keep augmentation on as the default for now.

## Limitations
- All runs used one desktop GPU; exact result values may differ on other hardware
- Results are on the validation set only; the test set was not used, so these numbers are for choosing settings, not for reporting final performance.