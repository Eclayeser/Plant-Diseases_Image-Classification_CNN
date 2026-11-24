# SGP-91: Dropout on the Wider+Deeper CNN (Full Dataset)

## Why
SGP-73 found that the Wider+Deeper CNN (`[32, 64, 128, 256]`) beats the Wider CNN on the full dataset (+0.86 pts mean val accuracy), but both models in that comparison used dropout. SGP-63 found that dropout makes no difference for the Wider CNN, but a deeper network has more capacity and could behave differently, so this ticket checks whether the deeper model needs dropout.

## Setup
Wider+Deeper `[32, 64, 128, 256]` without dropout (BasicCNN), compared with the SGP-73 Wider+Deeper runs with dropout (RegularisedCNN, `dropout_p: 0.2`). The only difference between the two is dropout, so the SGP-73 runs were reused rather than repeated.

Full PlantVillage dataset (`fraction: 1.0`), image size 128, batch size 32, augmentation on, `plant_village` normalisation, AdamW (lr 0.001, weight decay 0.01), early stopping on validation loss with `patience: 8`, epoch cap 50, `num_workers: 6`. Seeds 42, 43 and 44, same laptop GPU, SGP-16 determinism fix.

Configs: `configs/cnn_wider_deeper_early_stopping_full_seed{42,43,44}_patience8_num_workers6.yaml` (new) and `configs/cnn_wider_deeper_dropout_early_stopping_full_seed{42,43,44}_patience8_num_workers6.yaml` (SGP-73).

Decision rule, set before the runs: if no dropout is within about 0.5 pts mean val accuracy and not consistently worse, prefer no dropout (simpler and cheaper); if dropout is better on all three seeds by more than that, keep dropout.

## Results (best validation-loss epoch)
| Seed | Model | Best epoch | Stopped at | Val accuracy | Val F1 | Val loss | ClearML |
|---|---|---:|---:|---:|---:|---:|---|
| 42 | Dropout (SGP-73) | 23 | 31 | 0.967 | 0.957 | 0.102 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/d0e8070132b44c6f8e93024e11308fe4/output/log) |
| 42 | No dropout | 10 | 18 | 0.971 | 0.962 | 0.101 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/bce1416e27dc455592b5160505361e42/output/log) |
| 43 | Dropout (SGP-73) | 35 | 43 | 0.969 | 0.960 | 0.113 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/bcfc6ff7c5784474961e785e3ccc45ee/output/log) |
| 43 | No dropout | 18 | 26 | 0.972 | 0.966 | 0.116 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/fbef2e2da74149ba9d71fcb7fae85e70/output/log) |
| 44 | Dropout (SGP-73) | 37 | 45 | 0.970 | 0.959 | 0.089 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/d29cdcca59e8493e8af911df575aa546/output/log) |
| 44 | No dropout | 18 | 26 | 0.972 | 0.963 | 0.098 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/f5b6ae6cd9ef4ffdbed4b3031bb1f146/output/log) |

### Summary
| | Dropout | No dropout |
|---|---:|---:|
| Mean val accuracy | 0.969 | 0.971 |
| Range of val accuracy over seeds | 0.33 pts | 0.13 pts |
| Mean val F1 (macro) | 0.959 | 0.964 |
| Mean val loss | 0.101 | 0.105 |
| Mean epochs trained | 40 | 23 |

No dropout minus dropout per seed (val accuracy): +0.38, +0.31 and +0.14 pts, mean **+0.28 pts**. Macro F1: +0.58, +0.58 and +0.38 pts, mean +0.51 pts.

### Training time
| Seed | Model | Epochs | Time | Per epoch |
|---|---|---:|---:|---:|
| 42 | Dropout | 31 | 21 min | 0.69 min |
| 42 | No dropout | 18 | 11 min | 0.58 min |
| 43 | Dropout | 43 | 29 min | 0.68 min |
| 43 | No dropout | 26 | 16 min | 0.61 min |
| 44 | Dropout | 45 | 32 min | 0.70 min |
| 44 | No dropout | 26 | 16 min | 0.61 min |
| Total | Dropout | 119 | about 1 h 22 | |
| Total | No dropout | 70 | about 43 min | |

Times are approximate (from checkpoint timestamps) and depend on machine load.

## Observations
- **Dropout does not help the deeper model.** No dropout was slightly better on all three seeds (mean +0.28 pts accuracy, +0.51 pts macro F1). The gain is smaller than the run-to-run variation from augmentation alone measured in SGP-73 (up to 0.8 pts), so the two are best described as performing the same, with no sign that dropout helps.
- **Dropout slows training down.** Without dropout the best epochs were 10, 18 and 18, against 23, 35 and 37 with dropout, so no dropout needed about 40% fewer epochs and took about half the time (43 min vs 1 h 22 for three seeds). This matches SGP-63, where dropout needed about 1.2× as many epochs for the Wider CNN; the effect is larger here.
- **Dropout had slightly lower validation loss on two of three seeds** (mean 0.101 vs 0.105), meaning slightly more confident predictions, but no higher accuracy. The difference is smaller than the seed-to-seed variation in loss (0.089 to 0.116), so it does not change the recommendation. Because the app shows predicted probabilities, calibration of the final models could be checked during the test-set evaluation.
- **No dropout was also more consistent across seeds** (0.13 vs 0.33 pts range), though with three seeds this is not a meaningful difference.

## Recommendation
Use the Wider+Deeper BasicCNN without dropout (`[32, 64, 128, 256]`) as the final CNN: it matched or slightly beat the dropout version on every seed at about half the training cost. This agrees with SGP-63 (dropout gives no gain for the Wider CNN) and SGP-73 (depth helps on the full dataset).

Recommended settings for the final CNN: full data, augmentation on, `plant_village` normalisation, AdamW (lr 0.001, weight decay 0.01), `patience: 8`, epoch cap 50 (the no-dropout runs peaked at epochs 10 to 18, so the cap is not limiting).

## Limitations
- Three seeds per model; the small accuracy difference cannot be separated from run-to-run noise.
- The best epoch was chosen on the validation set, which makes the validation scores slightly optimistic for both models equally. The test set was not used.
- Only one dropout rate (0.2) was tested.
- Times are approximate and depend on machine load; the epoch counts are the more reliable measure of cost.

## Reproducing
```bash
python train.py --config configs/cnn_wider_deeper_early_stopping_full_seed42_patience8_num_workers6.yaml --responsible "Mario Westphalen Cavicchioli" --run-type comparison --issue-id SGP-91
```
Repeat with seeds 43 and 44. The dropout runs are in SGP-73.