# SGP-52: Dropout on the Full Dataset

## Setup
Compared the wider BasicCNN (`[32, 64, 128]`) with the same network plus dropout (RegularisedCNN, `dropout_p: 0.2`, `Dropout2d` after each ReLU). Both used the full PlantVillage dataset (`fraction: 1.0`), image size 128, batch size 32, augmentation on, `plant_village` normalisation, and AdamW (lr 0.001, weight decay 0.01). Training was capped at 30 epochs with early stopping on validation loss (`patience: 3`).

Each model was run with seeds 42, 43 and 44, all on the same laptop GPU (RTX 4070) with the SGP-16 determinism fix. The seed sets the weight initialisation, the data order and the train/val/test split, so the two models are compared within each seed first.

Configs: `configs/cnn_wider_plant_village_early_stopping_full_seed{42,43,44}.yaml` and `configs/cnn_wider_dropout_early_stopping_full_seed{42,43,44}.yaml`.

## Results (best validation-loss epoch)
| Seed | Model | Best epoch | Stopped at | Val accuracy | Val F1 | Val loss | ClearML |
|---|---|---:|---:|---:|---:|---:|---|
| 42 | No dropout | 5 | 8 | 0.944 | 0.929 | 0.179 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/f42ae22f53114dee8d6dc099b4f06ead/output/log) |
| 42 | Dropout | 16 | 19 | 0.958 | 0.941 | 0.138 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/bf07a89ca4ce4166b6dd5516ce326b50/output/log) |
| 43 | No dropout | 6 | 9 | 0.950 | 0.937 | 0.171 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/c17cd45a8cf549939d8c57199a4fb73c/output/log) |
| 43 | Dropout | 14 | 17 | 0.957 | 0.944 | 0.160 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/8c6a86b758ff4195b9d8bd20bd825cda/output/log) |
| 44 | No dropout | 11 | 14 | 0.963 | 0.955 | 0.128 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/70cc2c1bbf904bbda582b4ccac17c132/output/log) |
| 44 | Dropout | 13 | 16 | 0.961 | 0.947 | 0.131 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/a1ef66e4ef6d4a029f997e50236f4fe5/output/log) |

Dropout gain per seed (dropout minus no dropout):

| Seed | Val accuracy | Val F1 | Val loss |
|---|---:|---:|---:|
| 42 | +1.41 pts | +0.012 | −0.041 |
| 43 | +0.75 pts | +0.007 | −0.011 |
| 44 | −0.26 pts | −0.009 | +0.003 |
| Mean | +0.63 pts (SD 0.84) | +0.003 | −0.017 |

## Observations
- Dropout improved validation accuracy for two seeds and was slightly worse for the third. The mean gain (+0.63 pts) is smaller than its spread across seeds, so it is within the noise. This matches SGP-44 on 20% of the data (+0.9 pts, also within seed spread).
- Dropout runs were more consistent across seeds: their validation accuracy ranged over 0.34 pts, against 1.96 pts without dropout.
- The no-dropout runs reached their best epoch much earlier (5, 6 and 11) than the dropout runs (16, 14 and 13). The two no-dropout runs that peaked earliest were the two weakest, and the one that trained longest (seed 44) was the best of all six runs. Part of the no-dropout spread, and part of dropout's advantage, may therefore come from early stopping ending those runs too soon rather than from dropout itself.
- Dropout's training accuracy was below its validation accuracy in every run. This is expected, since dropout and augmentation are only active during training and training accuracy is averaged over the epoch, so it isn't a sign of underfitting on its own.
- Dropout runs took up to about 2.4× longer (roughly 28 vs 67 minutes for seed 42). Time per epoch was about the same; the extra cost comes from needing more epochs.

## Recommendation
The results are inconclusive. Dropout gave no clear accuracy gain on the full dataset, and its main advantage (more consistent results across seeds) may be caused by early stopping ending the no-dropout runs too soon. It also costs up to about 2.4× more training time.

Keep dropout as the default for now so ongoing experiments stay comparable, and decide after the follow-up with longer patience. If no-dropout becomes as consistent as dropout when given more patience, the simpler and faster BasicCNN should become the default.

## Limitations
- Three seeds per model is too few to confirm either the small accuracy gain or the difference in consistency.
- `patience: 3` may have stopped the no-dropout runs too early. This was checked in SGP-63 (`patience: 8`): no-dropout improved and the two models performed the same, so the dropout advantage reported here came mostly from early stopping. See `documents/experiments/SGP-63/`.
- All runs used one laptop GPU; exact values may differ on other hardware.
- Results are on the validation set only; the test set was not used, so these numbers are for choosing settings, not for reporting final performance.