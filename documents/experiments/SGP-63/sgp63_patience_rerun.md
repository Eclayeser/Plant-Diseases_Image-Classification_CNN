# SGP-63: Dropout on the Full Dataset with Longer Patience

## Why
SGP-52 compared the wider CNN with and without dropout on the full dataset using `patience: 3`. Dropout looked slightly better (+0.63 pts val accuracy) and more consistent across seeds (0.34 vs 1.96 pts), but the no-dropout runs peaked early (best epochs 5, 6 and 11) and the two that stopped earliest were the weakest. This rerun checks whether that advantage came from early stopping ending the no-dropout runs too soon.

## Setup
Identical to SGP-52 except for two settings: `patience: 8` (was 3) and `epochs: 40` (was 30), so the cap would not end runs before patience did.

Full PlantVillage dataset (`fraction: 1.0`), image size 128, batch size 32, augmentation on, `plant_village` normalisation, AdamW (lr 0.001, weight decay 0.01), early stopping on validation loss. No dropout = BasicCNN `[32, 64, 128]`; dropout = RegularisedCNN with `dropout_p: 0.2`. Seeds 42, 43 and 44, same laptop GPU, SGP-16 determinism fix.

Configs: `configs/cnn_wider_plant_village_early_stopping_full_seed{42,43,44}_patience8.yaml` and `configs/cnn_wider_dropout_early_stopping_full_seed{42,43,44}_patience8.yaml`.

## Results (best validation-loss epoch)
| Seed | Model | Best epoch | Stopped at | Val accuracy | Val F1 | Val loss | ClearML |
|---|---|---:|---:|---:|---:|---:|---|
| 42 | No dropout | 16 | 24 | 0.960 | 0.948 | 0.138 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/e9f7b9454482472ab75f60c95eb60c60/output/log) |
| 42 | Dropout | 30 | 38 | 0.965 | 0.954 | 0.111 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/03f268c6d6d54b698a7fc95e03c7e988/output/log) |
| 43 | No dropout | 24 | 32 | 0.968 | 0.959 | 0.144 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/59a0d5ed90f3431ea9c1fde92bd68c8c/output/log) |
| 43 | Dropout | 33 | 40 (cap) | 0.965 | 0.955 | 0.132 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/abc741a2761e40ee8eb45d04e16d71c5/output/log) |
| 44 | No dropout | 15 | 23 | 0.964 | 0.953 | 0.123 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/658805e8450741ba87edcecb6126603d/output/log) |
| 44 | Dropout | 13 | 21 | 0.961 | 0.947 | 0.131 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/5dfccb689f3e41f480d3eea3c042d098/output/log) |

### Compared with SGP-52 (patience 3)
| | Patience 3 (SGP-52) | Patience 8 (SGP-63) |
|---|---|---|
| Dropout gain per seed (val acc) | +1.41, +0.75, −0.26 pts | +0.55, −0.34, −0.38 pts |
| Mean dropout gain | +0.63 pts | −0.06 pts |
| No dropout: mean val acc / range over seeds | 0.952 / 1.96 pts | 0.964 / 0.81 pts |
| Dropout: mean val acc / range over seeds | 0.959 / 0.34 pts | 0.964 / 0.48 pts |

Change in val accuracy from patience 3 to 8:

| Seed | No dropout | Dropout |
|---|---|---|
| 42 | +1.62 pts (best epoch 5 → 16) | +0.77 pts (16 → 30) |
| 43 | +1.82 pts (6 → 24) | +0.73 pts (14 → 33) |
| 44 | +0.12 pts (11 → 15) | 0.00 pts (13 → 13) |

### Training time
| Seed | Model | Epochs | Time | Per epoch |
|---|---|---:|---:|---:|
| 42 | No dropout | 24 | 43 min | 1.8 min |
| 42 | Dropout | 38 | 56 min | 1.5 min |
| 43 | No dropout | 32 | 49 min | 1.5 min |
| 43 | Dropout | 40 | 62 min | 1.6 min |
| 44 | No dropout | 23 | 33 min | 1.4 min |
| 44 | Dropout | 21 | 32 min | 1.5 min |
| Total | No dropout | 79 | about 2 h 05 | |
| Total | Dropout | 99 | about 2 h 30 | |

Times are approximate (from checkpoint timestamps) and depend on machine load.

## Observations
- **`patience: 3` was cutting runs short.** With more patience, no-dropout improved by 1.6 to 1.8 pts for seeds 42 and 43, and their best epochs moved from 5 and 6 to 16 and 24. Dropout also improved, by less.
- **The two models now perform the same.** The dropout gain is positive for one seed and negative for two, with a mean of −0.06 pts, well within the seed-to-seed spread.
- **The consistency difference from SGP-52 has mostly gone.** The no-dropout range across seeds fell from 1.96 to 0.81 pts, against 0.48 pts for dropout; with three seeds this is not a meaningful difference. Most of the earlier spread came from early stopping, not from the lack of dropout.
- **Dropout still has slightly lower validation loss** for two of the three seeds (mean 0.124 vs 0.135), meaning it is a little more confident on the correct class, but this does not translate into higher accuracy.
- **Cost:** both models take about 1.5 minutes per epoch, but dropout needed more epochs to reach its best, so it took about 1.2× longer overall for the same accuracy.
- **Determinism check:** dropout seed 44 gave exactly the same result as in SGP-52 (best epoch 13, val acc 0.9606, val loss 0.131). Its best epoch came before SGP-52's run stopped, so both runs were identical up to that point, on a different day, which supports the SGP-16 fix.

## Recommendation
With enough patience, dropout and no dropout perform the same on the full dataset, within seed noise. In my view, when two options give the same results, the simpler and cheaper one should be preferred, so I recommend the plain BasicCNN (no dropout) as the default: it reached the same accuracy in about 20% less training time.

Separately, `patience: 3` is too short for full-data runs: it cost the no-dropout model up to 1.8 pts. I recommend `patience: 8` with an epoch cap of at least 40 (one run still reached 40) for future full-data experiments.

## Limitations
- Three seeds per model is too few to detect a small difference; "the same" means no difference was detectable with three seeds.
- Dropout seed 43 reached the 40-epoch cap (best epoch 33), so it might have improved slightly with more epochs.
- Times are approximate and depend on machine load, so the epoch counts are the more reliable measure of cost. Times cannot be compared with SGP-52's, which ran under different load.
- All runs used one laptop GPU, and results are on the validation set only; the test set was not used.