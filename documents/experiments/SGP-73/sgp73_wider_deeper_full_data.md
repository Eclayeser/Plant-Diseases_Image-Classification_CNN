# SGP-73: Wider+Deeper CNN on the Full Dataset

## Why
SGP-15 and SGP-40 compared depth only on 20% of the data for 5 epochs, where adding a fourth block (`[32, 64, 128, 256]`) did not beat the wider CNN (`[32, 64, 128]`). SGP-63 then showed that short runs penalise models that learn more slowly: with `patience: 3` the no-dropout CNN lost up to 1.8 pts. A deeper network is more likely to need longer training, and more data is where extra capacity is most likely to pay off, so this ticket repeats the depth comparison on the full dataset with the longer patience.

## Setup
Both models are RegularisedCNN with dropout (`dropout_p: 0.2`); the only difference is the channels: Wider `[32, 64, 128]` and Wider+Deeper `[32, 64, 128, 256]`.

Full PlantVillage dataset (`fraction: 1.0`), image size 128, batch size 32, augmentation on, `plant_village` normalisation, AdamW (lr 0.001, weight decay 0.01), early stopping on validation loss with `patience: 8`, epoch cap 50, `num_workers: 6`. Seeds 42, 43 and 44, same laptop GPU, SGP-16 determinism fix. The runs were done in pairs (Wider then Wider+Deeper for each seed).

The Wider runs were repeated rather than reused from SGP-63, because `num_workers: 6` changes which random augmentations each image gets (they are generated inside the worker processes), so both sides of the comparison had to use the same loader settings.

Configs: `configs/cnn_wider_dropout_early_stopping_full_seed{42,43,44}_patience8_num_workers6.yaml` and `configs/cnn_wider_deeper_dropout_early_stopping_full_seed{42,43,44}_patience8_num_workers6.yaml`.

## Results (best validation-loss epoch)
| Seed | Model | Best epoch | Stopped at | Val accuracy | Val F1 | Val loss | ClearML |
|---|---|---:|---:|---:|---:|---:|---|
| 42 | Wider | 29 | 37 | 0.961 | 0.950 | 0.123 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/97f0df79c39e46f69c0ff97f246e7119/output/log) |
| 42 | Wider+Deeper | 23 | 31 | 0.967 | 0.957 | 0.102 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/d0e8070132b44c6f8e93024e11308fe4/output/log) |
| 43 | Wider | 15 | 23 | 0.957 | 0.944 | 0.155 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/06f8876009204f0a8025844afc97c564/output/log) |
| 43 | Wider+Deeper | 35 | 43 | 0.969 | 0.960 | 0.113 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/bcfc6ff7c5784474961e785e3ccc45ee/output/log) |
| 44 | Wider | 22 | 30 | 0.963 | 0.952 | 0.124 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/a0574cb008a84968b455ea6b7785fd42/output/log) |
| 44 | Wider+Deeper | 37 | 45 | 0.970 | 0.959 | 0.089 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/d29cdcca59e8493e8af911df575aa546/output/log) |

### Summary
| | Wider | Wider+Deeper |
|---|---:|---:|
| Mean val accuracy | 0.960 | 0.969 |
| Range of val accuracy over seeds | 0.59 pts | 0.33 pts |
| Mean val F1 (macro) | 0.948 | 0.959 |
| Mean val loss | 0.134 | 0.101 |
| Mean epochs trained | 30 | 40 |

Gain from depth per seed (val accuracy): +0.61, +1.22 and +0.76 pts, mean **+0.86 pts**. Macro F1 gain: +0.69, +1.62 and +0.73 pts, mean +1.01 pts.

### Training time
| Seed | Model | Epochs | Time | Per epoch |
|---|---|---:|---:|---:|
| 42 | Wider | 37 | 24 min | 0.64 min |
| 42 | Wider+Deeper | 31 | 21 min | 0.69 min |
| 43 | Wider | 23 | 15 min | 0.66 min |
| 43 | Wider+Deeper | 43 | 29 min | 0.68 min |
| 44 | Wider | 30 | 19 min | 0.64 min |
| 44 | Wider+Deeper | 45 | 32 min | 0.70 min |
| Total | Wider | 90 | about 58 min | |
| Total | Wider+Deeper | 119 | about 1 h 22 | |

Times are approximate (from checkpoint timestamps) and depend on machine load.

## Observations
- **Depth helps on the full dataset.** Wider+Deeper was better on every seed, by 0.61 to 1.22 pts, and the mean gain (+0.86 pts) is larger than the seed-to-seed range of either model (0.59 and 0.33 pts). It also had lower validation loss and higher macro F1 on every seed. Going from 96.0% to 96.9% accuracy means about 22% fewer validation errors.
- **This reverses the 20%-data result** from SGP-15 and SGP-40. Those runs were too short (5 epochs) and too small to show the benefit: here the deeper model reached its best at epochs 23, 35 and 37.
- **The deeper model is not bigger where it matters for overfitting.** With four pooling steps the feature map before the classifier is 8×8×256 = 16,384 values, against 16×16×128 = 32,768 for Wider, so its Flatten→Linear head is half the size (checkpoints: 4.1 MB vs 5.5 MB). The gain comes from the extra convolutional block, not from a larger classifier.
- **Cost:** the time per epoch is almost the same, but Wider+Deeper trained for about 1.3× as many epochs, so it took about 40% longer overall (about 1 h 22 vs 58 min for three seeds).
- **`num_workers: 6` made training about 2.4× faster** than SGP-63 (about 0.66 vs 1.6 min per epoch).
- **Augmentation randomness alone moves results by up to about 0.8 pts.** The Wider runs here differ from the SGP-63 dropout runs only in `num_workers` (and so in the random augmentations): 0.961 vs 0.965, 0.957 vs 0.965 and 0.963 vs 0.961. This is why the baseline was rerun, and it is a useful estimate of run-to-run noise.

## Recommendation
Use the Wider+Deeper architecture ([32, 64, 128, 256]) as the default CNN for full-data runs: it was better on all three seeds by more than the seed noise, although it took about 40% longer to train (1 h 22 vs 58 min for three seeds).

These runs used dropout. SGP-63 found that dropout makes no difference for the Wider CNN, but that may not hold for the deeper one, so whether the final deeper model should use dropout is tested in a follow-up ticket.

## Limitations
- Three seeds per model. The gain was consistent on all three, but three seeds cannot give a precise size for it.
- The best epoch was chosen on the validation set, which makes the validation scores slightly optimistic; this applies to both models equally, so the comparison is still fair. The test set was not used.
- Wider+Deeper seeds 43 and 44 were still improving late (best epochs 35 and 37), so they might gain slightly more with more patience; this would only widen the gap.
- Only one deeper architecture and one dropout setting were tested.
- Times are approximate and depend on machine load; the epoch counts are the more reliable measure of cost.

## Reproducing
```bash
python train.py --config configs/cnn_wider_dropout_early_stopping_full_seed42_patience8_num_workers6.yaml --responsible "Mario Westphalen Cavicchioli" --run-type comparison --issue-id SGP-73
python train.py --config configs/cnn_wider_deeper_dropout_early_stopping_full_seed42_patience8_num_workers6.yaml --responsible "Mario Westphalen Cavicchioli" --run-type comparison --issue-id SGP-73
```
Repeat with seeds 43 and 44.
