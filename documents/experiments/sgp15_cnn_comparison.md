# SGP-15: CNN Architecture Comparison

## Setup
Within each run, all models use the same training settings so that only the architecture changes:
- Image size 128, batch size 32
- Run 1 uses 5% of PlantVillage (`fraction: 0.05`); Run 2 uses 20% (`fraction: 0.2`, same as `prototype.yaml`), so the results can be compared with the other models in the team.
- Training: 5 epochs, AdamW, lr 0.001, weight decay 0.01, seed 42

## Results (validation set, last epoch)
### Run 1: 5% of the data
| Model | Config | Channels | Parameters | Val accuracy | Val F1 | Checkpoint |
|---|---|---|---|---|---|---|
| Baseline | `prototype_small_data_frac.yaml` | [16, 32, 64] | 662,599 | 0.630 | 0.534 | `prototype_small_data_frac_20261106-101034_9934d5.pt` |
| Wider | `cnn_wider.yaml` | [32, 64, 128] | 1,371,239 | 0.668 | 0.566 | `cnn_wider_20261106-102850_dceb83.pt` |
| Deeper | `cnn_deeper.yaml` | [16, 32, 64, 128] | 416,967 | 0.572 | 0.454 | `cnn_deeper_20261106-103407_af0549.pt` |


### Run 2: 20% of the data

| Model | Config | Channels | Parameters | Val accuracy | Val F1 | Checkpoint |
|---|---|---|---|---|---|---|
| Baseline | `prototype.yaml` | [16, 32, 64] | 662,599 | 0.782 | 0.701 | `basiccnn_20261108-190231_d45e16.pt` |
| Wider | `cnn_wider.yaml` | [32, 64, 128] | 1,371,239 | 0.827 | 0.773 | `basiccnn_20261108-190708_c89b97.pt` |
| Deeper | `cnn_deeper.yaml` | [16, 32, 64, 128] | 416,967 | 0.776 | 0.698 | `basiccnn_20261108-191112_e0aa84.pt` |

## Observations
- About 96% of the baseline's parameters are in the final linear layer.
- The deeper model has fewer parameters than the baseline because each extra
  pooling layer halves the feature map size before the final linear layer.
- For the baseline, validation loss was still decreasing at epoch 5
  (1.570 → 1.371 → 1.208), so more epochs may improve it.
- The wider model improved validation accuracy by 3.8 points (0.630 → 0.668)
  and F1 by 3.2 points (0.534 → 0.566), but it has about twice as many parameters.
- For the wider model, validation loss increased slightly from epoch 4 to 5
  (1.114 → 1.138) while training loss kept decreasing, which may be an early
  sign of overfitting.
- The deeper model performed worst after 5 epochs (accuracy 0.572, F1 0.454),
  but its validation loss was still dropping quickly (1.809 → 1.680 → 1.405),
  so it is likely undertrained rather than a worse architecture. Deeper models
  often need more epochs to converge.
- **Run 2 (20% of the data):** more data improved every model by 15–20 accuracy points.
- Unlike Run 1, the wider model's validation loss kept decreasing in Run 2 (0.727 → 0.699 → 0.589), suggesting less evidence of overfitting with more data. It is still the best model (accuracy 0.827, F1 0.773).
- The deeper model improved the most (+20.4 points, compared with +15.2 for the baseline) and now almost matches the baseline (0.776 vs 0.782) with 37% fewer parameters. This suggests that the deeper model may benefit more from additional data or training epochs.

## Limitations
- All models use the same training settings (5 epochs, lr 0.001). This keeps the
  comparison controlled, but it may not be the best setting for each architecture.
- Each result comes from a single run with one seed, so small differences (e.g. baseline vs deeper in Run 2) may not be significant.
- Run 2 checkpoints are named after the model type, so all three start with `basiccnn_`; the table records which file belongs to which config.