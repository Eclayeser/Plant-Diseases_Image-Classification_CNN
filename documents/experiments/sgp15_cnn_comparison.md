# SGP-15: CNN Architecture Comparison

## Setup
All runs use the same settings so that only the architecture changes:
- Data: 5% of PlantVillage (`fraction: 0.05`), image size 128, batch size 32
- Training: 5 epochs, AdamW, lr 0.001, weight decay 0.01, seed 42

## Results (validation set, last epoch)

| Model | Config | Channels | Parameters | Val accuracy | Val F1 | Checkpoint |
|---|---|---|---|---|---|---|
| Baseline | `prototype_small_data_frac.yaml` | [16, 32, 64] | 662,599 | 0.630 | 0.534 | `prototype_small_data_frac_20261106-101034_9934d5.pt` |
| Wider | `cnn_wider.yaml` | [32, 64, 128] | 1,371,239 | 0.668 | 0.566 | `cnn_wider_20261106-102850_dceb83.pt` |
| Deeper | `cnn_deeper.yaml` | [16, 32, 64, 128] | 416,967 | 0.572 | 0.454 | `cnn_deeper_20261106-103407_af0549.pt` |

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

## Limitations
- All models use the same training settings (5 epochs, lr 0.001). This keeps the
  comparison controlled, but it may not be the best setting for each architecture.
- Only 5% of the data is used, so results may change with the full dataset.