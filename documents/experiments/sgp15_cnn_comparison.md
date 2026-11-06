# SGP-15: CNN Architecture Comparison

## Setup
All runs use the same settings so that only the architecture changes:
- Data: 5% of PlantVillage (`fraction: 0.05`), image size 128, batch size 32
- Training: 5 epochs, AdamW, lr 0.001, weight decay 0.01, seed 42

## Results (validation set, last epoch)

| Model | Config | Channels | Parameters | Val accuracy | Val F1 | Checkpoint |
|---|---|---|---|---|---|---|
| Baseline | `prototype_small_data_frac.yaml` | [16, 32, 64] | 662,599 | 0.630 | 0.534 | `prototype_small_data_frac_20261106-101034_9934d5.pt` |
| Wider | `cnn_wider.yaml` | [32, 64, 128] | 1,371,239 | | | |
| Deeper | `cnn_deeper.yaml` | [16, 32, 64, 128] | 416,967 | | | |

## Observations
- About 96% of the baseline's parameters are in the final linear layer.
- The deeper model has fewer parameters than the baseline because each extra
  pooling layer halves the feature map size before the final linear layer.
- For the baseline, validation loss was still decreasing at epoch 5
  (1.570 → 1.371 → 1.208), so more epochs may improve it.

## Limitations
- All models use the same training settings (5 epochs, lr 0.001). This keeps the
  comparison controlled, but it may not be the best setting for each architecture.
- Only 5% of the data is used, so results may change with the full dataset.