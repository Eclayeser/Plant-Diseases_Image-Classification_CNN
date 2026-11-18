# SGP-51: Global Average Pooling for the Wider CNN

## Purpose

In the wider CNN (`[32, 64, 128]`, image size 128), the final Linear layer takes 128 × 16 × 16 = 32,768 inputs, so it holds 1,277,991 of the model's 1,371,239 parameters (about 93%). SGP-31 saw possible signs of overfitting, so this experiment tests whether replacing Flatten with global average pooling (GAP) reduces overfitting and how it affects accuracy.

## Setup

Both runs use the same training settings. The Flatten run uses `cnn_wider_normalize_plant_village_early_stopping.yaml` (from SGP-22), and the GAP config `cnn_wider_avg_pooling_early_stopping.yaml` is a copy of it with `pooling: avg` added, so the final layer is the only difference:

- Fraction 0.2, seed 42, image size 128, batch size 32, plant_village normalisation
- AdamW, lr 0.001, weight decay 0.01
- Up to 30 epochs with early stopping (patience 3); each run is compared at its lowest validation loss
- Both runs were trained on the same machine and tracked in ClearML (project `Plant-Disease-Image-Classification`): [ClearML comparison](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/compare-tasks;ids=7d68ac7b6086499b83a96e390ac7bcad,0303978090ea4e1b864befae73ecce10/scalars/graph)

## Results (validation set, best epoch)

| Model | Config | Parameters | Best epoch | Stopped | Val accuracy | Val F1 | Val loss | Train accuracy | Time per epoch |
|---|---|---|---|---|---|---|---|---|---|
| Flatten | `cnn_wider_normalize_plant_village_early_stopping.yaml` | 1,371,239 | 9 | 12 | 0.895 | 0.875 | 0.392 | 0.937 | ~62 s |
| GAP | `cnn_wider_avg_pooling_early_stopping.yaml` | 98,279 | 18 | 21 | 0.832 | 0.783 | 0.521 | 0.831 | ~58 s |

The checkpoints are not committed to the repository. They are stored as artifacts (`final_checkpoint` and `best_checkpoint`) in each ClearML task.