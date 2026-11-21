# SGP-51: Global Average Pooling for the Wider CNN

## Purpose

In the wider CNN (`[32, 64, 128]`, image size 128), the final Linear layer takes 128 × 16 × 16 = 32,768 inputs, so it holds 1,277,991 of the model's 1,371,239 parameters (about 93%). SGP-31 saw possible signs of overfitting, so this experiment tests whether replacing Flatten with global average pooling (GAP) reduces overfitting and how it affects accuracy.

## Setup

Both models use the same training settings. The Flatten runs use `cnn_wider_normalize_plant_village_early_stopping.yaml` (from SGP-22), and the GAP config `cnn_wider_avg_pooling_early_stopping.yaml` is a copy of it with `pooling: avg` added, so the pooling method and classifier head are the only architectural differences.

- Fraction 0.2, image size 128, batch size 32, plant_village normalisation
- AdamW, lr 0.001, weight decay 0.01
- Up to 30 epochs with early stopping (patience 3); each run is compared at its lowest validation loss
- Each model was trained with seeds 42, 43 and 44. For seeds 43 and 44, a copy of each config was made under `outputs/sgp51_configs/` with only the seed changed. Within each seed, both models use the same data subset and split.
- All runs were trained on the same machine and tracked in ClearML (project `Plant-Disease-Image-Classification`): [ClearML comparison](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/compare-tasks;ids=7d68ac7b6086499b83a96e390ac7bcad,0303978090ea4e1b864befae73ecce10,1b041c6496c0429aab77f290a4caf7e8,8874946b21fa4c08b25c7cc4d9b3c6da,e434f5fee47a4fd1a1869fbf0f70424c,d9b82401305e4f9d99b3d865d04caf4d/scalars/graph)

## Results (validation set, best epoch)

| Model | Seed | Best epoch | Stopped | Val accuracy | Val F1 | Val loss | Train accuracy | ClearML task |
|---|---|---|---|---|---|---|---|---|
| Flatten | 42 | 9 | 12 | 0.895 | 0.875 | 0.392 | 0.937 | `7d68ac7b6086499b83a96e390ac7bcad` |
| Flatten | 43 | 13 | 16 | 0.896 | 0.857 | 0.388 | 0.949 | `1b041c6496c0429aab77f290a4caf7e8` |
| Flatten | 44 | 8 | 11 | 0.886 | 0.856 | 0.405 | 0.918 | `e434f5fee47a4fd1a1869fbf0f70424c` |
| GAP | 42 | 18 | 21 | 0.832 | 0.783 | 0.521 | 0.831 | `0303978090ea4e1b864befae73ecce10` |
| GAP | 43 | 21 | 24 | 0.860 | 0.800 | 0.445 | 0.852 | `8874946b21fa4c08b25c7cc4d9b3c6da` |
| GAP | 44 | 26 | 29 | 0.880 | 0.855 | 0.418 | 0.859 | `d9b82401305e4f9d99b3d865d04caf4d` |

Mean ± sample standard deviation across the three seeds:

| Model | Parameters | Checkpoint size | Val accuracy | Val F1 | Val loss | Best epoch |
|---|---|---|---|---|---|---|
| Flatten | 1,371,239 | 5.2 MB | 0.892 ± 0.006 | 0.863 ± 0.011 | 0.395 ± 0.009 | 10 ± 3 |
| GAP | 98,279 | 391 KB | 0.857 ± 0.024 | 0.813 ± 0.038 | 0.461 ± 0.053 | 22 ± 4 |

The checkpoints are not committed to the repository. They are stored as artifacts (`final_checkpoint` and `best_checkpoint`) in each ClearML task.

The charts below show the seed 42 runs.

![Validation accuracy](accuracy_validation.png)

![Validation loss](loss_validation.png)

![Validation F1](f1_validation.png)

## Observations

- **Accuracy.** Flatten did better on average: 3.5 percentage points higher validation accuracy (0.892 vs 0.857), 5.0 points higher macro F1 (0.863 vs 0.813) and lower validation loss (0.395 vs 0.461). However, the gap depends a lot on the seed: 6.3 points for seed 42, 3.6 for seed 43 and only 0.6 for seed 44.
- **Stability.** Flatten gave very similar results for every seed (standard deviation 0.6 points), while GAP varied much more (2.4 points).
- **Overfitting.** Flatten shows signs of overfitting in every run: at its best epoch, training accuracy is 3.2 to 5.3 points above validation accuracy, and validation loss rises again in the following epochs. GAP shows fewer signs of overfitting across the three runs: its training accuracy is the same as or slightly below its validation accuracy at the best epoch in all three runs.
- **Slower learning.** GAP reached its best validation loss later than Flatten (22 vs 10 epochs on average). Its validation accuracy also varied considerably across seeds. This suggests that GAP may need different training settings, but longer training alone has not been shown to improve its results. Its training accuracy (about 0.85) also stays below Flatten's (about 0.93), so it fits the training data less closely.
- **Training time.** Time per epoch was about the same for both models (~45 s for seeds 43 and 44), because most of the computation is in the convolutional layers. Since GAP needs more epochs, its runs took longer overall.
- **Model size.** GAP has about 14 times fewer parameters (98,279 vs 1,371,239), so its checkpoint is much smaller (391 KB vs 5.2 MB).

## Recommendation

- Keep `flatten` as the default for the wider CNN. It was more accurate on average and much more consistent across seeds, although it showed more signs of overfitting than GAP.
- The pooling: avg option stays in the code, since it does not change existing configs. GAP could be tried again with different training settings or if a smaller model is needed for deployment.

## Limitations

- **Three seeds.** The results vary a lot between seeds for GAP, so more runs would give a clearer picture.
- **Same settings for both models.** GAP used the same learning rate and patience as Flatten. Since it appears to learn more slowly, these settings may not suit it, but this was not tested.
- **20% of the data.** All runs use `fraction: 0.2`. Results on the full dataset may differ.
- **Validation numbers only.** The validation set was used to pick each run's best epoch, so the results may be somewhat optimistic. Test-set performance was not measured.