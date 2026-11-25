# SGP-69: Training ResNet18 on the Full Dataset

## Setup
Rerun ResNet18 on the full dataset.
Shared settings:
- `data`: `image_size: 128`, `fraction: 1.0`, `batch_size: 32`, `normalize: imagenet`, `augment: true`, `num_workers: 6`
- `model`: `type: ResNet18`
- `train`: `lr: 0.001`, `epochs: 50`, `weight_decay: 0.01`, `checkpoint_dir: outputs/models`, `patience: 8`

Configs: `resnet18_full_patience8_num_workers6_seed{42,43,44}.yaml`

## Results (run time + best validation-loss epoch)
| Seed | Model | Run time | Best epoch | Stopped at | Val accuracy | Val F1 | Val loss | ClearML |
|---|---|---|---:|---:|---:|---:|---:|---|
| 42 | ResNet18 | 29:51m | 28 | 36 | 0.991 | 0.986 | 0.030 | [Task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/experiments/22742f545aba4e6ebf34b31c2ee407e2/output/execution)|
| 43 | ResNet18 | 22:42m | 20 | 28 | 0.989 | 0.986 | 0.030 | [Task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/experiments/5288ed32352f458ba9cc868a7e30853b/output/execution)|
| 44 | ResNet18 | 25:22m | 23 | 31 | 0.991 | 0.988 | 0.030 | [Task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/experiments/23edd3eb3f8e436aacbeeef0bbc1775c/output/execution) |

The checkpoints are not committed to the repository. They are stored as artifacts (`final_checkpoint` and `best_checkpoint`) in each ClearML task.

## Evaluation
The results were evaluated against the validation data split using `evaluate.py`

| Seed | Top-1 | Top-3 | Top-5 | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.9905 | 0.9995 | 0.9998 | 0.9871 | 0.9847 | 0.9857 |
| 43 | 0.9894 | 0.9998 | 0.9999 | 0.9864 | 0.9851 | 0.9856 |
| 44 | 0.9909 | 0.9994 | 0.9999 | 0.9891 | 0.9875 | 0.9882 |

The validation split has 8317 images across 38 classes.

## Observations
- **Consistent across seeds.** Seed 44 has the highest top-1 accuracy and macro F1 at the best epoch, but the margins are within seed-to-seed noise.
- **Top-k accuracy is near perfect.** Top-3 is at least 0.9994 and top-5 at least 0.9998 for every seed, so the correct class is almost always among the model's top few guesses.
- **Training length varies, time per epoch does not.** Best epochs range from 20 to 28, and runs stopped between epoch 28 and 36. Time per epoch is steady at about 49 s, so the run time differences (22:42m to 29:51m) come from the number of epochs.
- **Validation loss is noisy.** It spikes from epoch to epoch even late in training (for example seed 44: 0.035 at epoch 18, 0.098 at epoch 22, 0.030 at epoch 23). The patience of 8 let the runs ride out these spikes, but the best epoch is sensitive to noise. For seed 42, epoch 28 (0.0304) beats epoch 25 (0.0311) by a tiny margin.
- **Corn leaf diseases are the most persistent weak spot.** *Corn Cercospora leaf spot / Gray leaf spot* is mistaken for *Corn Northern Leaf Blight* in every seed (9, 9 and 6 images). It is the top confused pair for seeds 42 and 43 and the second for seed 44. *Corn GLS* has the lowest or near-lowest F1 in each seed (0.94, 0.91, 0.95), with recall of 0.88, 0.88 and 0.92.
- **Tomato classes are the next source of errors.** *Tomato early blight* has F1 of 0.95–0.96 in every seed (recall 0.93–0.95). *Tomato target spot* and *Tomato spider mites* are confused with each other in seeds 43 and 44 (8 and 7 images, in opposite directions). Seed 42's second and third confusions are *Tomato late blight* predicted as *Potato late blight* (7) and *Tomato early blight* (5).

## Deployment

## Limitations
- While the seed spread is small, three runs are too few to estimate variance or support significance claims.
- Best-epoch metrics depend on noisy validation loss, and small differences between epochs are within the noise. Seed 43 shows that the lowest-loss checkpoint is not guaranteed to have the best F1.
- Runs used a ROCm build of PyTorch. Results are deterministic only given the SGP-16 fix, and may not match CUDA exactly.

## Reproducing
