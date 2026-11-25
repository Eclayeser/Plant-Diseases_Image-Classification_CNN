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


## Recommendation

## Limitations

## Reproducing
