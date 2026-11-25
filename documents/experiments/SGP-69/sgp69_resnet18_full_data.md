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

## Evaluation
The results were evaluated against the validation data split using `evaluate.py`

| Seed | Checkpoint | Top-1 | Top-3 | Top-5 | Macro precision | Macro recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 42 | `resnet18_20261124-205942_944b25.pt` | 0.9873 | 0.9996 | 0.9996 | 0.9821 | 0.9758 | 0.9775 |
| 43 | `resnet18_20261124-213547_56c944.pt` | 0.9894 | 0.9994 | 0.9999 | 0.9865 | 0.9862 | 0.9860 |
| 44 | `resnet18_20261124-220057_d21c08.pt` | 0.9867 | 0.9994 | 0.9998 | 0.9843 | 0.9838 | 0.9835 |
## Observations


## Recommendation

## Limitations

## Reproducing
