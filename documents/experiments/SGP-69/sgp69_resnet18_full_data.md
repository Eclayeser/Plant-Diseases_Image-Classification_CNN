# SGP-69: Training ResNet18 on the Full Dataset

## Setup
Rerun ResNet18 on the full dataset.
Configs:
- `data`: `image_size: 128`, `fraction: 1.0`, `batch_size: 32`, `normalize: imagenet`, `augment: true`, `num_workers: 6`
- `model`: `type: ResNet18`
- `train`: `lr: 0.001`, `epochs: 50`, `weight_decay: 0.01`, `checkpoint_dir: outputs/models`, `patience: 8`

## Results (best validation-loss epoch)


## Observations


## Recommendation


## Limitations
