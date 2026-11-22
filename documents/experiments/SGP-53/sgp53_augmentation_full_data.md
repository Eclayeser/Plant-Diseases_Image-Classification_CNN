# SGP-52: Dropout on the Full Dataset

## Setup
Compared the wider CNN with the same model but data augmentation turned off.
Shared settings:
- `data`: `image_size: 128`, full data (`fraction: 1.0`), `batch_size: 32`, `normalize: plant_village`
- `model`: `type: RegularisedCNN`, `channels: [32, 64, 128]`, `regularisation: dropout`, `dropout_p: 0.2`
- `train`: AdamW (`lr: 0.001`, `weight_decay 0.01`), early stopping (`patience: 3`, max 30 epochs `epochs: 30`)

Configs: `configs/cnn_wider_dropout_plant_village_full_dataset_early_stopping_seed{42,43,44}_augmentation_on_num_workers6` and `configs/cnn_wider_dropout_plant_village_full_dataset_early_stopping_seed{42,43,44}_augmentation_off_num_workers6`

## Results (best validation-loss epoch)



## Observations


## Recommendation


## Limitations
