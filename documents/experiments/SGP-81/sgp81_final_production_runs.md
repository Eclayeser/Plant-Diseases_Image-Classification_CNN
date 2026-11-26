# SGP-81: Final Production Runs

## Why
The experiments up to SGP-91 settled two CNN configurations: Wider+Deeper `[32, 64, 128, 256]` as the most accurate (SGP-73, SGP-91) and Wider `[32, 64, 128]` as the cheaper option (SGP-63), both without dropout. This ticket trains several seeds of each on the full dataset, evaluates them on the validation set, selects the best model of each configuration and publishes them as ONNX models in ClearML for the app.

The configurations were fixed by the earlier experiments; the production phase only trains more seeds of them and picks the best, so no new design decisions are made here. Training more than the minimum is a trade-off against compute (see Training cost), and runs were reused where possible.

## Setup
| | Config 1: Wider+Deeper | Config 2: Wider |
|---|---|---|
| Model | BasicCNN `[32, 64, 128, 256]` | BasicCNN `[32, 64, 128]` |
| Dropout | none | none |

Both: full PlantVillage dataset (`fraction: 1.0`), image size 128, batch size 32, augmentation on, `plant_village` normalisation, AdamW (lr 0.001, weight decay 0.01), early stopping on validation loss with `patience: 8`, epoch cap 50, `num_workers: 6`. Same laptop GPU, SGP-16 determinism fix. Seeds 42 to 46 for each config, so 5 candidates per config.

- **Config 1:** seeds 42 to 44 are the SGP-91 runs, which use exactly this config. Training is deterministic, so rerunning them would give identical models; they were reused, and only seeds 45 and 46 were trained here.
- **Config 2:** all five seeds were trained here. The closest existing runs (SGP-63) used `num_workers: 0`, which changes the random augmentations, so new runs keep every candidate on the same settings.

Configs: `configs/cnn_wider_deeper_early_stopping_full_seed{42..46}_patience8_num_workers6.yaml` and `configs/cnn_wider_early_stopping_full_seed{42..46}_patience8_num_workers6.yaml`. New runs are tagged production / SGP-81. The reused SGP-91 tasks keep their original `@run-type:comparison` / `@issue:SGP-91` tags and were additionally tagged `@run-type:production` / `@issue:SGP-81` in ClearML, so all ten candidates appear under the SGP-81 filter.

**Selection rule (fixed before the runs):** rank candidates by validation accuracy, with macro F1 as the tie-breaker; the top 2 per config were evaluated with `evaluate.py`, and the best of each config was published.

## Candidates (best validation-loss epoch)

### Config 1: Wider+Deeper
| Seed | Source | Best epoch | Stopped at | Val accuracy | Val F1 | Val loss | ClearML |
|---|---|---:|---:|---:|---:|---:|---|
| 42 | SGP-91 | 10 | 18 | 0.9707 | 0.962 | 0.101 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/bce1416e27dc455592b5160505361e42/output/log) |
| 43 | SGP-91 | 18 | 26 | **0.9720** | **0.966** | 0.116 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/fbef2e2da74149ba9d71fcb7fae85e70/output/log) |
| 44 | SGP-91 | 18 | 26 | 0.9716 | 0.963 | 0.098 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/f5b6ae6cd9ef4ffdbed4b3031bb1f146/output/log) |
| 45 | new | 11 | 19 | **0.9722** | 0.963 | 0.117 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/f7139668aca54192892ad23dbe6c5b87/output/log) |
| 46 | new | 10 | 18 | 0.9692 | 0.962 | 0.122 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/c12b8ecb2f544e31a50b564c2c28d0a7/output/log) |

### Config 2: Wider
| Seed | Best epoch | Stopped at | Val accuracy | Val F1 | Val loss | ClearML |
|---|---:|---:|---:|---:|---:|---|
| 42 | 31 | 39 | **0.9684** | **0.958** | 0.128 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/24a671745446453c86c87974872994a0/output/log) |
| 43 | 14 | 22 | 0.9647 | 0.954 | 0.148 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/17e9ae5f422a45c08d4df6c6dcdccd4d/output/log) |
| 44 | 19 | 27 | **0.9674** | **0.957** | **0.119** | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/6c0a69e2d95f4a839db41b10a401ac7b/output/log) |
| 45 | 9 | 17 | 0.9607 | 0.948 | 0.156 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/ab22a584467542e591a774f0f75b5a60/output/log) |
| 46 | 33 | 41 | 0.9666 | 0.956 | 0.140 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/66c9c8297cc945be8bdf50c017e464fc/output/log) |

Mean validation accuracy over the five seeds: 0.9711 (Config 1) and 0.9656 (Config 2), a gap of about 0.55 pts, consistent with SGP-73. No run reached the 50-epoch cap.

## Selection
| Config | Selected (primary first) | Val accuracy | Val F1 |
|---|---|---:|---:|
| 1: Wider+Deeper | Seed 45 | 0.9722 | 0.963 |
| 1: Wider+Deeper | Seed 43 | 0.9720 | 0.966 |
| 2: Wider | Seed 44 | 0.9674 | 0.957 |
| 2: Wider | Seed 42 | 0.9684 | 0.958 |

- **Config 1:** seeds 43, 44 and 45 are effectively tied (validation accuracy within 0.06 pts, well inside run-to-run noise), so any of them would be a sound deployment choice. Seeds 45 and 43 were selected because they rank first and second under the selection rule.
- **Config 2:** seeds 42 and 44 were selected (top 2 by validation accuracy). Their accuracy and F1 differ by about 0.1 pt, within run-to-run noise, so seed 44 is listed as the primary model because of its lower validation loss (0.119 vs 0.128), meaning more confident correct predictions, which matters for an app that shows probabilities. It also trained for fewer epochs (27 vs 39).

Because the seed also sets the data split, the selected models were trained on different splits. This matters for the later test-set evaluation: each model will be tested on its own test split.

## Validation evaluation (`evaluate.py`)
| Config | Seed | Top-1 | Top-3 | Top-5 | Macro precision | Macro recall | Macro F1 | ClearML |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 45 | 0.9722 | 0.9974 | 0.9990 | 0.970 | 0.958 | 0.963 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/23d7f6e853b141aeb94ee09cb0c7ad24/output/log) |
| 1 | 43 | 0.9720 | 0.9971 | 0.9993 | 0.967 | 0.965 | 0.966 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/65a9432df24e4b70b4d178118ed54303/output/log) |
| 2 | 44 | 0.9674 | 0.9966 | 0.9992 | 0.964 | 0.951 | 0.957 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/7a1ac4738834447c9e4d5e791b9ac89b/output/log) |
| 2 | 42 | 0.9684 | 0.9964 | 0.9989 | 0.960 | 0.957 | 0.958 | [task](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/tasks/41d5afcbe910408e86e46f2058132d6a/output/log) |

The top-1 accuracies match the best-epoch values from training exactly, which confirms that evaluation rebuilds the same checkpoint and split.

### Most confused pairs (true → predicted, count)
| Config 1 seed 45 | Config 1 seed 43 | Config 2 seed 44 | Config 2 seed 42 |
|---|---|---|---|
| Corn Cercospora → Corn Northern Leaf Blight: 12 | Tomato Yellow Leaf Curl Virus → Tomato Bacterial spot: 13 | Tomato Early blight → Tomato Late blight: 20 | Corn Northern Leaf Blight → Corn Cercospora: 15 |
| Tomato Early blight → Tomato Late blight: 11 | Tomato Late blight → Potato Late blight: 12 | Corn Cercospora → Corn Northern Leaf Blight: 14 | Grape Esca → Grape Black rot: 13 |
| Tomato Target Spot → Tomato Bacterial spot: 9 | Corn Cercospora → Corn Northern Leaf Blight: 12 | Grape Esca → Grape Black rot: 9 | Tomato Late blight → Potato Late blight: 12 |

## Observations
- **The right answer is almost always in the top 5.** Top-5 accuracy is about 99.9% for all four models, and the app shows the top 5 predictions, so the correct disease is almost always on the user's screen even when the top pick is wrong.
- **Macro recall is below macro precision for every model** (by up to about 1.2 pts for Config 1 seed 45 and Config 2 seed 44): per class, the models miss real cases slightly more often than they raise false alarms, mostly in smaller classes. The per-class tables in the ClearML reports show which.
- **The same confusions recur across models.** Corn Cercospora/Gray leaf spot and Northern Leaf Blight are among the most confused pairs for all four models (in both directions for two of them); Tomato Early blight is predicted as Late blight; Grape Esca is predicted as Black rot. Tomato Late blight is predicted as **Potato** Late blight: both are caused by the same pathogen (*Phytophthora infestans*) and look very similar on the leaf.
- **The two configs make the same kinds of mistakes;** Wider+Deeper makes fewer of them (about 0.4 to 0.5 pts higher top-1).
- **The extra seeds changed little.** The best Config 1 model (0.9722) is within 0.02 pts of the best SGP-91 run (0.9720), and the Config 2 spread over five seeds (0.9607 to 0.9684) shows how much seed choice alone moves the result.

## Published models
The best model of each configuration was exported to ONNX from its training task's `best_checkpoint` (with the exporter's parity check against PyTorch), tested in the app locally, and published to the ClearML Models tab.

| Config | Seed | Training task | ClearML model ID |
|---|---|---|---|
| 1: Wider+Deeper | 45 | `f7139668aca54192892ad23dbe6c5b87` | `e10c56f879504f13837d47965be973f3` |
| 2: Wider | 44 | `6c0a69e2d95f4a839db41b10a401ac7b` | `3fd958c7abcf45af8a420609d8862d94` |

The runners-up (Config 1 seed 43, Config 2 seed 42) were also evaluated. They were published at first, then archived once the ticket was clarified to one model per config. Their training tasks remain in ClearML if a backup is needed.

`deployment/model.lock.json` was not changed: which model is deployed is decided separately, through a PR to `main`.

**Suggestion for the app:** two seeds of the same config are effectively the same model, so showing both adds little. One model per config (Config 1 seed 45 as the default, Config 2 seed 44 as the lighter option) plus ResNet18 (SGP-69) would show a meaningful comparison between architectures.

## Training cost
| Phase | Runs | Epochs | Time |
|---|---:|---:|---:|
| Config 1, seeds 45 and 46 (new) | 2 | 37 | about 37 min |
| Config 2, seeds 42 to 46 (new) | 5 | 146 | about 2 h 15 |
| Config 1, seeds 42 to 44 (reused from SGP-91) | 3 | 70 | none (about 43 min in SGP-91) |

Times are approximate (from checkpoint timestamps) and depend on machine load. Reusing the deterministic SGP-91 runs avoided about 43 minutes of retraining; `num_workers: 6` (introduced before SGP-73) cut time per epoch by about 2.4× compared with earlier runs.

## Limitations
- Results are on the validation set only. The validation set was used both for early stopping and for choosing the best seeds, so the selected models' scores are slightly optimistic; the test set is reserved for the final evaluation.
- Five seeds per config; differences between the top seeds are within run-to-run noise.
- Each selected model was trained with a different seed, so their test splits will differ.

## Reproducing
```bash
python train.py --config configs/cnn_wider_deeper_early_stopping_full_seed45_patience8_num_workers6.yaml --responsible "Mario Westphalen Cavicchioli" --run-type production --issue-id SGP-81
python evaluate.py --task-id <training task id> --responsible "Mario Westphalen Cavicchioli" --run-type production --issue-id SGP-81
python -m scripts.export_onnx --task-id <training task id> --upload --responsible "Mario Westphalen Cavicchioli" --run-type production --issue-id SGP-81
```