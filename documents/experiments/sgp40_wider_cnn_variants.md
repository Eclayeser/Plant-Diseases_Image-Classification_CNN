# SGP-40: Wider CNN Variants on ClearML

## Setup

All three models use the same settings agreed in #main-discussions, so only the channels change:

- Fraction 0.2, seed 42, image size 128, batch size 32
- 5 epochs, AdamW, lr 0.001, weight decay 0.01
- All runs were tracked in ClearML (project `Plant-Disease-Image-Classification`): [ClearML comparison](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/compare-tasks;ids=b81a07a8dea94c2b9560d4589b6bdd3e,ddda6b13d5fc40e3b9e04024c932429d,d9572589657f4fe795da83688ca7ad5b/scalars/graph)

## Results (validation set, last epoch)

| Model | Config | Channels | Parameters | Val accuracy | Val F1 | Val loss | Checkpoint |
|---|---|---|---|---|---|---|---|
| Wider | `cnn_wider.yaml` | [32, 64, 128] | 1,371,239 | 0.833 | 0.773 | 0.550 | `cnn_wider_20261111-234519_53b550.pt` |
| Even wider | `cnn_even_wider.yaml` | [64, 128, 256] | 2,926,759 | 0.838 | 0.787 | 0.534 | `cnn_even_wider_20261111-232941_67ac4e.pt` |
| Wider + deeper | `cnn_wider_deeper.yaml` | [32, 64, 128, 256] | 1,027,431 | 0.823 | 0.776 | 0.584 | `cnn_wider_deeper_20261111-234033_42be13.pt` |

The checkpoints are in `documents/experiments/SGP-40/`.

![Validation accuracy](SGP-40/accuracy_validation.png)

![Validation loss](SGP-40/loss_validation.png)