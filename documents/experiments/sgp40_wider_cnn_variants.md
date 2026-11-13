# SGP-40: Wider CNN Variants on ClearML

## Setup

All three models use the same settings agreed in #main-discussions, so only the channels change:

- Fraction 0.2, seed 42, image size 128, batch size 32
- 5 epochs, AdamW, lr 0.001, weight decay 0.01
- All runs were tracked in ClearML (project `Plant-Disease-Image-Classification`): [ClearML comparison](https://app.clear.ml/projects/93fd565121f041b38b7914098b4df984/compare-tasks;ids=b81a07a8dea94c2b9560d4589b6bdd3e,ddda6b13d5fc40e3b9e04024c932429d,d9572589657f4fe795da83688ca7ad5b/scalars/graph)

