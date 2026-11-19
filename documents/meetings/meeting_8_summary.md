# Meeting number 8 - Summary
## Date: *19/11/2026*
## Type: *In-Person*
## Duration: *1 hour*


### Attendees (7/7):
- Illia, Mark, Mario, Pranay, Javier, Reon, Nil


### Completed Issues
Some of the completed issues include:
- Enable localhost deployment (SGP-5), Move prediction logic from app/app.py into leafapp/predict.py (SGP-17), Add pytest coverage report for leafapp and app, run in CI (SGP-25), `evaluate.py` run should also happen through ClearML (SGP-33), Compare CNN regularisation variants in ClearML (SGP-44), Upgrade model-storing logic for ClearML (SGP-29), Show used device at task startup (SGP-39)


### Overview
A quick catch-up meeting, as our experiment tracking stage on the small fraction of data (0.2) has mainly been completed. The agenda for the meeting was to discuss the results of the experiments and agree on the model architecture and parameters for the full data set run.
Everyone reported what they had done since the beginning of the week. We discussed the experiments and agreed on the parameters for testing on the full data set: wider CNN, plant_village normalisation, lr 0.001, AdamW and early stopping. We agreed to train the models on two different machines, with 3 different seeds, with and without dropout.
We assigned team members to start the report and code-cleaning practices. As the deadline is now only 2 weeks away, we also discussed ideas for the UI and extra features, such as which models users could choose from, and agreed to prepare more ideas and discuss them on Monday. We opened the first issues on UI features, with more to follow.
We also discussed using other metrics besides validation loss to choose the best checkpoint and created SGP-58 for it, with further discussion and a decision to follow.
The reminder to complete the Peer Assessment on Team Feedback was brought up as well.
Everyone reported that they are aware of their next duties.



### Decisions and Items completed:
1. Everyone had a short stand-up: "what issues they completed / which issues they were still working on", "any comments regarding their issue"
2. Agreed the settings for the full data set runs: wider CNN, plant_village normalisation, AdamW (lr 0.001), early stopping; dropout vs no dropout with 3 seeds on two machines (SGP-52, SGP-53)
3. Created SGP-58 to decide which metric(s) to use for best checkpoint selection
4. Discussed UI extras (model selector, nicer histogram); more ideas to be brought to Monday's meeting (SGP-54, SGP-55)
5. Agreed to start the report.


### New TODOs - Cycle 4
- **Deploying to Render.com with some clean refactoring changes**, Mark, SGP-49 *(in review)*
- **Add error analysis to evaluation**, Illia, SGP-48 
- **Testing on full data set - v1**, Mario, SGP-52
- **Testing on full data set - v2**, Reon, SGP-53
- **Improve the logic behind 'best_checkpoint.pt'**, Pranay, SGP-58
- **"Introduction" section of Final Report**, Pranay, SGP-57
- **Ensure high code quality of the codebase**, Javier, SGP-56
- **Additional UI feature - Make histogram look better**, Nil, SGP-55
- **Additional UI feature - Model Selector**, *Unassigned*, SGP-54
- **Make displayed matrixes easier to analyse**, *Unassigned*, SGP-59
- **Make cuDNN deterministic in train.py so same-seed GPU runs match**, Mario, SGP-16

