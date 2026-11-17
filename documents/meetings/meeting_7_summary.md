# Meeting number 7 - Summary
## Date: *16/11/2026*
## Type: *In-Person*
## Dyration: *1 hour*


### Attendees (7/7):
- Illia, Mark, Mario, Pranay, Javier, Reon, Nil


### Completed Issues
Some of the completed issues include:
- Make sure that ClearML uses the seed from the config (SGP-42), Expose augmentation parameter in the config (SGP-32), Upgrade tag logic for ClearML (SGP-30), Trigger pytests automatically on push (SGP-9), Implement early stopping when validation loss stops improving (SGP-22), Add ImageNet and plant_village normalisation to the wider CNN and compare them (SGP-31), Make optimiser a config option and justify chosen optimiser (SGP-27), Regularised CNN prototype (SGP-14), Soften noise claim in SGP-27 optimiser report (SGP-41), Compare wider CNN variants using ClearML (SGP-40), Remove checkpoint .pt files from documents/experiments/SGP-15 (SGP-15)


### Overview
According to the Agenda, everyone reported what they did in the past week. We were notified today that TeamFeedback was finally up, so we spent some time setting it up and making sure that everyone is on the same page regarding its use.
We continue to focus on experimenting, and we have almost completed tuning for all hyperparameters, except the augmentation option. Therefore, we expect to be finished exploring and have started training heavy models on bigger fractions of data by the end of this week.
Since we are getting closer to the end, we agreed to kickstart the report and code-cleaning practices (therefore, several tickets have been promoted from the "backlog" to "todo", which mostly focus on minor bug-fixing and clean-ups).
No major concerns were expressed, and everyone reported that they are aware of their next duties.


### Decisions and Items completed:
1. Everyone had a short stand-up: "what issues they completed / which issues they were still working on", "any comments regarding their issue"
2. Everyone completed setting up their account on Team Feedback + linked the GitHub repo to Team Feedback
3. Analysed the criteria list once more to identify gaps and next steps for meeting full requirements
4. Multiple issues created/moved from the backlog for the upcoming cycle


### New TODOs - Cycle 4
- **Make cuDNN deterministic in train.py so same-seed GPU runs match**, *Unassigned*, SGP-16
- **Add pytest coverage report for leafapp and app, run in CI**, *Unassigned*, SGP-25
- **Skip ImageNet weight download when loading ResNet18 from a checkpoint**, *Unassigned*, SGP-26
- **Move prediction logic from app/app.py into leafapp/predict.py**, *Unassigned*, SGP-17
- **Google Colab Deployment**, *Unassigned*, SGP-46
- **Conduct model training with Augmentation On/Off - Produce Analysis**, Pranay, SGP-45
- **.docm Report - Build foundation with Placeholders**, Mark, SGP-47

