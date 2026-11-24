# Meeting number 9 - Summary
## Date: *23/11/2026*
## Type: *In-Person*
## Duration: *1 hour*


### Attendees (7/7):
- Illia, Mark, Mario, Pranay, Javier, Reon, Nil


### Completed Issues
Some of the completed issues include:
- Deploying to Render.com with some clean refactoring changes (SGP-49), Testing on full data set - v1 (SGP-52), Testing on full data set - v2 (SGP-53), Rerun full-data dropout comparison with `patience: 8` (SGP-63), Conduct model training with Augmentation On/Off - Produce Analysis (SGP-45), Add error analysis to evaluation (SGP-48), Check plant_village normalisation stats on the full training split (SGP-68), Add optional Global Average Pooling to BasicCNN (SGP-51), Additional UI feature - Make histogram look better (SGP-55), Additional UI feature - Example image gallery (SGP-64), Make cuDNN deterministic in train.py so same-seed GPU runs match (SGP-16), .docm Report - Build foundation with Placeholders (SGP-47), Skip ImageNet weight download when loading ResNet18 from a checkpoint (SGP-26), Use full name in tracking names (SGP-60)


### Overview
With 10 days left before the deadline, the agenda was to agree how we choose the final model and to make sure every part of the marking scheme is covered.
Everyone reported what they had done since Thursday. The full data set experiments (SGP-52, SGP-53, SGP-63), the augmentation analysis (SGP-45), the Render deployment (SGP-49) and several UI improvements (SGP-55, SGP-64) were completed.
We discussed th results of the full data set runs. Before choosing the final configuration, we agreed to train the wider + deeper regularised CNN on the full data set (SGP-73) and ResNet18 as a transfer-learning comparison (SGP-69). The final models will be selected on the validation set in the final production runs (SGP-81), and only then evaluated once on the test set, including robustness checks (SGP-74).
We then went through what is still missing for each part of the marking scheme and created tickets for the remaining report sections, documentation, Responsible AI, testing, UI features, the demo video, the AI use statement and the final submission.
Everyone reported that they are aware of their next duties.



### Decisions and Items completed:
2. Reviewed the the results received from full dataset runs; agreed to test the wider + deeper regularised CNN on the full data set before fixing the final configuration (SGP-73)
3. Agreed to train ResNet18 on the full data set for the transfer-learning comparison (SGP-69)
4. Agreed that the final models are chosen on validation results (SGP-81) and evaluated once on the test set afterwards (SGP-74)
5. Created tickets for everything still missing per marking criterion: report sections, documentation, Responsible AI, testing, UI features, demo video, AI use statement and final submission
6. Agreed that all team members review everything before the submission (SGP-79)


### New TODOs - Cycle 5
- **Train and test Wider+Deeper Regularised CNN with full-data on validation set**, Mario, SGP-73
- **Train ResNet18 on the full data for the transfer-learning comparison**, Reon, SGP-69
- **Final Production Runs**, Mario, SGP-81
- **Evaluate the final models using testing data**, Mario, SGP-74
- **Exploratory data analysis and data card**, Mark, SGP-70
- **Write the architecture document with a component diagram**, Mark, SGP-85
- **Write the test strategy**, Nil, SGP-84
- **Responsible AI analysis and model card**, Illia, SGP-83
- **Responsible AI features in the app**, Illia, SGP-88
- **Complete "Data Preparation" section of Final Report**, Illia, SGP-75
- **Complete "User Interface and Deployment" section of Final Report**, Nil, SGP-76
- **Complete "Model Development and Training" section of Final Report**, *Unassigned*, SGP-77
- **Complete "Experimental Evaluation" section of Final Report**, *Unassigned*, SGP-78
- **Add an end-to-end pipeline test**, *Unassigned*, SGP-90
- **Additional UI feature - Flag wrong predictions**, *Unassigned*, SGP-87
- **Write requirements and success criteria**, *Unassigned*, SGP-89 *(backlog)*
- **Additional UI feature - Batch prediction**, *Unassigned*, SGP-71 *(backlog)*
- **Additional UI feature - Show training examples of the predicted class**, *Unassigned*, SGP-72 *(backlog)*
- **Write an `ai-use-statement.md`, submission-ready**, *Unassigned*, SGP-82 *(backlog)*
- **Record a submission-ready Demo Video**, *Unassigned*, SGP-80 *(backlog)*
- **FINAL PROJECT SUBMISSION**, *Unassigned*, SGP-79 *(backlog)*
- **Ensure high code quality of the codebase**, Javier, SGP-56 *(in review)*
- **Make displayed matrixes easier to analyse**, Pranay, SGP-59 *(in review)*
- **Additional UI feature - Model Selector**, Pranay, SGP-54
- **"Introduction" section of Final Report**, Pranay, SGP-57
- **"Project Management" section - Report**, Mark, SGP-61
- **Refactor and prepare for submission - README.md**, *Unassigned*, SGP-62