# Meeting number 10 - Summary
## Date: *26/11/2026*
## Type: *In-Person*
## Duration: *1 hour*


### Attendees (7/7):
- Illia, Mark, Mario, Pranay, Javier, Reon, Nil


### Completed Issues
Some of the completed issues include:
- Ensure high code quality of the codebase (SGP-56), Make displayed matrixes easier to analyse (SGP-59), Apply the connected configuration before training setup (SGP-35), Train and test Wider+Deeper Regularised CNN with full-data on validation set (SGP-73), Write the test strategy (SGP-84), Responsible AI features in the app (SGP-88), Write the architecture document with a component diagram (SGP-85), Reject epoch < 1 before training (SGP-36), Best checkpoint can be stale when a failure occurs during epoch reporting (SGP-67), Add an end-to-end pipeline test (SGP-90), Dropout vs no dropout on Wider+Deeper CNN (full data) (SGP-91), Improve code modularity and dependencies (SGP-65)


### Overview
A quick catch-up meeting to track progress one week before the final submission. The agenda was to check where every open ticket stands and to agree the last deadlines for the code.
Everyone reported what they had done since Monday. The Wider+Deeper CNN was trained on the full data set and beat the Wider CNN on every seed (SGP-73), and the dropout comparison showed that it does not need dropout (SGP-91). These results fixed the two CNN configurations for the final production runs (SGP-81), which were in progress together with the ResNet18 full-data runs (SGP-69). The architecture document (SGP-85), the test strategy (SGP-84), the Responsible AI features in the app (SGP-88) and the end-to-end pipeline test (SGP-90) were completed, along with code quality work and several bug fixes (SGP-56, SGP-65, SGP-35, SGP-36, SGP-67).
We agreed a code freeze for Sunday at 10 pm, with all pull requests merged by then, and to finish the clean-up by Monday at 10 pm. This leaves the last days for the report, the demo video and the submission itself.
We also agreed that the requirements and success criteria (SGP-89) have to be written before the final models are evaluated on the test set (SGP-74), so the criteria are fixed before the results are known.
We went through the remaining tickets and their owners. The comparison of the three final models (SGP-92) was cancelled as a separate ticket because it is covered by the test-set evaluation (SGP-74).
Everyone reported that they are aware of their next duties.



### Decisions and Items completed:
1. Everyone had a short stand-up: "what issues they completed / which issues they were still working on", "any comments regarding their issue"
2. Agreed a code freeze on Sunday 29/11 at 10 pm: all pull requests are merged by then
3. Agreed to finish the clean-up by Monday 30/11 at 10 pm
4. Agreed that the requirements and success criteria (SGP-89) are written before the test-set evaluation (SGP-74)
5. Cancelled SGP-92: the comparison of the three final models is done as part of SGP-74
6. Went through the open tickets and assigned the remaining ones (SGP-70, SGP-87, SGP-89)
7. Two new issues were opened after the meeting: large images exceeding Render's memory limit (SGP-93) and pytest coverage improvement (SGP-94)


### New TODOs - Cycle 5
- **Final Production Runs**, Mario, SGP-81
- **Evaluate the final models using testing data**, Mario, SGP-74
- **Train ResNet18 on the full data for the transfer-learning comparison**, Reon, SGP-69
- **Exploratory data analysis and data card**, Reon, SGP-70
- **Write requirements and success criteria**, Illia, SGP-89
- **Responsible AI analysis and model card**, Illia, SGP-83
- **Complete "Data Preparation" section of Final Report**, Illia, SGP-75
- **Additional UI feature - Model Selector**, Pranay, SGP-54
- **Additional UI feature - Flag wrong predictions**, Pranay, SGP-87
- **"Introduction" section of Final Report**, Pranay, SGP-57
- **Additional UI feature - Batch prediction**, Nil, SGP-71 *(in review)*
- **Refactor and prepare for submission - README.md**, Nil, SGP-62
- **Additional UI feature - Show training examples of the predicted class**, Mark, SGP-72
- **Large images can exceed Render's memory limit**, Mark, SGP-93
- **Pytest coverage improvement**, Mark, SGP-94
- **"Project Management" section - Report**, Mark, SGP-61
- **Improve code layout, docstrings, and comments**, Javier, SGP-66
- **Record a submission-ready Demo Video**, *Unassigned*, SGP-80
- **Complete "User Interface and Deployment" section of Final Report**, *Unassigned*, SGP-76 *(backlog)*
- **Complete "Model Development and Training" section of Final Report**, *Unassigned*, SGP-77 *(backlog)*
- **Complete "Experimental Evaluation" section of Final Report**, *Unassigned*, SGP-78 *(backlog)*
- **Write an `ai-use-statement.md`, submission-ready**, *Unassigned*, SGP-82 *(backlog)*
- **FINAL PROJECT SUBMISSION**, *Unassigned*, SGP-79 *(backlog)*