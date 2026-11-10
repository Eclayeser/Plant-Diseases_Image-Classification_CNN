# Meeting number 5 - Summary
## Date: *09/11/2026*
## Type: *In-Person*
## Dyration: *1 hour*

### Attendees (7/7):
- Illia, Mark, Mario, Pranay, Javier, Reon, Nil

### Completed Issues
Some of the completed issues include:
- Create baseline Gradio UI (SGP-4), Implement reusable evaluation functions + create evalution main() caller (SGP-12), Add ResNet18 builder (SGP-13),
Enable different model building via config (SGP-23), Add --config argument to train.py (SGP-19), Switch evaluate.py CLI to argparse (SGP-20)

### Overview
Following the meeting agenda everyone discussed their progress: we have finished laying foundation for data preparation and model development - now, any work in this area is expandatory. The UI foundation has also been concluded, however cannot be tested properly until deployment ticket is finished (SGP-5). SGP-5 is currently being a long-standing blocker since the unforseen limitation of using Gradio on HuggingFace, so additional effort goes into figuring out a workaround.
Meanwhile, the ClearML integration foundation is almost fully complete and expected to be delivered in 1 day at most (SGP-3). Reon set up a single shared account for ClearML workspace in order to deal with another unforseen risk (ClearML would not let more than 3 people to join the same workspace on Free Tier)
Since we have the monitoring system integrated, we decided that our main focus for the next cycle is as follows:
- Improve the monitoring system to meet "must-to-complete" requirements and make experiment-run analysis more accessible and profound
- Everyone to start experimenting with different models and parameters, steadily narrowing down to the best model and its settings to choose for the final shipment

### Decisions and Items completed:
1. Everyone had a short stand-up: "what issue they completed / which issues they were still working on", "any comment regarding their issue"
2. Marking discussed. All team members have solidified their understanding about importance of their consistent project contributions
3. Summarised and went over full "must" requirements checklist: marked what has been completed, currently in progress and future todos
4. Multiple issues created for the upcoming cycle

### New TODOs - Cycle 3
- **Make optimiser a config option and justify chosen optimiser**, Mario, SGP-27
- **Fail loudly when a checkpoint file to upload is missing**, Mark, SGP-28
- **Upgrade model-storing logic for ClearML**, *Unassigned*, SGP-29
- **Upgrade tag logic for ClearML**, *Unassigned*, SGP-30
- **Explore Image Normalisation Requirement**, Illia, SGP-31
- **Expose augmentation parameter in the config**, *Unassigned*, SGP-32
- **evaluate.py run should also happen through ClearML**, *Unassigned*, SGP-33
