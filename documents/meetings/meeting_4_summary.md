# Meeting number 4 - Summary
## Date: *04/11/2026*
## Type: *Online (Reading Week)*
## Dyration: *1 hour*

### Attendees (6/7):
- Illia, Mark, Mario, Pranay, Javier, Reon

### Completed Issues
Some of the completed issues include:
- Introduce the foundation for documentation maintenance. Prepare the ground for the deliverable report (SGP-8), Skeleton train loop (SGP-10)

### Overview
While mostly a catch-up meeting, we have discussed some important limitations and blockers regarding our work. Some limitations included lack of understanding whether we should use multiple credentials or single-super one for ClearML, and how to bypass paid subscription requirement for HF deployment with Gradio. Likewise, the blockers included Pranay requiring SGP-4 finished before efficient implementation of deployment, and Javier requiring SGP-10 and SGP-12 sorted before his implementation of regularised CNN.
Overall, most people continue working on the tasks they were assigned last week steadily.

### Decisions and Items completed:
1. Everyone had a short stand-up: "what issue they completed / which issues they were still working on", "any comment regarding their issue"
2. Limitations and risks discussed, including unforeseen limitation on deployment using Gradio unless paid subscription
3. As part of SGP-12, brainstormed and picked what metrics and methods should evaluation process include to allow for thorough analyses in future
4. Few issues created to accommodate those who will finish their tickets before next meeting

### TODOs - Cycle 2
*all issues unassigned and will be picked up by those who complete their current issues first*
- **Enable different model building via config**, SGP-23
- **Add --config argument to evaluate.py**, SGP-20
- **Add --config argument to train.py**, SGP-19
- **Save best run during training loop**, SGP-21