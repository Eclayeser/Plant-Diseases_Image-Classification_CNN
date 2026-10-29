# Plant-Diseases_Image-Classification_CNN

Our aim is to classify plant diseases from leaf images using a PyTorch CNN and a Gradio interface.
The repository currently contains a project skeleton, with purpose comments to guide future implementation.

Shared Python code belongs in `leafapp/`, the interface in `app/`, and configuration files in `configs/`.
The `train.py` and `evaluate.py` files are entry points for training and evaluating the model.
The `notebooks/`, `tests/`, and `docs/` folders hold exploratory work, automated checks, and project documentation.

The prototype and final product will share the same codebase.
Prototype settings will be kept in `configs/prototype.yaml`, and ClearML tags will identify prototype runs.
A `v0.1-prototype` Git tag will mark the version delivered as the prototype.

## Tools

Despite the **Technology Stack** used directly for product development, out team uses range of different software tools to accelerate the speed of our work and improve the code quality.

| Tool | Use | Why this tool | How to View |
|---|---|---|---|
| **Linear** | Issue tracking, development cycles management | Easy to learn and intuitive to use; commonly used across businesses; some team memebrs have previous experience using this tool; Free tier available; supports integration with Github | Join workspace via a link (cannot be shared publicly since no read-only access available; request one of the team members for access) |
| **Slack** | Real-time (via Huddles) and asynchronousteam communication (via different dedicated channels) | Free to use; supports integration with Linear; commonly used across businesses; some team memebrs have previous experience using this tool; highly customiseable which allows for clear information exchange | Join workspace via a link (cannot be shared publicly since no read-only access available; request one of the team members for access) |
| **Coderabbit AI**| AI-driven code reviews for PRs; integrated as a required check before PR can be merged | Catches bugs early and allows developers to debug quicklier without waiting for others to review; provides useful PR summaries; free to use and easy to integrate; common practice across most companies | N/A |