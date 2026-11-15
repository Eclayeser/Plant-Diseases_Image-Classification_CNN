# SGP-5 — Local Deployment

## Objective
Deploy the plant disease classification prototype locally using SGP-4's Gradio interface application publicly accessible.

## Implementation
- Train a CNN using `train.py`.
- Load a selected `.pt` checkpoint through `CHECKPOINT_PATH`.
- Run the Gradio interface at `http://localhost:8080`.
- Support macOS/Linux and Windows.
- Verify checkpoint loading, inference, and Gradio integration through automated tests.

## Documentation
See [Local Deployment Guide](../deployment/LOCAL_DEPLOYMENT.md) for setup and launch instructions.

## Status
Local deployment implemented and tested.