# AI use

## leafapp/data.py (SGP-7, Illia Putintsev)

- Tool: Claude (Anthropic), used as a tutor and reviewer.
- `load_subset()`: code generated in full by Claude; I tested it.
- `split()`, `make_transforms()`, `make_batch_transform()`, `get_loaders()`: written by me from my prototype notebook (itself written with Claude's help). Claude reviewed each function and pointed out bugs, such as a hardcoded seed in the split and a wrong image size in the training transforms, which I fixed.
- Docstrings: mostly suggested by Claude.
- Approximate share of AI-generated code: 50%