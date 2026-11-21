import json
from pathlib import Path


def load_model_registry(path):
    """Read the model configuration and validate selectable models."""
    path = Path(path)

    with path.open(encoding="utf-8") as file:
        registry = json.load(file)

    default_model_id = registry["model_id"]

    if "models" not in registry:
        registry["models"] = [
            {"name": "Default", "model_id": default_model_id}
        ]

    models = registry["models"]

    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")

    names = []
    model_ids = []

    for model in models:
        if not isinstance(model, dict):
            raise ValueError("each model must be an object")

        name = model.get("name")
        model_id = model.get("model_id")

        if not isinstance(name, str) or not name.strip():
            raise ValueError("each model must have a valid name")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("each model must have a valid model_id")

        names.append(name)
        model_ids.append(model_id)

    if len(names) != len(set(names)):
        raise ValueError("duplicate model names are not allowed")
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("duplicate model IDs are not allowed")
    if default_model_id not in model_ids:
        raise ValueError("default model must be registered")
    return registry