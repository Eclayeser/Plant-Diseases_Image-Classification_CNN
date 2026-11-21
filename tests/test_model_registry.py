import json
import pytest
from leafapp.model_registry import load_model_registry

def test_load_legacy_single_model(tmp_path):
    """Existing deployments with one model remain supported."""
    path = tmp_path / "model.lock.json"

    path.write_text(json.dumps({
        "model_id": "model-a",
    }))

    registry = load_model_registry(path)

    assert registry == {
        "model_id": "model-a",
        "models": [
            {"name": "Default", "model_id": "model-a"}
        ],
    }

def test_load_three_models(tmp_path):
    """A configuration can register three selectable models."""
    path = tmp_path / "model.lock.json"

    models = [
        {"name": "Model 1", "model_id": "model-a"},
        {"name": "Model 2", "model_id": "model-b"},
        {"name": "Model 3", "model_id": "model-c"},
    ]

    path.write_text(json.dumps({
        "model_id": "model-a",
        "models": models,
    }))

    registry = load_model_registry(path)

    assert registry["model_id"] == "model-a"
    assert registry["models"] == models

def test_reject_duplicate_model_names(tmp_path):
    """Dropdown options must have unique names."""
    path = tmp_path / "model.lock.json"

    path.write_text(json.dumps({
        "model_id": "model-a",
        "models": [
            {"name": "Model 1", "model_id": "model-a"},
            {"name": "Model 1", "model_id": "model-b"},
        ],
    }))

    with pytest.raises(ValueError, match="duplicate"):
        load_model_registry(path)

def test_default_model_must_be_registered(tmp_path):
    """The default model must also appear among selectable models."""
    path = tmp_path / "model.lock.json"

    path.write_text(json.dumps({
        "model_id": "model-a",
        "models": [
            {"name": "Model 2", "model_id": "model-b"}
        ],
    }))

    with pytest.raises(ValueError, match="default"):
        load_model_registry(path)