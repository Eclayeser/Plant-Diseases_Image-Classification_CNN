from pathlib import Path
import pytest
from leafapp.model_manager import ModelManager

class FakeClassifier:
    """Simulates an ONNX classifier without loading real weights."""

    def __init__(self, name):
        self.name = name
        self.calls = []

    def top_k(self, image, k=5):
        self.calls.append((image, k))
        return [(self.name, 1.0)]

def make_manager(tmp_path):
    """Create three fake models and track which ones are loaded."""
    model_dirs = {
        "Model 1": tmp_path / "model-1",
        "Model 2": tmp_path / "model-2",
        "Model 3": tmp_path / "model-3",
    }

    loaded = []

    def fake_loader(path):
        path = Path(path)
        loaded.append(path)
        return FakeClassifier(path.name)

    manager = ModelManager(
        model_dirs=model_dirs,
        loader=fake_loader,
    )

    return manager, model_dirs, loaded

def test_models_are_loaded_on_demand(tmp_path):
    """Creating the manager should not load all three models."""

    manager, model_dirs, loaded = make_manager(tmp_path)
    assert loaded == []
    prediction = manager.top_k("Model 1", object())
    assert prediction == [("model-1", 1.0)]
    assert loaded == [model_dirs["Model 1"]]

def test_reusing_model_does_not_reload_it(tmp_path):
    """Repeated predictions with the same model reuse its session."""

    manager, model_dirs, loaded = make_manager(tmp_path)

    manager.top_k("Model 1", object())
    manager.top_k("Model 1", object())

    assert loaded == [model_dirs["Model 1"]]

def test_switching_models_loads_correct_weights(tmp_path):
    """Each selection must use its corresponding model."""

    manager, model_dirs, loaded = make_manager(tmp_path)

    first = manager.top_k("Model 1", object())
    second = manager.top_k("Model 2", object())
    third = manager.top_k("Model 3", object())

    assert first == [("model-1", 1.0)]
    assert second == [("model-2", 1.0)]
    assert third == [("model-3", 1.0)]

    assert loaded == [
        model_dirs["Model 1"],
        model_dirs["Model 2"],
        model_dirs["Model 3"],
    ]

def test_switching_back_reloads_previous_model(tmp_path):
    """Only the active classifier should be cached."""

    manager, model_dirs, loaded = make_manager(tmp_path)

    manager.top_k("Model 1", object())
    manager.top_k("Model 2", object())
    manager.top_k("Model 1", object())

    assert loaded == [
        model_dirs["Model 1"],
        model_dirs["Model 2"],
        model_dirs["Model 1"],
    ]

def test_unknown_model_is_rejected(tmp_path):
    manager, _, loaded = make_manager(tmp_path)
    with pytest.raises(ValueError, match="Unknown model"):
        manager.top_k("Model 4", object())
    assert loaded == []