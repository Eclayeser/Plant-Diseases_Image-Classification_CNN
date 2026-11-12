"""
Test 1: Verify that a trained CNN checkpoint can be loaded.
Test 2: Verify that the CNN processes an image and returns valid probabilities.
Test 3: Verify that Gradio's predict() function returns the top five classifications.
"""

import importlib
import os
from pathlib import Path
import pandas as pd
import pytest
import torch
from PIL import Image
from leafapp.checkpoints import load_checkpoint
from leafapp.data import make_transforms


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained_model():
    """Load the specified checkpoint or the latest available one."""
    checkpoint = os.getenv("CHECKPOINT_PATH")

    if checkpoint:
        path = Path(checkpoint).expanduser()
        if not path.is_absolute():
            path = ROOT / path
    else:
        checkpoints = list((ROOT / "outputs/models").glob("*.pt"))
        assert checkpoints, "No trained checkpoint found."
        path = max(checkpoints, key=lambda p: p.stat().st_mtime)

    assert path.is_file(), f"Checkpoint not found: {path}"
    return load_checkpoint(path)


@pytest.fixture(scope="module")
def inference_setup(trained_model):
    """Share the CNN and preprocessing between inference tests."""
    model, class_names, _, config = trained_model

    _, transform = make_transforms(
        config["data"]["image_size"],
        augment=False,
        normalize=config["data"].get("normalize", False),
    )

    return model, class_names, transform


def test_checkpoint_loads(trained_model):
    """Test 1: Confirm that the trained model loads correctly."""
    model, class_names, _, config = trained_model

    assert model is not None
    assert not model.training
    assert len(class_names) > 0
    assert config["data"]["image_size"] > 0


def test_cnn_inference(inference_setup):
    """Test 2: Confirm that the CNN produces valid probabilities."""
    model, class_names, transform = inference_setup

    image = Image.new("RGB", (256, 256), color="green")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

    assert logits.shape == (1, len(class_names))
    assert torch.isfinite(probabilities).all()
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1e-5)


def test_gradio_cnn_integration(inference_setup, monkeypatch):
    """Test 3: Confirm that Gradio uses the CNN for predictions."""
    model, class_names, transform = inference_setup

    ui = importlib.import_module("app.app")
    monkeypatch.setattr(ui, "MODEL", model)
    monkeypatch.setattr(ui, "CLASS_NAMES", class_names)
    monkeypatch.setattr(ui, "TRANSFORM", transform)

    assert ui.demo.fn is ui.predict

    image = Image.new("L", (256, 256), color=120)
    predictions = ui.predict(image)

    assert isinstance(predictions, pd.DataFrame)
    assert list(predictions.columns) == ["Disease", "Probability"]
    assert len(predictions) == min(5, len(class_names))

    assert predictions["Disease"].notna().all()
    assert predictions["Probability"].between(0, 1).all()
    assert predictions["Probability"].is_monotonic_decreasing