"""
Test 1: Verify that a trained CNN checkpoint can be loaded.
Test 2: Verify that the CNN processes an image and returns valid probabilities.
Test 3: Verify that Gradio's predict() function returns the top five classifications.
Test 4: Proves whether Gradio automatically loads the specified checkpoint
"""

import importlib
import os
import pandas as pd
import pytest
import torch
from PIL import Image
from leafapp.checkpoints import load_checkpoint, save_checkpoint
from leafapp.models.cnn import BasicCNN
from leafapp.data import make_transforms
import subprocess
import sys
from leafapp.config import ROOT


@pytest.fixture(scope="module")
def checkpoint_path(tmp_path_factory):
    """Create a temporary CNN checkpoint for testing."""
    config = {
        "model": {"type": "BasicCNN", "channels": [8]},
        "data": {"image_size": 32, "normalize": False},
    }
    class_names = ["healthy", "rust", "blight"]

    model = BasicCNN(
        num_classes=len(class_names),
        channels=config["model"]["channels"],
        image_size=config["data"]["image_size"],
    )

    path = tmp_path_factory.mktemp("sgp5") / "test_model.pt"
    save_checkpoint(path, model, config, class_names, history_metrics={})

    return path


@pytest.fixture(scope="module")
def trained_model(checkpoint_path):
    """Load the temporary checkpoint."""
    return load_checkpoint(checkpoint_path)


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


def test_checkpoint_loaded_on_startup(checkpoint_path):
    """Test 4: Verify that Gradio loads the checkpoint at startup."""

    # Use the temporary checkpoint created by pytest.
    env = os.environ.copy()
    env["CHECKPOINT_PATH"] = str(checkpoint_path)

    # Start a fresh Python process to test app initialization.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib

ui = importlib.import_module("app.app")

assert ui.MODEL is not None
assert ui.CLASS_NAMES
assert ui.TRANSFORM is not None
assert not ui.MODEL.training
assert ui.demo.fn is ui.predict
""",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, result.stdout + result.stderr