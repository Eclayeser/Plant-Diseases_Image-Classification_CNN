import os
from pathlib import Path

import pytest

from leafapp.checkpoints import load_checkpoint

import torch
from PIL import Image
from leafapp.data import make_transforms

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained_model():
    """Load a saved CNN checkpoint for testing."""
    specified = os.getenv("CHECKPOINT_PATH")

    if specified:
        path = Path(specified).expanduser()
        if not path.is_absolute():
            path = ROOT / path
    else:
        checkpoints = list((ROOT / "outputs/models").glob("*.pt"))
        assert checkpoints, "No trained checkpoint found."
        path = max(checkpoints, key=lambda p: p.stat().st_mtime)

    assert path.is_file(), f"Checkpoint not found: {path}"
    return load_checkpoint(path)


def test_checkpoint_loads(trained_model):
    """Verify that the trained CNN can be restored."""
    model, class_names, _, config = trained_model

    assert model is not None
    assert not model.training
    assert len(class_names) > 0
    assert config["data"]["image_size"] > 0


def test_cnn_inference(trained_model):
    """Verify that the CNN produces valid predictions from an image."""
    model, class_names, _, config = trained_model

    # Apply the same preprocessing used during evaluation.
    _, transform = make_transforms(
        config["data"]["image_size"],
        augment=False,
        normalize=config["data"].get("normalize", False),
    )

    # Create a synthetic image for testing.
    image = Image.new("RGB", (256, 256), color="green")
    input_tensor = transform(image).unsqueeze(0)

    # Run inference.
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

    # Validate model output.
    assert logits.shape == (1, len(class_names))
    assert torch.isfinite(probabilities).all()
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1e-5)