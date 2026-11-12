import os
from pathlib import Path

import pytest

from leafapp.checkpoints import load_checkpoint

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