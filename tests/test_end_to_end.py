# Runs the whole pipeline once on tiny fake data: train, evaluate, export to ONNX and predict.
from unittest.mock import MagicMock

import train
from utils import make_fake_loader

CLASS_NAMES = ["a", "b", "c", "d", "e"]


def fake_get_loaders(**kwargs):
    return make_fake_loader(), make_fake_loader(), make_fake_loader(), CLASS_NAMES


def test_end_to_end_pipeline(monkeypatch, tmp_path):
    cfg = {
        "seed": 0,
        "data": {
            "fraction": 1.0,
            "image_size": 32,
            "batch_size": 4,
            "normalize": "none",
            "augment": False,
        },
        "model": {
            "type": "BasicCNN",
            "channels": [8],
        },
        "train": {
            "lr": 0.001,
            "weight_decay": 0.0,
            "epochs": 1,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    }

    # Use fake data and configuration
    monkeypatch.setattr(train, "load_config", lambda path: cfg)
    monkeypatch.setattr(train, "get_loaders", fake_get_loaders)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)

    # Fake ClearML
    task = MagicMock()
    monkeypatch.setattr(train, "initialise_task", lambda **kwargs: (task, cfg))
    monkeypatch.setattr(train, "upload_checkpoint", MagicMock())
    monkeypatch.setattr(train, "close_task", MagicMock())

    # Train the model
    train.main("fake.yaml", train.NAMES[0], "debugging", "")

    # Find the saved checkpoint
    checkpoints = list((tmp_path / "checkpoints").glob("*_best.pt"))
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]