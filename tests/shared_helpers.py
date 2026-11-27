# Helpers shared by several test files.
from unittest.mock import MagicMock

import torch
from torchvision import models

from leafapp.models.cnn import BasicCNN
from leafapp.tracking import NAMES

VALID_NAME = NAMES[0]

# The fake loaders, the tiny model and the class names have to agree with each other.
FAKE_CLASS_NAMES = ["a", "b", "c", "d", "e"]
FAKE_IMAGE_SIZE = 32


def make_fake_loader():
    return [
        {
            "pixels": torch.randn(4, 3, FAKE_IMAGE_SIZE, FAKE_IMAGE_SIZE),
            "label": torch.randint(0, len(FAKE_CLASS_NAMES), (4,)),
        }
        for _ in range(2)
    ]


def fake_get_loaders(**kwargs):
    """Stands in for get_loaders: train, validation and test loaders plus the class names."""
    return make_fake_loader(), make_fake_loader(), make_fake_loader(), FAKE_CLASS_NAMES


def make_tiny_model(num_classes=len(FAKE_CLASS_NAMES)):
    return BasicCNN(num_classes=num_classes, channels=(8,), image_size=FAKE_IMAGE_SIZE)


def make_train_cfg(checkpoint_dir, epochs=1, data=None, model=None):
    """A small training config; data and model entries are added to the defaults."""
    return {
        "seed": 0,
        "data": {"fraction": 1.0, "image_size": FAKE_IMAGE_SIZE, "batch_size": 4, **(data or {})},
        "model": {"type": "BasicCNN", **(model or {})},
        "train": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": epochs,
            "checkpoint_dir": str(checkpoint_dir),
        },
    }


def patch_train_basics(monkeypatch, cfg, task=None):
    """Make train.main use cfg, the fake loaders, the CPU and a fake ClearML task.

    get_loaders and initialise_task are MagicMocks, so tests can read what they were
    called with, e.g. train.get_loaders.call_args.kwargs.
    """
    import train  # imported here so files that don't test train.py don't load it

    task = task or MagicMock()
    monkeypatch.setattr(train, "load_config", lambda path: cfg)
    monkeypatch.setattr(train, "get_loaders", MagicMock(side_effect=fake_get_loaders))
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train, "initialise_task", MagicMock(side_effect=lambda **kwargs: (task, kwargs["config"])))
    return task


def expect_resnet18_weights(monkeypatch, expected, message):
    """Fail if torchvision's resnet18 is asked for other weights, without downloading any."""
    real_resnet18 = models.resnet18

    def check_weights(weights=None):
        assert weights == expected, message
        return real_resnet18(weights=None)

    monkeypatch.setattr(models, "resnet18", check_weights)
