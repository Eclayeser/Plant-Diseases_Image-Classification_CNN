import pytest
import torch
from leafapp.models.cnn import BasicCNN
from leafapp.model_constructor import build_model

DATA_SPECS = {"image_size": 32}
CLASS_NAMES = ["a", "b", "c"]


def test_build_model_returns_basiccnn():
    settings = {"type": "BasicCNN", "channels": [8, 16]}
    model = build_model(settings, DATA_SPECS, CLASS_NAMES)
    assert isinstance(model, BasicCNN), "BasicCNN type should build a BasicCNN"


def test_build_model_uses_config_values():
    settings = {"type": "BasicCNN", "channels": [8, 16]}
    model = build_model(settings, DATA_SPECS, CLASS_NAMES)
    x = torch.randn(2, 3, 32, 32)
    assert model(x).shape == (2, len(CLASS_NAMES)), "Output size should match the number of classes"
    assert len(model.features) == 2, "Should build one conv block per channel entry"


def test_build_model_missing_type_raises():
    with pytest.raises(ValueError, match="model type"):
        build_model({"channels": [8]}, DATA_SPECS, CLASS_NAMES)


def test_build_model_unknown_type_raises():
    with pytest.raises(ValueError, match="NotAModel"):
        build_model({"type": "NotAModel", "channels": [8]}, DATA_SPECS, CLASS_NAMES)
