import pytest
import torch
from torch import nn

from leafapp.data import IMAGENET_MEAN, IMAGENET_STD, PLANT_VILLAGE_MEAN, PLANT_VILLAGE_STD
from scripts.export_onnx import build_export_model


@pytest.mark.parametrize("normalize, mean, std", [
    ("imagenet", IMAGENET_MEAN, IMAGENET_STD),
    (True, IMAGENET_MEAN, IMAGENET_STD),        # older configs
    ("plant_village", PLANT_VILLAGE_MEAN, PLANT_VILLAGE_STD),
])
def test_build_export_model_prepends_normalization(normalize, mean, std):
    x = torch.rand(1, 3, 8, 8)
    out = build_export_model(nn.Identity(), normalize)(x)
    expected = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    assert torch.allclose(out, expected)


def test_build_export_model_without_normalization_leaves_input_unchanged():
    x = torch.rand(1, 3, 8, 8)
    model = build_export_model(nn.Identity(), False)
    assert torch.equal(model(x), x)
    assert not model.training
