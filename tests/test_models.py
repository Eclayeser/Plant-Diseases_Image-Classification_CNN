import torch
from leafapp.models.cnn import BasicCNN
from leafapp.models.resnet import build_resnet18
from PIL import Image
from leafapp.data import make_transforms


def test_basiccnn_output_shape():
    model = BasicCNN()
    x = torch.randn(8, 3, 128, 128)
    out = model(x)
    assert out.shape == (8, 39), "BasicCNN should output [8, 39] logits"


def test_resnet_output_shape():
    model = build_resnet18()
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 39), "ResNet18 should output [2, 39] logits"


def test_normalize_gives_imagenet_values():
    pic = Image.new("RGB", (8, 8), color=(102, 204, 51))
    train_tf, eval_tf = make_transforms(8, normalize=True)
    x = eval_tf(pic)
    assert torch.allclose(
        x[:, 0, 0], torch.tensor([-0.37, 1.54, -0.92]), atol=0.01
    ), "Normalize should give the ImageNet values"
