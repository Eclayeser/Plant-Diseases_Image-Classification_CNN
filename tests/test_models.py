import pytest
import torch
from torch import nn
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


@pytest.mark.parametrize("regularisation", ["none", "dropout", "batchnorm"])
def test_variants_support_forward_and_backward(regularisation):
    torch.manual_seed(42)

    model = BasicCNN(
        num_classes=5,
        channels=(8, 16),
        image_size=32,
        regularisation=regularisation,
        dropout_p=0.2,
    )
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])

    model.train()
    outputs = model(images)

    assert outputs.shape == (4, 5)
    assert torch.isfinite(outputs).all()

    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()

    first_conv_gradient = model.features[0][0].weight.grad
    assert first_conv_gradient is not None
    assert torch.isfinite(first_conv_gradient).all()
    assert torch.count_nonzero(first_conv_gradient) > 0


def test_dropout_matches_baseline_in_eval_mode():
    torch.manual_seed(42)

    settings = {
        "num_classes": 5,
        "channels": (8,),
        "image_size": 32,
    }
    baseline = BasicCNN(**settings)
    regularised = BasicCNN(
        **settings,
        regularisation="dropout",
        dropout_p=0.5,
    )

    # Use identical weights so any difference comes from dropout.
    regularised.load_state_dict(baseline.state_dict())
    images = torch.randn(4, 3, 32, 32)

    baseline.eval()
    regularised.eval()

    with torch.no_grad():
        expected = baseline(images)
        actual = regularised(images)

    torch.testing.assert_close(actual, expected)

    # Dropout becomes active again when training mode is restored.
    regularised.train()
    torch.manual_seed(42)

    with torch.no_grad():
        training_output = regularised(images)

    assert not torch.allclose(training_output, expected)


def test_batchnorm_statistics_update_only_during_training():
    torch.manual_seed(42)

    model = BasicCNN(
        num_classes=5,
        channels=(8,),
        image_size=32,
        regularisation="batchnorm",
    )
    batchnorm = next(
        layer
        for layer in model.modules()
        if isinstance(layer, nn.BatchNorm2d)
    )
    images = torch.randn(4, 3, 32, 32)
    initial_batches = batchnorm.num_batches_tracked.item()

    model.train()
    with torch.no_grad():
        model(images)

    assert batchnorm.num_batches_tracked.item() == initial_batches + 1

    saved_mean = batchnorm.running_mean.clone()
    saved_variance = batchnorm.running_var.clone()
    saved_batches = batchnorm.num_batches_tracked.clone()

    model.eval()
    with torch.no_grad():
        model(images)

    assert torch.equal(batchnorm.running_mean, saved_mean)
    assert torch.equal(batchnorm.running_var, saved_variance)
    assert torch.equal(batchnorm.num_batches_tracked, saved_batches)


def test_unknown_regularisation_is_rejected():
    with pytest.raises(ValueError, match="regularisation"):
        BasicCNN(regularisation="unknown")


@pytest.mark.parametrize("dropout_p", [-0.1, 1.0])
def test_invalid_dropout_probability_is_rejected(dropout_p):
    with pytest.raises(ValueError, match="dropout_p"):
        BasicCNN(
            regularisation="dropout",
            dropout_p=dropout_p,
        )
