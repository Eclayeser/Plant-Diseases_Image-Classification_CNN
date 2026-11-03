import pytest
import torch
from torch import nn
from leafapp.models.cnn import BasicCNN
from leafapp.eval_functions import loss_accuracy_per_loader
from utils import make_fake_loader
from train import train_one_epoch

def test_loss_accuracy_per_loader_does_not_update_weights():
    model = BasicCNN(num_classes=5, channels=(8,), image_size=32)
    criterion = nn.CrossEntropyLoss()

    before = [p.clone() for p in model.parameters()]
    loss_accuracy_per_loader(model, make_fake_loader(), criterion, torch.device("cpu"))
    after = list(model.parameters())

    unchanged = all(torch.equal(b, a) for b, a in zip(before, after))

    assert unchanged, "loss_accuracy_per_loader changed the model's weights"

def test_train_rejects_empty_loader():
    """train_one_epoch() raises a clear error when the loader is empty."""
    model = BasicCNN(num_classes=5, channels=(8,), image_size=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError):
        train_one_epoch(model, [], nn.CrossEntropyLoss(), optimizer, torch.device("cpu"))

def test_loss_accuracy_per_loader_rejects_empty_loader():
    """loss_accuracy_per_loader() raises a clear error when the loader is empty."""
    model = BasicCNN(num_classes=5, channels=(8,), image_size=32)
    with pytest.raises(ValueError):
        loss_accuracy_per_loader(model, [], nn.CrossEntropyLoss(), torch.device("cpu"))