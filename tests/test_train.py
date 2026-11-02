# Checks the training process and model checkpoint handling.
import pytest
import torch
from torch import nn
from leafapp.models.cnn import BasicCNN
from train import train_one_epoch, evaluate

def make_fake_loader():
    return [
        {"pixels": torch.randn(4, 3, 32, 32), "label": torch.randint(0, 5, (4,))}
        for _ in range(2)
    ]

def test_train_one_epoch_returns_valid_metrics():
    model = BasicCNN(num_classes=5, channels=(8,), image_size=32)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss, acc = train_one_epoch(model, make_fake_loader(), criterion, optimizer, torch.device("cpu"))
    assert loss >= 0, "Negative loss"
    assert 0 <= acc <= 1, "Acc out of range"

def test_train_one_epoch_updates_weights():
    model = BasicCNN(num_classes=5, channels=(8,), image_size=32)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    before = [p.clone() for p in model.parameters()]
    train_one_epoch(model, make_fake_loader(), criterion, optimizer, torch.device("cpu"))
    after = list(model.parameters())

    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "train_one_epoch did not update any weights"

def test_evaluate_does_not_update_weights():
    model = BasicCNN(num_classes=5, channels=(8,), image_size=32)
    criterion = nn.CrossEntropyLoss()

    before = [p.clone() for p in model.parameters()]
    evaluate(model, make_fake_loader(), criterion, torch.device("cpu"))
    after = list(model.parameters())

    unchanged = all(torch.equal(b, a) for b, a in zip(before, after))

    assert unchanged, "evaluate changed the model's weights"

def test_train_rejects_empty_loader():
    """train_one_epoch() raises a clear error when the loader is empty."""
    model = BasicCNN(num_classes=5, channels=(8,), image_size=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError):
        train_one_epoch(model, [], nn.CrossEntropyLoss(), optimizer, torch.device("cpu"))

def test_evaluate_rejects_empty_loader():
    """evaluate() raises a clear error when the loader is empty."""
    model = BasicCNN(num_classes=5, channels=(8,), image_size=32)
    with pytest.raises(ValueError):
        evaluate(model, [], nn.CrossEntropyLoss(), torch.device("cpu"))