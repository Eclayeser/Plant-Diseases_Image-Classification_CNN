# Checks the training process and model checkpoint handling.
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