import pytest
import torch
from torch import nn
from leafapp.models.cnn import BasicCNN
from train import train_one_epoch, parse_args
from utils import make_fake_loader

def test_train_one_epoch_returns_valid_metrics():
    model = BasicCNN(num_classes=5, channels=(8,), image_size=32)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = train_one_epoch(model, make_fake_loader(), criterion, optimizer, torch.device("cpu"))
    assert metrics["loss"] >= 0, "Negative loss"
    assert 0 <= metrics["accuracy"] <= 1, "Acc out of range"
    assert 0 <= metrics["f1"] <= 1, "F1 out of range"

def test_train_one_epoch_updates_weights():
    model = BasicCNN(num_classes=5, channels=(8,), image_size=32)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    before = [p.clone() for p in model.parameters()]
    train_one_epoch(model, make_fake_loader(), criterion, optimizer, torch.device("cpu"))
    after = list(model.parameters())

    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "train_one_epoch did not update any weights"

def test_train_one_epoch_rejects_empty_loader():
    model = BasicCNN(num_classes=5, channels=(8,), image_size=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError):
        train_one_epoch(model, [], nn.CrossEntropyLoss(), optimizer, torch.device("cpu"))

def test_parse_args_gives_default_path():
    args = parse_args([])
    assert args.config == "configs/prototype.yaml", "Did not return default path"

def test_parse_args_reads_custom_config():
    args = parse_args(["--config", "configs/resnet18.yaml"])
    assert args.config == "configs/resnet18.yaml", "Did not read the --config value"