from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from leafapp.models.cnn import BasicCNN
import train
from train import train_one_epoch, parse_args, RUN_TYPES
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
    args = parse_args(["--responsible", "tester"])
    assert args.config == "configs/prototype.yaml", "Did not return default path"

def test_parse_args_reads_custom_config():
    args = parse_args(["--config", "configs/resnet18.yaml", "--responsible", "tester"])
    assert args.config == "configs/resnet18.yaml", "Did not read the --config value"

def test_parse_args_requires_responsible():
    with pytest.raises(SystemExit):
        parse_args([])

def test_parse_args_reads_responsible():
    args = parse_args(["--responsible", "Mark"])
    assert args.responsible == "Mark", "Did not read the --responsible value"

def test_parse_args_run_type_default():
    args = parse_args(["--responsible", "tester"])
    assert args.run_type == "debugging", "Did not return the default run type"
    assert args.run_type in RUN_TYPES

@pytest.mark.parametrize("run_type", RUN_TYPES)
def test_parse_args_accepts_every_run_type(run_type):
    args = parse_args(["--responsible", "tester", "--run-type", run_type])
    assert args.run_type == run_type, "Did not read the --run-type value"

def test_parse_args_rejects_invalid_run_type():
    with pytest.raises(SystemExit):
        parse_args(["--responsible", "tester", "--run-type", "not-a-run-type"])

def test_main_tags_task_with_run_details(monkeypatch, tmp_path):
    cfg = {
        "seed": 0,
        "data": {"fraction": 1.0, "image_size": 32, "batch_size": 4},
        "model": {"type": "BasicCNN"},
        "train": {"lr": 1e-3, "weight_decay": 0.0, "epochs": 1, "checkpoint_dir": str(tmp_path)},
    }
    captured = {}

    def fake_initialise_task(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(train, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        train, "get_loaders",
        lambda **kwargs: (make_fake_loader(), make_fake_loader(), None, ["a", "b", "c", "d", "e"]),
    )
    monkeypatch.setattr(
        train, "build_model",
        lambda model_cfg, data_cfg, class_names: BasicCNN(num_classes=5, channels=(8,), image_size=32),
    )
    monkeypatch.setattr(train, "initialise_task", fake_initialise_task)
    monkeypatch.setattr(
        train, "calculate_metrics_per_loader",
        lambda **kwargs: {"loss": 1.0, "accuracy": 0.5, "f1": 0.5},
    )
    for name in ("log_epoch_metrics", "save_checkpoint", "upload_checkpoint", "close_task"):
        monkeypatch.setattr(train, name, MagicMock())

    train.main("configs/prototype.yaml", "Mark", "production")

    assert captured["tags"] == [
        "@run-type:production",
        "@model:BasicCNN",
        "@responsible:Mark",
        "@task-type:train.py",
    ]