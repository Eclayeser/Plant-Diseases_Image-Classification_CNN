# Checks that the optimiser is built from the config correctly.
import pytest
import torch
from torch import nn
from leafapp.optimizer_constructor import choose_optimizer


def make_params():
    return nn.Linear(2, 1).parameters()


def test_missing_optimizer_defaults_to_adamw():
    settings = {"lr": 0.001, "weight_decay": 0.01}          # no "optimizer" key
    optimizer = choose_optimizer(make_params(), settings)
    assert type(optimizer) is torch.optim.AdamW, "Missing key should default to AdamW"


def test_adam_config_builds_adam():
    settings = {"optimizer": "Adam", "lr": 0.001, "weight_decay": 0.01}
    optimizer = choose_optimizer(make_params(), settings)
    assert type(optimizer) is torch.optim.Adam, "Adam config should build Adam"


def test_lr_and_weight_decay_are_passed_through():
    lr, weight_decay = 0.0003, 0.05
    settings = {"optimizer": "AdamW", "lr": lr, "weight_decay": weight_decay}
    optimizer = choose_optimizer(make_params(), settings)
    assert optimizer.param_groups[0]["lr"] == lr, "lr not passed through"
    assert optimizer.param_groups[0]["weight_decay"] == weight_decay, "weight_decay not passed through"


def test_unknown_optimizer_raises():
    settings = {"optimizer": "adamw", "lr": 0.001, "weight_decay": 0.01}
    with pytest.raises(ValueError):
        choose_optimizer(make_params(), settings)