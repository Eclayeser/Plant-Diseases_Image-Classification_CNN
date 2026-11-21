from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from leafapp.models.cnn import BasicCNN
import train
from train import train_one_epoch, parse_args, RUN_TYPES, NAMES
from utils import make_fake_loader

VALID_NAME = NAMES[0]

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
    args = parse_args(["--responsible", VALID_NAME])
    assert args.config == "configs/prototype.yaml", "Did not return default path"
    assert args.save_interval == 5, "Did not return default checkpoint interval"

def test_parse_args_reads_custom_config():
    args = parse_args(["--config", "configs/resnet18.yaml", "--responsible", VALID_NAME])
    assert args.config == "configs/resnet18.yaml", "Did not read the --config value"

def test_parse_args_requires_responsible():
    with pytest.raises(SystemExit):
        parse_args([])

def test_parse_args_reads_responsible():
    args = parse_args(["--responsible", "Mark Tarnavskyi"])
    assert args.responsible == "Mark Tarnavskyi", "Did not read the --responsible value"

def test_parse_args_rejects_unknown_responsible():
    with pytest.raises(SystemExit):
        parse_args(["--responsible", "Not A Teammate"])

def test_parse_args_run_type_default():
    args = parse_args(["--responsible", VALID_NAME])
    assert args.run_type == "debugging", "Did not return the default run type"
    assert args.run_type in RUN_TYPES

def test_parse_args_reads_save_interval():
    args = parse_args(["--responsible", VALID_NAME, "--save-interval", "1"])
    assert args.save_interval == 1, "Did not read the --save-interval value"

@pytest.mark.parametrize("run_type", RUN_TYPES)
def test_parse_args_accepts_every_run_type(run_type):
    args = parse_args(["--responsible", VALID_NAME, "--run-type", run_type])
    assert args.run_type == run_type, "Did not read the --run-type value"

def test_parse_args_rejects_invalid_run_type():
    with pytest.raises(SystemExit):
        parse_args(["--responsible", VALID_NAME, "--run-type", "not-a-run-type"])

def test_main_tags_task_with_run_details(monkeypatch, tmp_path, capsys):
    cfg = {
        "seed": 0,
        "data": {"fraction": 1.0, "image_size": 32, "batch_size": 4, "augmentation": {"rotation_degrees": 10}},
        "model": {"type": "BasicCNN"},
        "train": {"lr": 1e-3, "weight_decay": 0.0, "epochs": 1, "checkpoint_dir": str(tmp_path)},
    }
    captured = {}

    def fake_initialise_task(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(train, "load_config", lambda path: cfg)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)
    loaders = MagicMock(return_value=(make_fake_loader(), make_fake_loader(), None, ["a", "b", "c", "d", "e"]))
    monkeypatch.setattr(train, "get_loaders", loaders)
    monkeypatch.setattr(
        train, "build_model",
        lambda model_cfg, data_cfg, class_names: BasicCNN(num_classes=5, channels=(8,), image_size=32),
    )
    monkeypatch.setattr(train, "initialise_task", fake_initialise_task)
    monkeypatch.setattr(
        train, "calculate_metrics_per_loader",
        lambda **kwargs: {"loss": 1.0, "accuracy": 0.5, "f1": 0.5, "confusion-matrix": [[0] * 5 for _ in range(5)]},
    )
    for name in ("log_epoch_metrics", "save_checkpoint", "upload_checkpoint", "close_task"):
        monkeypatch.setattr(train, name, MagicMock())

    train.main("configs/prototype.yaml", "Mark Tarnavskyi", "production", "")

    assert "Training device: CPU (CUDA unavailable; falling back to CPU)" in capsys.readouterr().out
    assert captured["tags"] == [
        "@run-type:production",
        "@model:BasicCNN",
        "@responsible:mark-tarnavskyi",
        "@task-type:train.py",
        "@config-name:prototype",
    ]
    assert captured["task_name"] == "BasicCNN - prototype"
    assert captured["config"]["train"]["checkpoint_interval"] == 5
    assert loaders.call_args.kwargs["augment"] is True
    assert loaders.call_args.kwargs["normalize"] is False
    settings = captured["config"]["data"]["augmentation"]
    assert settings == {"flip_probability": 0.5, "rotation_degrees": 10, "brightness": 0.2, "saturation": 0.2}
    assert loaders.call_args.kwargs["augmentation"] == settings
    assert train.save_checkpoint.call_args.kwargs["cfg"]["data"]["augmentation"] == settings

@pytest.mark.parametrize("overrides, expected", [({}, 0.2), ({"dropout_p": 0.4}, 0.4)])
def test_training_records_effective_dropout_probability(monkeypatch, tmp_path, overrides, expected):
    cfg = {
        "seed": 0,
        "data": {"fraction": 1.0, "image_size": 32, "batch_size": 4},
        "model": {"type": "RegularisedCNN", "channels": [8], "regularisation": "dropout", **overrides},
        "train": {"lr": 0.001, "weight_decay": 0.0, "epochs": 1, "checkpoint_dir": str(tmp_path)},
    }
    monkeypatch.setattr(train, "load_config", lambda path: cfg)
    loaders = (make_fake_loader(), make_fake_loader(), None, ["a", "b", "c", "d", "e"])
    monkeypatch.setattr(train, "get_loaders", lambda **kwargs: loaders)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)
    task_init = MagicMock()
    monkeypatch.setattr(train, "initialise_task", task_init)

    train.main("experiment.yaml", VALID_NAME, "comparison", "")

    assert task_init.call_args.kwargs["config"]["model"]["dropout_p"] == expected
    checkpoint = torch.load(next(tmp_path.glob("*_best.pt")), map_location="cpu")
    assert checkpoint["config"]["model"]["dropout_p"] == expected


def test_main_saves_and_uploads_checkpoints_at_requested_interval(
    monkeypatch, tmp_path
):
    cfg = {
        "seed": 0,
        "data": {"fraction": 1.0, "image_size": 32, "batch_size": 4, "augment": False},
        "model": {"type": "BasicCNN"},
        "train": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 5,
            "checkpoint_dir": str(tmp_path),
        },
    }
    saved_paths = []
    uploaded_artifacts = []
    task = MagicMock()

    monkeypatch.setattr(train, "load_config", lambda path: cfg)
    loaders = MagicMock(return_value=(make_fake_loader(), make_fake_loader(), None, ["a", "b", "c", "d", "e"]))
    monkeypatch.setattr(train, "get_loaders", loaders)
    monkeypatch.setattr(
        train,
        "build_model",
        lambda model_cfg, data_cfg, class_names: BasicCNN(
            num_classes=5, channels=(8,), image_size=32
        ),
    )
    initialise_task = MagicMock(return_value=task)
    monkeypatch.setattr(train, "initialise_task", initialise_task)
    monkeypatch.setattr(
        train,
        "train_one_epoch",
        lambda *args, **kwargs: {"loss": 0.5, "accuracy": 0.8, "f1": 0.7, "confusion-matrix": [[0] * 5 for _ in range(5)]},
    )
    monkeypatch.setattr(
        train,
        "calculate_metrics_per_loader",
        lambda **kwargs: {"loss": 0.4, "accuracy": 0.8, "f1": 0.7, "confusion-matrix": [[0] * 5 for _ in range(5)]},
    )
    monkeypatch.setattr(train, "log_epoch_metrics", MagicMock())
    def save_checkpoint(path, cfg, **kwargs):
        assert cfg["train"]["checkpoint_interval"] == 2
        saved_paths.append(path)

    monkeypatch.setattr(train, "save_checkpoint", save_checkpoint)
    monkeypatch.setattr(
        train,
        "upload_checkpoint",
        lambda task, path, name: uploaded_artifacts.append((path, name)),
    )
    monkeypatch.setattr(train, "close_task", MagicMock())

    train.main(
        "configs/prototype.yaml",
        VALID_NAME,
        "production",
        "",
        checkpoint_interval=2,
    )

    assert initialise_task.call_args.kwargs["config"]["train"]["checkpoint_interval"] == 2
    assert loaders.call_args.kwargs["augment"] is False
    periodic_paths = [path for path in saved_paths if "_epoch_" in path.name]
    assert [path.name.rsplit("_epoch_", 1)[1] for path in periodic_paths] == [
        "2.pt",
        "4.pt",
    ]
    assert [name for _, name in uploaded_artifacts if name.startswith("checkpoint_epoch_")] == [
        "checkpoint_epoch_2",
        "checkpoint_epoch_4",
    ]

def test_main_continues_when_periodic_checkpoint_upload_fails(
    monkeypatch, tmp_path, caplog
):
    cfg = {
        "seed": 0,
        "data": {"fraction": 1.0, "image_size": 32, "batch_size": 4},
        "model": {"type": "BasicCNN"},
        "train": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 5,
            "checkpoint_dir": str(tmp_path),
        },
    }
    task = MagicMock()
    saved_paths = []
    uploaded_artifacts = []
    trained_epochs = []

    monkeypatch.setattr(train, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        train,
        "get_loaders",
        lambda **kwargs: (
            make_fake_loader(),
            make_fake_loader(),
            None,
            ["a", "b", "c", "d", "e"],
        ),
    )
    monkeypatch.setattr(
        train,
        "build_model",
        lambda model_cfg, data_cfg, class_names: BasicCNN(
            num_classes=5, channels=(8,), image_size=32
        ),
    )
    monkeypatch.setattr(train, "initialise_task", lambda **kwargs: task)

    def train_epoch(*args, **kwargs):
        trained_epochs.append(len(trained_epochs) + 1)
        return {"loss": 0.5, "accuracy": 0.8, "f1": 0.7}

    def save_checkpoint(path, **kwargs):
        path.write_bytes(b"checkpoint")
        saved_paths.append(path)

    def upload(task, path, name):
        if name == "checkpoint_epoch_2":
            raise OSError("storage unavailable")
        uploaded_artifacts.append((path, name))

    monkeypatch.setattr(train, "train_one_epoch", train_epoch)
    monkeypatch.setattr(
        train,
        "calculate_metrics_per_loader",
        lambda **kwargs: {
            "loss": 0.4,
            "accuracy": 0.8,
            "f1": 0.7,
            "confusion-matrix": [[0] * 5 for _ in range(5)],
        },
    )
    monkeypatch.setattr(train, "save_checkpoint", save_checkpoint)
    monkeypatch.setattr(train, "upload_checkpoint", upload)
    monkeypatch.setattr(train, "log_epoch_metrics", MagicMock())
    monkeypatch.setattr(train, "close_task", MagicMock())

    train.main(
        "configs/prototype.yaml",
        VALID_NAME,
        "production",
        "",
        checkpoint_interval=2,
    )

    failed_periodic_checkpoint = next(
        path for path in saved_paths if path.name.endswith("_epoch_2.pt")
    )
    assert failed_periodic_checkpoint.is_file()
    assert trained_epochs == [1, 2, 3, 4, 5]
    assert {name for _, name in uploaded_artifacts} == {
        "checkpoint_epoch_4",
        "final_checkpoint",
        "best_checkpoint",
    }
    assert "Could not upload periodic checkpoint for epoch 2" in caplog.text
    assert "storage unavailable" in caplog.text

def test_main_retries_pending_best_checkpoint_save_after_failure(
    monkeypatch, tmp_path
):
    cfg = {
        "seed": 0,
        "data": {"fraction": 1.0, "image_size": 32, "batch_size": 4},
        "model": {"type": "BasicCNN"},
        "train": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 2,
            "checkpoint_dir": str(tmp_path),
        },
    }
    task = MagicMock()
    best_save_epochs = []
    uploaded_artifacts = []
    validation_losses = iter((0.6, 0.4))

    monkeypatch.setattr(train, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        train,
        "get_loaders",
        lambda **kwargs: (
            make_fake_loader(),
            make_fake_loader(),
            None,
            ["a", "b", "c", "d", "e"],
        ),
    )
    monkeypatch.setattr(
        train,
        "build_model",
        lambda model_cfg, data_cfg, class_names: BasicCNN(
            num_classes=5, channels=(8,), image_size=32
        ),
    )
    monkeypatch.setattr(train, "initialise_task", lambda **kwargs: task)
    monkeypatch.setattr(
        train,
        "train_one_epoch",
        lambda *args, **kwargs: {"loss": 0.5, "accuracy": 0.8, "f1": 0.7},
    )
    monkeypatch.setattr(
        train,
        "calculate_metrics_per_loader",
        lambda **kwargs: {
            "loss": next(validation_losses),
            "accuracy": 0.8,
            "f1": 0.7,
            "confusion-matrix": [[0] * 5 for _ in range(5)],
        },
    )

    def save_checkpoint(path, history_metrics, **kwargs):
        if path.stem.endswith("_best"):
            epoch = len(history_metrics["losses"]["validation"])
            best_save_epochs.append(epoch)
            if epoch == 2 and best_save_epochs.count(2) == 1:
                raise OSError("best checkpoint write failed")
            path.write_text(f"best epoch {epoch}")
        else:
            path.write_text("final snapshot")

    monkeypatch.setattr(train, "save_checkpoint", save_checkpoint)
    monkeypatch.setattr(
        train,
        "upload_checkpoint",
        lambda task, path, name: uploaded_artifacts.append((path, name)),
    )
    monkeypatch.setattr(train, "log_epoch_metrics", MagicMock())
    monkeypatch.setattr(train, "close_task", MagicMock())

    with pytest.raises(OSError, match="best checkpoint write failed"):
        train.main(
            "configs/prototype.yaml",
            VALID_NAME,
            "production",
            "",
            checkpoint_interval=5,
        )

    best_checkpoint_path = next(
        path for path, name in uploaded_artifacts if name == "best_checkpoint"
    )
    assert best_save_epochs == [1, 2, 2]
    assert best_checkpoint_path.read_text() == "best epoch 2"

@pytest.mark.parametrize("failure_epoch", [1, 2])
def test_main_uploads_checkpoints_when_training_fails(
    monkeypatch, tmp_path, failure_epoch, capsys
):
    cfg = {
        "seed": 0,
        "data": {"fraction": 1.0, "image_size": 32, "batch_size": 4},
        "model": {"type": "BasicCNN"},
        "train": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 2,
            "checkpoint_dir": str(tmp_path),
        },
    }
    task = MagicMock()
    uploaded_artifacts = []
    train_calls = 0

    monkeypatch.setattr(train, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        train,
        "get_loaders",
        lambda **kwargs: (
            make_fake_loader(),
            make_fake_loader(),
            None,
            ["a", "b", "c", "d", "e"],
        ),
    )
    monkeypatch.setattr(
        train,
        "build_model",
        lambda model_cfg, data_cfg, class_names: BasicCNN(
            num_classes=5, channels=(8,), image_size=32
        ),
    )
    monkeypatch.setattr(train, "initialise_task", lambda **kwargs: task)

    def fail_on_second_epoch(*args, **kwargs):
        nonlocal train_calls
        train_calls += 1
        if train_calls == failure_epoch:
            raise RuntimeError("training failed")
        return {"loss": 0.5, "accuracy": 0.8, "f1": 0.7}

    def save_checkpoint(path, **kwargs):
        path.write_bytes(b"checkpoint")
        saved_paths.append(path)

    saved_paths = []
    monkeypatch.setattr(train, "train_one_epoch", fail_on_second_epoch)
    monkeypatch.setattr(
        train,
        "calculate_metrics_per_loader",
        lambda **kwargs: {"loss": 0.4, "accuracy": 0.8, "f1": 0.7, "confusion-matrix": [[0] * 5 for _ in range(5)]},
    )
    monkeypatch.setattr(train, "save_checkpoint", save_checkpoint)
    monkeypatch.setattr(
        train,
        "upload_checkpoint",
        lambda task, path, name: uploaded_artifacts.append((path, name)),
    )
    close_task = MagicMock()
    monkeypatch.setattr(train, "close_task", close_task)

    with pytest.raises(RuntimeError, match="training failed"):
        train.main("configs/prototype.yaml", VALID_NAME, "production", "")

    output = capsys.readouterr().out
    final_checkpoint_path = next(path for path in saved_paths if not path.stem.endswith("_best"))
    assert f"Final checkpoint saved to {final_checkpoint_path}" in output
    best_checkpoint_paths = [
        path for path in saved_paths if path.stem.endswith("_best")
    ]
    if failure_epoch == 1:
        assert best_checkpoint_paths == []
        assert "Best checkpoint saved to" not in output
        assert {name for _, name in uploaded_artifacts} == {"final_checkpoint"}
    else:
        assert len(best_checkpoint_paths) == 1
        assert f"Best checkpoint saved to {best_checkpoint_paths[0]}" in output
        assert {name for _, name in uploaded_artifacts} == {
            "final_checkpoint",
            "best_checkpoint",
        }
    close_task.assert_called_once_with(
        task, failed=True, message="training failed"
    )
