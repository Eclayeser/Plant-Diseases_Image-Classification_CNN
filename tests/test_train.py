from unittest.mock import MagicMock

import copy
import pytest
import torch
from torch import nn
import train
from train import train_one_epoch, parse_args, RUN_TYPES
from tests.shared_helpers import (
    FAKE_CLASS_NAMES,
    VALID_NAME,
    make_fake_loader,
    make_tiny_model,
    make_train_cfg,
    patch_train_basics,
)


def parse(*args):
    """parse_args with a valid --responsible already filled in."""
    return parse_args(["--responsible", VALID_NAME, *args])


def fake_metrics(loss=0.4, with_matrix=True):
    metrics = {"loss": loss, "accuracy": 0.8, "f1": 0.7}
    if with_matrix:
        size = len(FAKE_CLASS_NAMES)
        metrics["confusion-matrix"] = [[0] * size for _ in range(size)]
    return metrics


def patch_fake_training(monkeypatch, cfg):
    """patch_train_basics plus a tiny model and fixed metrics, so no real training runs."""
    task = patch_train_basics(monkeypatch, cfg)
    monkeypatch.setattr(train, "build_model", lambda model_cfg, data_cfg, class_names: make_tiny_model())
    monkeypatch.setattr(train, "train_one_epoch", lambda *args, **kwargs: fake_metrics(loss=0.5, with_matrix=False))
    monkeypatch.setattr(train, "calculate_metrics_per_loader", lambda **kwargs: fake_metrics())
    monkeypatch.setattr(train, "log_epoch_metrics", MagicMock())
    monkeypatch.setattr(train, "close_task", MagicMock())
    return task


def record_checkpoints(monkeypatch):
    """Replace saving and uploading with stubs that write a small file and record what happened."""
    saved_paths = []
    uploaded_artifacts = []

    def save_checkpoint(path, **kwargs):
        path.write_bytes(b"checkpoint")
        saved_paths.append(path)

    monkeypatch.setattr(train, "save_checkpoint", save_checkpoint)
    monkeypatch.setattr(
        train, "upload_checkpoint", lambda task, path, name: uploaded_artifacts.append((path, name)),
    )
    return saved_paths, uploaded_artifacts


@pytest.fixture
def model_and_optimizer():
    model = make_tiny_model()
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


def test_train_one_epoch_returns_valid_metrics(model_and_optimizer):
    model, optimizer = model_and_optimizer

    metrics = train_one_epoch(model, make_fake_loader(), nn.CrossEntropyLoss(), optimizer, torch.device("cpu"))
    assert metrics["loss"] >= 0, "Negative loss"
    assert 0 <= metrics["accuracy"] <= 1, "Acc out of range"
    assert 0 <= metrics["f1"] <= 1, "F1 out of range"

def test_train_one_epoch_updates_weights(model_and_optimizer):
    model, optimizer = model_and_optimizer

    before = [p.clone() for p in model.parameters()]
    train_one_epoch(model, make_fake_loader(), nn.CrossEntropyLoss(), optimizer, torch.device("cpu"))
    after = list(model.parameters())

    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "train_one_epoch did not update any weights"

def test_train_one_epoch_rejects_empty_loader(model_and_optimizer):
    model, optimizer = model_and_optimizer
    with pytest.raises(ValueError):
        train_one_epoch(model, [], nn.CrossEntropyLoss(), optimizer, torch.device("cpu"))

def test_parse_args_gives_default_path():
    args = parse()
    assert args.config == "configs/prototype.yaml", "Did not return default path"
    assert args.save_interval == 5, "Did not return default checkpoint interval"

def test_parse_args_reads_custom_config():
    args = parse("--config", "configs/resnet18.yaml")
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
    args = parse()
    assert args.run_type == "debugging", "Did not return the default run type"
    assert args.run_type in RUN_TYPES

def test_parse_args_reads_save_interval():
    args = parse("--save-interval", "1")
    assert args.save_interval == 1, "Did not read the --save-interval value"

@pytest.mark.parametrize("run_type", RUN_TYPES)
def test_parse_args_accepts_every_run_type(run_type):
    args = parse("--run-type", run_type)
    assert args.run_type == run_type, "Did not read the --run-type value"

def test_parse_args_rejects_invalid_run_type():
    with pytest.raises(SystemExit):
        parse("--run-type", "not-a-run-type")

@pytest.mark.parametrize("value, expected", [(None, None), ("", None), ("None", None), ("8", 8), (3, 3)])
def test_parse_patience_converts_connected_values(value, expected):
    assert train.parse_patience(value) == expected

@pytest.mark.parametrize("value", ["0", -1, "abc", 2.5, "2.5", True])
def test_parse_patience_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="patience must be a positive integer"):
        train.parse_patience(value)

@pytest.mark.parametrize("value, expected", [(1, 1), (30, 30), ("5", 5)])
def test_parse_epochs_converts_connected_values(value, expected):
    assert train.parse_epochs(value) == expected

@pytest.mark.parametrize("value", [0, "0", -1, "abc", 2.5, "2.5", True, None])
def test_parse_epochs_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        train.parse_epochs(value)

def test_main_rejects_zero_epochs_set_in_connected_config(monkeypatch, tmp_path):
    task = patch_train_basics(monkeypatch, make_train_cfg(tmp_path))

    def fake_initialise_task(**kwargs):
        connected = copy.deepcopy(kwargs["config"])
        connected["train"]["epochs"] = 0  # what a cloned task edited in the UI returns
        return task, connected

    monkeypatch.setattr(train, "initialise_task", fake_initialise_task)
    upload_checkpoint = MagicMock()
    monkeypatch.setattr(train, "upload_checkpoint", upload_checkpoint)
    close_task = MagicMock()
    monkeypatch.setattr(train, "close_task", close_task)

    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        train.main("configs/prototype.yaml", VALID_NAME, "debugging", "")

    train.get_loaders.assert_not_called()
    upload_checkpoint.assert_not_called()
    close_task.assert_called_once()
    assert close_task.call_args.kwargs["failed"] is True

def test_main_tags_task_with_run_details(monkeypatch, tmp_path, capsys):
    patch_fake_training(monkeypatch, make_train_cfg(tmp_path))
    for name in ("save_checkpoint", "upload_checkpoint"):
        monkeypatch.setattr(train, name, MagicMock())

    train.main("configs/prototype.yaml", "Mark Tarnavskyi", "production", "")

    captured = train.initialise_task.call_args.kwargs
    loader_settings = train.get_loaders.call_args.kwargs
    assert "Training device: CPU (CUDA unavailable; falling back to CPU)" in capsys.readouterr().out
    assert captured["tags"] == [
        "@run-type:production",
        "@model:BasicCNN",
        "@responsible:mark-tarnavskyi",
        "@task-type:train.py",
        "@config-name:prototype",
    ]
    assert captured["task_name"] == "BasicCNN - prototype"
    assert loader_settings["num_workers"] == 0
    assert "checkpoint_interval" not in captured["config"]["train"]
    assert loader_settings["augment"] is True
    assert loader_settings["normalize"] == "none"
    assert captured["config"]["data"]["augment"] is True


@pytest.mark.parametrize("overrides, expected", [({}, 0.2), ({"dropout_p": 0.4}, 0.4)])
def test_training_records_effective_dropout_probability(monkeypatch, tmp_path, overrides, expected):
    cfg = make_train_cfg(
        tmp_path,
        model={"type": "RegularisedCNN", "channels": [8], "regularisation": "dropout", **overrides},
    )
    patch_train_basics(monkeypatch, cfg)

    train.main("experiment.yaml", VALID_NAME, "comparison", "")

    assert train.initialise_task.call_args.kwargs["config"]["model"]["dropout_p"] == expected
    checkpoint = torch.load(next(tmp_path.glob("*_best.pt")), map_location="cpu")
    assert checkpoint["config"]["model"]["dropout_p"] == expected


def test_main_saves_and_uploads_checkpoints_at_requested_interval(monkeypatch, tmp_path):
    patch_fake_training(monkeypatch, make_train_cfg(tmp_path, epochs=5, data={"augment": False}))
    saved_paths, uploaded_artifacts = record_checkpoints(monkeypatch)

    train.main("configs/prototype.yaml", VALID_NAME, "production", "", checkpoint_interval=2)

    assert "checkpoint_interval" not in train.initialise_task.call_args.kwargs["config"]["train"]
    assert train.get_loaders.call_args.kwargs["augment"] is False
    periodic_paths = [path for path in saved_paths if "_epoch_" in path.name]
    assert [path.name.rsplit("_epoch_", 1)[1] for path in periodic_paths] == ["2.pt", "4.pt"]
    assert [name for _, name in uploaded_artifacts if name.startswith("checkpoint_epoch_")] == [
        "checkpoint_epoch_2",
        "checkpoint_epoch_4",
    ]

def test_main_continues_when_periodic_checkpoint_upload_fails(monkeypatch, tmp_path, caplog):
    patch_fake_training(monkeypatch, make_train_cfg(tmp_path, epochs=5))
    saved_paths, uploaded_artifacts = record_checkpoints(monkeypatch)
    trained_epochs = []

    def train_epoch(*args, **kwargs):
        trained_epochs.append(len(trained_epochs) + 1)
        return fake_metrics(loss=0.5, with_matrix=False)

    def upload(task, path, name):
        if name == "checkpoint_epoch_2":
            raise OSError("storage unavailable")
        uploaded_artifacts.append((path, name))

    monkeypatch.setattr(train, "train_one_epoch", train_epoch)
    monkeypatch.setattr(train, "upload_checkpoint", upload)

    train.main("configs/prototype.yaml", VALID_NAME, "production", "", checkpoint_interval=2)

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

def test_main_retries_pending_best_checkpoint_save_after_failure(monkeypatch, tmp_path):
    patch_fake_training(monkeypatch, make_train_cfg(tmp_path, epochs=2))
    _, uploaded_artifacts = record_checkpoints(monkeypatch)
    best_save_epochs = []
    validation_losses = iter((0.6, 0.4))

    monkeypatch.setattr(
        train, "calculate_metrics_per_loader", lambda **kwargs: fake_metrics(loss=next(validation_losses)),
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

    with pytest.raises(OSError, match="best checkpoint write failed"):
        train.main("configs/prototype.yaml", VALID_NAME, "production", "", checkpoint_interval=5)

    best_checkpoint_path = next(
        path for path, name in uploaded_artifacts if name == "best_checkpoint"
    )
    assert best_save_epochs == [1, 2, 2]
    assert best_checkpoint_path.read_text() == "best epoch 2"

@pytest.mark.parametrize("report_name", ["log_epoch_metrics", "log_confusion_matrix"])
def test_main_saves_improved_best_checkpoint_after_interruption(monkeypatch, tmp_path, report_name):
    task = patch_fake_training(monkeypatch, make_train_cfg(tmp_path))
    saved_paths, uploaded_artifacts = record_checkpoints(monkeypatch)

    def interrupt_reporting(*args, **kwargs):
        assert list(tmp_path.glob("*_best.pt")), "Best checkpoint should have been saved before reporting"
        raise KeyboardInterrupt

    monkeypatch.setattr(train, report_name, interrupt_reporting)

    with pytest.raises(KeyboardInterrupt):
        train.main("configs/prototype.yaml", VALID_NAME, "production", "")

    best_checkpoint_path = [path for path in saved_paths if path.stem.endswith("_best")]
    assert len(best_checkpoint_path) == 1
    assert best_checkpoint_path[0].read_text() == "checkpoint"
    assert {name for _, name in uploaded_artifacts} == {"final_checkpoint", "best_checkpoint"}

    train.close_task.assert_called_once_with(task, failed=True, message="")

@pytest.mark.parametrize("failure_epoch", [1, 2])
def test_main_uploads_checkpoints_when_training_fails(monkeypatch, tmp_path, failure_epoch, capsys):
    task = patch_fake_training(monkeypatch, make_train_cfg(tmp_path, epochs=2))
    saved_paths, uploaded_artifacts = record_checkpoints(monkeypatch)
    train_calls = 0

    def fail_on_chosen_epoch(*args, **kwargs):
        nonlocal train_calls
        train_calls += 1
        if train_calls == failure_epoch:
            raise RuntimeError("training failed")
        return fake_metrics(loss=0.5, with_matrix=False)

    monkeypatch.setattr(train, "train_one_epoch", fail_on_chosen_epoch)

    with pytest.raises(RuntimeError, match="training failed"):
        train.main("configs/prototype.yaml", VALID_NAME, "production", "")

    output = capsys.readouterr().out
    final_checkpoint_path = next(path for path in saved_paths if not path.stem.endswith("_best"))
    assert f"Final checkpoint saved to {final_checkpoint_path}" in output
    best_checkpoint_paths = [path for path in saved_paths if path.stem.endswith("_best")]
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
    train.close_task.assert_called_once_with(task, failed=True, message="training failed")

@pytest.mark.parametrize(
    "normalize, expected",
    [(False, "none"), (True, "imagenet"), ("plant_village", "plant_village")],
)
def test_main_builds_from_connected_config(monkeypatch, tmp_path, normalize, expected):
    patch_fake_training(monkeypatch, make_train_cfg(tmp_path, data={"augment": False, "normalize": normalize}))
    optimizer_settings = {}

    def fake_initialise_task(**kwargs):
        connected = copy.deepcopy(kwargs["config"])  # what a cloned task edited in the UI returns
        assert connected["data"]["normalize"] == expected
        assert connected["train"]["optimizer"] == "AdamW"
        assert connected["train"]["patience"] is None
        assert connected["train"]["min_delta"] == 0.0
        assert connected["model"]["pooling"] == "flatten"
        connected["train"]["lr"] = 0.0123
        connected["data"]["batch_size"] = 7
        return MagicMock(), connected

    def recording_choose_optimizer(parameters, train_settings):
        optimizer_settings.update(train_settings)
        return train.torch.optim.AdamW(parameters, lr=train_settings["lr"])

    monkeypatch.setattr(train, "initialise_task", fake_initialise_task)
    monkeypatch.setattr(train, "choose_optimizer", recording_choose_optimizer)

    train.main("experiment.yaml", VALID_NAME, "comparison", "")

    assert optimizer_settings["lr"] == 0.0123, "Optimiser should use the lr from the connected config"
    assert train.get_loaders.call_args.kwargs["batch_size"] == 7, (
        "Loaders should use the batch size from the connected config"
    )

def test_main_marks_task_failed_when_loading_data_fails(monkeypatch):
    task = patch_train_basics(monkeypatch, make_train_cfg("unused"))

    def broken_loaders(**kwargs):
        raise FileNotFoundError("dataset missing")

    monkeypatch.setattr(train, "get_loaders", broken_loaders)
    close_task = MagicMock()
    monkeypatch.setattr(train, "close_task", close_task)

    with pytest.raises(FileNotFoundError, match="dataset missing"):
        train.main("configs/prototype.yaml", VALID_NAME, "debugging", "")

    close_task.assert_called_once_with(task, failed=True, message="dataset missing")
