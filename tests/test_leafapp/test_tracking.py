import argparse
from unittest.mock import MagicMock

import clearml
import pytest

from pathlib import Path
from leafapp.tracking import (
    build_tags,
    close_task,
    download_checkpoint,
    download_onnx_model_from_clearml,
    get_task,
    initialise_task,
    log_confusion_matrix,
    log_epoch_metrics,
    log_final_metrics,
    log_image,
    to_plain_config,
    upload_checkpoint,
    upload_file,
    upload_onnx_model_to_clearml,
    validate_issue_id,
)
from clearml.utilities.proxy_object import ProxyDictPostWrite
from leafapp.checkpoints import save_checkpoint, load_checkpoint
from tests.shared_helpers import FAKE_CLASS_NAMES, FAKE_IMAGE_SIZE, VALID_NAME, make_tiny_model


def make_connected_config(values):
    """The proxy dict ClearML's task.connect returns."""
    return ProxyDictPostWrite(None, lambda *args: None, values)


@pytest.fixture
def fake_clearml(monkeypatch):
    """Replace the ClearML classes tracking imports, so nothing talks to a server."""
    fakes = MagicMock()
    for name in ("Task", "OutputModel", "Model"):
        monkeypatch.setattr(clearml, name, getattr(fakes, name))
    return fakes


class FakeLogger:
    def __init__(self):
        self.scalars = []
        self.single_values = []
        self.images = []
        self.confusion_matrices = []
        self.plots = []

    def report_scalar(self, **kwargs):
        self.scalars.append(kwargs)

    def report_single_value(self, **kwargs):
        self.single_values.append(kwargs)

    def report_image(self, **kwargs):
        self.images.append(kwargs)

    def report_confusion_matrix(self, **kwargs):
        self.confusion_matrices.append(kwargs)

    def report_plotly(self, **kwargs):
        self.plots.append(kwargs)

class FakeTask:
    def __init__(self):
        self.artifacts = []
        self.tags = []
        self.closed = False
        self.failed = False
        self.failure_message = None
        self.close_error = None
        self.events = []

    def upload_artifact(self, **kwargs):
        self.artifacts.append(kwargs)

    def add_tags(self, tags):
        if isinstance(tags, str):
            tags = [tags]
        self.tags.extend(tags)
        self.events.append("add_tags")

    def mark_failed(self, ignore_errors=True, status_reason=None, status_message=None, force=False):
        self.failed = True
        self.failure_message = status_message
        self.events.append("mark_failed")

    def close(self):
        self.closed = True
        self.events.append("close")
        if self.close_error is not None:
            raise self.close_error

class FakeArtifact:
    def __init__(self, local_path):
        self.local_path = local_path
        self.calls = []

    def get_local_copy(self, **kwargs):
        self.calls.append(kwargs)
        return str(self.local_path)


class FakeTrainingTask:
    def __init__(self, artifacts):
        self.id = "train-task-42"
        self.artifacts = artifacts


def test_log_epoch_metrics_reports_each_split_and_metric():
    logger = FakeLogger()

    log_epoch_metrics(
        logger,
        epoch=3,
        training_metrics={"loss": 0.4, "accuracy": 0.8},
        validation_metrics={"loss": 0.5, "accuracy": 0.7},
    )

    assert logger.scalars == [
        {"title": "loss", "series": "training", "value": 0.4, "iteration": 3},
        {"title": "accuracy", "series": "training", "value": 0.8, "iteration": 3},
        {"title": "loss", "series": "validation", "value": 0.5, "iteration": 3},
        {"title": "accuracy", "series": "validation", "value": 0.7, "iteration": 3},
    ]


def test_log_final_metrics_reports_numbers_and_skips_others():
    logger = FakeLogger()

    log_final_metrics(
        logger,
        {"accuracy": 0.8, "f1": 0.7, "confusion-matrix": [[1, 0], [0, 1]]},
    )

    assert logger.single_values == [
        {"name": "accuracy", "value": 0.8},
        {"name": "f1", "value": 0.7},
    ], "Should report each number and skip the confusion matrix"

def test_log_confusion_matrix_reports_matrix_with_labels():
    logger = FakeLogger()
    matrix = [[5, 2], [1, 7]]
    class_names = ["healthy", "rust"]

    log_confusion_matrix(logger, epoch=3, matrix=matrix, class_names=class_names)

    assert len(logger.confusion_matrices) == 1
    plot = logger.confusion_matrices[0]

    assert plot["title"] == "Confusion Matrix (%)"
    assert plot["series"] == "validation"
    assert plot["iteration"] == 3
    assert plot["xlabels"] == class_names
    assert plot["ylabels"] == class_names
    assert plot["yaxis_reversed"] is True

    assert plot["matrix"][0, 0] == pytest.approx(5 / 7 * 100)
    assert plot["matrix"][1, 1] == pytest.approx(7 / 8 * 100)

    layout = plot["extra_layout"]
    assert layout["width"] == 1800
    assert layout["height"] == 1500
    assert layout["margin"]["l"] >= 350
    assert layout["margin"]["b"] >= 300
    assert layout["xaxis"]["tickangle"] <= -45
    
def test_log_confusion_matrix_reports_normalised_percentages():
    logger = FakeLogger()

    matrix = [
        [3, 1, 0],
        [0, 0, 0],
        [1, 1, 2],
    ]
    class_names = ["healthy", "rust", "mold"]

    log_confusion_matrix(
        logger,
        epoch=3,
        matrix=matrix,
        class_names=class_names,
    )

    assert len(logger.confusion_matrices) == 1
    percentages = logger.confusion_matrices[0]

    assert percentages["title"] == "Confusion Matrix (%)"
    assert percentages["iteration"] == 3
    assert percentages["xlabels"] == class_names
    assert percentages["ylabels"] == class_names

    assert percentages["matrix"].tolist() == [
        [75.0, 25.0, 0.0],
        [0.0, 0.0, 0.0],
        [25.0, 25.0, 50.0],
    ]

def test_confusion_matrices_use_white_to_red_scale():
    logger = FakeLogger()

    log_confusion_matrix(
        logger,
        epoch=1,
        matrix=[[0, 1], [2, 3]],
        class_names=["healthy", "rust"],
    )

    assert len(logger.confusion_matrices) == 1
    layout = logger.confusion_matrices[0]["extra_layout"]

    assert layout["colorscale"][0] == [0.0, "#FFFFFF"]
    assert layout["colorscale"][-1] == [1.0, "#7F0000"]
    assert layout["zmin"] == 0
    assert layout["zmax"] == 100

def test_confusion_matrices_outline_non_zero_cells():
    logger = FakeLogger()

    log_confusion_matrix(
        logger,
        epoch=1,
        matrix=[[0, 1], [2, 0]],
        class_names=["healthy", "rust"],
    )

    assert len(logger.confusion_matrices) == 1
    layout = logger.confusion_matrices[0]["extra_layout"]
    shapes = layout["shapes"]
    assert len(shapes) == 2

    for shape in shapes:
        assert shape["type"] == "rect"
        assert shape["line"]["width"] > 0

def test_upload_checkpoint_registers_artifact_path(tmp_path):
    task = FakeTask()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"random text")

    upload_checkpoint(task, checkpoint, "best_checkpoint")

    assert task.artifacts == [
        {
            "name": "best_checkpoint",
            "artifact_object": checkpoint,
            "wait_on_upload": True,
        }
    ]


def test_upload_raises_when_missing_file(tmp_path):
    task = FakeTask()
    missing = tmp_path / "does_not_exist.pt"

    with pytest.raises(FileNotFoundError, match="best_checkpoint"):
        upload_checkpoint(task, missing, "best_checkpoint")

    assert task.artifacts == []


def test_upload_raises_when_path_is_a_directory(tmp_path):
    task = FakeTask()

    with pytest.raises(FileNotFoundError, match="best_checkpoint"):
        upload_checkpoint(task, tmp_path, "best_checkpoint")

    assert task.artifacts == []

def test_close_task_marks_task_as_closed_and_failed():
    task = FakeTask()

    close_task(task, failed=True, message="some error")

    assert "@status:failed" in task.tags
    assert task.closed is True
    assert task.failed is True
    assert task.failure_message == "some error"
    assert task.events == ["add_tags", "mark_failed", "close"]

def test_close_task_logs_mark_failed_error_and_still_closes(caplog):
    task = FakeTask()

    def fail_marking_task_failed(**kwargs):
        raise RuntimeError("ClearML unavailable")

    task.mark_failed = fail_marking_task_failed

    close_task(task, failed=True, message="training failed")

    assert "Could not mark the ClearML task as failed" in caplog.text
    assert "ClearML unavailable" in caplog.text
    assert task.closed is True

def test_close_task_propagates_close_error_when_not_already_failed():
    task = FakeTask()
    task.close_error = RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="close failed"):
        close_task(task)

def test_close_task_logs_close_error_when_preserving_failure(caplog):
    task = FakeTask()
    task.close_error = RuntimeError("close failed")

    close_task(task, failed=True, message="training failed")

    assert "Could not close the ClearML task after marking it as failed" in caplog.text
    assert "@status:failed" in task.tags
    assert task.failed is True
    assert task.failure_message == "training failed"


def test_upload_file_registers_report_artifact(tmp_path):
    task = FakeTask()
    report = tmp_path / "eval_report.md"
    report.write_text("# Evaluation report")

    upload_file(task, report, "evaluation_report")

    assert task.artifacts == [
        {
            "name": "evaluation_report",
            "artifact_object": report,
            "wait_on_upload": True,
        }
    ]


def test_upload_file_raises_and_uploads_nothing_when_file_missing(tmp_path):
    task = FakeTask()
    missing = tmp_path / "no_such_report.md"

    with pytest.raises(FileNotFoundError):
        upload_file(task, missing, "evaluation_report")

    assert task.artifacts == []


def test_log_image_reports_image_at_iteration_zero(tmp_path):
    logger = FakeLogger()
    image = tmp_path / "fake_image.png"
    image.write_bytes(b"fake png")

    log_image(logger, "evaluation", "confusion_matrix", image)

    assert logger.images == [
        {
            "title": "evaluation",
            "series": "confusion_matrix",
            "iteration": 0,
            "local_path": str(image),
        }
    ]


def test_log_image_raises_and_logs_nothing_when_file_missing(tmp_path):
    logger = FakeLogger()
    missing = tmp_path / "fake_image.png"

    with pytest.raises(FileNotFoundError):
        log_image(logger, "evaluation", "confusion_matrix", missing)

    assert logger.images == []


def test_download_checkpoint_returns_local_path(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    task = FakeTrainingTask({"best_checkpoint": FakeArtifact(checkpoint)})

    result = download_checkpoint(task, "best_checkpoint")

    assert isinstance(result, Path)
    assert result == checkpoint


def test_download_checkpoint_asks_clearml_to_fail_loudly(tmp_path):
    artifact = FakeArtifact(tmp_path / "best.pt")
    task = FakeTrainingTask({"best_checkpoint": artifact})

    download_checkpoint(task, "best_checkpoint")

    assert artifact.calls == [{"raise_on_error": True}]


def test_download_checkpoint_lists_available_when_missing(tmp_path):
    task = FakeTrainingTask({"best_checkpoint": FakeArtifact(tmp_path / "best.pt")})

    with pytest.raises(KeyError, match="best_checkpoint"):
        download_checkpoint(task, "final_checkpoint")

def test_reports_top_ten_classes_by_error_rate():
    logger = FakeLogger()

    # 11 classes with samples, plus one empty class.
    matrix = [[0] * 12 for _ in range(12)]

    for i in range(11):
        matrix[i][i] = 10 - i
        matrix[i][(i + 1) % 11] = i

    names = [f"Class {i}" for i in range(12)]

    log_confusion_matrix(
        logger, epoch=5, matrix=matrix, class_names=names
    )

    assert len(logger.confusion_matrices) == 1
    assert len(logger.plots) == 1

    plot = logger.plots[0]

    assert plot["title"] == "Highest Class Error Rates"
    assert plot["iteration"] == 5

    bar = plot["figure"]["data"][0]

    assert bar["type"] == "bar"
    assert bar["orientation"] == "h"
    assert bar["y"] == [f"Class {i}" for i in range(10, 0, -1)]
    assert bar["x"] == [float(i * 10) for i in range(10, 0, -1)]
    assert "Class 11" not in bar["y"]

def test_confusion_matrix_displays_non_zero_raw_counts():
    logger = FakeLogger()

    log_confusion_matrix(
        logger,
        epoch=1,
        matrix=[[180, 20], [0, 100]],
        class_names=["healthy", "rust"],
    )

    plot = logger.confusion_matrices[0]
    layout = plot["extra_layout"]

    assert layout["text"] == [
        ["180", "20"],
        ["", "100"],
    ]
    assert layout["texttemplate"] == "%{text}"
    assert plot["matrix"][0, 0] == pytest.approx(90)
    assert plot["matrix"][0, 1] == pytest.approx(10)

def test_empty_confusion_matrix_has_no_labels():
    logger = FakeLogger()

    log_confusion_matrix(
        logger,
        epoch=1,
        matrix=[[0, 0], [0, 0]],
        class_names=["healthy", "rust"],
    )

    layout = logger.confusion_matrices[0]["extra_layout"]

    assert layout["text"] == [["", ""], ["", ""]]
    assert layout["shapes"] == []
    assert logger.plots == []


def test_connected_config_can_be_saved_and_loaded(tmp_path):
    connected = make_connected_config({
        "model": {"type": "BasicCNN", "channels": [8]},
        "data": {"image_size": FAKE_IMAGE_SIZE},
        "train": {"lr": 0.0123},
    })
    cfg = to_plain_config(connected)

    assert type(cfg) is dict and type(cfg["model"]) is dict, "Connected config should be plain dicts"
    path = tmp_path / "model.pt"
    save_checkpoint(
        path=path,
        model=make_tiny_model(),
        cfg=cfg,
        class_names=FAKE_CLASS_NAMES,
        history_metrics={},
    )

    _, _, _, loaded_config = load_checkpoint(path)
    assert loaded_config["train"]["lr"] == 0.0123, "Checkpoint should load and keep the connected values"


@pytest.mark.parametrize("issue_id", ["", "SGP-94"])
def test_validate_issue_id_accepts_empty_or_sgp_ids(issue_id):
    assert validate_issue_id(issue_id) == issue_id


def test_validate_issue_id_rejects_other_formats():
    with pytest.raises(argparse.ArgumentTypeError, match="SGP-XXX"):
        validate_issue_id("94")


def test_build_tags_adds_issue_tag_only_when_given():
    with_issue = build_tags("debugging", "BasicCNN", VALID_NAME, "training", "SGP-94", extra=("@extra",))
    without_issue = build_tags("debugging", "BasicCNN", VALID_NAME, "training")

    assert with_issue == [
        "@run-type:debugging",
        "@model:BasicCNN",
        "@responsible:mark-tarnavskyi",
        "@task-type:training",
        "@extra",
        "@issue:SGP-94",
    ]
    assert not any(tag.startswith("@issue:") for tag in without_issue)


def test_initialise_task_returns_config_edited_in_clearml(fake_clearml):
    task = fake_clearml.Task.init.return_value
    task.connect.return_value = make_connected_config({"train": {"lr": 0.5}})

    returned_task, cfg = initialise_task({"train": {"lr": 0.1}}, "project", "run")

    assert returned_task is task
    assert cfg == {"train": {"lr": 0.5}}, "Should return the values from ClearML, not the original"
    assert type(cfg["train"]) is dict, "Returned config should be plain dicts"
    init_kwargs = fake_clearml.Task.init.call_args.kwargs
    assert init_kwargs["tags"] == []
    assert init_kwargs["reuse_last_task_id"] is False, "Every run should get a new task"


def test_get_task_fetches_task_by_id(fake_clearml):
    assert get_task("abc123") is fake_clearml.Task.get_task.return_value
    fake_clearml.Task.get_task.assert_called_once_with(task_id="abc123")


def test_upload_onnx_model_publishes_all_files_and_returns_id(fake_clearml, tmp_path):
    model = fake_clearml.OutputModel.return_value
    model.id = "model-7"
    files = [tmp_path / "model.onnx", tmp_path / "labels.json"]

    model_id = upload_onnx_model_to_clearml(
        task="task", files=files, name="cnn", tags=["@model:BasicCNN"], comment="best run", config={}
    )

    assert model_id == "model-7"
    upload_kwargs = model.update_weights_package.call_args.kwargs
    assert upload_kwargs["weights_filenames"] == [str(path) for path in files]
    assert upload_kwargs["auto_delete_file"] is False, "Local files should be kept"
    assert model.comment == "best run"
    model.publish.assert_called_once()


def test_download_onnx_model_returns_extracted_folder(fake_clearml, tmp_path):
    model = fake_clearml.Model.return_value
    model.get_weights_package.return_value = str(tmp_path)

    assert download_onnx_model_from_clearml("model-7") == tmp_path
    fake_clearml.Model.assert_called_once_with(model_id="model-7")
    assert model.get_weights_package.call_args.kwargs["raise_on_error"] is True


def test_close_task_does_nothing_without_task():
    close_task(None, failed=True, message="failed before the task was created")


def test_close_task_logs_add_tags_error_and_still_fails_and_closes(caplog):
    task = FakeTask()

    def fail_adding_tags(tags):
        raise RuntimeError("ClearML unavailable")

    task.add_tags = fail_adding_tags

    close_task(task, failed=True, message="training failed")

    assert "Could not add the @status:failed tag" in caplog.text
    assert task.failed is True
    assert task.closed is True