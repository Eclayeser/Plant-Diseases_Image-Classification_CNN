import pytest

from pathlib import Path
from leafapp.tracking import close_task, log_epoch_metrics, log_confusion_matrix, upload_checkpoint, log_final_metrics, upload_file, log_image, download_checkpoint


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

def report_plotly(self, **kwargs):
    self.plots.append(kwargs)