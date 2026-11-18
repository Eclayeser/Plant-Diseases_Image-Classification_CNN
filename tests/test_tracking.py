import pytest

from leafapp.tracking import close_task, log_epoch_metrics, log_confusion_matrix, upload_checkpoint


class FakeLogger:
    def __init__(self):
        self.scalars = []
        self.confusion_matrices = []
        self.failure_message = None

    def report_scalar(self, **kwargs):
        self.scalars.append(kwargs)

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

    def set_tags(self, tags):
        self.tags = tags

    def add_tags(self, tags):
        if isinstance(tags, str):
            tags = [tags]
        self.tags.extend(tags)
        self.events.append("add_tags")

    def get_tags(self):
        return list(self.tags)

    def set_status(self, status):
        self.failed = status == "failed"

    def mark_failed(self, ignore_errors=True, status_reason=None, status_message=None, force=False):
        self.failed = True
        self.failure_message = status_message
        self.events.append("mark_failed")

    def close(self):
        self.closed = True
        self.events.append("close")
        if self.close_error is not None:
            raise self.close_error
        self.closed = True

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

def test_log_confusion_matrix_reports_matrix_with_labels():
    logger = FakeLogger()
    matrix = [[5, 2], [1, 7]]
    class_names = ["healthy", "rust"]

    log_confusion_matrix(logger, epoch=3, matrix=matrix, class_names=class_names)

    assert logger.confusion_matrices == [
        {
            "title": "Confusion Matrix",
            "series": "validation",
            "matrix": matrix,
            "iteration": 3,
            "xaxis": "Predicted label",
            "yaxis": "True label",
            "xlabels": class_names,
            "ylabels": class_names,
            "yaxis_reversed": True,
        }
    ]

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
