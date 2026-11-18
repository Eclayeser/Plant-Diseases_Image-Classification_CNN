import pytest

from leafapp.tracking import close_task, log_epoch_metrics, log_confusion_matrix, upload_checkpoint


class FakeLogger:
    def __init__(self):
        self.scalars = []
        self.confusion_matrices = []

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

    def upload_artifact(self, **kwargs):
        self.artifacts.append(kwargs)

    def set_tags(self, tags):
        self.tags = tags

    def add_tags(self, tags):
        self.tags.extend(tags)

    def get_tags(self):
        return list(self.tags)

    def set_status(self, status):
        self.failed = status == "failed"

    def mark_failed(self, message=None):
        self.failed = True

    def close(self):
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

    assert "failed" in task.tags
    assert task.closed is True
    assert task.failed is True
