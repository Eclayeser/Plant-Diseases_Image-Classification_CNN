import pytest

from leafapp.tracking import log_epoch_metrics, upload_checkpoint, log_final_metrics, upload_file, log_image


class FakeLogger:
    def __init__(self):
        self.scalars = []
        self.single_values = []
        self.images = []

    def report_scalar(self, **kwargs):
        self.scalars.append(kwargs)

    def report_single_value(self, **kwargs):
        self.single_values.append(kwargs)

    def report_image(self, **kwargs):
        self.images.append(kwargs)


class FakeTask:
    def __init__(self):
        self.artifacts = []

    def upload_artifact(self, **kwargs):
        self.artifacts.append(kwargs)


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