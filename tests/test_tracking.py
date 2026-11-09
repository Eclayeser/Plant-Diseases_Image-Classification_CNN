from pathlib import Path

from leafapp.tracking import log_epoch_metrics, upload_checkpoint


class FakeLogger:
    def __init__(self):
        self.scalars = []

    def report_scalar(self, **kwargs):
        self.scalars.append(kwargs)


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


def test_upload_checkpoint_registers_artifact_path():
    task = FakeTask()
    checkpoint = Path("outputs/models/best.pt")

    upload_checkpoint(task, checkpoint, "best_checkpoint")

    assert task.artifacts == [
        {
            "name": "best_checkpoint",
            "artifact_object": str(checkpoint),
            "wait_on_upload": True,
        }
    ]
