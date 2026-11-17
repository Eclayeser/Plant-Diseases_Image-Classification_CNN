"""Reusable ClearML helpers for experiment tracking."""
import argparse

from pathlib import Path
from typing import Any, Mapping
from clearml import Task


RUN_TYPES = ("debugging", "exploring", "comparison", "production")    # see README.md to understand difference
NAMES = ("Mark Tarnavskyi", "Pranay Singh", "Illia Putintsev", "Javier Fimia", "Reon Carroll-Ito", "Nil Kayalioglu", "Mario Westphalen Cavicchioli")

def sanitise_name(value):
    """
    Sanitises name into lowercase kebab format.
    """
    slug = "-".join(value.lower().split())
    return slug


def validate_issue_id(value):
    """
    Ensures issue is in the right format. Raises otherwise
    """
    if value and not value.startswith("SGP-"):
        raise argparse.ArgumentTypeError("--issue-id must be in 'SGP-XXX' format.")
    return value


def initialise_task(
    config: Mapping[str, Any],
    project_name: str,
    task_name: str,
    tags: list[str] | None = None,
) -> Any:
    """Create a ClearML task and connect the complete run configuration."""
    from clearml import Task

    Task.set_random_seed(None)

    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        tags=tags or [],
        auto_connect_frameworks=False,
    )
    task.connect(dict(config))
    return task


def log_epoch_metrics(
    logger: Any,
    epoch: int,
    training_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
) -> None:
    """Report each training and validation metric for an epoch."""
    for phase, metrics in (
        ("training", training_metrics),
        ("validation", validation_metrics),
    ):
        for metric_name, value in metrics.items():
            logger.report_scalar(
                title=metric_name,
                series=phase,
                value=float(value),
                iteration=epoch,
            )


def log_final_metrics(logger: Any, metrics: Mapping[str, Any]) -> None:
    """Report each single-number metric (e.g. accuracy, f1) as a ClearML single value."""
    for name, value in metrics.items():
        if not isinstance(value, (int, float)):
             continue
        logger.report_single_value(name=name, value=value)


def log_image(logger: Any, title: str, series: str, image_path: str | Path) -> None:
    """Log an image file to ClearML as a single evaluation image."""
    picture_path = Path(image_path)
    if not picture_path.is_file():
        raise FileNotFoundError(f"Image not found at {image_path}")
    logger.report_image(iteration=0, local_path=str(picture_path), title=title, series=series)


def upload_file(task: Any, path: str | Path, name: str) -> None:
    """Upload file as a ClearML task artifact."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"File with name {name} not found at {path}")
    task.upload_artifact(
        name=name,
        artifact_object=artifact_path,
        wait_on_upload=True,
    )


def upload_checkpoint(task: Any, checkpoint_path: str | Path, name: str) -> None:
    """Upload a saved checkpoint as a ClearML task artifact."""
    upload_file(task, checkpoint_path, name)


def download_checkpoint(task: Any, artifact_name: str) -> Path:
    """Download a checkpoint artifact from a ClearML task and return its local path."""

    if artifact_name not in task.artifacts:
        raise KeyError(
            f"Task {task.id} has no artifact '{artifact_name}'. "
            f"Available: {list(task.artifacts)}"
        )

    local_path = task.artifacts[artifact_name].get_local_copy(raise_on_error=True)
    return Path(local_path)


def close_task(task: Any) -> None:
    """Close the ClearML task after all metrics and artifacts are logged."""
    task.close()