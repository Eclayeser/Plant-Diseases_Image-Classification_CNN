"""Reusable ClearML helpers for experiment tracking."""

from pathlib import Path
from typing import Any, Mapping


def initialise_task(
    config: Mapping[str, Any],
    project_name: str,
    task_name: str,
    tags: list[str] | None = None,
) -> Any:
    """Create a ClearML task and connect the complete run configuration."""
    from clearml import Task

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


def upload_checkpoint(task: Any, checkpoint_path: str | Path, name: str) -> None:
    """Upload a saved checkpoint as a ClearML task artifact."""
    task.upload_artifact(
        name=name,
        artifact_object=str(checkpoint_path),
        wait_on_upload=True,
    )


def close_task(task: Any) -> None:
    """Close the ClearML task after all metrics and artifacts are logged."""
    task.close()
