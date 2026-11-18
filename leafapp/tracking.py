"""Reusable ClearML helpers for experiment tracking."""

import logging
from pathlib import Path
from typing import Any, Mapping

LOGGER = logging.getLogger(__name__)


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
        reuse_last_task_id=False,
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

def log_confusion_matrix(
    logger: Any,
    epoch: int,
    matrix: Any,
    class_names: list[str],
) -> None:
    """Report a validation confusion matrix with true labels on rows and predicted labels on columns."""
    logger.report_confusion_matrix(
        title="Confusion Matrix",
        series="validation",
        matrix=matrix,
        iteration=epoch,
        xaxis="Predicted label",
        yaxis="True label",
        xlabels=class_names,
        ylabels=class_names,
        yaxis_reversed=True,
    )

def upload_checkpoint(task: Any, checkpoint_path: str | Path, name: str) -> None:
    """Upload a saved checkpoint as a ClearML task artifact."""
    artifact_path = Path(checkpoint_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Checkpoint with name {name} not found at {checkpoint_path}")
    
    task.upload_artifact(
        name=name,
        artifact_object=artifact_path,
        wait_on_upload=True,
    )


def close_task(task: Any, *, failed: bool = False, message: str | None = None) -> None:
    """Close the ClearML task and mark it as failed when the run errors out."""
    if task is None:
        return

    if failed:
        try:
            task.add_tags("@status:failed")
        except Exception:
            LOGGER.exception("Could not add the @status:failed tag to the ClearML task")

        try:
            task.mark_failed(status_message=message or "Task failed")
        except Exception:
            LOGGER.exception("Could not mark the ClearML task as failed")
    try:
        task.close()
    except Exception:
        if failed:
            LOGGER.exception("Could not close the ClearML task after marking it as failed")
        else:
            raise