"""Reusable ClearML helpers for experiment tracking."""
import argparse

import logging
from pathlib import Path
from typing import Any, Mapping
from clearml import Task
import numpy as np


LOGGER = logging.getLogger(__name__)
RUN_TYPES = ("debugging", "exploring", "comparison", "production")    # see README.md to understand difference
NAMES = ("Mark Tarnavskyi", "Pranay Singh", "Illia Putintsev", "Javier Fimia", "Reon Carroll-Ito", "Gul Nil Kayalioglu", "Mario Westphalen Cavicchioli")
DEFAULT_ARTIFACT = "best_checkpoint"
ARTIFACTS = (DEFAULT_ARTIFACT, "final_checkpoint")
PROJECT_NAME = "Plant-Disease-Image-Classification"


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
    output_uri: str | bool | None = None,
) -> Any:
    """Create a ClearML task and connect the complete run configuration."""
    from clearml import Task

    Task.set_random_seed(None)

    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        tags=tags or [],
        auto_connect_frameworks=False,
        output_uri=output_uri,
        reuse_last_task_id=False,
    )
    task.connect(dict(config))
    return task


def get_task(task_id: str) -> Any:
    """Fetch an existing ClearML task by its ID."""
    from clearml import Task
    return Task.get_task(task_id=task_id)


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

def log_confusion_matrix(logger, epoch, matrix, class_names):
    counts = np.asarray(matrix, dtype=float)
    row_totals = counts.sum(axis=1, keepdims=True)

    percentages = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts),
        where=row_totals > 0,
    ) * 100

    logger.report_confusion_matrix(
        title="Confusion Matrix (%)",
        series="validation",
        matrix=percentages,
        iteration=epoch,
        xaxis="Predicted label",
        yaxis="True label",
        xlabels=class_names,
        ylabels=class_names,
        yaxis_reversed=True,
        extra_layout=layout,
    )
    
    """Report a readable validation confusion matrix to ClearML."""

    layout = {
        "width": 1800,
        "height": 1500,
        "margin": {
            "l": 440,
            "r": 80,
            "t": 70,
            "b": 380,
        },
        "xaxis": {
            "tickangle": -60,
            "tickfont": {"size": 10},
            "automargin": True,
        },
        "yaxis": {
            "tickfont": {"size": 10},
            "automargin": True,
        },
    }

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
        extra_layout=layout,
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


def build_tags(run_type, model_type, responsible, task_type, issue_id="", extra=()):
    """Return the ClearML tags for a task."""
    tags = [
        f"@run-type:{run_type}",
        f"@model:{model_type}",
        f"@responsible:{sanitise_name(responsible)}",
        f"@task-type:{task_type}",
        *extra,
    ]
    if issue_id:
        tags.append(f"@issue:{issue_id}")
    return tags


def upload_onnx_model_to_clearml(
        task: Any,
        files: list[Path],
        name: str,
        tags: list[str],
        comment: str,
        config: dict
    ) -> str:
    """Upload files as one published ONNX model in the ClearML Models tab and return its ID."""
    from clearml import OutputModel

    model = OutputModel(
        task=task,
        name=name,
        tags=tags,
        framework="ONNX",
        config_dict=config
    )
    model.update_weights_package(
        weights_filenames=[str(path) for path in files],
        auto_delete_file=False,
        async_enable=False,
    )
    model.comment = comment
    model.publish()
    return model.id


def download_onnx_model_from_clearml(model_id: str) -> Path:
    """Download a published ClearML model package and return the folder it was extracted to."""
    from clearml import Model

    model = Model(model_id=model_id)
    return Path(model.get_weights_package(return_path=True, raise_on_error=True))


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
