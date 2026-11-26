"""Plots for evaluation reports."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import seaborn
import torch

from leafapp.config import ROOT
from leafapp.labels import format_class_name


def _save_figure(fig: Figure, plot_path: str | Path, show: bool) -> None:
    fig.tight_layout()
    path = ROOT / plot_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def prepare_image_for_plotting(image, mean=None, std=None) -> torch.Tensor:
    picture = image.permute(1, 2, 0)
    if mean is not None:
        picture = picture * torch.tensor(std) + torch.tensor(mean)
    return picture.clamp(0, 1)


def plot_and_save_misclassified_grid(
        misclassified: list[tuple[torch.Tensor, int, int]],
        class_names: list[str],
        plot_path: str | Path,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        show: bool = False
    ) -> None:
    columns = 4
    rows = max(1, math.ceil(len(misclassified) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(columns * 4, rows * 4), squeeze=False
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, (image, true_class, guessed_class) in zip(axes.flat, misclassified):
        ax.imshow(prepare_image_for_plotting(image, mean, std))
        true_name = format_class_name(class_names[true_class])
        predicted_name = format_class_name(class_names[guessed_class])
        ax.set_title(f"true: {true_name}\npred: {predicted_name}", fontsize=8)

    _save_figure(fig, plot_path, show)


def plot_and_save_training_curve(
        history_metrics: dict[str, dict[str, list[float]]],
        plot_path: str | Path,
        exclude_metrics: list[str]|None = None,
        exclude_run_types: list[str]|None = None,
        show: bool = False
    ) -> None:
    """
    Plot per-epoch historical metrics, one subplot per metric, and save the figure.
    """
    exclude_metrics = exclude_metrics or []
    exclude_run_types = exclude_run_types or []


    metrics_to_plot = {
        name: runs for name, runs in history_metrics.items() if name not in exclude_metrics
    }
    if not metrics_to_plot:
        return


    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(12, 4), squeeze=False)
    i = 0
    axes = axes[0]

    for metric_name, metrics in metrics_to_plot.items():
        for run_type in metrics.keys():
            if run_type in exclude_run_types:
                continue

            metrics_array = metrics[run_type]
            epochs = range(1, len(metrics_array) + 1)
            axes[i].plot(epochs, metrics_array, label=run_type.capitalize())

        axes[i].set_title(f"CNN {metric_name.capitalize()}")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(f"{metric_name.capitalize()}")
        axes[i].legend()
        axes[i].grid(alpha=0.3)

        i += 1

    _save_figure(fig, plot_path, show)


def plot_and_save_confusion_matrix(
        confusion_matrix: np.ndarray,
        class_names: list[str],
        plot_path: str | Path,
        show: bool = False
    ) -> None:
    fig = plt.figure(figsize=(20, 17))

    seaborn.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("CNN Confusion Matrix")

    plt.xticks(rotation=60, ha="right")
    _save_figure(fig, plot_path, show)
