"""Shared evaluation functions"""

from typing import Any
from pathlib import Path
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import torch
import matplotlib.pyplot as plt
import seaborn

ROOT = Path(__file__).resolve().parent.parent


def calculate_f1_score(true_labels, predicted_labels):
    f1 = f1_score(
        true_labels, predicted_labels, average="macro", zero_division=0
    )
    return round(f1, 6)


def calculate_precision(true_labels, predicted_labels):
    precision = precision_score(
        true_labels, predicted_labels, average="macro", zero_division=0
    )
    return round(precision, 6)


def calculate_accuracy(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    return round(accuracy, 6)


def calculate_recall(true_labels, predicted_labels):
    recall = recall_score(
        true_labels, predicted_labels, average="macro", zero_division=0
    )
    return round(recall, 6)


def calculate_confusion_matrix(true_labels, predicted_labels, labels_pool: list[str]):
    cm = confusion_matrix(
        true_labels, predicted_labels, labels=list(range(len(labels_pool)))
    )
    return cm

# Computes Top-K Accuracy
def calculate_top_k_accuracy(true_labels, logits, k: int):
    top_k_predictions = logits.topk(k, dim=1).indices
    correct = true_labels.unsqueeze(1)

    top_k_accuracy = (
        (top_k_predictions[:, :k] == correct)
        .any(dim=1)
        .float()
        .mean()
        .item()
    )

    return round(top_k_accuracy, 6)


# Computes a per-class precision/recall/f1/support | accuracy/macro avg and weighted avg
def calculate_per_class_metrics(true_labels, predicted_labels, labels_pool: list[str]):
    per_class_dict = classification_report(
        true_labels, predicted_labels,
        labels=list(range(len(labels_pool))),
        target_names=labels_pool,
        output_dict=True,
        zero_division=0
    )
    per_class_to_print = classification_report(
        true_labels, predicted_labels,
        labels=list(range(len(labels_pool))),
        target_names=labels_pool,
        zero_division=0
    )
    return (per_class_dict, per_class_to_print)


# Computes various metrics per-loader on demand
# Can be used for both per-epoch and at the end for test data loader
def calculate_metrics_per_loader(
        model, loader, device, class_names,
        criterion=None,
        metrics_list=None,
    ) -> dict[str, Any]:

    final_metrics = {}
    if metrics_list is None and criterion is None:
            # avoid the run as no metrics were requested
            return final_metrics

    model.eval()
    logits_batches, labels_batches = [], []

    calc_dispatcher = {
        "accuracy": calculate_accuracy,
        "precision": calculate_precision,
        "f1": calculate_f1_score,
        "recall": calculate_recall,
    }
    other_accepted_metrics = [
        "top-3-accuracy",
        "top-5-accuracy",
        "per-class-metrics",
        "confusion-matrix"
    ]

    with torch.no_grad():
        for batch in loader:
            images, labels = batch["pixels"], batch["label"]
            images, labels = images.to(device), labels.to(device)

            logits = model(images)

            logits_batches.append(logits.cpu())
            labels_batches.append(labels.cpu())
            

    if not logits_batches:
        raise ValueError("The loader produced no samples, so the metrics can't be computed")

    final_logits = torch.cat(logits_batches)
    true_labels = torch.cat(labels_batches)

    # calculate loss with regard to provided criterion
    if criterion is not None:
        final_metrics["loss"] = criterion(final_logits, true_labels).item()

    # calculate other metrics
    if metrics_list is not None:
        predicted_labels = final_logits.argmax(dim=1)

        for metric in metrics_list:
            if metric in calc_dispatcher:
                final_metrics[metric] = calc_dispatcher[metric](true_labels, predicted_labels)
            elif metric in other_accepted_metrics:
                if metric.startswith("top-") and metric.endswith("-accuracy"):
                    final_metrics[metric] = calculate_top_k_accuracy(true_labels, final_logits, int(metric.split("-")[1]))
                elif metric == "per-class-metrics":
                    final_metrics[metric] = calculate_per_class_metrics(true_labels, predicted_labels, class_names)
                elif metric == "confusion-matrix":
                    final_metrics[metric] = calculate_confusion_matrix(true_labels, predicted_labels, class_names)
                
            else:
                raise ValueError(f"{metric} metric is not recognised")

    return final_metrics


# --- Plotting Functions ----

# for historical data
def plot_and_save_training_curve(
        history_metrics: dict[str, dict[str, list[float]]],
        plot_path: str | Path,
        exclude_metrics: list[str]|None = None,
        exclude_run_types: list[str]|None = None,
        show: bool = False
    ) -> None:

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

    plt.tight_layout()

    # save
    plot_path = ROOT / plot_path
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()

def plot_and_save_confusion_matrix(
        confusion_matrix,
        class_names,
        plot_path: str | Path,
        show: bool = False
    ) -> None:
    plt.figure(figsize=(20, 17))

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
    plt.tight_layout()

    # save
    plot_path = ROOT / plot_path
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def save_evaluation_report(written_report: str, report_path: str|Path) -> None:
    report_path = ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(written_report, encoding="utf-8")
