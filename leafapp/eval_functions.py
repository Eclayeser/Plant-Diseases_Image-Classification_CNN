"""Shared evaluation metrics and error analysis."""

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
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from leafapp.config import ROOT


def calculate_f1_score(true_labels, predicted_labels) -> float:
    f1 = f1_score(
        true_labels, predicted_labels, average="macro", zero_division=0
    )
    return round(f1, 6)


def calculate_precision(true_labels, predicted_labels) -> float:
    precision = precision_score(
        true_labels, predicted_labels, average="macro", zero_division=0
    )
    return round(precision, 6)


def calculate_accuracy(true_labels, predicted_labels) -> float:
    accuracy = accuracy_score(true_labels, predicted_labels)
    return round(accuracy, 6)


def calculate_recall(true_labels, predicted_labels) -> float:
    recall = recall_score(
        true_labels, predicted_labels, average="macro", zero_division=0
    )
    return round(recall, 6)


def calculate_confusion_matrix(true_labels, predicted_labels, labels_pool: list[str]) -> np.ndarray:
    cm = confusion_matrix(
        true_labels, predicted_labels, labels=list(range(len(labels_pool)))
    )
    return cm


def calculate_most_confused_pairs(cm, class_names, top_k=5) -> list[tuple[int, str, str]]:
    """
    List the most common mistakes in a confusion matrix, biggest first.

    cm has true classes as rows and predicted classes as columns. Each record is
    (count, true class, predicted class), so the direction of a mistake is kept:
    Correct predictions (the diagonal) and empty cells are skipped, and at most
    top_k records are returned.
    """
    out = []
    n = len(cm)
    for row in range(n):
        for col in range(n):
            num = int(cm[row][col])
            if col != row and num > 0:
                out.append((num, class_names[row], class_names[col]))
    
    return sorted(out, reverse=True)[:top_k]


def format_most_confused_pairs(pairs) -> str:
    confused_pair_lines = []
    for count, true_name, predicted_name in pairs:
        confused_pair_lines.append(f" - {true_name} -> {predicted_name}: {count}")
    return "\n".join(confused_pair_lines)

    
def calculate_top_k_accuracy(true_labels, logits, k: int) -> float:
    """
    Fraction of samples whose true class is among the k highest logits.
    """
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


def calculate_per_class_metrics(
        true_labels: torch.Tensor,
        predicted_labels: torch.Tensor,
        labels_pool: list[str]
    ) -> tuple[dict[str, Any], str]:
    """Per-class precision/recall/f1/support plus accuracy, macro avg and weighted avg.

    Args:
        true_labels: Ground-truth class indices, shape (N,).
        predicted_labels: Predicted class indices, shape (N,).
        labels_pool: Class names, indexed by class index.

    Returns:
        A (report_dict, report_text) tuple: the sklearn report as a nested
        dict, and the same report formatted as a printable table.
    """
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


def calculate_metrics_per_loader(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device | str,
        class_names: list[str],
        criterion: nn.Module | None = None,
        metrics_list: list[str] | None = None,
    ) -> dict[str, Any]:
    """Run the model over a loader and compute the requested metrics on demand.

    Works both per epoch (for validation) and once at the end on the validation/test loader.
    All logits are collected first, so metrics are computed over the whole loader
    rather than averaged per batch.

    Returns:
        A dictionary of requested metrics, with "loss" when a criterion is
        supplied. "per-class-metrics" contains (report_dict, printable_report).
    """
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

def collect_misclassified_images(model, loader, device, wanted_pairs, max_images=16):
    """Collect examples matching the (true class ID, predicted class ID) pairs in wanted_pairs.

    Return up to max_images examples across all pairs in loader order.
    Each example is a (CPU image tensor, true class ID, predicted class ID) tuple.
    """
    if max_images < 0:
        raise ValueError(f"max_images must be 0 or more, got {max_images}")
    if max_images == 0 or not wanted_pairs:
        return []
    
    model.eval()
    misclassified = []
    with torch.no_grad():
        for batch in loader:
            images, labels = batch["pixels"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            wrong = predictions != labels
            
            for image, true_label, predicted_label in zip(
                images[wrong], labels[wrong], predictions[wrong]
            ):
                true_class, guessed_class = int(true_label), int(predicted_label)
                if (true_class, guessed_class) in wanted_pairs:
                    misclassified.append((image.cpu(), true_class, guessed_class))
                    if len(misclassified) == max_images:
                        return misclassified
                
    return misclassified


def save_evaluation_report(written_report: str, report_path: str|Path) -> None:
    report_path = ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(written_report, encoding="utf-8")
