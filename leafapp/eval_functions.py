"""Shared evaluation functions"""

from typing import Any
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import torch

# ROOT = Path(__file__).resolve().parent


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
    per_class_report = classification_report(
        true_labels, predicted_labels,
        labels=list(range(len(labels_pool))),
        target_names=labels_pool,
        output_dict=True,
        zero_division=0
    )
    return per_class_report


# TODO(most likely): plotting from passed metrics / from loaded saved metrix
# TODO(possibily): some other morre advanced reporting

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
