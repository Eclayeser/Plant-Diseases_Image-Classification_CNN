"""Shared evaluation functions"""

from typing import Any
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
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


def calculate_confusion_matrix(true_labels, predicted_labels):
    cm = confusion_matrix(
        true_labels, predicted_labels
    )
    return cm


# Computes various metrics per-loader on demand
# Can be used for both per-epoch and at the end for test data loader
def calculate_metrics_per_loader(
        model, loader, device,
        criterion=None,
        metrics_list=None
    ) -> dict[str, Any]:

    final_metrics = {}
    if metrics_list is None and criterion is None:
            # avoid the run as no metrics were requested
            return final_metrics

    model.eval()
    total_loss, total = 0, 0
    all_preds, all_labels = [], []

    calc_dispatcher = {
        "accuracy": calculate_accuracy,
        "precision": calculate_precision,
        "f1": calculate_f1_score,
        "recall": calculate_recall,
        "confusion_matrix": calculate_confusion_matrix
    }

    with torch.no_grad():
        for batch in loader:
            images, labels = batch["pixels"], batch["label"]
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            batch_size = labels.size(0)
            total += batch_size

            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * batch_size

            if metrics_list is not None:
                all_preds.append(logits.argmax(dim=1).cpu())
                all_labels.append(labels.cpu())
            

    if total == 0:
        raise ValueError("The loader produced no samples, so the metrics can't be computed")

    if criterion is not None:
        final_metrics["loss"] = total_loss / total

    if metrics_list is not None:
        true_labels, predicted_labels = torch.cat(all_labels), torch.cat(all_preds)
        
        for metric in metrics_list:
            if metric in calc_dispatcher:
                final_metrics[metric] = calc_dispatcher[metric](true_labels, predicted_labels)
            else:
                raise ValueError(f"{metric} metric is not recognised")

    return final_metrics


# TODO: Top-K Accuracy
# TODO: Per-class metrics
# TODO(most likely): plotting from passed metrics / from loaded saved metrix
# TODO(possibily): some other morre advanced reporting