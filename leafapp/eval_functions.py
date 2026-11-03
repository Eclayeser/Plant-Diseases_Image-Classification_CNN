"""Shared evaluation functions"""
from pathlib import Path

from leafapp.config import load_config
from leafapp.data import get_loaders
from leafapp.models.cnn import BasicCNN

import torch

# ROOT = Path(__file__).resolve().parent


# Computes per-batch total loss and accuracy
def loss_accuracy_per_batch(loss, logits, true_labels):
    batch_size = true_labels.size(0)
    batch_loss = loss.item() * batch_size
    batch_correct = (logits.argmax(dim=1) == true_labels).sum().item()
    return batch_loss, batch_correct, batch_size

# Computes per-loader average loss and accuracy
# Can be used for both per-epoch and at the end for test data loader
def loss_accuracy_per_loader(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            images, labels = batch["pixels"], batch["label"]
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_loss, batch_correct, batch_size = loss_accuracy_per_batch(loss, logits, labels)
            total_loss += batch_loss
            total_correct += batch_correct
            total += batch_size

    if total == 0:
        raise ValueError("The loader produced no samples, so the metrics can't be computed")

    loader_loss = total_loss / total
    loader_accuracy = total_correct / total
    return loader_loss, loader_accuracy


# takes all calculated metrics troughout the run
# and returns them nicely in an object to be stored at a checkpoint
def construct_full_metrics_object():

    pass

# TODO: functions that calculate different metrcis using validation set
# -> to be used for tunning purposes with validation set;
# -> to be used at the very end for report with test set; after tunning
# Do not use test set to tune hyperparameters to avoid bias

# Metrics include, but not limited to:
#   - accuracy and loss
#   - per-class precision, recall and F1, macro-F1
#   - confusion matrix

# TODO: function that saves progress metrics [what evaluation() in train.py does already] (calculated every n-th epoch using validation set)
# TODO: function that saves final metrics using single run of trained model (calculated using test/validation set at the end of the training)
# the above 2 actually may be the same depending how flexible the signature is
# I think JSON format could be the best fit for easy parsing

# TODO (possibly): loads saved evaluation and conducts error analysis - summary of evaluation