"""Shared evaluation functions"""
from pathlib import Path

from leafapp.config import load_config
from leafapp.data import get_loaders
from leafapp.models.cnn import BasicCNN


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