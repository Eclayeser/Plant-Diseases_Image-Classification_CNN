import pytest
import torch
from torch import nn
from torch.nn import functional as F
from leafapp.eval_functions import calculate_metrics_per_loader

CLASS_NAMES = ["healthy", "rust", "blight"]


def make_logits_loader():
    """Two uneven batches (4 + 2 samples) whose pixels are the logits themselves.

    Used together with nn.Identity() so predictions are known exactly:
    batch 1 gets 3/4 right, batch 2 gets 0/2 right -> 3/6 overall,
    which differs from the per-batch mean (0.375).
    """
    def batch(labels, predictions):
        logits = 3.0 * F.one_hot(torch.tensor(predictions), num_classes=3).float()
        return {"pixels": logits, "label": torch.tensor(labels)}

    return [
        batch([0, 1, 2, 0], [0, 1, 2, 1]),
        batch([0, 1], [2, 2]),
    ]


def test_metrics_are_computed_over_whole_loader_not_per_batch():
    loader = make_logits_loader()
    criterion = nn.CrossEntropyLoss()

    metrics = calculate_metrics_per_loader(
        nn.Identity(), loader, "cpu", CLASS_NAMES,
        criterion=criterion, metrics_list=["accuracy"],
    )

    all_logits = torch.cat([b["pixels"] for b in loader])
    all_labels = torch.cat([b["label"] for b in loader])
    assert metrics["accuracy"] == 0.5
    assert metrics["loss"] == pytest.approx(F.cross_entropy(all_logits, all_labels).item())


def test_returns_empty_dict_without_running_model_when_nothing_requested():
    class ExplodingModel(nn.Module):
        def forward(self, x):
            raise AssertionError("model must not run when no metrics are requested")

    metrics = calculate_metrics_per_loader(
        ExplodingModel(), make_logits_loader(), "cpu", CLASS_NAMES,
        criterion=None, metrics_list=None,
    )

    assert metrics == {}


def test_raises_on_unknown_metric_and_on_empty_loader():
    with pytest.raises(ValueError, match="not recognised"):
        calculate_metrics_per_loader(
            nn.Identity(), make_logits_loader(), "cpu", CLASS_NAMES,
            metrics_list=["accuracy", "not-a-metric"],
        )

    with pytest.raises(ValueError, match="no samples"):
        calculate_metrics_per_loader(
            nn.Identity(), [], "cpu", CLASS_NAMES, metrics_list=["accuracy"],
        )


def test_structured_metrics_have_expected_shape_and_content():
    metrics = calculate_metrics_per_loader(
        nn.Identity(), make_logits_loader(), "cpu", CLASS_NAMES,
        metrics_list=["top-3-accuracy", "confusion-matrix", "per-class-metrics"],
    )

    # with 3 classes the true class is always within the top 3
    assert metrics["top-3-accuracy"] == 1.0

    confusion = metrics["confusion-matrix"]
    assert confusion.shape == (3, 3)
    assert confusion.sum() == 6
    assert confusion.trace() == 3  # correct predictions sit on the diagonal

    report_dict, report_text = metrics["per-class-metrics"]
    assert all(name in report_dict for name in CLASS_NAMES)
    assert all(name in report_text for name in CLASS_NAMES)
