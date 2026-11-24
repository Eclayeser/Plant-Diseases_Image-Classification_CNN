import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from leafapp.data import PLANT_VILLAGE_MEAN, PLANT_VILLAGE_STD
from leafapp.eval_functions import (
    calculate_metrics_per_loader,
    calculate_most_confused_pairs,
    collect_misclassified_images,
)
from leafapp.eval_plots import (
    prepare_image_for_plotting,
    plot_and_save_misclassified_grid,
)

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
    
    
def test_confused_pairs_sorted_biggest_first():
    cm = [[50, 2, 1], [3, 40, 9], [0, 12, 35]]
    pairs = calculate_most_confused_pairs(cm, CLASS_NAMES, top_k=2) 
    expected = [(12, "blight", "rust"), (9, "rust", "blight")]
    assert pairs == expected, "Should list (count, true, predicted), biggest first"
    
    
def test_confused_pairs_list_true_class_before_predicted():
    cm = [[40, 10], [0, 50]]
    names = ["early blight", "late blight"]
    pairs = calculate_most_confused_pairs(cm, names, top_k=2)
    expected = [(10, "early blight", "late blight")]
    assert pairs == expected, "Should name the true class first, then the predicted one"


def test_misclassified_keeps_only_mistakes_from_wanted_pairs():
    # make_logits_loader's mistakes, in loader order: (0, 1), then (0, 2) and (1, 2)
    found = collect_misclassified_images(
        nn.Identity(), make_logits_loader(), "cpu", wanted_pairs={(0, 1), (1, 2)}
    )

    pairs = [(true_class, guessed_class) for _, true_class, guessed_class in found]
    assert pairs == [(0, 1), (1, 2)], (
        "Should keep only the wanted mistakes, in loader order"
    )
    # each photo's pixels are its logits here, so (0, 1) is the photo scored [0, 3, 0]
    assert torch.equal(found[0][0], torch.tensor([0.0, 3.0, 0.0])), (
        "Should store each mistake with its own photo"
    )


def test_misclassified_stops_reading_once_it_has_max_images():
    def loader():
        yield make_logits_loader()[0]  # its only mistake is (0, 1)
        raise AssertionError("asked for another batch after max_images was reached")

    found = collect_misclassified_images(
        nn.Identity(), loader(), "cpu", wanted_pairs={(0, 1), (1, 2)}, max_images=1
    )

    pairs = [(true_class, guessed_class) for _, true_class, guessed_class in found]
    assert pairs == [(0, 1)], "Should stop as soon as it has max_images mistakes"


def test_prepare_image_for_plotting_undoes_normalisation():
    photo = torch.rand(3, 8, 8)
    normalised = transforms.Normalize(PLANT_VILLAGE_MEAN, PLANT_VILLAGE_STD)(photo)

    picture = prepare_image_for_plotting(
        normalised, PLANT_VILLAGE_MEAN, PLANT_VILLAGE_STD
    )

    assert picture.shape == (8, 8, 3), "Should put the colours last for matplotlib"
    assert torch.allclose(picture, photo.permute(1, 2, 0), atol=1e-6), (
        "Should give back the photo as it was before normalising"
    )


def test_misclassified_grid_saves_a_picture_even_with_no_mistakes(tmp_path):
    plot_path = tmp_path / "grid.png"

    plot_and_save_misclassified_grid([], CLASS_NAMES, plot_path)

    assert plot_path.is_file(), "Should save an empty grid instead of crashing"
    

def test_misclassified_returns_nothing_without_running_model_when_nothing_wanted():
    class ExplodingModel(nn.Module):
        def forward(self, x):
            raise AssertionError("model must not run when no images are wanted")

    found_with_zero_limit = collect_misclassified_images(
        ExplodingModel(), make_logits_loader(), "cpu", wanted_pairs={(0, 1)}, max_images=0
    )
    found_with_no_pairs = collect_misclassified_images(
        ExplodingModel(), make_logits_loader(), "cpu", wanted_pairs=set()
    )

    assert found_with_zero_limit == [], "Should return nothing when max_images is 0"
    assert found_with_no_pairs == [], "Should return nothing when no pairs are wanted"


def test_misclassified_raises_on_negative_max_images():
    with pytest.raises(ValueError, match="0 or more"):
        collect_misclassified_images(
            nn.Identity(), make_logits_loader(), "cpu", wanted_pairs={(0, 1)}, max_images=-1
        )