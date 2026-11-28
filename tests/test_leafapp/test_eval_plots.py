import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from leafapp import eval_plots
from leafapp.data import PLANT_VILLAGE_MEAN, PLANT_VILLAGE_STD
from leafapp.eval_plots import (
    prepare_image_for_plotting,
    plot_and_save_misclassified_grid,
    plot_and_save_training_curve,
)
from tests.shared_helpers import FAKE_CLASS_NAMES

HISTORY = {
    "loss": {"training": [0.9, 0.5], "validation": [1.0, 0.7]},
    "accuracy": {"training": [0.6, 0.8], "validation": [0.5, 0.7]},
}


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

    plot_and_save_misclassified_grid([], ["healthy", "rust", "blight"], plot_path)

    assert plot_path.is_file(), "Should save an empty grid instead of crashing"


def test_show_displays_the_figure(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(eval_plots.plt, "show", lambda: shown.append(True))

    plot_and_save_misclassified_grid([], FAKE_CLASS_NAMES, tmp_path / "grid.png", show=True)

    assert shown == [True], "show=True should display the figure once"


def test_training_curve_skips_excluded_metrics_and_run_types(tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr(eval_plots, "_save_figure", lambda fig, path, show: saved.append(fig))

    plot_and_save_training_curve(
        HISTORY, tmp_path / "curve.png",
        exclude_metrics=["loss"], exclude_run_types=["validation"],
    )

    (axis,) = saved[0].axes
    plt.close(saved[0])
    assert axis.get_title() == "CNN Accuracy", "Only the accuracy subplot should be drawn"
    assert [line.get_label() for line in axis.get_lines()] == ["Training"]


def test_training_curve_saves_nothing_when_every_metric_is_excluded(tmp_path):
    plot_path = tmp_path / "curve.png"

    plot_and_save_training_curve(HISTORY, plot_path, exclude_metrics=["loss", "accuracy"])

    assert not plot_path.exists(), "Should not save an empty figure"
