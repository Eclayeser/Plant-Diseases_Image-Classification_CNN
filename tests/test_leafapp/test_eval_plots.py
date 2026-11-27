import torch
from torchvision import transforms
from leafapp.data import PLANT_VILLAGE_MEAN, PLANT_VILLAGE_STD
from leafapp.eval_plots import (
    prepare_image_for_plotting,
    plot_and_save_misclassified_grid,
)


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
