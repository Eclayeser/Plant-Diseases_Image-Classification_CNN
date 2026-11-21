from unittest.mock import Mock

import pytest
import torch

import compute_norm_stats


@pytest.fixture
def get_loaders(monkeypatch):
    train_batches = [{"pixels": torch.zeros(1, 3, 2, 2)}]
    loader = Mock(return_value=(train_batches, None, None, None))
    monkeypatch.setattr(compute_norm_stats, "get_loaders", loader)
    return loader


def test_default_uses_prototype_settings(get_loaders):
    compute_norm_stats.main([])

    get_loaders.assert_called_once_with(
        fraction=0.2, image_size=128, batch_size=32, seed=42, augment=False, normalize=False,
    )


def test_custom_config_disables_augmentation_and_normalisation(tmp_path, get_loaders):
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "seed: 44\n"
        "data:\n"
        "  fraction: 1.0\n"
        "  image_size: 96\n"
        "  batch_size: 8\n"
        "  augment: true\n"
        "  normalize: plant_village\n",
        encoding="utf-8",
    )

    compute_norm_stats.main(["--config", str(config)])

    get_loaders.assert_called_once_with(
        fraction=1.0, image_size=96, batch_size=8, seed=44, augment=False, normalize=False,
    )
