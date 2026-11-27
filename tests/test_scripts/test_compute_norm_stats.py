from unittest.mock import Mock

import torch

from scripts import compute_norm_stats


def test_stats_use_original_training_split_without_augmentation_or_normalisation(monkeypatch):
    train_batches = [{"pixels": torch.zeros(1, 3, 2, 2)}]
    loader = Mock(return_value=(train_batches, None, None, None))
    monkeypatch.setattr(compute_norm_stats, "get_loaders", loader)
    compute_norm_stats.main()

    loader.assert_called_once_with(
        fraction=0.2, image_size=128, batch_size=32, seed=42, augment=False, normalize=False,
    )
