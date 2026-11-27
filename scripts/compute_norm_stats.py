# Computes the mean and std of each colour over the plant_village training photos.
import torch

from leafapp.data import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FRACTION,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SEED,
    get_loaders,
)


def main():
    train_loader, _, _, _ = get_loaders(
        fraction=DEFAULT_FRACTION,
        image_size=DEFAULT_IMAGE_SIZE,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=DEFAULT_SEED,
        augment=False,
        normalize=False,
    )

    total = torch.zeros(3, dtype=torch.float64)
    squares = torch.zeros(3, dtype=torch.float64)
    count = 0

    for batch in train_loader:
        pixels = batch["pixels"]
        total += pixels.sum(dim=(0, 2, 3))
        squares += (pixels ** 2).sum(dim=(0, 2, 3))
        count += pixels.shape[0] * pixels.shape[2] * pixels.shape[3]

    mean = total / count
    std = (squares / count - mean ** 2).sqrt()
    print("mean:", mean)
    print("std:", std)


if __name__ == "__main__":
    main()
