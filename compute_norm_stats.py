# Computes the mean and std of each colour over the plant_village training photos.
import argparse

import torch

from leafapp.config import ROOT, load_config
from leafapp.data import get_loaders


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute RGB statistics on the training split.")
    parser.add_argument("--config", default="configs/prototype.yaml", help="Path to the experiment YAML.")
    args = parser.parse_args(argv)

    cfg = load_config(ROOT / args.config)
    data = cfg["data"]
    train_loader, _, _, _ = get_loaders(
        fraction=data["fraction"],
        image_size=data["image_size"],
        batch_size=data["batch_size"],
        seed=cfg["seed"],
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
