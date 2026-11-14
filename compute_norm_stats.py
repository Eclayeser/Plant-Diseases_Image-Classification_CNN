# Computes the mean and std of each colour over the plant_village training photos.
import torch
from leafapp.data import get_loaders

train_loader, _, _, _ = get_loaders(
    fraction=0.2, image_size=128, batch_size=32, seed=42,
    augment=False, normalize=False,
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