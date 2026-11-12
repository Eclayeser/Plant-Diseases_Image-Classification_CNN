import torch

def make_fake_loader():
    return [
        {"pixels": torch.randn(4, 3, 32, 32), "label": torch.randint(0, 5, (4,))}
        for _ in range(2)
    ]