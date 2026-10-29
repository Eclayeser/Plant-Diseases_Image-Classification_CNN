# Runs model training and validation, records results and saves checkpoints.
import torch
from torch import nn
from leafapp.config import load_config
from leafapp.data import get_loaders
from leafapp.models.cnn import BasicCNN

cfg = load_config("configs/prototype.yaml")
train_loader, val_loader, test_loader, class_names = get_loaders(
    fraction=cfg["data"]["fraction"],
    image_size=cfg["data"]["image_size"],
    batch_size=cfg["data"]["batch_size"],
)

model = BasicCNN(
    num_classes=len(class_names),
    channels=cfg["model"]["channels"],
    image_size=cfg["data"]["image_size"],
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])