# Runs model training and validation, records results and saves checkpoints.
from pathlib import Path
from datetime import datetime
import uuid

import torch
from torch import nn
from leafapp.config import load_config
from leafapp.data import get_loaders
from leafapp.models.cnn import BasicCNN
from leafapp.checkpoints import save_checkpoint

from leafapp.eval_functions import loss_accuracy_per_epoch

ROOT = Path(__file__).resolve().parent

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one pass over the loader, updating its weights.
    
    Raises ValueError if the loader is empty.
    Returns (average loss, accuracy) over all batches in the epoch.
    """
    model.train()

    total_loss, correct, total = 0, 0, 0

    for batch in loader:
        images, labels = batch["pixels"], batch["label"]
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()             # clear last batch's gradients
        logits = model(images)            # predict
        loss = criterion(logits, labels)  # how wrong?
        loss.backward()                   # work out how each weight should change
        optimizer.step()                  # change them

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise ValueError("The loader produced no samples, so the metrics can't be computed")

    return total_loss / total, correct / total


def main():
    """
    Train BasicCNN using configs/prototype.yaml and save a checkpoint.

    Prints train/val loss and accuracy each epoch, then saves the model,
    config and class names to a uniquely named file (timestamp + short id)
    in checkpoint_dir.
    """
    cfg = load_config(ROOT / "configs" / "prototype.yaml")
    torch.manual_seed(cfg["seed"])
    train_loader, val_loader, test_loader, class_names = get_loaders(
        fraction=cfg["data"]["fraction"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        seed=cfg["seed"],
    )

    model = BasicCNN(
        num_classes=len(class_names),
        channels=cfg["model"]["channels"],
        image_size=cfg["data"]["image_size"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    for epoch in range(cfg["train"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = loss_accuracy_per_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}: train loss {train_loss:.3f}, train acc {train_acc:.2f} | val loss {val_loss:.3f}, val acc {val_acc:.2f}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:6]   # e.g. "a3f9c1": 6 random hex characters
    checkpoint_path = ROOT / cfg["train"]["checkpoint_dir"] / f"basic_cnn_{timestamp}_{run_id}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        cfg=cfg,
        class_names=class_names,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()