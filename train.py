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

from leafapp.eval_functions import (
    calculate_metrics_per_loader,
    calculate_f1_score,
    calculate_accuracy,
)

ROOT = Path(__file__).resolve().parent

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one pass over the loader, updating its weights.
    
    Raises ValueError if the loader is empty.
    Returns (average loss, accuracy) over all batches in the epoch.
    """
    model.train()

    total_loss, total = 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        images, labels = batch["pixels"], batch["label"]
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()             # clear last batch's gradients
        logits = model(images)            # predict
        loss = criterion(logits, labels)  # how wrong?
        loss.backward()                   # work out how each weight should change
        optimizer.step()                  # change them
 
        batch_size = labels.size(0)
        total += batch_size

        total_loss += loss.item() * batch_size
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    if total == 0:
        raise ValueError("The loader produced no samples, so the metrics can't be computed")

    true_labels, predicted_labels = torch.cat(all_labels), torch.cat(all_preds)
    f1 = calculate_f1_score(true_labels, predicted_labels)
    accuracy = calculate_accuracy(true_labels, predicted_labels)

    epoch_metrics = {
        "loss": total_loss / total,
        "accuracy": accuracy,
        "f1": f1
    }

    return epoch_metrics



def main():
    """
    Train BasicCNN using configs/prototype.yaml and save a checkpoint.

    Prints train/val loss and accuracy each epoch, then saves the model,
    config and class names to a uniquely named file (timestamp + short id)
    in checkpoint_dir.
    """
    cfg = load_config(ROOT / "configs" / "prototype.yaml")
    torch.manual_seed(cfg["seed"])
    train_loader, val_loader, _, class_names = get_loaders(
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

    # metrics which are to record per-epoch
    history_metrics = {
        "train_losses": [],
        "train_accuracies": [],
        "train_f1s": [],
        "val_losses": [],
        "val_accuracies": [],
        "val_f1s": []
    }

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    for epoch in range(cfg["train"]["epochs"]):
        train_epoch_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_epoch_metrics = calculate_metrics_per_loader(
            model=model,
            loader=val_loader,
            device=device,
            class_names=class_names,
            criterion=criterion,                # triggers loss calculation
            metrics_list=["accuracy", "f1"]     # + the indicated metrics
        )
        
        history_metrics["train_losses"].append(train_epoch_metrics["loss"]) 
        history_metrics["train_accuracies"].append(train_epoch_metrics["accuracy"])
        history_metrics["train_f1s"].append(train_epoch_metrics["f1"])
        history_metrics["val_losses"].append(val_epoch_metrics["loss"])
        history_metrics["val_accuracies"].append(val_epoch_metrics["accuracy"])
        history_metrics["val_f1s"].append(val_epoch_metrics["f1"])

        print(
                f"Epoch {epoch + 1}: "
                f"train loss {train_epoch_metrics["loss"]:.3f}, train acc {train_epoch_metrics["accuracy"]:.3f}, train f1 {train_epoch_metrics["f1"]:.3f} | "
                f"val loss {val_epoch_metrics["loss"]:.3f}, val acc {val_epoch_metrics["accuracy"]:.3f}, val f1 {val_epoch_metrics["f1"]:.3f}"
            )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:6]   # e.g. "a3f9c1": 6 random hex characters
    checkpoint_path = ROOT / cfg["train"]["checkpoint_dir"] / f"basic_cnn_{timestamp}_{run_id}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        cfg=cfg,
        class_names=class_names,
        history_metrics=history_metrics,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()