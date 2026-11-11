# Runs model training and validation, records results and saves checkpoints.
import argparse
from pathlib import Path
from datetime import datetime
import uuid

import torch
from torch import nn
from leafapp.config import load_config
from leafapp.model_constructor import build_model
from leafapp.optimizer_constructor import choose_optimizer
from leafapp.data import get_loaders
from leafapp.checkpoints import save_checkpoint
from leafapp.eval_functions import (
    calculate_metrics_per_loader,
    calculate_f1_score,
    calculate_accuracy,
)
from leafapp.tracking import (
    close_task,
    initialise_task,
    log_epoch_metrics,
    upload_checkpoint,
)

ROOT = Path(__file__).resolve().parent

def parse_args(argv=None):
    """
    Read the command-line arguments and return them (currently just --config).
    
    If argv is None, the real command line is used.
    """

    parser = argparse.ArgumentParser(description="Train a CNN model and save a checkpoint.")
    parser.add_argument("--config", default="configs/prototype.yaml", help="Path to the YAML config file for this run.")
    return parser.parse_args(argv)

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one pass over the loader, updating its weights.
    
    Raises ValueError if the loader is empty.
    Returns a dict with the epoch's average "loss", "accuracy" and "f1" (macro).
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

def main(config_path):
    """
    Train a model using the config at config_path and save a checkpoint.

    Prints train/val loss, accuracy and f1 each epoch and records them as
    history, then saves the model, config, class names and history to a
    uniquely named file (timestamp + short id) in checkpoint_dir.
    """
    cfg = load_config(ROOT / config_path) 
    torch.manual_seed(cfg["seed"])
    train_loader, val_loader, _, class_names = get_loaders(
        fraction=cfg["data"]["fraction"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        seed=cfg["seed"],
        normalize=cfg["data"].get("normalize", False),
    )

    model = build_model(cfg["model"], cfg["data"], class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    config_name = Path(config_path).stem
    task = initialise_task(
        config=cfg,
        project_name="Plant-Disease-Image-Classification",
        task_name=f"{cfg['model']['type']} - {config_name}",
        tags=["training", config_name],
    )
    logger = task.get_logger()

    # metrics which are to record per-epoch
    history_metrics = {
        "losses": {
            "training": [],
            "validation": []
        },
        "accuracies": {
            "training": [],
            "validation": [] 
        },
        "f1s": {
            "training": [],
            "validation": [] 
        }
    }

    criterion = nn.CrossEntropyLoss()
    optimizer = choose_optimizer(model.parameters(), cfg["train"])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    checkpoint_dir = ROOT / cfg["train"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{cfg['model']['type'].lower()}_{timestamp}_{run_id}.pt"
    best_checkpoint_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_best.pt"
    )
    best_validation_loss = float("inf")

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
        
        history_metrics["losses"]["training"].append(train_epoch_metrics["loss"]) 
        history_metrics["accuracies"]["training"].append(train_epoch_metrics["accuracy"])
        history_metrics["f1s"]["training"].append(train_epoch_metrics["f1"])
        history_metrics["losses"]["validation"].append(val_epoch_metrics["loss"])
        history_metrics["accuracies"]["validation"].append(val_epoch_metrics["accuracy"])
        history_metrics["f1s"]["validation"].append(val_epoch_metrics["f1"])

        log_epoch_metrics(
            logger=logger,
            epoch=epoch + 1,
            training_metrics=train_epoch_metrics,
            validation_metrics=val_epoch_metrics,
        )

        if val_epoch_metrics["loss"] < best_validation_loss:
            best_validation_loss = val_epoch_metrics["loss"]
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                cfg=cfg,
                class_names=class_names,
                history_metrics=history_metrics,
            )

        print(
                f"Epoch {epoch + 1}: "
                f"train loss {train_epoch_metrics['loss']:.3f}, train acc {train_epoch_metrics['accuracy']:.3f}, train f1 {train_epoch_metrics['f1']:.3f} | "
                f"val loss {val_epoch_metrics['loss']:.3f}, val acc {val_epoch_metrics['accuracy']:.3f}, val f1 {val_epoch_metrics['f1']:.3f}"
            )

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        cfg=cfg,
        class_names=class_names,
        history_metrics=history_metrics,
    )
    print(f"Saved checkpoint to {checkpoint_path}")
    upload_checkpoint(task, checkpoint_path, "final_checkpoint")
    upload_checkpoint(task, best_checkpoint_path, "best_checkpoint")
    close_task(task)

if __name__ == "__main__":
    args = parse_args()
    main(args.config)