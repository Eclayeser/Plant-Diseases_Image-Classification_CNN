# Runs model training and validation, records results and saves checkpoints.
import argparse
import logging
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
from leafapp.early_stopping import EarlyStopping
from leafapp.eval_functions import (
    calculate_metrics_per_loader,
    calculate_f1_score,
    calculate_accuracy,
)
from leafapp.tracking import (
    close_task,
    initialise_task,
    log_epoch_metrics,
    log_confusion_matrix,
    upload_checkpoint,
)

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
RUN_TYPES = ("debugging", "exploring", "comparison", "production")    # see README.md to understand difference
NAMES = ("Mark Tarnavskyi", "Pranay Singh", "Illia Putintsev", "Javier Fimia", "Reon Carroll-Ito", "Nil Kayalioglu", "Mario Westphalen Cavicchioli")

def parse_args(argv=None):
    """
    Read the command-line arguments and return them.
    
    If argv is None, the real command line is used.
    """

    parser = argparse.ArgumentParser(description="Train a CNN model and save a checkpoint.")
    parser.add_argument("--config", default="configs/prototype.yaml", help="Path to the YAML config file for this run.")
    parser.add_argument("--responsible", required=True, choices=NAMES, help="Name who is responsible for / triggered the run.")
    parser.add_argument("--run-type", choices=RUN_TYPES, default="debugging", help="Describe the purpose of the run.")
    parser.add_argument("--issue-id", default="", help="Pin issue identifier in format 'SGP-XXX'", type=validate_issue_id)
    parser.add_argument("--checkpoint", default=5, metavar="CHECKPOINT_INTERVAL", help="Saves a checkpoint every N epoch(s) (default: 5).", type=validate_checkpoint_interval)
    return parser.parse_args(argv)


def sanitise_name(value):
    """
    Sanitises name into lowercase kebab format.
    """
    slug = "-".join(value.lower().split())
    return slug


def validate_issue_id(value):
    """
    Ensures issue is in the right format. Raises otherwise
    """
    if value and not value.startswith("SGP-"):
        raise argparse.ArgumentTypeError("--issue-id must be in 'SGP-XXX' format.")
    return value

def validate_checkpoint_interval(value):
    """
    Ensures checkpoint interval is a positive integer. Raises otherwise
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("--checkpoint must be a positive integer.")
    return ivalue

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

def main(config_path, responsible, run_type, issue_id, checkpoint_interval=5):
    """
    Train a model using the config at config_path and save a checkpoint.

    Prints train/val loss, accuracy and f1 each epoch and records them as
    history, then saves the model, config, class names and history to a
    uniquely named file (timestamp + short id) in checkpoint_dir.
    """
    task = None
    model = None
    class_names = None
    history_metrics = None
    checkpoint_path = None
    best_checkpoint_path = None
    best_checkpoint_pending = False
    final_checkpoint_uploaded = False
    best_checkpoint_uploaded = False
    try:
        cfg = load_config(ROOT / config_path) 
        cfg["data"].setdefault("normalize", False)  # record the default for ClearML visibility
        cfg["train"].setdefault("patience", None) # None = early stopping off
        cfg["train"].setdefault("min_delta", 0.0) # 0.0 = any drop counts as improvement
        torch.manual_seed(cfg["seed"])
        train_loader, val_loader, _, class_names = get_loaders(
            fraction=cfg["data"]["fraction"],
            image_size=cfg["data"]["image_size"],
            batch_size=cfg["data"]["batch_size"],
            seed=cfg["seed"],
            normalize=cfg["data"].get("normalize", False),
            augment=cfg["data"].setdefault("augment", True),
        )

        model = build_model(cfg["model"], cfg["data"], class_names)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print(f"Training device: GPU ({torch.cuda.get_device_name(device)})")
        else:
            print("Training device: CPU (CUDA unavailable; falling back to CPU)")
        model.to(device)
        optimizer = choose_optimizer(model.parameters(), cfg["train"])
        config_name = Path(config_path).stem

        tags=[
            f"@run-type:{run_type}",
            f"@model:{cfg['model']['type']}",
            f"@responsible:{sanitise_name(responsible)}",
            "@task-type:train.py",
            f"@config-name:{config_name}"
        ]
        if issue_id:
            tags.append(f"@issue:{issue_id}")

        task = initialise_task(
            config=cfg,
            project_name="Plant-Disease-Image-Classification",
            task_name=f"{cfg['model']['type']} - {config_name}",
            tags=tags,
            
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

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = uuid.uuid4().hex[:6]
        checkpoint_dir = ROOT / cfg["train"]["checkpoint_dir"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{cfg['model']['type'].lower()}_{timestamp}_{run_id}.pt"
        best_checkpoint_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_best.pt"
        )
        best_validation_loss = float("inf")
        patience = cfg["train"]["patience"]
        if patience is not None:
            early_stopping = EarlyStopping(patience=patience, min_delta=cfg["train"]["min_delta"])
        else:
            early_stopping = None

        for epoch in range(cfg["train"]["epochs"]):
            train_epoch_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_epoch_metrics = calculate_metrics_per_loader(
                model=model,
                loader=val_loader,
                device=device,
                class_names=class_names,
                criterion=criterion,                # triggers loss calculation
                metrics_list=["accuracy", "f1", "confusion-matrix"]     # + the indicated metrics
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
                validation_metrics={
                    name: value
                    for name, value in val_epoch_metrics.items()
                    if name != "confusion-matrix"
                }, 
            )
            log_confusion_matrix(
                logger=logger,
                epoch=epoch + 1,
                matrix=val_epoch_metrics["confusion-matrix"],
                class_names=class_names,
            )

            if val_epoch_metrics["loss"] < best_validation_loss:
                best_validation_loss = val_epoch_metrics["loss"]
                best_checkpoint_pending = True
                save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    cfg=cfg,
                    class_names=class_names,
                    history_metrics=history_metrics,
                )
                best_checkpoint_pending = False

            if (epoch + 1) % checkpoint_interval == 0:
                epoch_checkpoint_path = checkpoint_path.with_name(
                    f"{checkpoint_path.stem}_epoch_{epoch + 1}.pt"
                )
                save_checkpoint(
                    path=epoch_checkpoint_path,
                    model=model,
                    cfg=cfg,
                    class_names=class_names,
                    history_metrics=history_metrics,
                )
                print(f"Saved checkpoint to {epoch_checkpoint_path}")
                try:
                    upload_checkpoint(
                        task,
                        epoch_checkpoint_path,
                        f"checkpoint_epoch_{epoch + 1}",
                    )
                except Exception:
                    LOGGER.exception(
                        "Could not upload periodic checkpoint for epoch %s; "
                        "local checkpoint saved to %s",
                        epoch + 1,
                        epoch_checkpoint_path,
                    )

            print(
                    f"Epoch {epoch + 1}: "
                    f"train loss {train_epoch_metrics['loss']:.3f}, train acc {train_epoch_metrics['accuracy']:.3f}, train f1 {train_epoch_metrics['f1']:.3f} | "
                    f"val loss {val_epoch_metrics['loss']:.3f}, val acc {val_epoch_metrics['accuracy']:.3f}, val f1 {val_epoch_metrics['f1']:.3f}"
                )
            
            if early_stopping is not None and early_stopping.step(val_epoch_metrics["loss"]):
                print(f"Stopped early after epoch {epoch + 1}: no improvement for {patience} epochs")
                logger.report_single_value("stopped_epoch", epoch + 1)
                break

        save_checkpoint(
            path=checkpoint_path,
            model=model,
            cfg=cfg,
            class_names=class_names,
            history_metrics=history_metrics,
        )
        print(f"Saved checkpoint to {checkpoint_path}")
        upload_checkpoint(task, checkpoint_path, "final_checkpoint")
        final_checkpoint_uploaded = True
        upload_checkpoint(task, best_checkpoint_path, "best_checkpoint")
        best_checkpoint_uploaded = True
        close_task(task)
    except BaseException as exc:
        if task is not None:
            if (
                not final_checkpoint_uploaded
                and checkpoint_path is not None
                and model is not None
                and class_names is not None
                and history_metrics is not None):
                try:
                    save_checkpoint(
                        path=checkpoint_path,
                        model=model,
                        cfg=cfg,
                        class_names=class_names,
                        history_metrics=history_metrics,
                    )
                except Exception:
                    LOGGER.exception("Could not save the final checkpoint while handling a failed run")

            if (
                best_checkpoint_pending
                and best_checkpoint_path is not None
                and model is not None
                and class_names is not None
                and history_metrics is not None):
                try:
                    save_checkpoint(
                        path=best_checkpoint_path,
                        model=model,
                        cfg=cfg,
                        class_names=class_names,
                        history_metrics=history_metrics,
                    )
                except Exception:
                    LOGGER.exception(
                        "Could not retry saving the pending best checkpoint while handling a failed run"
                    )
                else:
                    best_checkpoint_pending = False

            if (
                not best_checkpoint_uploaded
                and best_checkpoint_path is not None
                and not best_checkpoint_path.is_file()
                and checkpoint_path is not None
                and checkpoint_path.is_file()
                and model is not None
                and class_names is not None
                and history_metrics is not None):

                LOGGER.warning("No validation-best checkpoint exists; using the failure snapshot "
                    "as the best-checkpoint fallback")
                try:
                    save_checkpoint(
                        path=best_checkpoint_path,
                        model=model,
                        cfg=cfg,
                        class_names=class_names,
                        history_metrics=history_metrics,
                    )
                except Exception:
                    LOGGER.exception("Could not save the best checkpoint fallback")

            for path, checkpoint_name in (
                (checkpoint_path, "Final"),
                (best_checkpoint_path, "Best"),
            ):
                if path is not None and path.is_file():
                    print(f"{checkpoint_name} checkpoint saved to {path}")
            
            for path, name, already_uploaded in (
                (checkpoint_path, "final_checkpoint", final_checkpoint_uploaded),
                (best_checkpoint_path, "best_checkpoint", best_checkpoint_uploaded)):
                if path is None or already_uploaded or not path.is_file():
                    continue
                try:
                    upload_checkpoint(task, path, name)
                except Exception:
                    LOGGER.exception(f"Could not upload the {name} checkpoint while handling a failed run")
            
            close_task(task, failed=True, message=str(exc))
        raise

if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.responsible, args.run_type, args.issue_id, args.checkpoint)