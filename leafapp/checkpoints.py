# Saves and loads model weights together with the information needed to use them.
from pathlib import Path
import torch
from leafapp.models.cnn import BasicCNN

ROOT = Path(__file__).resolve().parent.parent

def save_checkpoint(path, model, cfg, class_names, history_metrics):
    """
    Save the model weights, the run's config, the class names and history metrcis.

    The config lets the model be rebuilt and the same data split be
    recreated (seed, fraction) when the checkpoint is evaluated later.
    """
    checkpoint = {
        "state_dict":model.state_dict(),
        "config":cfg,
        "class_names":class_names,
        "history_metrics": history_metrics,
    }

    path = Path(path)                      # works whether a str or Path was passed
    tmp_path = path.with_suffix(".tmp")    # e.g. outputs/basic_cnn.tmp
    torch.save(checkpoint, tmp_path)       # if this fails, the old checkpoint is untouched
    tmp_path.replace(path)                 # swap the finished file into place

def load_checkpoint(path):
    """
    Rebuild a BasicCNN from a checkpoint saved with save_checkpoint.
    A relative path is resolved against the project root, not the current directory.

    Returns (model, class_names, history_metrics, config). The model is in eval mode.
    """
    checkpoint = torch.load(ROOT / path, map_location="cpu")
    config = checkpoint["config"]

    model = BasicCNN(
        num_classes=len(checkpoint["class_names"]),
        channels=config["model"]["channels"],
        image_size=config["data"]["image_size"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, checkpoint["class_names"], checkpoint["history_metrics"], config