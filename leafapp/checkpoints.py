# Saves and loads model weights together with the information needed to use them.
import torch
from leafapp.models.cnn import BasicCNN

def save_checkpoint(path, model, channels, image_size, class_names):
    """
    Save the model weights plus the settings needed to rebuild it.
    """
    checkpoint = {
        "state_dict":model.state_dict(),
        "channels":channels,
        "image_size":image_size,
        "class_names":class_names,
    }

    torch.save(checkpoint, path)

def load_checkpoint(path):
    """
    Rebuild a BasicCNN from a checkpoint saved with save_checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu")

    model = BasicCNN(
        num_classes=len(checkpoint["class_names"]),
        channels=checkpoint["channels"],
        image_size=checkpoint["image_size"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, checkpoint["class_names"], checkpoint["image_size"]