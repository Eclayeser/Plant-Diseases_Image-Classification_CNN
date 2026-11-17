# Loads a trained model and returns predictions for an uploaded image.

from leafapp.checkpoints import load_checkpoint
from leafapp.data import make_transforms

def load_model(checkpoint_path):
    # Load the trained CNN model
    if not checkpoint_path:
        return None, None, None

    model, class_names, _, config = load_checkpoint(checkpoint_path)

    # Use the same image preprocessing as evaluation
    _, transform = make_transforms(
        config["data"]["image_size"],
        augment=False,
        normalize=config["data"].get("normalize", False),
    )

    return model, class_names, transform


