# Loads a trained model and returns predictions for an uploaded image.
import pandas as pd
import torch

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
def preprocess_image(image, transform):
    # Prepare the uploaded image in the same way as during evaluation
    image = image.convert("RGB")
    return transform(image)

def get_cnn_predictions(image, model, class_names):
    model.eval()

    # Get predictions from the CNN
    with torch.no_grad():
        output = model(image.unsqueeze(0))

    # Convert the output into probabilities
    probabilities = torch.softmax(output, dim=1)[0]

    # Find the 5 most likely diseases
    top5 = torch.topk(probabilities, min(5, len(class_names)))

    diseases = []
    for i in top5.indices:
        name = class_names[int(i)].replace("___", " - ").replace("_", " ")
        diseases.append(name)

    return pd.DataFrame({
        "Disease": diseases,
        "Probability": top5.values.tolist()
    })

