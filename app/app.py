
import os

import gradio as gr
import pandas as pd
import torch

from leafapp.checkpoints import load_checkpoint
from leafapp.data import make_transforms


CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")


def load_model():
    # Load the trained CNN model
    if not CHECKPOINT_PATH:
        return None, None, None

    model, class_names, _, config = load_checkpoint(CHECKPOINT_PATH)

    # Use the same image preprocessing as evaluation
    _, transform = make_transforms(
    config["data"]["image_size"],
    augment=False,
    normalize=config["data"].get("normalize", False),
    )

    return model, class_names, transform


def get_cnn_predictions(image, model, class_names):
    model.eval()

    # Get predictions from the CNN
    with torch.no_grad():
        output = model(image.unsqueeze(0))

    # Convert the output into probabilities
    probabilities = torch.softmax(output, dim=1)[0]

    # Find the 5 most likely diseases
    top5 = torch.topk(probabilities, 5)

    diseases = []
    for i in top5.indices:
        name = class_names[int(i)].replace("___", " - ").replace("_", " ")
        diseases.append(name)

    return pd.DataFrame({
        "Disease": diseases,
        "Probability": top5.values.tolist()
    })


def predict(image):
    
    if image is None:
        return None

    if MODEL is None:
        raise gr.Error("No trained model loaded. Set CHECKPOINT_PATH to a trained checkpoint.")

    # Prepare the uploaded image
    image = image.convert("RGB")
    processed_image = TRANSFORM(image)

    return get_cnn_predictions(processed_image, MODEL, CLASS_NAMES)


MODEL, CLASS_NAMES, TRANSFORM = load_model()


# Create the Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a leaf image"),
    outputs=gr.BarPlot(
        x="Disease",
        y="Probability",
        title="Top 5 Predictions",
        x_label_angle=45,
        sort="-y"
    ),
    title="Plant Disease Classification",
    description="Upload a plant leaf image to identify its disease.",
    api_name="predict"
)


if __name__ == "__main__":
    demo.launch()
