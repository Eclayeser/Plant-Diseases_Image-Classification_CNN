import os

import gradio as gr
import pandas as pd


from PIL import Image
from leafapp.predict import load_model, predict_top_k


Image.MAX_IMAGE_PIXELS = 20_000_000


CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")

if CHECKPOINT_PATH:
    MODEL, CLASS_NAMES, TRANSFORM = load_model(CHECKPOINT_PATH)
else:
    MODEL, CLASS_NAMES, TRANSFORM = None, None, None



def predict(image):
    
    if image is None:
        return None

    if MODEL is None:
        raise gr.Error("No trained model loaded. Set CHECKPOINT_PATH to a trained checkpoint.")

    predictions = predict_top_k(image, MODEL, CLASS_NAMES, TRANSFORM)

    # Gradio's bar plot expects a DataFrame
    return pd.DataFrame(predictions, columns=["Disease", "Probability"])




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
    demo.launch(max_file_size="10mb")
