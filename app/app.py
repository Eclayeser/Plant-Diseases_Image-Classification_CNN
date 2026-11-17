import os
import gradio as gr
from PIL import Image
from leafapp.predict import load_model, get_cnn_predictions, preprocess_image


Image.MAX_IMAGE_PIXELS = 20_000_000


CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")

MODEL, CLASS_NAMES, TRANSFORM = load_model(CHECKPOINT_PATH)

def predict(image):
    
    if image is None:
        return None

    if MODEL is None:
        raise gr.Error("No trained model loaded. Set CHECKPOINT_PATH to a trained checkpoint.")

    # Prepare the uploaded image
    processed_image = preprocess_image(image, TRANSFORM)

    return get_cnn_predictions(processed_image, MODEL, CLASS_NAMES)
    

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
