import os

import gradio as gr
import pandas as pd
from PIL import Image

from leafapp.config import MODEL_DIR
from leafapp.labels import format_class_name
from leafapp.predict import LeafClassifier

Image.MAX_IMAGE_PIXELS = 20_000_000


def build_ui(classifier):
    """Build the Gradio interface"""

    def predict(image):
        if image is None:
            return None

        predictions = classifier.top_k(image, k=5)
        return pd.DataFrame({
            "Disease": [format_class_name(name) for name, _ in predictions],
            "Probability": [probability for _, probability in predictions],
        })

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
        api_name="predict",
        flagging_mode="never"
    )
    return demo


if __name__ == "__main__":
    classifier = LeafClassifier.from_dir(MODEL_DIR)
    demo = build_ui(classifier)
    demo.launch(server_port=int(os.getenv("PORT", 8080)), max_file_size="10mb")