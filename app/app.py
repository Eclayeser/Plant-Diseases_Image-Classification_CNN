import os

import gradio as gr
import pandas as pd
from PIL import Image

from leafapp.config import MODEL_DIR
from leafapp.predict import LeafClassifier

Image.MAX_IMAGE_PIXELS = 20_000_000
COLORS = ["#2e7d32", "#1565c0", "#ef8f00", "#d84315", "#6a1b9a"]  # one colour per rank


def format_class_name(name):
    """Turn a label such as "Tomato___Leaf_Mold" into "Tomato - Leaf Mold"."""
    return name.replace("___", " - ").replace("_", " ")


def prediction_list_html(ranks, names, percents):
    """Show predictions with matching colours."""
    rows = []

    for rank, name, percent, color in zip(ranks, names, percents, COLORS):
        text = f"{rank} {name}: {percent}%"
        weight = "bold" if rank == "#1" else "normal"
        rows.append(f'<p style="color:{color}; font-weight:{weight};">{text}</p>')

    return "".join(rows)


def build_ui(classifier):
    """Build the Gradio interface"""

    def predict(image):
        if image is None:
            return None, None

        predictions = classifier.top_k(image, k=5)
        ranks = [f"#{i + 1}" for i in range(len(predictions))]
        names = [format_class_name(name) for name, _ in predictions]
        percents = [round(probability * 100, 1) for _, probability in predictions]

        chart = pd.DataFrame({"Rank": ranks, "Disease": names, "Probability (%)": percents})
        return chart, prediction_list_html(ranks, names, percents)

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload a leaf image"),
        outputs=[
            gr.BarPlot(
                x="Rank",
                y="Probability (%)",
                y_title="Probability (%)",
                y_lim=[0, 100],
                title="Top 5 Predictions",
                x_title=" ",
                color="Rank",
                color_map=dict(zip(["#1", "#2", "#3", "#4", "#5"], COLORS)),
                color_title=" ",
                height=350,
                tooltip="all",
            ),
            gr.HTML(label="Predicted diseases"),
        ],
        title="Plant Disease Classification",
        description="Upload a plant leaf image to identify its disease. The bars and the list below use the same colours.",
        api_name="predict",
        flagging_mode="never"
    )
    return demo


if __name__ == "__main__":
    classifier = LeafClassifier.from_dir(MODEL_DIR)
    demo = build_ui(classifier)
    demo.launch(server_port=int(os.getenv("PORT", 8080)), max_file_size="10mb")