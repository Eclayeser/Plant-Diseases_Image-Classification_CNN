import os

import gradio as gr
from PIL import Image

from leafapp.config import MODEL_DIR
from leafapp.predict import LeafClassifier

Image.MAX_IMAGE_PIXELS = 20_000_000
COLORS = ["#2e7d32", "#1565c0", "#ef8f00", "#d84315", "#6a1b9a"]  # one colour per rank


def format_class_name(name):
    """Turn a label such as "Tomato___Leaf_Mold" into "Tomato - Leaf Mold"."""
    return name.replace("___", " - ").replace("_", " ")


def prediction_chart_html(names, percents):
    """Show the top 5 predictions as horizontal bars."""
    rows = ["<h3>Top 5 Predictions</h3>"]

    for i, (name, percent, color) in enumerate(zip(names, percents, COLORS)):
        weight = "bold" if i == 0 else "normal"

        name_html = f'<p style="font-weight:{weight}; margin:10px 0 4px;">{name}</p>'

        # Grey background with a coloured bar showing the percentage
        bar_html = (
            f'<div style="flex:1; background:#eeeeee;">'
            f'<div style="width:{percent}%; background:{color}; height:22px;"></div>'
            f'</div>'
        )

        percent_html = f'<span style="width:55px; text-align:right;">{percent}%</span>'

        bar_row = f'<div style="display:flex; align-items:center; gap:8px;">{bar_html}{percent_html}</div>'

        row = f'<div>{name_html}{bar_row}</div>'

        rows.append(row)

    return "".join(rows)


def build_ui(classifier):
    """Build the Gradio interface"""

    def predict(image):
        if image is None:
            return None

        predictions = classifier.top_k(image, k=5)
        names = [format_class_name(name) for name, _ in predictions]
        percents = [round(probability * 100, 1) for _, probability in predictions]
        return prediction_chart_html(names, percents)

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload a leaf image"),
        outputs=gr.HTML(label="Top 5 Predictions"),
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