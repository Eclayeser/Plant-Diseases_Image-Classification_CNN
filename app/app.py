import os
from pathlib import Path
import gradio as gr
from PIL import Image
from app.components import LeafImage
from leafapp.config import MODEL_DIR
from leafapp.labels import format_class_name
from leafapp.predict import LeafClassifier

Image.MAX_IMAGE_PIXELS = 20_000_000
COLORS = ["#2e7d32", "#1565c0", "#ef8f00", "#d84315", "#6a1b9a"]  # one colour per rank
APP_DIR = Path(__file__).parent
GALLERY_DIR = APP_DIR / "gallery"
ABOUT_MODELS = (APP_DIR / "about_models.md").read_text(encoding="utf-8")
APP_CSS = (APP_DIR / "style.css").read_text(encoding="utf-8")
NO_LEAF_CLASS = "Background_without_leaves"
NO_LEAF_MESSAGE = "<p><strong>No leaf found.</strong> Please, try a photo of a single leaf.</p>"


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


def predict_batch(classifier, files):
    """Return one table row (file name, top prediction, confidence) for each uploaded image."""
    rows = []

    if not files:
        return rows

    for path in files:
        file_name = Path(path).name

        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except (OSError, Image.DecompressionBombError):
            rows.append([file_name, "Couldn't read this file", ""])
            continue

        top_name, probability = classifier.top_k(image, k=1)[0]

        if top_name == NO_LEAF_CLASS:
            prediction = "No leaf found"
        else:
            prediction = format_class_name(top_name)

        confidence = f"{probability * 100:.1f}%"

        rows.append([file_name, prediction, confidence])

    return rows


def build_ui(classifier):
    """Build the Gradio interface"""

    def predict(image):
        if image is None:
            return None

        predictions = classifier.top_k(image, k=5)
        names = [format_class_name(name) for name, _ in predictions]
        percents = [round(probability * 100, 1) for _, probability in predictions]
        chart = prediction_chart_html(names, percents)

        if predictions[0][0] == NO_LEAF_CLASS:
            return NO_LEAF_MESSAGE + chart
        return chart

    demo = gr.Interface(
        fn=predict,
        inputs=LeafImage(type="pil", label="Upload a leaf image"),
        outputs=gr.HTML(label="Top 5 Predictions"),
        title="Plant Disease Classification",
        description=(
            "Upload a plant leaf image to identify its disease, or pick one from the gallery below. "
            "It gives a first opinion for the crops and diseases the model was trained on and is not a diagnosis."
        ),
        examples=[[str(path)] for path in sorted(GALLERY_DIR.glob("*.jpg"))],
        api_name="predict",
        flagging_mode="never",
        delete_cache=(600, 600),  # every 10 minutes, delete uploads older than 10 minutes
        article=ABOUT_MODELS,
    )
    return demo


if __name__ == "__main__":
    classifier = LeafClassifier.from_dir(MODEL_DIR)
    demo = build_ui(classifier)
    demo.launch(server_port=int(os.getenv("PORT", 8080)), max_file_size="10mb", css=APP_CSS)