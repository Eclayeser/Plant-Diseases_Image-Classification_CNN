import os

import gradio as gr
from PIL import Image

from leafapp.config import MODEL_DIR, MODEL_FILE, LABELS_FILE, MODEL_CONFIG_FILE, ROOT
from leafapp.predict import LeafClassifier
from leafapp.model_registry import load_model_registry
from leafapp.model_manager import ModelManager

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


def build_ui(classifier, model_names=None, default_model=None):
    def format_predictions(predictions):
        """Convert the top predictions into the existing HTML chart."""
        names = [
            format_class_name(name)
            for name, _ in predictions
        ]

        percents = [
            round(probability * 100, 1)
            for _, probability in predictions
        ]

        return prediction_chart_html(names, percents)

    def predict_single(image):
        if image is None:
            return None

        predictions = classifier.top_k(image, k=5)
        return format_predictions(predictions)

    def predict_selected(image, model_name):
        if image is None:
            return None

        predictions = classifier.top_k(
            model_name,
            image,
            k=5,
        )

        return format_predictions(predictions)

    image_input = gr.Image(
        type="pil",
        label="Upload a leaf image",
    )

    if model_names is None:
        inputs = image_input
        predict_fn = predict_single

    else:
        if not model_names:
            raise ValueError("At least one model must be available")

        if default_model is None:
            default_model = model_names[0]

        if default_model not in model_names:
            raise ValueError("Default model must be selectable")
        model_selector = gr.Dropdown(
            choices=model_names,
            value=default_model,
            label="Classification Model",
            interactive=True,
        )
        inputs = [image_input, model_selector]
        predict_fn = predict_selected

    demo = gr.Interface(
        fn=predict_fn,
        inputs=inputs,
        outputs=gr.HTML(label="Top 5 Predictions"),
        title="Plant Disease Classification",
        description="Upload a plant leaf image to identify its disease.",
        api_name="predict",
        flagging_mode="never",
    )
    return demo


if __name__ == "__main__":
    lock_file = ROOT / "deployment" / "model.lock.json"
    registry = load_model_registry(lock_file)
    models = registry["models"]

    if len(models) == 1:
        classifier = LeafClassifier.from_dir(MODEL_DIR)
        demo = build_ui(classifier)
    else:
        model_dirs = {
            model["name"]: MODEL_DIR / f"model_{index}"
            for index, model in enumerate(models)
        }

        required_files = MODEL_FILE, LABELS_FILE, MODEL_CONFIG_FILE
        for model_dir in model_dirs.values():
            for file_name in required_files:
                file_path = model_dir / file_name

                if not file_path.is_file():
                    raise FileNotFoundError(
                        f"Missing model file: {file_path}. "
                        "Run: python -m scripts.fetch_model_from_clearml"
                    )

        manager = ModelManager(model_dirs=model_dirs)

        default_model_id = registry["model_id"]
        default_model = next(
            model["name"]
            for model in models
            if model["model_id"] == default_model_id
        )

        demo = build_ui(
            manager,
            model_names=list(model_dirs.keys()),
            default_model=default_model,
        )

    demo.launch(
        server_port=int(os.getenv("PORT", 8080)),
        max_file_size="10mb",
    )