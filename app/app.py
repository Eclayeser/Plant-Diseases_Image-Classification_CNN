import os
from pathlib import Path
import gradio as gr
from PIL import Image
from app.components import LeafImage, IMAGE_READ_ERRORS
from leafapp.config import MODEL_DIR, MODEL_FILE, LABELS_FILE, MODEL_CONFIG_FILE, ROOT
from leafapp.model_registry import load_model_registry
from leafapp.model_manager import ModelManager
from leafapp.labels import format_class_name
from leafapp.predict import LeafClassifier

Image.MAX_IMAGE_PIXELS = 6_500_000  # Pillow errors at 2x this: 13M pixels
COLORS = ["#2e7d32", "#1565c0", "#ef8f00", "#d84315", "#6a1b9a"]  # one colour per rank
APP_DIR = Path(__file__).parent
GALLERY_DIR = APP_DIR / "gallery"
ABOUT_MODELS = (APP_DIR / "about_models.md").read_text(encoding="utf-8")
APP_CSS = (APP_DIR / "style.css").read_text(encoding="utf-8")
NO_LEAF_CLASS = "Background_without_leaves"
NO_LEAF_MESSAGE = "<p><strong>No leaf found.</strong> Please, try a photo of a single leaf.</p>"
MAX_BATCH_FILES = 20  # keeps memory and request time low on Render's free tier
APP_TITLE = "Plant Disease Classification"
DISCLAIMER = "It gives a first opinion for the crops and diseases the model was trained on and is not a diagnosis."
UPLOAD_CLEANUP = (600, 600)  # every 10 minutes, delete uploads older than 10 minutes
SHARED_SETTINGS = {  # settings used by both tabs
    "title": APP_TITLE,
    "flagging_mode": "never",
    "delete_cache": UPLOAD_CLEANUP,
    "article": ABOUT_MODELS,
}

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


def display_name(class_name):
    """Readable class name; the background class is shown as "No leaf found"."""
    if class_name == NO_LEAF_CLASS:
        return "No leaf found"
    return format_class_name(class_name)


def predict_batch(classifier, files, model_name=None):
    """Return one table row (file name, top prediction, confidence) for each uploaded image."""
    rows = []

    if not files:
        return rows

    if len(files) > MAX_BATCH_FILES:
        raise gr.Error(f"Please upload at most {MAX_BATCH_FILES} images at a time.")

    for path in files:
        file_name = Path(path).name

        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except Image.DecompressionBombError:
            rows.append([file_name, f"Image too large (max {2 * Image.MAX_IMAGE_PIXELS // 1_000_000} megapixels)", ""])
            continue
        except IMAGE_READ_ERRORS:
            rows.append([file_name, "Couldn't read this file", ""])
            continue

        if model_name is None:
            predictions = classifier.top_k(image, k=1)
        else:
            predictions = classifier.top_k(model_name, image, k=1)
        top_name, probability = predictions[0]
        prediction = display_name(top_name)

        confidence = f"{probability * 100:.1f}%"

        rows.append([file_name, prediction, confidence])

    return rows


def build_model_selector(model_names, default_model=None):
    """Create a validated dropdown shared by both prediction interfaces."""
    if not model_names:
        raise ValueError("At least one model must be available")
    if default_model is None:
        default_model = model_names[0]
    if default_model not in model_names:
        raise ValueError("Default model must be selectable")
    return gr.Dropdown(
        choices=model_names, value=default_model,
        label="Classification Model", interactive=True,
    )


def format_predictions(predictions):
    """Render the prediction chart and the existing no-leaf warning."""
    names = [display_name(name) for name, _ in predictions]
    percents = [round(probability * 100, 1) for _, probability in predictions]
    chart = prediction_chart_html(names, percents)
    if predictions and predictions[0][0] == NO_LEAF_CLASS:
        return NO_LEAF_MESSAGE + chart
    return chart


def build_ui(classifier, model_names=None, default_model=None):
    """Build the single-image tab, optionally with model selection."""
    def predict_single(image):
        if image is None:
            return None
        return format_predictions(classifier.top_k(image, k=5))

    def predict_selected(image, model_name):
        if image is None:
            return None
        return format_predictions(classifier.top_k(model_name, image, k=5))

    image_input = LeafImage(type="pil", label="Upload a leaf image")
    gallery = sorted(GALLERY_DIR.glob("*.jpg"))
    if model_names is None:
        inputs = image_input
        predict_fn = predict_single
        examples = [[str(path)] for path in gallery]
    else:
        selector = build_model_selector(model_names, default_model)
        inputs = [image_input, selector]
        predict_fn = predict_selected
        examples = [[str(path), selector.value] for path in gallery]

    return gr.Interface(
        fn=predict_fn,
        inputs=inputs,
        outputs=gr.HTML(label="Top 5 Predictions"),
        description=f"Upload a plant leaf image to identify its disease, or pick one from the gallery below. {DISCLAIMER}",
        examples=examples,
        api_name="predict",
        **SHARED_SETTINGS,
    )


def build_batch_ui(classifier, model_names=None, default_model=None):
    """Build the batch tab, optionally with model selection."""
    def predict(files):
        return predict_batch(classifier, files)

    def predict_selected(files, model_name):
        return predict_batch(classifier, files, model_name=model_name)

    file_input = gr.File(file_count="multiple", file_types=["image"],
                         type="filepath", label="Upload leaf images")
    if model_names is None:
        inputs = file_input
        predict_fn = predict
    else:
        inputs = [file_input, build_model_selector(model_names, default_model)]
        predict_fn = predict_selected

    return gr.Interface(
        fn=predict_fn,
        inputs=inputs,
        outputs=gr.Dataframe(
            headers=["File", "Prediction", "Confidence"],
            label="Predictions",
            column_widths=["30%", "38%", "32%"],
            wrap=True,
        ),
        description=f"Upload up to {MAX_BATCH_FILES} leaf photos at once to get the top prediction for each one. {DISCLAIMER}",
        api_name="predict_batch",
        **SHARED_SETTINGS,
    )


def build_deployed_interfaces():
    """Construct both tabs using the same deployed classifier or model manager."""
    registry = load_model_registry(ROOT / "deployment" / "model.lock.json")
    models = registry["models"]
    if len(models) == 1:
        classifier = LeafClassifier.from_dir(MODEL_DIR)
        return build_ui(classifier), build_batch_ui(classifier)

    model_dirs = {
        model["name"]: MODEL_DIR / f"model_{index}"
        for index, model in enumerate(models)
    }
    for model_dir in model_dirs.values():
        for file_name in (MODEL_FILE, LABELS_FILE, MODEL_CONFIG_FILE):
            file_path = model_dir / file_name
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"Missing model file: {file_path}. "
                    "Run: python -m scripts.fetch_model_from_clearml"
                )

    manager = ModelManager(model_dirs=model_dirs)
    default_model = next(
        model["name"] for model in models
        if model["model_id"] == registry["model_id"]
    )
    model_names = list(model_dirs)
    return (
        build_ui(manager, model_names, default_model),
        build_batch_ui(manager, model_names, default_model),
    )


def build_deployed_ui():
    """Build the complete application with both prediction tabs."""
    demo = gr.TabbedInterface(
        list(build_deployed_interfaces()),
        tab_names=["Single image", "Batch prediction"],
    )
    demo.title = APP_TITLE
    demo.delete_cache = UPLOAD_CLEANUP
    return demo


if __name__ == "__main__":
    demo = build_deployed_ui()
    demo.launch(
        server_port=int(os.getenv("PORT", 8080)),
        max_file_size="10mb",
        css=APP_CSS,
    )
