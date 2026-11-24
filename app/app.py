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
GALLERY_DIR = Path(__file__).parent / "gallery"
NO_LEAF_CLASS = "Background_without_leaves"
NO_LEAF_MESSAGE = "<p><strong>No leaf found.</strong> Please, try a photo of a single leaf.</p>"
ABOUT_MODELS = """
<p class="ai-note">ⓘ Predictions come from AI models and can be wrong.</p>

<details class="about-models">
<summary>Read more</summary>

- **Intended use.** We realise many people may treat our models' predictions as a source of truth. However, we recommend using the models only for a first opinion on the likely disease, and we encourage you to confirm the result with an agronomist or plant-health expert before any treatment.
- **Training data.** Our models learned from 55,447 photographs in the PlantVillage dataset, each showing a single leaf on a plain background in a lab. They cover 26 diseases and 12 types of healthy leaf, plus photos without a leaf, so the app can tell you when it can't find one.
- **Crops covered.** Our models can recognise leaves from 14 crops: apple, blueberry, cherry, corn, grape, orange, peach, bell pepper, potato, raspberry, soybean, squash, strawberry and tomato. The models haven't been trained on any other plants, so a leaf from another plant will very likely get a wrong prediction.
- **Known limitations.** To get the most reliable result, it helps to know where our models can go wrong:
    - Photos taken outdoors, with busy backgrounds, uneven lighting or several leaves, look different from the photos our models learned from, so results may be less accurate. A clear photo of one leaf on a plain surface works best for our models.
    - The models always pick one of their 39 classes, even for a disease they don't know, so an unfamiliar disease will still get a label, and that label will be wrong.
    - Unfortunately, our training data does not cover every disease for every crop. For blueberry, raspberry, soybean, orange and squash, the models have seen only healthy leaves or only a single disease, so please take some extra time to cross-check the results for these crops.
    - Some leaf types had far fewer training photos than others, so the models are more reliable for some crops than for others.
- **Evaluation.** We tested each model on photos it never saw during training. Please see the results, including how well each crop and disease is recognised, in our [model card](https://github.com/Eclayeser/Plant-Diseases_Image-Classification_CNN/blob/main/documents/model_card.md).

</details>
"""

APP_CSS = """
.ai-note { padding: 8px 12px; border-left: 3px solid var(--color-accent); background: var(--background-fill-secondary); }
.about-models summary { cursor: pointer; display: inline-block; list-style: none; padding: 6px 14px; border: 1px solid var(--border-color-primary); border-radius: var(--radius-lg); font-weight: 600; }
.about-models summary::-webkit-details-marker { display: none; }
.about-models summary::after { content: " ▾"; }
.about-models[open] summary::after { content: " ▴"; }
.about-models summary:hover { background: var(--background-fill-secondary); }
.about-models[open] summary { margin-bottom: 12px; }
"""


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