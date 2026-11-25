import runpy
import sys

import gradio as gr
from gradio.components.image import ImageData
import pytest
from PIL import Image

import leafapp.config
from app.app import GALLERY_DIR, build_batch_ui, build_ui, format_class_name, predict_batch
from leafapp.predict import LeafClassifier


class FakeClassifier:
    """Stands in for LeafClassifier and returns fixed predictions."""

    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def top_k(self, image, k=5):
        self.calls.append((image, k))
        return self.predictions[:k]


PREDICTIONS = [
    ("Tomato___Leaf_Mold", 0.6),
    ("Tomato___healthy", 0.2),
    ("Apple___Apple_scab", 0.1),
    ("Corn_(maize)___Common_rust_", 0.06),
    ("Potato___Late_blight", 0.03),
    ("Grape___Black_rot", 0.01),
]


def test_format_class_name():
    """Dataset labels are shown as "Plant - Disease"."""
    assert format_class_name("Tomato___Leaf_Mold") == "Tomato - Leaf Mold"
    assert format_class_name("Apple___healthy") == "Apple - healthy"


def test_build_ui_returns_interface():
    """build_ui returns a Gradio Interface with an image input."""
    demo = build_ui(FakeClassifier(PREDICTIONS))

    assert isinstance(demo, gr.Interface)
    assert isinstance(demo.input_components[0], gr.Image)


def test_predict_returns_top_5_html():
    """Check that the top 5 predictions are shown in the HTML chart."""
    classifier = FakeClassifier(PREDICTIONS)
    demo = build_ui(classifier)
    image = Image.new("RGB", (300, 200), color="green")

    html = demo.fn(image)

    assert classifier.calls == [(image, 5)]

    for name, probability in PREDICTIONS[:5]:
        assert format_class_name(name) in html
        assert f"{round(probability * 100, 1)}%" in html

    assert format_class_name(PREDICTIONS[5][0]) not in html


def test_predict_without_image():
    """With no image uploaded, predict returns None for both outputs and doesn't call the model."""
    classifier = FakeClassifier(PREDICTIONS)
    demo = build_ui(classifier)

    assert demo.fn(None) is None
    assert classifier.calls == []


def run_app_main(monkeypatch):
    """Run app/app.py the way `python -m app.app` does.

    app.app is imported above, so it's removed from sys.modules first, or
    runpy warns about running a module that is already imported.
    """
    monkeypatch.delitem(sys.modules, "app.app", raising=False)
    runpy.run_module("app.app", run_name="__main__")


def test_main_fails_without_model(tmp_path, monkeypatch):
    """Without a fetched model, the app stops at startup instead of serving errors."""
    monkeypatch.setattr(leafapp.config, "MODEL_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="fetch_model_from_clearml"):
        run_app_main(monkeypatch)


def test_main_launches_on_port_env(monkeypatch):
    """The app listens on $PORT (set by Render) and limits upload size."""
    launched = {}
    monkeypatch.setattr(LeafClassifier, "from_dir", classmethod(lambda cls, model_dir: FakeClassifier(PREDICTIONS)))
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, **kwargs: launched.update(kwargs))
    monkeypatch.setenv("PORT", "10000")

    run_app_main(monkeypatch)

    assert launched["server_port"] == 10000
    assert launched["max_file_size"] == "10mb"


def test_main_default_port(monkeypatch):
    """Without $PORT, the app listens on 8080 as in the local deployment docs."""
    launched = {}
    monkeypatch.setattr(LeafClassifier, "from_dir", classmethod(lambda cls, model_dir: FakeClassifier(PREDICTIONS)))
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, **kwargs: launched.update(kwargs))
    monkeypatch.delenv("PORT", raising=False)

    run_app_main(monkeypatch)

    assert launched["server_port"] == 8080


def test_example_images_can_be_opened():
    """The gallery folder has leaf images that can be opened."""
    images = list(GALLERY_DIR.glob("*.jpg"))

    assert len(images) > 0, "There should be gallery images in app/gallery"

    for path in images:
        image = Image.open(path)
        assert image.width > 0, f"{path.name} should be a valid image"
        image.close()


def test_build_ui_shows_example_images():
    """build_ui adds every gallery image to the interface."""
    demo = build_ui(FakeClassifier(PREDICTIONS))

    images = sorted(GALLERY_DIR.glob("*.jpg"))
    expected = []

    for path in images:
        expected.append([str(path)])

    assert demo.examples == expected, "All example images should be shown in the app"


def test_predict_works_on_example_images():
    """Each gallery image can be sent to predict and gives a chart."""
    classifier = FakeClassifier(PREDICTIONS)
    demo = build_ui(classifier)

    images = list(GALLERY_DIR.glob("*.jpg"))

    for path in images:
        image = Image.open(path)
        result = demo.fn(image)
        image.close()

        assert "Top 5 Predictions" in result, f"{path.name} should give a prediction chart"

    assert len(classifier.calls) == len(images), "The model should be called once per example"


def test_build_ui_deletes_old_uploads():
    """Uploaded photos are deleted from the server after about 10 minutes."""
    demo = build_ui(FakeClassifier(PREDICTIONS))

    assert demo.delete_cache == (600, 600)


def test_description_says_not_a_diagnosis():
    """The app tells users its prediction is not a diagnosis."""
    demo = build_ui(FakeClassifier(PREDICTIONS))

    assert "not a diagnosis" in demo.description


def test_predict_warns_when_no_leaf_found():
    """A photo whose top class is the background class gets a "No leaf found" message."""
    predictions = [("Background_without_leaves", 0.9)] + PREDICTIONS[:4]
    demo = build_ui(FakeClassifier(predictions))
    image = Image.new("RGB", (300, 200), color="green")

    html = demo.fn(image)

    assert "No leaf found" in html


def test_predict_has_no_warning_for_a_leaf():
    """A normal leaf prediction shows the chart without the warning."""
    demo = build_ui(FakeClassifier(PREDICTIONS))
    image = Image.new("RGB", (300, 200), color="green")

    html = demo.fn(image)

    assert "No leaf found" not in html


def test_unreadable_file_shows_a_clear_error(tmp_path):
    """A file that isn't a readable image gives a clear message instead of a generic error."""
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    demo = build_ui(FakeClassifier(PREDICTIONS))

    with pytest.raises(gr.Error, match="Couldn't read this file"):
        demo.input_components[0].preprocess(ImageData(path=str(broken)))


def test_about_panel_explains_the_models():
    """The About panel warns that predictions can be wrong and lists the known limitations."""
    demo = build_ui(FakeClassifier(PREDICTIONS))

    assert "can be wrong" in demo.article
    assert "Known limitations" in demo.article


def test_image_input_uses_the_normal_image_box():
    """The custom input is drawn with Gradio's normal image component in the browser."""
    demo = build_ui(FakeClassifier(PREDICTIONS))

    assert demo.input_components[0].get_block_name() == "image"


def test_main_adds_the_app_styles(monkeypatch):
    """The app is launched with the styles for the AI note and the Read more button."""
    launched = {}
    monkeypatch.setattr(LeafClassifier, "from_dir", classmethod(lambda cls, model_dir: FakeClassifier(PREDICTIONS)))
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, **kwargs: launched.update(kwargs))

    run_app_main(monkeypatch)

    assert ".about-models summary" in launched["css"]


def test_predict_batch_returns_one_row_per_image(tmp_path):
    """Each uploaded image gets its file name, top prediction and confidence."""
    paths = []

    for name in ["leaf1.jpg", "leaf2.jpg"]:
        path = tmp_path / name
        Image.new("RGB", (64, 64), "green").save(path)
        paths.append(str(path))

    rows = predict_batch(FakeClassifier(PREDICTIONS), paths)

    assert rows == [
        ["leaf1.jpg", "Tomato - Leaf Mold", "60.0%"],
        ["leaf2.jpg", "Tomato - Leaf Mold", "60.0%"],
    ]


def test_predict_batch_skips_unreadable_files(tmp_path):
    """A broken file gets an error row instead of stopping the whole batch."""
    broken = tmp_path / "broken.jpg"
    broken.write_text("not an image")

    good = tmp_path / "good.jpg"
    Image.new("RGB", (64, 64), "green").save(good)

    rows = predict_batch(
        FakeClassifier(PREDICTIONS),
        [str(broken), str(good)]
    )

    assert rows == [
        ["broken.jpg", "Couldn't read this file", ""],
        ["good.jpg", "Tomato - Leaf Mold", "60.0%"],
    ]


def test_predict_batch_without_files():
    """No uploaded files gives an empty table."""
    assert predict_batch(FakeClassifier(PREDICTIONS), None) == []


def test_build_batch_ui_returns_interface(tmp_path):
    """The batch tab takes several files and shows the predictions in a table."""
    demo = build_batch_ui(FakeClassifier(PREDICTIONS))

    path = tmp_path / "leaf.jpg"
    Image.new("RGB", (64, 64), "green").save(path)

    assert isinstance(demo, gr.Interface)
    assert isinstance(demo.input_components[0], gr.File)
    assert isinstance(demo.output_components[0], gr.Dataframe)

    rows = demo.fn([str(path)])

    assert rows == [["leaf.jpg", "Tomato - Leaf Mold", "60.0%"]]


def test_main_shows_single_and_batch_tabs(monkeypatch):
    """The app has one tab for a single image and one for many images."""
    launched = {}

    monkeypatch.setattr(
        LeafClassifier,
        "from_dir",
        classmethod(lambda cls, model_dir: FakeClassifier(PREDICTIONS))
    )

    monkeypatch.setattr(
        gr.Blocks,
        "launch",
        lambda self, **kwargs: launched.update(demo=self, delete_cache=self.delete_cache)
    )

    run_app_main(monkeypatch)

    demo = launched["demo"]
    assert isinstance(demo, gr.TabbedInterface)

    tabs = demo.children[0].children

    assert len(tabs) == 2
    assert tabs[0].label == "Single image"
    assert tabs[1].label == "Batch prediction"
    assert launched["delete_cache"] == (600, 600)


def test_predict_batch_skips_files_pillow_cannot_parse(tmp_path, monkeypatch):
    """Some broken PNGs make Pillow raise ValueError instead of OSError."""
    good = tmp_path / "good.jpg"
    bad = tmp_path / "bad.png"
    Image.new("RGB", (64, 64), "green").save(good)

    real_open = Image.open

    def fake_open(path):
        if path == str(bad):
            raise ValueError("Broken image")
        return real_open(path)

    monkeypatch.setattr(Image, "open", fake_open)

    rows = predict_batch(FakeClassifier(PREDICTIONS), [str(bad), str(good)])

    assert rows == [
        ["bad.png", "Couldn't read this file", ""],
        ["good.jpg", "Tomato - Leaf Mold", "60.0%"],
    ]
