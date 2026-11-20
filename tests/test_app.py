import runpy
import sys

import gradio as gr
import pandas as pd
import pytest
from PIL import Image

import leafapp.config
from app.app import build_ui, format_class_name
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
