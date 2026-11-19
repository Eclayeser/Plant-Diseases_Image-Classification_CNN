import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from PIL import Image

from leafapp.config import LABELS_FILE, MODEL_CONFIG_FILE, MODEL_FILE
from leafapp.predict import LeafClassifier, preprocess_image, softmax

CLASS_NAMES = ["red", "green", "blue"]
IMAGE_SIZE = 16


def write_model_dir(model_dir, class_names=CLASS_NAMES, image_size=IMAGE_SIZE):
    """Write the 3 files export_onnx produces, with a tiny model instead of a CNN.

    The model averages each colour channel, so the logits are the mean red,
    green and blue values and a plain red image is classified as "red".
    """
    graph = helper.make_graph(
        nodes=[
            helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["logits"]),
        ],
        name="channel_means",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, image_size, image_size])],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    model_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(model_dir / MODEL_FILE))
    (model_dir / LABELS_FILE).write_text(json.dumps(class_names), encoding="utf-8")
    (model_dir / MODEL_CONFIG_FILE).write_text(json.dumps({"image_size": image_size}), encoding="utf-8")
    return model_dir


class FakeSession:
    """Stands in for an onnxruntime session and returns fixed logits."""

    def __init__(self, logits, input_name="pixels"):
        self.logits = np.asarray([logits], dtype=np.float32)
        self.input_name = input_name
        self.inputs = None

    def get_inputs(self):
        class Input:
            name = self.input_name

        return [Input()]

    def run(self, output_names, inputs):
        self.inputs = inputs
        return [self.logits]


def test_preprocess_image():
    """A grayscale image becomes a [1, 3, 32, 32] array."""
    image = Image.new("L", (100, 80))

    x = preprocess_image(image, 32)

    assert x.shape == (1, 3, 32, 32)


def test_preprocess_image_scales_to_0_1_in_rgb_order():
    """Pixels are float32 in 0-1, with channels in RGB order."""
    image = Image.new("RGB", (50, 40), color=(255, 0, 0))

    x = preprocess_image(image, 8)

    assert x.dtype == np.float32
    assert np.allclose(x[0, 0], 1.0)
    assert np.allclose(x[0, 1:], 0.0)


def test_preprocess_image_matches_eval_transform():
    """preprocess_image gives the same pixels as make_transforms(augment=False)."""
    pytest.importorskip("torch")
    from leafapp.data import make_transforms

    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (90, 120, 3), dtype=np.uint8))
    _, eval_tf = make_transforms(32, augment=False, normalize=False)

    expected = eval_tf(image).unsqueeze(0).numpy()

    assert np.allclose(preprocess_image(image, 32), expected, atol=1e-6)


def test_softmax():
    """Probabilities sum to 1, keep the order of the logits and don't overflow."""
    probabilities = softmax(np.array([1000.0, 999.0, 0.0]))

    assert np.isfinite(probabilities).all()
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_top_k_sorted_pairs():
    """top_k returns k (name, probability) pairs, most likely first."""
    session = FakeSession([0.1, 3.0, 2.0, -1.0])
    classifier = LeafClassifier(session, ["a", "b", "c", "d"], IMAGE_SIZE)

    predictions = classifier.top_k(Image.new("RGB", (20, 20)), k=3)

    assert [name for name, _ in predictions] == ["b", "c", "a"]
    probabilities = [p for _, p in predictions]
    assert all(isinstance(p, float) for p in probabilities)
    assert probabilities == sorted(probabilities, reverse=True)


def test_top_k_feeds_preprocessed_image():
    """The session gets the preprocessed image under the model's input name."""
    session = FakeSession([0.0, 1.0])
    classifier = LeafClassifier(session, ["a", "b"], IMAGE_SIZE)

    classifier.top_k(Image.new("RGB", (40, 30)))

    assert list(session.inputs) == ["pixels"]
    assert session.inputs["pixels"].shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_top_k_with_fewer_classes_than_k():
    """With fewer classes than k, every class is returned once."""
    classifier = LeafClassifier(FakeSession([0.0, 1.0]), ["a", "b"], IMAGE_SIZE)

    predictions = classifier.top_k(Image.new("RGB", (20, 20)), k=5)

    assert sorted(name for name, _ in predictions) == ["a", "b"]


def test_from_dir_loads_export_files(tmp_path):
    """from_dir reads the labels and image size, and the model runs end to end."""
    model_dir = write_model_dir(tmp_path / "model")

    classifier = LeafClassifier.from_dir(model_dir)
    predictions = classifier.top_k(Image.new("RGB", (64, 48), color=(255, 0, 0)))

    assert classifier.class_names == CLASS_NAMES
    assert classifier.image_size == IMAGE_SIZE
    assert predictions[0][0] == "red"
    assert np.isclose(sum(p for _, p in predictions), 1.0)


def test_from_dir_without_model(tmp_path):
    """A missing model.onnx raises an error that says how to fetch it."""
    with pytest.raises(FileNotFoundError, match="fetch_model_from_clearml"):
        LeafClassifier.from_dir(tmp_path)
