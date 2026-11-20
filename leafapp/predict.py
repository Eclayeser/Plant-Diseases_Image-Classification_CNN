import json
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image

from leafapp.config import (
    LABELS_FILE,
    MODEL_CONFIG_FILE,
    MODEL_DIR,
    MODEL_FILE,
)


def preprocess_image(image, image_size):
    """Prepare an image the way make_transforms(augment=False) does, without torch.

    Return a float32 array of shape (1, 3, image_size, image_size) with values
    in 0-1. Normalisation is not done here: it is part of the ONNX graph.
    """
    image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return pixels.transpose(2, 0, 1)[np.newaxis]


def softmax(logits):
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


class LeafClassifier:
    """An exported ONNX model with its class names and input size."""

    def __init__(self, session, class_names, image_size):
        self.session = session
        self.class_names = class_names
        self.image_size = image_size
        self.input_name = session.get_inputs()[0].name

    @classmethod
    def from_dir(cls, model_dir=MODEL_DIR):
        """Load the model, labels and config that export_onnx wrote to model_dir."""
        model_dir = Path(model_dir)
        model_path = model_dir / MODEL_FILE
        if not model_path.exists():
            raise FileNotFoundError(f"No model at {model_path}. Run: python -m scripts.fetch_model_from_clearml")

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1  # Render free tier has 0.1 CPU
        options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )

        class_names = json.loads((model_dir / LABELS_FILE).read_text(encoding="utf-8"))
        config = json.loads((model_dir / MODEL_CONFIG_FILE).read_text(encoding="utf-8"))
        return cls(session, class_names, config["image_size"])

    def top_k(self, image, k=5):
        """Return the k most likely (class_name, probability) pairs, most likely first."""
        pixels = preprocess_image(image, self.image_size)
        logits = self.session.run(None, {self.input_name: pixels})[0][0]
        probabilities = softmax(logits)
        best = np.argsort(probabilities)[::-1][:k]
        return [(self.class_names[i], float(probabilities[i])) for i in best]
