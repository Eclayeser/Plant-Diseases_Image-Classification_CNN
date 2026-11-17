# Checks image prediction and the format of the returned results
import torch
from PIL import Image
from torch import nn

from leafapp.checkpoints import save_checkpoint
from leafapp.data import make_transforms
from leafapp.models.cnn import BasicCNN
from leafapp.predict import get_cnn_predictions, load_model, preprocess_image

def test_load_model(tmp_path):
    """The model, class names and transform are loaded from a checkpoint."""
    cfg = {"model": {"type": "BasicCNN", "channels": [8]}, "data": {"image_size": 32}}
    class_names = ["a", "b", "c"]
    model = BasicCNN(num_classes=3, channels=[8], image_size=32)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, cfg, class_names, history_metrics={})

    model, loaded_names, transform = load_model(path)

    assert model is not None
    assert loaded_names == class_names
    assert transform is not None


def test_load_model_without_checkpoint():
    """No model is loaded when there is no checkpoint path."""
    assert load_model(None) == (None, None, None)

def test_preprocess_image():
    """A grayscale image becomes a [3, 32, 32] tensor."""
    _, transform = make_transforms(32, augment=False)
    image = Image.new("L", (100, 80))

    x = preprocess_image(image, transform)

    assert x.shape == (3, 32, 32)

def test_get_cnn_predictions():
    """The top 5 classes are returned in order with readable names."""
    class_names = ["Apple___scab", "Tomato___Leaf_Mold", "Corn___healthy",
                   "Grape___Black_rot", "Potato___Late_blight", "Soybean___healthy"]
    logits = torch.tensor([1.0, 6.0, 2.0, 5.0, 3.0, 4.0])

    predictions = get_cnn_predictions(logits, nn.Identity(), class_names)

    assert list(predictions.columns) == ["Disease", "Probability"]
    assert len(predictions) == 5
    assert predictions["Disease"][0] == "Tomato - Leaf Mold"
    assert predictions["Probability"].is_monotonic_decreasing


