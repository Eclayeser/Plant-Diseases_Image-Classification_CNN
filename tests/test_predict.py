# Checks image prediction and the format of the returned results
from PIL import Image

from leafapp.checkpoints import save_checkpoint
from leafapp.data import make_transforms
from leafapp.models.cnn import BasicCNN
from leafapp.predict import load_model, preprocess_image

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


