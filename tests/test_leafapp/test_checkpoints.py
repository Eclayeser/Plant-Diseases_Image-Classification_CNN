# Checks that checkpoints save and load a model without losing anything.
import torch
from leafapp.models.cnn import BasicCNN
from leafapp.checkpoints import save_checkpoint, load_checkpoint
from torchvision import models
from leafapp.models.resnet import build_resnet18


def test_checkpoint_round_trip_gives_same_outputs(tmp_path):
    cfg = {"model": {"type": "BasicCNN", "channels": [8]}, "data": {"image_size": 32}}
    class_names = ["a", "b", "c"]
    model = BasicCNN(num_classes=len(class_names), channels=cfg["model"]["channels"], image_size=cfg["data"]["image_size"])
    model.eval()

    path = tmp_path / "model.pt"
    save_checkpoint(path, model, cfg, class_names, history_metrics={})
    loaded_model, _, _, _ = load_checkpoint(path)
    assert not loaded_model.training, "Loaded model should be in eval mode"

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(loaded_model(x), model(x)), "Loaded model gives different outputs"

def test_checkpoint_keeps_config_class_names_and_history_metrics(tmp_path):
    cfg = {"model": {"type": "BasicCNN", "channels": [8]}, "data": {"image_size": 32}, "seed": 42}
    class_names = ["a", "b", "c"]
    model = BasicCNN(num_classes=len(class_names), channels=cfg["model"]["channels"], image_size=cfg["data"]["image_size"])

    path = tmp_path / "model.pt"
    history_metrics = {"train_loss": [0.9, 0.5], "val_acc": [0.4, 0.7]}
    save_checkpoint(path, model, cfg, class_names, history_metrics)
    _, loaded_names, loaded_history, loaded_config = load_checkpoint(path)

    assert loaded_names == class_names, "Class names changed after loading"
    assert loaded_history == history_metrics, "History metrics changed after loading"
    assert loaded_config == cfg, "Config changed after loading"


def test_resnet_checkpoint_loads_without_pretrained_weights(tmp_path, monkeypatch):
    cfg = {"model": {"type": "ResNet18"}, "data": {"image_size": 32}}
    class_names = ["a", "b", "c"]
    model = build_resnet18(num_classes=len(class_names), pretrained=False)

    path = tmp_path / "resnet.pt"
    save_checkpoint(path, model, cfg, class_names, history_metrics={})

    # Fail the test if loading the checkpoint asks for ImageNet weights
    real_resnet18 = models.resnet18

    def check_weights(weights=None):
        assert weights is None, "Loading a checkpoint should not download ImageNet weights"
        return real_resnet18(weights=None)

    monkeypatch.setattr(models, "resnet18", check_weights)
    loaded_model, _, _, _ = load_checkpoint(path)

    assert torch.equal(loaded_model.fc.weight, model.fc.weight), "Loaded ResNet18 has different weights"