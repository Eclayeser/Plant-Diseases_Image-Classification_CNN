# Checks that checkpoints save and load a model without losing anything.
import torch
from leafapp.checkpoints import save_checkpoint, load_checkpoint
from leafapp.models.resnet import build_resnet18
from tests.shared_helpers import FAKE_IMAGE_SIZE, expect_resnet18_weights, make_tiny_model

CLASS_NAMES = ["a", "b", "c"]
CFG = {"model": {"type": "BasicCNN", "channels": [8]}, "data": {"image_size": FAKE_IMAGE_SIZE}}


def test_checkpoint_round_trip_gives_same_outputs(tmp_path):
    model = make_tiny_model(num_classes=len(CLASS_NAMES))
    model.eval()

    path = tmp_path / "model.pt"
    save_checkpoint(path, model, CFG, CLASS_NAMES, history_metrics={})
    loaded_model, _, _, _ = load_checkpoint(path)
    assert not loaded_model.training, "Loaded model should be in eval mode"

    x = torch.randn(2, 3, FAKE_IMAGE_SIZE, FAKE_IMAGE_SIZE)
    with torch.no_grad():
        assert torch.equal(loaded_model(x), model(x)), "Loaded model gives different outputs"

def test_checkpoint_keeps_config_class_names_and_history_metrics(tmp_path):
    cfg = {**CFG, "seed": 42}
    model = make_tiny_model(num_classes=len(CLASS_NAMES))

    path = tmp_path / "model.pt"
    history_metrics = {"train_loss": [0.9, 0.5], "val_acc": [0.4, 0.7]}
    save_checkpoint(path, model, cfg, CLASS_NAMES, history_metrics)
    _, loaded_names, loaded_history, loaded_config = load_checkpoint(path)

    assert loaded_names == CLASS_NAMES, "Class names changed after loading"
    assert loaded_history == history_metrics, "History metrics changed after loading"
    assert loaded_config == cfg, "Config changed after loading"


def test_resnet_checkpoint_loads_without_pretrained_weights(tmp_path, monkeypatch):
    cfg = {"model": {"type": "ResNet18"}, "data": {"image_size": FAKE_IMAGE_SIZE}}
    model = build_resnet18(num_classes=len(CLASS_NAMES), pretrained=False)

    path = tmp_path / "resnet.pt"
    save_checkpoint(path, model, cfg, CLASS_NAMES, history_metrics={})

    # Fail the test if loading the checkpoint asks for ImageNet weights
    expect_resnet18_weights(monkeypatch, None, "Loading a checkpoint should not download ImageNet weights")
    loaded_model, _, _, _ = load_checkpoint(path)

    assert torch.equal(loaded_model.fc.weight, model.fc.weight), "Loaded ResNet18 has different weights"
